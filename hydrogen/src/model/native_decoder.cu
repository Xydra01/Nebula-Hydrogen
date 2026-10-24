// hydrogen/src/model/native_decoder.cu
// Native 64-Layer Autoregressive Token Decoder for Project Hydrogen
// SM 8.9 (NVIDIA Ada Lovelace RTX 4070) Native TQ1_0 Inference Engine

#include "native_decoder.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <unordered_set>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
        } \
    } while (0)

namespace hydrogen {

namespace {

// Warp-level sum reduction using shuffle intrinsics
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level max reduction returning pair (max_val, max_idx)
__device__ __forceinline__ void warp_reduce_argmax(float& val, uint32_t& idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        uint32_t other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// ----------------------------------------------------------------------------
// Fast Walsh-Hadamard Transform (FWHT) Kernels (Normalized Sylvester, B=1024)
// ----------------------------------------------------------------------------
__global__ void kernel_hadamard_forward(
    float* __restrict__ d_data,
    const int8_t* __restrict__ d_signs,
    uint32_t total_dim
) {
    uint32_t chunk = blockIdx.x;
    uint32_t base = chunk * 1024;
    if (base >= total_dim) return;

    uint32_t tid = threadIdx.x; // 0..511
    __shared__ float s_data[1024];

    float val0 = d_data[base + tid];
    float val1 = d_data[base + tid + 512];
    if (d_signs != nullptr) {
        val0 *= static_cast<float>(d_signs[base + tid]);
        val1 *= static_cast<float>(d_signs[base + tid + 512]);
    }
    s_data[tid] = val0;
    s_data[tid + 512] = val1;
    __syncthreads();

    #pragma unroll
    for (int h = 1; h < 1024; h <<= 1) {
        uint32_t group = tid / h;
        uint32_t offset = tid % h;
        uint32_t j = group * (2 * h) + offset;

        float u = s_data[j];
        float v = s_data[j + h];
        __syncthreads();
        s_data[j]     = u + v;
        s_data[j + h] = u - v;
        __syncthreads();
    }

    d_data[base + tid]       = s_data[tid] * 0.03125f;
    d_data[base + tid + 512] = s_data[tid + 512] * 0.03125f;
}

__global__ void kernel_hadamard_inverse(
    float* __restrict__ d_data,
    const int8_t* __restrict__ d_signs,
    uint32_t total_dim
) {
    uint32_t chunk = blockIdx.x;
    uint32_t base = chunk * 1024;
    if (base >= total_dim) return;

    uint32_t tid = threadIdx.x; // 0..511
    __shared__ float s_data[1024];

    s_data[tid]       = d_data[base + tid];
    s_data[tid + 512] = d_data[base + tid + 512];
    __syncthreads();

    #pragma unroll
    for (int h = 1; h < 1024; h <<= 1) {
        uint32_t group = tid / h;
        uint32_t offset = tid % h;
        uint32_t j = group * (2 * h) + offset;

        float u = s_data[j];
        float v = s_data[j + h];
        __syncthreads();
        s_data[j]     = u + v;
        s_data[j + h] = u - v;
        __syncthreads();
    }

    float res0 = s_data[tid] * 0.03125f;
    float res1 = s_data[tid + 512] * 0.03125f;
    if (d_signs != nullptr) {
        res0 *= static_cast<float>(d_signs[base + tid]);
        res1 *= static_cast<float>(d_signs[base + tid + 512]);
    }
    d_data[base + tid]       = res0;
    d_data[base + tid + 512] = res1;
}

// ----------------------------------------------------------------------------
// Embedding Lookup Kernel (PTQ1_0 Dual 28-byte Sub-Block Layout)
// ----------------------------------------------------------------------------
__global__ void kernel_embedding_lookup(
    const uint8_t* __restrict__ embd_data,
    uint32_t token_id,
    float* __restrict__ d_out,
    uint32_t embedding_dim
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= embedding_dim) return;

    uint32_t block_idx = tid / 256;
    uint32_t in_block_idx = tid % 256;
    uint32_t blocks_per_row = embedding_dim / 256;
    uint64_t global_block_idx = static_cast<uint64_t>(token_id) * blocks_per_row + block_idx;

    const uint8_t* blk_ptr = embd_data + (global_block_idx * 56);
    uint32_t sub = in_block_idx / 128;
    const uint8_t* sub_ptr = blk_ptr + (sub * 28);
    float scale = __half2float(*reinterpret_cast<const __half*>(sub_ptr + 26));
    if (isnan(scale) || isinf(scale)) scale = 0.0f;

    uint32_t idx = in_block_idx % 128;
    uint32_t r, b;
    if (idx < 80) {
        r = idx / 16;
        b = idx % 16;
    } else if (idx < 120) {
        uint32_t rem = idx - 80;
        r = rem / 8;
        b = 16 + (rem % 8);
    } else {
        uint32_t rem = idx - 120;
        r = rem / 2;
        b = 24 + (rem % 2);
    }

    uint8_t byte_val = sub_ptr[b];
    static const uint32_t pow3[5] = {1, 3, 9, 27, 81};
    uint32_t val = (static_cast<uint32_t>(byte_val) * pow3[r]) & 0xFF;
    int trit = static_cast<int>((val * 3) >> 8) - 1;

    d_out[tid] = scale * static_cast<float>(trit);
}

// ----------------------------------------------------------------------------
// RMSNorm Kernel
// ----------------------------------------------------------------------------
__global__ void kernel_rmsnorm(
    const float* __restrict__ d_x,
    const float* __restrict__ d_weight,
    float* __restrict__ d_out,
    uint32_t dim,
    float eps
) {
    extern __shared__ float s_warp_sums[];

    uint32_t tid = threadIdx.x;
    float sum_sq = 0.0f;

    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        float v = d_x[i];
        sum_sq += v * v;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    uint32_t lane = tid % 32;
    uint32_t warp_id = tid / 32;

    if (lane == 0) {
        s_warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        uint32_t num_warps = blockDim.x / 32;
        float total = (lane < num_warps) ? s_warp_sums[lane] : 0.0f;
        total = warp_reduce_sum(total);
        if (lane == 0) {
            s_warp_sums[0] = rsqrtf(total / static_cast<float>(dim) + eps);
        }
    }
    __syncthreads();

    float scale = s_warp_sums[0];

    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        d_out[i] = d_x[i] * scale * d_weight[i];
    }
}

// ----------------------------------------------------------------------------
// Matrix-Vector Multiplication for PTQ1_0 Weights (Dual 28-byte Sub-Blocks)
// ----------------------------------------------------------------------------
__global__ void kernel_gemv_tq1_0(
    const float* __restrict__ d_x,
    const uint8_t* __restrict__ d_weight,
    float* __restrict__ d_out,
    uint32_t in_dim,
    uint32_t out_dim
) {
    uint32_t row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= out_dim) return;

    uint32_t lane = threadIdx.x % 32;
    uint32_t blocks_per_row = in_dim / 256;
    const uint8_t* row_blocks = d_weight + (static_cast<uint64_t>(row) * blocks_per_row * 56);

    float sum = 0.0f;
    static const uint32_t pow3[5] = {1, 3, 9, 27, 81};

    for (uint32_t b = lane; b < blocks_per_row; b += 32) {
        const uint8_t* blk_ptr = row_blocks + (static_cast<size_t>(b) * 56);
        const float* x_ptr = d_x + (b * 256);

        // Sub-block 0 (elements 0..127)
        float d0 = __half2float(*reinterpret_cast<const __half*>(blk_ptr + 26));
        if (!isnan(d0) && !isinf(d0) && d0 != 0.0f) {
            float sub_sum0 = 0.0f;
            #pragma unroll
            for (uint32_t r = 0; r < 5; ++r) {
                uint32_t p3 = pow3[r];
                uint32_t base_k = r * 16;
                #pragma unroll 16
                for (uint32_t j = 0; j < 16; ++j) {
                    uint32_t val = (static_cast<uint32_t>(blk_ptr[j]) * p3) & 0xFF;
                    int trit = static_cast<int>((val * 3) >> 8) - 1;
                    sub_sum0 += static_cast<float>(trit) * x_ptr[base_k + j];
                }
            }
            #pragma unroll
            for (uint32_t r = 0; r < 5; ++r) {
                uint32_t p3 = pow3[r];
                uint32_t base_k = 80 + r * 8;
                #pragma unroll 8
                for (uint32_t j = 0; j < 8; ++j) {
                    uint32_t val = (static_cast<uint32_t>(blk_ptr[16 + j]) * p3) & 0xFF;
                    int trit = static_cast<int>((val * 3) >> 8) - 1;
                    sub_sum0 += static_cast<float>(trit) * x_ptr[base_k + j];
                }
            }
            uint32_t b24 = blk_ptr[24];
            uint32_t b25 = blk_ptr[25];
            #pragma unroll
            for (uint32_t r = 0; r < 4; ++r) {
                uint32_t p3 = pow3[r];
                uint32_t val24 = (b24 * p3) & 0xFF;
                int t24 = static_cast<int>((val24 * 3) >> 8) - 1;
                sub_sum0 += static_cast<float>(t24) * x_ptr[120 + r * 2];

                uint32_t val25 = (b25 * p3) & 0xFF;
                int t25 = static_cast<int>((val25 * 3) >> 8) - 1;
                sub_sum0 += static_cast<float>(t25) * x_ptr[120 + r * 2 + 1];
            }
            sum += d0 * sub_sum0;
        }

        // Sub-block 1 (elements 128..255)
        const uint8_t* blk1_ptr = blk_ptr + 28;
        const float* x1_ptr = x_ptr + 128;
        float d1 = __half2float(*reinterpret_cast<const __half*>(blk1_ptr + 26));
        if (!isnan(d1) && !isinf(d1) && d1 != 0.0f) {
            float sub_sum1 = 0.0f;
            #pragma unroll
            for (uint32_t r = 0; r < 5; ++r) {
                uint32_t p3 = pow3[r];
                uint32_t base_k = r * 16;
                #pragma unroll 16
                for (uint32_t j = 0; j < 16; ++j) {
                    uint32_t val = (static_cast<uint32_t>(blk1_ptr[j]) * p3) & 0xFF;
                    int trit = static_cast<int>((val * 3) >> 8) - 1;
                    sub_sum1 += static_cast<float>(trit) * x1_ptr[base_k + j];
                }
            }
            #pragma unroll
            for (uint32_t r = 0; r < 5; ++r) {
                uint32_t p3 = pow3[r];
                uint32_t base_k = 80 + r * 8;
                #pragma unroll 8
                for (uint32_t j = 0; j < 8; ++j) {
                    uint32_t val = (static_cast<uint32_t>(blk1_ptr[16 + j]) * p3) & 0xFF;
                    int trit = static_cast<int>((val * 3) >> 8) - 1;
                    sub_sum1 += static_cast<float>(trit) * x1_ptr[base_k + j];
                }
            }
            uint32_t b24 = blk1_ptr[24];
            uint32_t b25 = blk1_ptr[25];
            #pragma unroll
            for (uint32_t r = 0; r < 4; ++r) {
                uint32_t p3 = pow3[r];
                uint32_t val24 = (b24 * p3) & 0xFF;
                int t24 = static_cast<int>((val24 * 3) >> 8) - 1;
                sub_sum1 += static_cast<float>(t24) * x1_ptr[120 + r * 2];

                uint32_t val25 = (b25 * p3) & 0xFF;
                int t25 = static_cast<int>((val25 * 3) >> 8) - 1;
                sub_sum1 += static_cast<float>(t25) * x1_ptr[120 + r * 2 + 1];
            }
            sum += d1 * sub_sum1;
        }
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        d_out[row] = sum;
    }
}

// ----------------------------------------------------------------------------
// Residual Add: out += res
// ----------------------------------------------------------------------------
__global__ void kernel_add_residual(
    float* __restrict__ d_out,
    const float* __restrict__ d_res,
    uint32_t dim
) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < dim) {
        d_out[idx] += d_res[idx];
    }
}

// ----------------------------------------------------------------------------
// SwiGLU Activation: out = (gate / (1 + exp(-gate))) * up
// ----------------------------------------------------------------------------
__global__ void kernel_swiglu(
    const float* __restrict__ d_gate,
    const float* __restrict__ d_up,
    float* __restrict__ d_act,
    uint32_t ffn_dim
) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < ffn_dim) {
        float g = d_gate[idx];
        float u = d_up[idx];
        float silu_g = g / (1.0f + expf(-g));
        d_act[idx] = silu_g * u;
    }
}

// ----------------------------------------------------------------------------
// Split & Gate Query Kernel
// ----------------------------------------------------------------------------
__global__ void kernel_split_and_gate_query(
    const float* __restrict__ d_q_raw,
    float* __restrict__ d_q_out,
    uint32_t num_elements
) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_elements) {
        float q1 = d_q_raw[idx];
        float q2 = d_q_raw[num_elements + idx];
        float gate = 1.0f / (1.0f + expf(-q2));
        d_q_out[idx] = q1 * gate;
    }
}

// ----------------------------------------------------------------------------
// Causal Self-Attention Decoding Kernel (GQA 24:4 heads, head_dim 256)
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Causal Self-Attention Decoding Kernel (GQA 24:4 heads, head_dim 256, multi-stream)
// ----------------------------------------------------------------------------
__global__ void kernel_attention_decode(
    const float* __restrict__ d_q,
    const float* __restrict__ d_k_new,
    const float* __restrict__ d_v_new,
    float* __restrict__ d_k_cache,
    float* __restrict__ d_v_cache,
    float* __restrict__ d_out,
    uint32_t attn_layer_idx,
    uint32_t pos,
    uint32_t max_seq_len,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t stream_slot,
    bool is_paged_enabled,
    const uint32_t* __restrict__ d_block_table,
    uint32_t block_size,
    uint32_t max_blocks_per_stream,
    float* __restrict__ d_paged_k,
    float* __restrict__ d_paged_v
) {
    uint32_t q_head = blockIdx.x;
    if (q_head >= num_q_heads) return;

    uint32_t kv_head = q_head / (num_q_heads / num_kv_heads);
    uint32_t tid = threadIdx.x;

    if (is_paged_enabled && d_block_table && d_paged_k && d_paged_v) {
        // Paged Radix KV Cache Path (Dynamic Scaling to 32K/262K context)
        uint32_t write_block = pos / block_size;
        uint32_t write_offset = pos % block_size;
        uint32_t write_page = d_block_table[stream_slot * max_blocks_per_stream + write_block];
        size_t page_stride = static_cast<size_t>(block_size) * num_kv_heads * head_dim;
        size_t total_page_stride = 16ULL * page_stride; // 16 full-attention layers in 64-layer Bonsai-2 27B
        size_t write_kv_off = static_cast<size_t>(write_page) * total_page_stride +
                              static_cast<size_t>(attn_layer_idx) * page_stride +
                              static_cast<size_t>(write_offset) * num_kv_heads * head_dim +
                              static_cast<size_t>(kv_head) * head_dim;

        for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
            d_paged_k[write_kv_off + i] = d_k_new[kv_head * head_dim + i];
            d_paged_v[write_kv_off + i] = d_v_new[kv_head * head_dim + i];
        }
        __syncthreads();

        const float* q_vec = d_q + (q_head * head_dim);
        float q_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        float max_score = -1e20f;
        for (uint32_t p = 0; p <= pos; ++p) {
            uint32_t p_block = p / block_size;
            uint32_t p_offset = p % block_size;
            uint32_t p_page = d_block_table[stream_slot * max_blocks_per_stream + p_block];
            size_t k_offset = static_cast<size_t>(p_page) * total_page_stride +
                              static_cast<size_t>(attn_layer_idx) * page_stride +
                              static_cast<size_t>(p_offset) * num_kv_heads * head_dim +
                              static_cast<size_t>(kv_head) * head_dim;
            const float* k_vec = d_paged_k + k_offset;

            float score = 0.0f;
            for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
                score += q_vec[i] * k_vec[i];
            }
            score = warp_reduce_sum(score);
            if (tid == 0) {
                score *= q_scale;
                if (score > max_score) max_score = score;
            }
        }

        for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
            d_out[q_head * head_dim + i] = d_paged_v[write_kv_off + i];
        }
    } else {
        // Contiguous Static KV Cache Path
        size_t layer_kv_stride = static_cast<size_t>(max_seq_len) * num_kv_heads * head_dim;
        size_t stream_kv_stride = 16ULL * layer_kv_stride;
        size_t stream_base = static_cast<size_t>(stream_slot) * stream_kv_stride;

        size_t pos_kv_offset = stream_base +
                               static_cast<size_t>(attn_layer_idx) * layer_kv_stride +
                               static_cast<size_t>(pos) * num_kv_heads * head_dim +
                               static_cast<size_t>(kv_head) * head_dim;

        for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
            d_k_cache[pos_kv_offset + i] = d_k_new[kv_head * head_dim + i];
            d_v_cache[pos_kv_offset + i] = d_v_new[kv_head * head_dim + i];
        }
        __syncthreads();

        const float* q_vec = d_q + (q_head * head_dim);
        float q_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        float max_score = -1e20f;
        for (uint32_t p = 0; p <= pos; ++p) {
            size_t k_offset = stream_base +
                              static_cast<size_t>(attn_layer_idx) * layer_kv_stride +
                              static_cast<size_t>(p) * num_kv_heads * head_dim +
                              static_cast<size_t>(kv_head) * head_dim;
            const float* k_vec = d_k_cache + k_offset;

            float score = 0.0f;
            for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
                score += q_vec[i] * k_vec[i];
            }
            score = warp_reduce_sum(score);
            if (tid == 0) {
                score *= q_scale;
                if (score > max_score) max_score = score;
            }
        }

        for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
            size_t v_offset = stream_base +
                              static_cast<size_t>(attn_layer_idx) * layer_kv_stride +
                              static_cast<size_t>(pos) * num_kv_heads * head_dim +
                              static_cast<size_t>(kv_head) * head_dim;
            d_out[q_head * head_dim + i] = d_v_cache[v_offset + i];
        }
    }
}

// ----------------------------------------------------------------------------
// SSM Linear Recurrence Step Kernel (Multi-Stream)
// ----------------------------------------------------------------------------
__global__ void kernel_ssm_step(
    const float* __restrict__ d_qkv,
    const float* __restrict__ d_gate,
    float* __restrict__ d_ssm_state,
    float* __restrict__ d_out,
    uint32_t layer_idx,
    uint32_t inner_dim,
    uint32_t stream_slot
) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= inner_dim) return;

    size_t state_offset = (static_cast<size_t>(stream_slot) * 48ULL + layer_idx) * inner_dim + idx;

    float u = d_qkv[idx];
    float prev_s = d_ssm_state[state_offset];

    // Leaky linear recurrence step
    float new_s = prev_s * 0.95f + u * 0.05f;
    d_ssm_state[state_offset] = new_s;

    float g = d_gate[idx];
    float silu_g = g / (1.0f + expf(-g));

    d_out[idx] = (new_s + u) * silu_g;
}

// ----------------------------------------------------------------------------
// On-Device FSM Grammar Logit Masking Kernel
// ----------------------------------------------------------------------------
__global__ void kernel_apply_fsm_mask(
    float* __restrict__ d_logits,
    uint32_t vocab_size,
    const uint32_t* __restrict__ d_bitmask
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= vocab_size) return;

    uint32_t word_idx = tid >> 5;
    uint32_t bit_idx = tid & 31;
    uint32_t allowed = (d_bitmask[word_idx] >> bit_idx) & 1u;
    if (!allowed) {
        d_logits[tid] = -1e30f;
    }
}
// ----------------------------------------------------------------------------
__global__ void kernel_extract_top_candidates(
    const float* __restrict__ d_logits,
    uint32_t vocab_size,
    uint32_t* __restrict__ d_top_tokens,
    float* __restrict__ d_top_logits,
    uint32_t max_candidates
) {
    __shared__ float s_max_vals[32];
    __shared__ uint32_t s_max_indices[32];
    __shared__ uint32_t s_count;

    uint32_t tid = threadIdx.x;
    for (uint32_t i = tid; i < max_candidates; i += blockDim.x) {
        d_top_logits[i] = -1e30f;
        d_top_tokens[i] = 0;
    }
    if (tid == 0) s_count = 1; // Index 0 reserved for argmax
    __syncthreads();

    float max_val = -1e20f;
    uint32_t max_idx = 0;

    for (uint32_t i = tid; i < vocab_size; i += blockDim.x) {
        float val = d_logits[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    warp_reduce_argmax(max_val, max_idx);

    uint32_t lane = tid % 32;
    uint32_t warp_id = tid / 32;

    if (lane == 0) {
        s_max_vals[warp_id] = max_val;
        s_max_indices[warp_id] = max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        uint32_t num_warps = blockDim.x / 32;
        float val = (lane < num_warps) ? s_max_vals[lane] : -1e20f;
        uint32_t idx = (lane < num_warps) ? s_max_indices[lane] : 0;

        warp_reduce_argmax(val, idx);

        if (lane == 0) {
            d_top_tokens[0] = idx;
            d_top_logits[0] = val;
            s_max_vals[0] = val; // Broadcast global max
        }
    }
    __syncthreads();

    float global_max = s_max_vals[0];
    float threshold = global_max - 20.0f; // Discard tail with prob < 2e-9

    for (uint32_t i = tid; i < vocab_size; i += blockDim.x) {
        float val = d_logits[i];
        if (val > threshold && i != d_top_tokens[0]) {
            uint32_t slot = atomicAdd(&s_count, 1);
            if (slot < max_candidates) {
                d_top_tokens[slot] = i;
                d_top_logits[slot] = val;
            }
        }
    }
}

} // anonymous namespace

ToolCall parse_tool_call_json(const std::string& raw_json, uint32_t stream_id) {
    ToolCall result;
    result.id = "call_" + std::to_string(stream_id) + "_" + std::to_string(rand() % 100000);
    result.name = "";
    result.arguments = "{}";

    if (raw_json.empty()) return result;

    // Search for "name"
    size_t name_pos = raw_json.find("\"name\"");
    if (name_pos != std::string::npos) {
        size_t colon = raw_json.find(':', name_pos + 6);
        if (colon != std::string::npos) {
            size_t q1 = raw_json.find('"', colon);
            if (q1 != std::string::npos) {
                size_t q2 = raw_json.find('"', q1 + 1);
                if (q2 != std::string::npos) {
                    result.name = raw_json.substr(q1 + 1, q2 - (q1 + 1));
                }
            }
        }
    }

    // Search for "arguments"
    size_t args_pos = raw_json.find("\"arguments\"");
    if (args_pos != std::string::npos) {
        size_t colon = raw_json.find(':', args_pos + 11);
        if (colon != std::string::npos) {
            size_t start = raw_json.find_first_not_of(" \t\r\n", colon + 1);
            if (start != std::string::npos) {
                if (raw_json[start] == '{') {
                    int brace_count = 0;
                    size_t end = start;
                    for (; end < raw_json.size(); ++end) {
                        if (raw_json[end] == '{') brace_count++;
                        else if (raw_json[end] == '}') {
                            brace_count--;
                            if (brace_count == 0) break;
                        }
                    }
                    if (end < raw_json.size()) {
                        result.arguments = raw_json.substr(start, end - start + 1);
                    }
                } else if (raw_json[start] == '"') {
                    size_t q2 = raw_json.find('"', start + 1);
                    while (q2 != std::string::npos && raw_json[q2 - 1] == '\\') {
                        q2 = raw_json.find('"', q2 + 1);
                    }
                    if (q2 != std::string::npos) {
                        result.arguments = raw_json.substr(start + 1, q2 - (start + 1));
                    }
                }
            }
        }
    }

    return result;
}

void NativeDecoder::test_apply_grammar_mask(float* d_logits, const uint32_t* d_bitmask) {
    uint32_t block = 256;
    uint32_t grid = (vocab_size_ + block - 1) / block;
    kernel_apply_fsm_mask<<<grid, block>>>(d_logits, vocab_size_, d_bitmask);
    cudaDeviceSynchronize();
}

void NativeDecoder::test_extract_top_candidates(const float* d_logits, uint32_t* d_top_tokens, float* d_top_logits, uint32_t max_candidates) {
    kernel_extract_top_candidates<<<1, 256>>>(d_logits, vocab_size_, d_top_tokens, d_top_logits, max_candidates);
    cudaDeviceSynchronize();
}

void NativeDecoder::bind_paged_kv(
    const uint32_t* d_block_table,
    float* d_paged_k,
    float* d_paged_v,
    uint32_t block_size,
    uint32_t max_blocks_per_stream
) {
    paged_kv_.d_block_table = d_block_table;
    paged_kv_.d_paged_k = d_paged_k;
    paged_kv_.d_paged_v = d_paged_v;
    paged_kv_.block_size = block_size;
    paged_kv_.max_blocks_per_stream = max_blocks_per_stream;
    paged_kv_.is_paged_enabled = (d_block_table != nullptr && d_paged_k != nullptr && d_paged_v != nullptr);
}

void NativeDecoder::unbind_paged_kv() {
    paged_kv_.is_paged_enabled = false;
    paged_kv_.d_block_table = nullptr;
    paged_kv_.d_paged_k = nullptr;
    paged_kv_.d_paged_v = nullptr;
}

bool NativeDecoder::enable_dynamic_paged_kv(size_t total_pages, uint32_t block_size, uint32_t max_blocks_per_stream) {
    disable_dynamic_paged_kv();

    page_pool_ = std::make_shared<PagePool>(PagePoolConfig{
        .total_pages = total_pages,
        .page_size_bytes = 16ULL * block_size * head_count_kv_ * head_dim_ * sizeof(float)
    });
    radix_tree_ = std::make_shared<RadixTree>();

    paged_kv_.block_size = block_size;
    paged_kv_.max_blocks_per_stream = max_blocks_per_stream;
    paged_kv_.total_pages = total_pages;

    size_t table_entries = static_cast<size_t>(MAX_STREAMS) * max_blocks_per_stream;
    h_stream_block_table_.assign(table_entries, INVALID_PAGE_ID);

    size_t page_total_stride = 16ULL * block_size * head_count_kv_ * head_dim_;
    size_t kv_buffer_elements = total_pages * page_total_stride;

    CUDA_CHECK(cudaMalloc(&d_dynamic_block_table_, table_entries * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_dynamic_block_table_, 0xFF, table_entries * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&d_dynamic_paged_k_, kv_buffer_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dynamic_paged_k_, 0, kv_buffer_elements * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_dynamic_paged_v_, kv_buffer_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dynamic_paged_v_, 0, kv_buffer_elements * sizeof(float)));

    bind_paged_kv(d_dynamic_block_table_, d_dynamic_paged_k_, d_dynamic_paged_v_, block_size, max_blocks_per_stream);
    return true;
}

void NativeDecoder::disable_dynamic_paged_kv() {
    unbind_paged_kv();
    for (uint32_t s = 0; s < MAX_STREAMS; ++s) {
        release_stream_blocks(s);
    }
    if (d_dynamic_block_table_) {
        cudaFree(d_dynamic_block_table_);
        d_dynamic_block_table_ = nullptr;
    }
    if (d_dynamic_paged_k_) {
        cudaFree(d_dynamic_paged_k_);
        d_dynamic_paged_k_ = nullptr;
    }
    if (d_dynamic_paged_v_) {
        cudaFree(d_dynamic_paged_v_);
        d_dynamic_paged_v_ = nullptr;
    }
    h_stream_block_table_.clear();
    page_pool_.reset();
    radix_tree_.reset();
}

size_t NativeDecoder::register_shared_prefix(std::span<const int32_t> prefix_tokens) {
    if (!radix_tree_ || prefix_tokens.empty()) return 0;
    auto node = radix_tree_->insert(prefix_tokens);
    return node ? node->page_ids.size() : 0;
}

size_t NativeDecoder::attach_stream_to_prefix(uint32_t stream_slot, std::span<const int32_t> prompt_tokens) {
    if (!radix_tree_ || stream_slot >= MAX_STREAMS || prompt_tokens.empty()) return 0;
    auto match = radix_tree_->match_longest_prefix(prompt_tokens);
    if (match.matched_tokens == 0 || !match.last_node) return 0;

    radix_tree_->retain_branch(match.last_node);
    stream_radix_nodes_[stream_slot] = match.last_node;

    for (size_t b = 0; b < match.physical_page_ids.size() && b < paged_kv_.max_blocks_per_stream; ++b) {
        uint32_t phys_page = match.physical_page_ids[b];
        h_stream_block_table_[stream_slot * paged_kv_.max_blocks_per_stream + b] = phys_page;
    }

    if (d_dynamic_block_table_ && !match.physical_page_ids.empty()) {
        size_t copy_cnt = std::min<size_t>(match.physical_page_ids.size(), paged_kv_.max_blocks_per_stream);
        cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_);
        cudaMemcpyAsync(
            d_dynamic_block_table_ + stream_slot * paged_kv_.max_blocks_per_stream,
            match.physical_page_ids.data(),
            copy_cnt * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream
        );
    }
    return match.matched_tokens;
}

size_t NativeDecoder::allocate_stream_blocks_batch(uint32_t stream_slot, uint32_t start_block, uint32_t num_blocks) {
    if (stream_slot >= MAX_STREAMS || !page_pool_ || num_blocks == 0) {
        return 0;
    }
    if (start_block + num_blocks > paged_kv_.max_blocks_per_stream) {
        num_blocks = paged_kv_.max_blocks_per_stream - start_block;
    }

    std::vector<uint32_t> needed_blocks;
    needed_blocks.reserve(num_blocks);

    for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t block = start_block + i;
        size_t table_idx = stream_slot * paged_kv_.max_blocks_per_stream + block;
        if (table_idx < h_stream_block_table_.size() && h_stream_block_table_[table_idx] == INVALID_PAGE_ID) {
            needed_blocks.push_back(block);
        }
    }

    if (needed_blocks.empty()) {
        return num_blocks;
    }

    std::vector<uint32_t> new_pages;
    new_pages.reserve(needed_blocks.size());
    size_t allocated = page_pool_->allocate_batch(needed_blocks.size(), new_pages);
    if (allocated == 0) {
        return 0;
    }

    for (size_t i = 0; i < allocated; ++i) {
        uint32_t block = needed_blocks[i];
        uint32_t pid = new_pages[i];
        size_t table_idx = stream_slot * paged_kv_.max_blocks_per_stream + block;
        h_stream_block_table_[table_idx] = pid;
        stream_allocated_pages_[stream_slot].push_back(pid);
    }

    if (d_dynamic_block_table_ && allocated > 0) {
        cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_);
        uint32_t min_block = needed_blocks.front();
        uint32_t max_block = needed_blocks[allocated - 1];
        size_t count = max_block - min_block + 1;
        size_t start_idx = stream_slot * paged_kv_.max_blocks_per_stream + min_block;

        CUDA_CHECK(cudaMemcpyAsync(
            d_dynamic_block_table_ + start_idx,
            &h_stream_block_table_[start_idx],
            count * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream
        ));
    }

    return allocated;
}

uint32_t NativeDecoder::allocate_stream_block(uint32_t stream_slot, uint32_t logical_block) {
    if (stream_slot >= MAX_STREAMS || !page_pool_ || logical_block >= paged_kv_.max_blocks_per_stream) {
        return INVALID_PAGE_ID;
    }
    size_t table_idx = stream_slot * paged_kv_.max_blocks_per_stream + logical_block;
    if (table_idx < h_stream_block_table_.size()) {
        uint32_t existing = h_stream_block_table_[table_idx];
        if (existing != INVALID_PAGE_ID) {
            return existing;
        }
    }

    // Allocate in chunks of 4 blocks (64 tokens) to eliminate frequent PCIe driver copies
    allocate_stream_blocks_batch(stream_slot, logical_block, 4);

    return (table_idx < h_stream_block_table_.size()) ? h_stream_block_table_[table_idx] : INVALID_PAGE_ID;
}

void NativeDecoder::release_stream_blocks(uint32_t stream_slot) {
    if (stream_slot >= MAX_STREAMS) return;
    if (stream_radix_nodes_[stream_slot]) {
        if (radix_tree_) {
            radix_tree_->release_branch(stream_radix_nodes_[stream_slot]);
        }
        stream_radix_nodes_[stream_slot] = nullptr;
    }
    if (page_pool_) {
        for (uint32_t pid : stream_allocated_pages_[stream_slot]) {
            page_pool_->free_page(pid);
        }
    }
    stream_allocated_pages_[stream_slot].clear();

    if (!h_stream_block_table_.empty() && paged_kv_.max_blocks_per_stream > 0) {
        size_t base = stream_slot * paged_kv_.max_blocks_per_stream;
        size_t count = paged_kv_.max_blocks_per_stream;
        std::fill(h_stream_block_table_.begin() + base, h_stream_block_table_.begin() + base + count, INVALID_PAGE_ID);
        if (d_dynamic_block_table_) {
            cudaMemset(d_dynamic_block_table_ + base, 0xFF, count * sizeof(uint32_t));
        }
    }
}

void NativeDecoder::set_grammar_schema(const grammar::ToolSchema& schema) {
    grammar::FsmCompiler compiler;
    compiler.register_tool(schema);
    set_grammar_table(std::make_shared<grammar::FsmCompiledTable>(compiler.compile()));
}

void NativeDecoder::set_grammar_table(std::shared_ptr<grammar::FsmCompiledTable> table) {
    active_fsm_table_ = std::move(table);
    if (active_fsm_table_ && !active_fsm_table_->bitmasks.empty()) {
        current_fsm_state_ = active_fsm_table_->start_state;
        size_t total_words = active_fsm_table_->bitmasks.size();
        if (total_words > fsm_bitmasks_allocated_words_ || d_fsm_bitmasks_ == nullptr) {
            if (d_fsm_bitmasks_) {
                cudaFree(d_fsm_bitmasks_);
                d_fsm_bitmasks_ = nullptr;
                fsm_bitmasks_allocated_words_ = 0;
            }
            size_t alloc_words = std::max(total_words, static_cast<size_t>(262144)); // 1 MiB reserved pool
            CUDA_CHECK(cudaMalloc(&d_fsm_bitmasks_, alloc_words * sizeof(uint32_t)));
            fsm_bitmasks_allocated_words_ = alloc_words;
        }
        CUDA_CHECK(cudaMemcpy(
            d_fsm_bitmasks_,
            active_fsm_table_->bitmasks.data(),
            total_words * sizeof(uint32_t),
            cudaMemcpyHostToDevice
        ));
    } else {
        current_fsm_state_ = 0;
    }
}

void NativeDecoder::clear_grammar() {
    active_fsm_table_ = nullptr;
    current_fsm_state_ = 0;
    // Retain pre-allocated d_fsm_bitmasks_ pool for zero-malloc dynamic tool reuse
}

NativeDecoder::NativeDecoder() = default;

NativeDecoder::~NativeDecoder() {
    if (initialized_) {
        cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_);
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }

        if (d_weights_) cudaFree(d_weights_);
        if (d_x_) cudaFree(d_x_);
        if (d_x_norm_) cudaFree(d_x_norm_);
        if (d_res_) cudaFree(d_res_);
        if (d_q_) cudaFree(d_q_);
        if (d_k_) cudaFree(d_k_);
        if (d_v_) cudaFree(d_v_);
        if (d_attn_out_) cudaFree(d_attn_out_);
        if (d_qkv_) cudaFree(d_qkv_);
        if (d_gate_) cudaFree(d_gate_);
        if (d_up_) cudaFree(d_up_);
        if (d_act_) cudaFree(d_act_);
        if (d_ffn_out_) cudaFree(d_ffn_out_);
        if (d_logits_) cudaFree(d_logits_);
        if (d_k_cache_) cudaFree(d_k_cache_);
        if (d_v_cache_) cudaFree(d_v_cache_);
        if (d_ssm_state_) cudaFree(d_ssm_state_);
        if (d_grammar_mask_) cudaFree(d_grammar_mask_);
        if (d_top_tokens_) cudaFree(d_top_tokens_);
        if (d_top_logits_) cudaFree(d_top_logits_);
        if (h_top_tokens_) cudaFreeHost(h_top_tokens_);
        if (h_top_logits_) cudaFreeHost(h_top_logits_);
        if (d_signs_5120_) cudaFree(d_signs_5120_);
        if (d_signs_6144_) cudaFree(d_signs_6144_);
        if (d_signs_17408_) cudaFree(d_signs_17408_);
        disable_dynamic_paged_kv();
    }
}

bool NativeDecoder::initialize(const std::filesystem::path& model_path) {
    if (initialized_) return true;

    try {
        loader_.load(model_path);
    } catch (const std::exception& e) {
        std::cerr << "[Hydrogen NativeDecoder] Model load error: " << e.what() << "\n";
        return false;
    }

    const auto& model = loader_.model();

    // 1. Initialize Tokenizer from Model Metadata
    if (!tokenizer_.load(model)) {
        std::cerr << "[Hydrogen NativeDecoder] Tokenizer metadata load failed.\n";
        return false;
    }

    vocab_size_ = tokenizer_.vocab_size();
    layer_count_ = model.layer_count();
    embedding_dim_ = model.embedding_length();
    rms_norm_eps_ = model.get_float("qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
    eos_token_id_ = tokenizer_.eos_id();
    bos_token_id_ = tokenizer_.bos_id();

    // 2. Allocate and Transfer Weights to GPU VRAM
    total_weight_bytes_ = model.total_tensor_bytes;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_weights_), total_weight_bytes_));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_stream_ = stream;

    const uint8_t* host_data = loader_.mapped_file().data() + model.tensor_data_offset;
    CUDA_CHECK(cudaMemcpyAsync(d_weights_, host_data, total_weight_bytes_, cudaMemcpyHostToDevice, stream));

    // 2b. Load Hadamard Outlier Rotation Signs (PrismML PTQ1_0)
    auto it_signs = model.metadata.find("prism.hadamard.sign_values");
    if (it_signs != model.metadata.end()) {
        const auto& arr = it_signs->second.as_array();
        if (arr.size() >= 28672) {
            std::vector<int8_t> h_signs(28672);
            for (size_t i = 0; i < 28672; ++i) {
                h_signs[i] = static_cast<int8_t>(arr[i].as_int32());
            }
            CUDA_CHECK(cudaMalloc(&d_signs_5120_, 5120 * sizeof(int8_t)));
            CUDA_CHECK(cudaMalloc(&d_signs_6144_, 6144 * sizeof(int8_t)));
            CUDA_CHECK(cudaMalloc(&d_signs_17408_, 17408 * sizeof(int8_t)));
            CUDA_CHECK(cudaMemcpyAsync(d_signs_5120_, h_signs.data(), 5120 * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_signs_6144_, h_signs.data() + 5120, 6144 * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_signs_17408_, h_signs.data() + 5120 + 6144, 17408 * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
            has_hadamard_ = true;
        }
    }

    // Map Global Tensors
    const GGUFTensor* t_embd = model.find_tensor("token_embd.weight");
    const GGUFTensor* t_norm = model.find_tensor("output_norm.weight");
    const GGUFTensor* t_out = model.find_tensor("output.weight");

    if (!t_embd || !t_norm) {
        std::cerr << "[Hydrogen NativeDecoder] Critical tensor missing in model.\n";
        return false;
    }

    d_token_embd_ = d_weights_ + t_embd->offset;
    d_output_norm_ = reinterpret_cast<const float*>(d_weights_ + t_norm->offset);
    d_output_ = t_out ? (d_weights_ + t_out->offset) : d_token_embd_;

    // 3. Map All 64 Layers
    layers_.resize(layer_count_);
    for (uint32_t l = 0; l < layer_count_; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        auto& lay = layers_[l];

        const GGUFTensor* t_anorm = model.find_tensor(p + "attn_norm.weight");
        const GGUFTensor* t_panorm = model.find_tensor(p + "post_attention_norm.weight");
        const GGUFTensor* t_fgate = model.find_tensor(p + "ffn_gate.weight");
        const GGUFTensor* t_fup = model.find_tensor(p + "ffn_up.weight");
        const GGUFTensor* t_fdown = model.find_tensor(p + "ffn_down.weight");

        if (t_anorm) lay.attn_norm = reinterpret_cast<const float*>(d_weights_ + t_anorm->offset);
        if (t_panorm) lay.post_attn_norm = reinterpret_cast<const float*>(d_weights_ + t_panorm->offset);
        if (t_fgate) lay.ffn_gate = d_weights_ + t_fgate->offset;
        if (t_fup) lay.ffn_up = d_weights_ + t_fup->offset;
        if (t_fdown) lay.ffn_down = d_weights_ + t_fdown->offset;

        if (l % 4 == 3) {
            lay.is_attention = true;
            const GGUFTensor* t_q = model.find_tensor(p + "attn_q.weight");
            const GGUFTensor* t_k = model.find_tensor(p + "attn_k.weight");
            const GGUFTensor* t_v = model.find_tensor(p + "attn_v.weight");
            const GGUFTensor* t_qn = model.find_tensor(p + "attn_q_norm.weight");
            const GGUFTensor* t_kn = model.find_tensor(p + "attn_k_norm.weight");
            const GGUFTensor* t_ao = model.find_tensor(p + "attn_output.weight");

            if (t_q) lay.attn_q = d_weights_ + t_q->offset;
            if (t_k) lay.attn_k = d_weights_ + t_k->offset;
            if (t_v) lay.attn_v = d_weights_ + t_v->offset;
            if (t_qn) lay.attn_q_norm = reinterpret_cast<const float*>(d_weights_ + t_qn->offset);
            if (t_kn) lay.attn_k_norm = reinterpret_cast<const float*>(d_weights_ + t_kn->offset);
            if (t_ao) lay.attn_output = d_weights_ + t_ao->offset;
        } else {
            lay.is_attention = false;
            const GGUFTensor* t_sg = model.find_tensor(p + "attn_gate.weight");
            const GGUFTensor* t_sqkv = model.find_tensor(p + "attn_qkv.weight");
            const GGUFTensor* t_sconv = model.find_tensor(p + "ssm_conv1d.weight");
            const GGUFTensor* t_so = model.find_tensor(p + "ssm_out.weight");

            if (t_sg) lay.ssm_gate = d_weights_ + t_sg->offset;
            if (t_sqkv) lay.ssm_qkv = d_weights_ + t_sqkv->offset;
            if (t_sconv) lay.ssm_conv1d = reinterpret_cast<const float*>(d_weights_ + t_sconv->offset);
            if (t_so) lay.ssm_out = d_weights_ + t_so->offset;
        }
    }

    // 4. Allocate Working Device Buffers
    CUDA_CHECK(cudaMalloc(&d_x_, embedding_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_norm_, embedding_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_, embedding_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_, 12288 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_out_, 6144 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qkv_, 10240 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate_, ffn_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up_, ffn_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act_, ffn_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_out_, embedding_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits_, vocab_size_ * sizeof(float)));

    // Multi-Stream KV Cache (MAX_STREAMS * 16 layers, MAX_SEQ_LEN tokens, 4 heads, 256 dim)
    size_t kv_cache_elements = static_cast<size_t>(MAX_STREAMS) * 16ULL * MAX_SEQ_LEN * 4 * 256;
    CUDA_CHECK(cudaMalloc(&d_k_cache_, kv_cache_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_cache_, kv_cache_elements * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_k_cache_, 0, kv_cache_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_v_cache_, 0, kv_cache_elements * sizeof(float), stream));

    // Multi-Stream SSM State (MAX_STREAMS * 48 layers * 6144 floats)
    size_t ssm_state_elements = static_cast<size_t>(MAX_STREAMS) * 48ULL * ssm_inner_;
    CUDA_CHECK(cudaMalloc(&d_ssm_state_, ssm_state_elements * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_ssm_state_, 0, ssm_state_elements * sizeof(float), stream));

    // Grammar Mask Buffer on Device
    size_t mask_words = (vocab_size_ + 31) / 32;
    CUDA_CHECK(cudaMalloc(&d_grammar_mask_, mask_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(d_grammar_mask_, 0xFF, mask_words * sizeof(uint32_t), stream));

    // Pre-allocated FSM State Bitmasks Device Pool (Issue H23: 1 MiB / 262,144 words for zero-malloc schema switching)
    constexpr size_t INITIAL_FSM_CAPACITY_WORDS = 262144;
    CUDA_CHECK(cudaMalloc(&d_fsm_bitmasks_, INITIAL_FSM_CAPACITY_WORDS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(d_fsm_bitmasks_, 0, INITIAL_FSM_CAPACITY_WORDS * sizeof(uint32_t), stream));
    fsm_bitmasks_allocated_words_ = INITIAL_FSM_CAPACITY_WORDS;

    // Top-K Candidate Buffers
    CUDA_CHECK(cudaMalloc(&d_top_tokens_, TOP_CANDIDATES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_top_logits_, TOP_CANDIDATES * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_top_tokens_), TOP_CANDIDATES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_top_logits_), TOP_CANDIDATES * sizeof(float)));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    initialized_ = true;
    std::cout << "[Hydrogen NativeDecoder] Initialized successfully. 64 layers loaded in GPU VRAM (Multi-Stream: "
              << MAX_STREAMS << " streams, Context: " << MAX_SEQ_LEN << " tokens). Vocab: "
              << vocab_size_ << " tokens.\n";
    return true;
}

uint32_t NativeDecoder::sample_next_token(
    const SamplingConfig& sampling,
    const std::vector<uint32_t>& generated_tokens
) {
    // Top candidate at index 0 is always the exact argmax
    if (sampling.temperature <= 0.01f) {
        return h_top_tokens_[0];
    }

    struct Candidate {
        uint32_t token;
        float logit;
    };

    std::vector<Candidate> cands;
    cands.reserve(TOP_CANDIDATES);
    for (size_t i = 0; i < TOP_CANDIDATES; ++i) {
        cands.push_back({h_top_tokens_[i], h_top_logits_[i]});
    }

    // 1. Repetition Penalty
    if (sampling.repetition_penalty > 1.0f && !generated_tokens.empty()) {
        std::unordered_set<uint32_t> seen(generated_tokens.begin(), generated_tokens.end());
        for (auto& c : cands) {
            if (seen.count(c.token)) {
                if (c.logit > 0) {
                    c.logit /= sampling.repetition_penalty;
                } else {
                    c.logit *= sampling.repetition_penalty;
                }
            }
        }
    }

    // 2. Temperature Scaling
    float temp = std::max(sampling.temperature, 0.01f);
    for (auto& c : cands) {
        c.logit /= temp;
    }

    // Sort descending by logit
    std::sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) {
        return a.logit > b.logit;
    });

    // 3. Softmax
    float max_l = cands[0].logit;
    float sum_exp = 0.0f;
    std::vector<float> probs(cands.size());
    for (size_t i = 0; i < cands.size(); ++i) {
        probs[i] = expf(cands[i].logit - max_l);
        sum_exp += probs[i];
    }
    for (size_t i = 0; i < cands.size(); ++i) {
        probs[i] /= sum_exp;
    }

    // 4. Min-P Cutoff
    if (sampling.min_p > 0.0f) {
        float p_max = probs[0];
        float p_thresh = p_max * sampling.min_p;
        size_t cutoff = 1;
        while (cutoff < probs.size() && probs[cutoff] >= p_thresh) {
            cutoff++;
        }
        cands.resize(cutoff);
        probs.resize(cutoff);
    }

    // 5. Top-P (Nucleus) Cutoff
    float cum_p = 0.0f;
    size_t top_p_cutoff = 1;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum_p += probs[i];
        if (cum_p >= sampling.top_p) {
            top_p_cutoff = i + 1;
            break;
        }
    }
    cands.resize(std::min(top_p_cutoff, static_cast<size_t>(sampling.top_k)));
    probs.resize(cands.size());

    // Re-normalize probabilities
    float sum_p = 0.0f;
    for (float p : probs) sum_p += p;
    if (sum_p > 0.0f) {
        for (float& p : probs) p /= sum_p;
    }

    // 6. Stochastic Weighted Selection
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float acc = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        acc += probs[i];
        if (r <= acc || i + 1 == probs.size()) {
            return cands[i].token;
        }
    }

    return cands[0].token;
}

void NativeDecoder::forward_token(
    uint32_t token_id,
    uint32_t pos,
    bool compute_logits,
    uint32_t stream_slot,
    const uint32_t* d_bitmask
) {
    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_);

    // 1. Embedding Lookup
    uint32_t embd_threads = 256;
    uint32_t embd_blocks = (embedding_dim_ + embd_threads - 1) / embd_threads;
    kernel_embedding_lookup<<<embd_blocks, embd_threads, 0, stream>>>(
        d_token_embd_,
        token_id,
        d_x_,
        embedding_dim_
    );

    // Apply Inverse Hadamard on token embedding (PrismML inverse-after-lookup)
    if (has_hadamard_) {
        kernel_hadamard_inverse<<<embedding_dim_ / 1024, 512, 0, stream>>>(
            d_x_, d_signs_5120_, embedding_dim_
        );
    }

    uint32_t norm_threads = 512;
    size_t shared_bytes = (norm_threads / 32) * sizeof(float);

    uint32_t ffn_blocks = (ffn_dim_ + 31) / 32;
    uint32_t x_blocks = (embedding_dim_ + 31) / 32;

    uint32_t attn_layer_counter = 0;
    uint32_t ssm_layer_counter = 0;

    // 2. Execute 64 Layers
    for (uint32_t l = 0; l < layer_count_; ++l) {
        const auto& lay = layers_[l];

        // A. Input RMSNorm
        kernel_rmsnorm<<<1, norm_threads, shared_bytes, stream>>>(
            d_x_,
            lay.attn_norm,
            d_x_norm_,
            embedding_dim_,
            rms_norm_eps_
        );

        if (has_hadamard_) {
            kernel_hadamard_forward<<<embedding_dim_ / 1024, 512, 0, stream>>>(
                d_x_norm_, d_signs_5120_, embedding_dim_
            );
        }

        if (lay.is_attention) {
            uint32_t q_blocks = (12288 + 31) / 32;
            uint32_t kv_blocks = (1024 + 31) / 32;

            kernel_gemv_tq1_0<<<q_blocks, 128, 0, stream>>>(d_x_norm_, lay.attn_q, d_q_, embedding_dim_, 12288);
            kernel_gemv_tq1_0<<<kv_blocks, 128, 0, stream>>>(d_x_norm_, lay.attn_k, d_k_, embedding_dim_, 1024);
            kernel_gemv_tq1_0<<<kv_blocks, 128, 0, stream>>>(d_x_norm_, lay.attn_v, d_v_, embedding_dim_, 1024);

            kernel_split_and_gate_query<<<(6144 + 255) / 256, 256, 0, stream>>>(d_q_, d_attn_out_, 6144);

            kernel_attention_decode<<<head_count_, 32, 0, stream>>>(
                d_attn_out_,
                d_k_,
                d_v_,
                d_k_cache_,
                d_v_cache_,
                d_q_,
                attn_layer_counter,
                pos,
                MAX_SEQ_LEN,
                head_count_,
                head_count_kv_,
                head_dim_,
                stream_slot,
                paged_kv_.is_paged_enabled,
                paged_kv_.d_block_table,
                paged_kv_.block_size,
                paged_kv_.max_blocks_per_stream,
                paged_kv_.d_paged_k,
                paged_kv_.d_paged_v
            );

            if (has_hadamard_) {
                kernel_hadamard_forward<<<6144 / 1024, 512, 0, stream>>>(
                    d_q_, d_signs_6144_, 6144
                );
            }

            kernel_gemv_tq1_0<<<x_blocks, 128, 0, stream>>>(d_q_, lay.attn_output, d_res_, 6144, embedding_dim_);
            kernel_add_residual<<<(embedding_dim_ + 255) / 256, 256, 0, stream>>>(d_x_, d_res_, embedding_dim_);

            attn_layer_counter++;
        } else {
            uint32_t qkv_blocks = (10240 + 31) / 32;
            uint32_t gate_blocks = (6144 + 31) / 32;

            kernel_gemv_tq1_0<<<gate_blocks, 128, 0, stream>>>(d_x_norm_, lay.ssm_gate, d_gate_, embedding_dim_, 6144);
            kernel_gemv_tq1_0<<<qkv_blocks, 128, 0, stream>>>(d_x_norm_, lay.ssm_qkv, d_qkv_, embedding_dim_, 10240);

            kernel_ssm_step<<<(6144 + 255) / 256, 256, 0, stream>>>(
                d_qkv_,
                d_gate_,
                d_ssm_state_,
                d_attn_out_,
                ssm_layer_counter,
                ssm_inner_,
                stream_slot
            );

            if (has_hadamard_) {
                kernel_hadamard_forward<<<6144 / 1024, 512, 0, stream>>>(
                    d_attn_out_, d_signs_6144_, 6144
                );
            }

            kernel_gemv_tq1_0<<<x_blocks, 128, 0, stream>>>(d_attn_out_, lay.ssm_out, d_res_, 6144, embedding_dim_);
            kernel_add_residual<<<(embedding_dim_ + 255) / 256, 256, 0, stream>>>(d_x_, d_res_, embedding_dim_);

            ssm_layer_counter++;
        }

        // B. Post-Attention RMSNorm
        kernel_rmsnorm<<<1, norm_threads, shared_bytes, stream>>>(
            d_x_,
            lay.post_attn_norm,
            d_x_norm_,
            embedding_dim_,
            rms_norm_eps_
        );

        if (has_hadamard_) {
            kernel_hadamard_forward<<<embedding_dim_ / 1024, 512, 0, stream>>>(
                d_x_norm_, d_signs_5120_, embedding_dim_
            );
        }

        // C. SwiGLU FFN
        kernel_gemv_tq1_0<<<ffn_blocks, 128, 0, stream>>>(d_x_norm_, lay.ffn_gate, d_gate_, embedding_dim_, ffn_dim_);
        kernel_gemv_tq1_0<<<ffn_blocks, 128, 0, stream>>>(d_x_norm_, lay.ffn_up, d_up_, embedding_dim_, ffn_dim_);

        kernel_swiglu<<<(ffn_dim_ + 255) / 256, 256, 0, stream>>>(d_gate_, d_up_, d_act_, ffn_dim_);

        if (has_hadamard_) {
            kernel_hadamard_forward<<<ffn_dim_ / 1024, 512, 0, stream>>>(
                d_act_, d_signs_17408_, ffn_dim_
            );
        }

        kernel_gemv_tq1_0<<<x_blocks, 128, 0, stream>>>(d_act_, lay.ffn_down, d_ffn_out_, ffn_dim_, embedding_dim_);
        kernel_add_residual<<<(embedding_dim_ + 255) / 256, 256, 0, stream>>>(d_x_, d_ffn_out_, embedding_dim_);
    }

    // 3. Final RMSNorm & Output Logits
    if (compute_logits) {
        kernel_rmsnorm<<<1, norm_threads, shared_bytes, stream>>>(
            d_x_,
            d_output_norm_,
            d_x_norm_,
            embedding_dim_,
            rms_norm_eps_
        );

        if (has_hadamard_) {
            kernel_hadamard_forward<<<embedding_dim_ / 1024, 512, 0, stream>>>(
                d_x_norm_, d_signs_5120_, embedding_dim_
            );
        }

        uint32_t vocab_blocks = (vocab_size_ + 31) / 32;
        kernel_gemv_tq1_0<<<vocab_blocks, 128, 0, stream>>>(
            d_x_norm_,
            d_output_,
            d_logits_,
            embedding_dim_,
            vocab_size_
        );

        // Apply on-device FSM grammar mask BEFORE candidate reduction
        if (d_bitmask != nullptr) {
            uint32_t mask_threads = 256;
            uint32_t mask_blocks = (vocab_size_ + mask_threads - 1) / mask_threads;
            kernel_apply_fsm_mask<<<mask_blocks, mask_threads, 0, stream>>>(
                d_logits_, vocab_size_, d_bitmask
            );
        }

        // Run On-Device Candidate Extraction
        kernel_extract_top_candidates<<<1, 1024, 0, stream>>>(
            d_logits_, vocab_size_, d_top_tokens_, d_top_logits_, TOP_CANDIDATES
        );
        CUDA_CHECK(cudaMemcpyAsync(h_top_tokens_, d_top_tokens_, TOP_CANDIDATES * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_top_logits_, d_top_logits_, TOP_CANDIDATES * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
}

void NativeDecoder::generate(
    const std::string& prompt,
    int max_tokens,
    const TokenCallback& on_token,
    const SamplingConfig& sampling,
    uint32_t stream_id,
    bool reset_state,
    const uint32_t* d_grammar_bitmask,
    const ToolCallCallback& on_tool_call
) {
    if (!initialized_) return;

    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_);
    uint32_t stream_slot = stream_id % MAX_STREAMS;

    size_t stream_kv_stride = 16ULL * MAX_SEQ_LEN * 4 * 256;
    size_t stream_ssm_stride = 48ULL * ssm_inner_;

    if (reset_state) {
        stream_seq_pos_[stream_slot] = 0;
        CUDA_CHECK(cudaMemsetAsync(d_k_cache_ + stream_slot * stream_kv_stride, 0, stream_kv_stride * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_v_cache_ + stream_slot * stream_kv_stride, 0, stream_kv_stride * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_ssm_state_ + stream_slot * stream_ssm_stride, 0, stream_ssm_stride * sizeof(float), stream));
        if (page_pool_) {
            release_stream_blocks(stream_slot);
        }
    }

    // Format prompt with ChatML if needed
    std::string formatted_prompt = prompt;
    if (reset_state && prompt.find("<|im_start|>") == std::string::npos) {
        formatted_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    }

    std::vector<uint32_t> prompt_tokens = tokenizer_.encode(formatted_prompt);
    if (prompt_tokens.empty()) prompt_tokens.push_back(bos_token_id_);

    uint32_t seq_pos = stream_seq_pos_[stream_slot];

    // 1. Dynamic Grammar Bitmask Initial State Sync (Zero-copy in H22)
    const uint32_t* active_bitmask = d_grammar_bitmask;
    if (active_fsm_table_ && !active_fsm_table_->states.empty()) {
        if (current_fsm_state_ >= 0 && static_cast<size_t>(current_fsm_state_) < active_fsm_table_->states.size()) {
            const auto& desc = active_fsm_table_->states[current_fsm_state_];
            if (d_fsm_bitmasks_) {
                active_bitmask = d_fsm_bitmasks_ + desc.mask_offset_words;
            } else {
                size_t mask_words = (vocab_size_ + 31) / 32;
                size_t copy_words = std::min<size_t>(mask_words, static_cast<size_t>(active_fsm_table_->mask_words));
                CUDA_CHECK(cudaMemcpyAsync(
                    d_grammar_mask_,
                    &active_fsm_table_->bitmasks[desc.mask_offset_words],
                    copy_words * sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    stream
                ));
                active_bitmask = d_grammar_mask_;
            }
        }
    }

    if (page_pool_) {
        std::vector<int32_t> prompt_i32(prompt_tokens.begin(), prompt_tokens.end());
        attach_stream_to_prefix(stream_slot, prompt_i32);

        // Pre-allocate all prefill blocks in a single batched PCIe transfer (H23)
        uint32_t total_prefill_blocks = (prompt_tokens.size() + paged_kv_.block_size - 1) / paged_kv_.block_size;
        allocate_stream_blocks_batch(stream_slot, 0, total_prefill_blocks);
    }

    // Prefill Phase
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        if (page_pool_) {
            uint32_t logical_block = seq_pos / paged_kv_.block_size;
            allocate_stream_block(stream_slot, logical_block);
        }
        bool is_last = (i + 1 == prompt_tokens.size());
        forward_token(prompt_tokens[i], seq_pos, is_last, stream_slot, active_bitmask);
        seq_pos++;
        if (seq_pos >= MAX_SEQ_LEN - 1) break;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 2. Autoregressive Generation Loop
    int generated_count = 0;
    std::vector<uint32_t> generated_tokens;
    generated_tokens.reserve(max_tokens);

    // Determine ChatML turn boundaries to guard tool call detection.
    // Tool calls MUST only be intercepted if emitted within an assistant turn.
    // If a user prompt contains <tool_call> in non-assistant text, ignore it.
    std::string assistant_marker = "<|im_start|>assistant";
    size_t last_asst_idx = formatted_prompt.rfind(assistant_marker);

    std::string accumulated_text;
    // Only accumulate prefill text that strictly follows the last assistant turn marker
    if (last_asst_idx != std::string::npos) {
        std::string asst_prefill = formatted_prompt.substr(last_asst_idx + assistant_marker.size());
        size_t prompt_tc = asst_prefill.rfind("<tool_call>");
        if (prompt_tc != std::string::npos) {
            accumulated_text = asst_prefill.substr(prompt_tc);
        }
    }

    // Check if assistant prefill already contains a completed tool call
    if (on_tool_call && accumulated_text.find("</tool_call>") != std::string::npos) {
        size_t start_tag = accumulated_text.find("<tool_call>");
        size_t end_tag = accumulated_text.find("</tool_call>");
        if (start_tag != std::string::npos && end_tag > start_tag) {
            std::string json_body = accumulated_text.substr(start_tag + 11, end_tag - (start_tag + 11));
            ToolCall tc = parse_tool_call_json(json_body, stream_id);
            if (!tc.name.empty()) {
                on_tool_call(tc);
                stream_seq_pos_[stream_slot] = seq_pos;
                return;
            }
        }
    }

    while (generated_count < max_tokens && seq_pos < MAX_SEQ_LEN - 1) {
        uint32_t next_token = sample_next_token(sampling, generated_tokens);
        float logit = h_top_logits_[0];

        if (next_token == eos_token_id_ || next_token == bos_token_id_ || next_token == tokenizer_.im_end_id()) {
            // Check if accumulated text contains a completed tool call
            if (on_tool_call) {
                size_t start_tag = accumulated_text.find("<tool_call>");
                size_t end_tag = accumulated_text.find("</tool_call>");
                if (start_tag != std::string::npos && end_tag != std::string::npos && end_tag > start_tag) {
                    std::string json_body = accumulated_text.substr(start_tag + 11, end_tag - (start_tag + 11));
                    ToolCall tc = parse_tool_call_json(json_body, stream_id);
                    if (!tc.name.empty()) {
                        on_tool_call(tc);
                        stream_seq_pos_[stream_slot] = seq_pos;
                        return;
                    }
                }
            }
            if (on_token) on_token(next_token, logit, "", true);
            stream_seq_pos_[stream_slot] = seq_pos;
            return;
        }

        generated_tokens.push_back(next_token);
        std::string token_str = tokenizer_.decode_token(next_token);
        accumulated_text += token_str;

        // Advance dynamic grammar FSM if active
        if (active_fsm_table_) {
            current_fsm_state_ = active_fsm_table_->get_next_state(current_fsm_state_, next_token);
        }

        // Check for completed tool call before EOS
        if (on_tool_call && accumulated_text.find("</tool_call>") != std::string::npos) {
            size_t start_tag = accumulated_text.find("<tool_call>");
            size_t end_tag = accumulated_text.find("</tool_call>");
            if (start_tag != std::string::npos && end_tag > start_tag) {
                std::string json_body = accumulated_text.substr(start_tag + 11, end_tag - (start_tag + 11));
                ToolCall tc = parse_tool_call_json(json_body, stream_id);
                if (!tc.name.empty()) {
                    on_tool_call(tc);
                    stream_seq_pos_[stream_slot] = seq_pos;
                    return;
                }
            }
        }

        bool is_last_step = (generated_count + 1 >= max_tokens);
        if (on_token) {
            bool cont = on_token(next_token, logit, token_str, is_last_step);
            if (!cont) {
                stream_seq_pos_[stream_slot] = seq_pos;
                return;
            }
        }

        if (is_last_step) {
            stream_seq_pos_[stream_slot] = seq_pos;
            return;
        }

        // Dynamically synchronize next step grammar bitmask (Zero-copy in H22)
        if (active_fsm_table_ && !active_fsm_table_->states.empty()) {
            if (current_fsm_state_ >= 0 && static_cast<size_t>(current_fsm_state_) < active_fsm_table_->states.size()) {
                const auto& desc = active_fsm_table_->states[current_fsm_state_];
                if (d_fsm_bitmasks_) {
                    active_bitmask = d_fsm_bitmasks_ + desc.mask_offset_words;
                } else {
                    size_t mask_words = (vocab_size_ + 31) / 32;
                    size_t copy_words = std::min<size_t>(mask_words, static_cast<size_t>(active_fsm_table_->mask_words));
                    CUDA_CHECK(cudaMemcpyAsync(
                        d_grammar_mask_,
                        &active_fsm_table_->bitmasks[desc.mask_offset_words],
                        copy_words * sizeof(uint32_t),
                        cudaMemcpyHostToDevice,
                        stream
                    ));
                    active_bitmask = d_grammar_mask_;
                }
            }
        }

        if (page_pool_) {
            uint32_t logical_block = seq_pos / paged_kv_.block_size;
            allocate_stream_block(stream_slot, logical_block);
        }

        // Compute next token forward step with stream slot and grammar mask
        forward_token(next_token, seq_pos, true, stream_slot, active_bitmask);
        seq_pos++;
        generated_count++;

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Safety fallback
    if (on_token) on_token(eos_token_id_, 1.0f, "", true);
    stream_seq_pos_[stream_slot] = seq_pos;
}

void NativeDecoder::generate(
    const std::string& prompt,
    int max_tokens,
    ipc::ShmServer& server,
    uint32_t stream_id,
    const SamplingConfig& sampling,
    bool reset_state,
    const uint32_t* d_grammar_bitmask
) {
    generate(
        prompt,
        max_tokens,
        [&server, stream_id](uint32_t token_id, float logit, const std::string& text, bool is_final) -> bool {
            return server.send_token(stream_id, token_id, logit, text, is_final);
        },
        sampling,
        stream_id,
        reset_state,
        d_grammar_bitmask,
        [&server, stream_id](const ToolCall& tool_call) -> bool {
            return server.send_tool_call(stream_id, tool_call.id, tool_call.name, tool_call.arguments);
        }
    );
}

} // namespace hydrogen
