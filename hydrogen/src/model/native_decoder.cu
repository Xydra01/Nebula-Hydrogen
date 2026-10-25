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

// TQ1_0 PrismML block definition (256 elements in 56 bytes, 8-byte aligned)
struct BlockTQ1_0 {
    uint16_t d;         // delta (fp16)
    uint8_t packed[52]; // packed base-3 trits
    uint16_t _pad;      // 8-byte alignment padding (total 56 bytes)
};

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
// Embedding Lookup Kernel (Exact PrismML PTQ1_0 28-byte Dual Sub-block Layout)
// ----------------------------------------------------------------------------
__device__ __forceinline__ int decode_ptq1_0_trit(uint32_t in_sub, const uint8_t* sub_bytes) {
    if (in_sub < 80) {
        uint32_t p_idx = in_sub >> 4;   // in_sub / 16
        uint32_t j = in_sub & 15;       // in_sub % 16
        uint32_t p = (p_idx == 0) ? 1 : (p_idx == 1) ? 3 : (p_idx == 2) ? 9 : (p_idx == 3) ? 27 : 81;
        uint32_t b = sub_bytes[j];
        return static_cast<int>(((b * p) & 0xff) * 3 >> 8) - 1;
    } else if (in_sub < 120) {
        uint32_t rem = in_sub - 80;
        uint32_t p_idx = rem >> 3;      // rem / 8
        uint32_t j = 16 + (rem & 7);    // 16 + (rem % 8)
        uint32_t p = (p_idx == 0) ? 1 : (p_idx == 1) ? 3 : (p_idx == 2) ? 9 : (p_idx == 3) ? 27 : 81;
        uint32_t b = sub_bytes[j];
        return static_cast<int>(((b * p) & 0xff) * 3 >> 8) - 1;
    } else {
        uint32_t rem = in_sub - 120;
        uint32_t p_idx = rem >> 1;      // rem / 2
        uint32_t byte_sel = rem & 1;    // rem % 2
        uint32_t p = (p_idx == 0) ? 1 : (p_idx == 1) ? 3 : (p_idx == 2) ? 9 : 27;
        uint32_t b = sub_bytes[24 + byte_sel];
        return static_cast<int>(((b * p) & 0xff) * 3 >> 8) - 1;
    }
}

__global__ void kernel_embedding_lookup(
    const uint8_t* __restrict__ embd_data,
    uint32_t token_id,
    float* __restrict__ d_out,
    uint32_t embedding_dim
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= embedding_dim) return;

    uint32_t block_idx = tid / 256;
    uint32_t in_block = tid % 256;
    uint32_t sub_idx = in_block / 128;
    uint32_t in_sub = in_block % 128;
    uint32_t blocks_per_row = embedding_dim / 256;
    uint64_t global_block_idx = static_cast<uint64_t>(token_id) * blocks_per_row + block_idx;

    const uint8_t* sub_bytes = embd_data + global_block_idx * 56 + sub_idx * 28;
    float d = __half2float(*reinterpret_cast<const __half*>(sub_bytes + 26));
    if (isnan(d) || isinf(d)) d = 0.0f;

    int trit = decode_ptq1_0_trit(in_sub, sub_bytes);
    d_out[tid] = d * static_cast<float>(trit);
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
        float w = (d_weight != nullptr) ? d_weight[i] : 1.0f;
        d_out[i] = d_x[i] * scale * w;
    }
}

// ----------------------------------------------------------------------------
// Matrix-Vector Multiplication for GGUF TQ1_0 Weights (Exact PTQ1_0)
// ----------------------------------------------------------------------------
__global__ void kernel_gemv_tq1_0(
    const float* __restrict__ d_x,
    const uint8_t* __restrict__ d_weight,
    float* __restrict__ d_out,
    uint32_t in_dim,
    uint32_t out_dim
) {
    uint32_t warp_global_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_global_id >= out_dim) return;

    uint32_t lane = threadIdx.x % 32;
    uint32_t blocks_per_row = in_dim / 256;
    const uint8_t* row_bytes = d_weight + static_cast<uint64_t>(warp_global_id) * blocks_per_row * 56;

    float acc = 0.0f;

    for (uint32_t b = 0; b < blocks_per_row; ++b) {
        const uint8_t* blk = row_bytes + b * 56;
        uint32_t blk_base = b * 256;

        #pragma unroll
        for (uint32_t sub = 0; sub < 2; ++sub) {
            const uint8_t* sub_bytes = blk + sub * 28;
            float d = __half2float(*reinterpret_cast<const __half*>(sub_bytes + 26));
            if (isnan(d) || isinf(d)) d = 0.0f;
            uint32_t sub_base = blk_base + sub * 128;

            float lane_acc = 0.0f;
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint32_t in_sub = lane * 4 + k;
                int trit = decode_ptq1_0_trit(in_sub, sub_bytes);
                lane_acc += static_cast<float>(trit) * d_x[sub_base + in_sub];
            }

            lane_acc = warp_reduce_sum(lane_acc);
            acc += d * lane_acc;
        }
    }

    if (lane == 0) {
        d_out[warp_global_id] = acc;
    }
}

// ----------------------------------------------------------------------------
// Multi-Stream & Batched Matrix-Vector Multiplication for GGUF TQ1_0 Weights (Issue H22B)
// Computes Y = W * X for B concurrent tokens / streams in a single kernel launch
// ----------------------------------------------------------------------------
__global__ void kernel_batched_gemv_tq1_0(
    const float* __restrict__ d_x,         // [batch_size, in_dim]
    const uint8_t* __restrict__ d_weight,  // [out_dim, blocks_per_row * 56]
    float* __restrict__ d_out,             // [batch_size, out_dim]
    uint32_t in_dim,
    uint32_t out_dim,
    uint32_t batch_size
) {
    uint32_t warp_global_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    uint32_t b_idx = blockIdx.y;
    if (warp_global_id >= out_dim || b_idx >= batch_size) return;

    uint32_t lane = threadIdx.x % 32;
    uint32_t blocks_per_row = in_dim / 256;
    const uint8_t* row_bytes = d_weight + static_cast<uint64_t>(warp_global_id) * blocks_per_row * 56;
    const float* x_ptr = d_x + static_cast<uint64_t>(b_idx) * in_dim;

    float acc = 0.0f;

    for (uint32_t b = 0; b < blocks_per_row; ++b) {
        const uint8_t* blk = row_bytes + b * 56;
        uint32_t blk_base = b * 256;

        #pragma unroll
        for (uint32_t sub = 0; sub < 2; ++sub) {
            const uint8_t* sub_bytes = blk + sub * 28;
            float d = __half2float(*reinterpret_cast<const __half*>(sub_bytes + 26));
            if (isnan(d) || isinf(d)) d = 0.0f;
            uint32_t sub_base = blk_base + sub * 128;

            float lane_acc = 0.0f;
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint32_t in_sub = lane * 4 + k;
                int trit = decode_ptq1_0_trit(in_sub, sub_bytes);
                lane_acc += static_cast<float>(trit) * x_ptr[sub_base + in_sub];
            }

            lane_acc = warp_reduce_sum(lane_acc);
            acc += d * lane_acc;
        }
    }

    if (lane == 0) {
        d_out[static_cast<uint64_t>(b_idx) * out_dim + warp_global_id] = acc;
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
// ----------------------------------------------------------------------------
// Attention Output Sigmoid Gate Kernel (Qwen 3.5 Gated Attention)
// ----------------------------------------------------------------------------
__global__ void kernel_gate_attention_output(
    float* __restrict__ d_out,
    const float* __restrict__ d_gate,
    uint32_t num_elements
) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_elements) {
        float g = d_gate[idx];
        float sig = 1.0f / (1.0f + expf(-g));
        d_out[idx] = d_out[idx] * sig;
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Deinterleave Q and Gate Kernel (24 heads * 512 -> Q: 24*256, Gate: 24*256)
// ----------------------------------------------------------------------------
__global__ void kernel_deinterleave_q_gate(
    const float* __restrict__ d_q_raw,   // [12288]
    float* __restrict__ d_q_heads,       // [6144]
    float* __restrict__ d_attn_gate      // [6144]
) {
    uint32_t head = blockIdx.x; // 0..23
    uint32_t tid = threadIdx.x; // 0..255

    uint32_t raw_offset = head * 512;
    uint32_t out_offset = head * 256;

    d_q_heads[out_offset + tid]   = d_q_raw[raw_offset + tid];
    d_attn_gate[out_offset + tid] = d_q_raw[raw_offset + 256 + tid];
}

// ----------------------------------------------------------------------------
// Per-Head QK-RMSNorm and Rotary Position Embeddings (RoPE) Kernel (NEOX pairing)
// ----------------------------------------------------------------------------
__global__ void kernel_apply_qk_norm_and_rope(
    float* __restrict__ d_q,           // [num_q_heads * head_dim] = [24 * 256]
    float* __restrict__ d_k,           // [num_kv_heads * head_dim] = [4 * 256]
    const float* __restrict__ q_norm,  // [head_dim] = [256]
    const float* __restrict__ k_norm,  // [head_dim] = [256]
    uint32_t num_q_heads,              // 24
    uint32_t num_kv_heads,             // 4
    uint32_t head_dim,                 // 256
    uint32_t rope_dim,                 // 64
    float freq_base,                   // 10000000.0f
    uint32_t pos
) {
    uint32_t head_idx = blockIdx.x; // 0..27 (0..23 is Q, 24..27 is K)
    uint32_t tid = threadIdx.x;     // 0..31

    bool is_q = (head_idx < num_q_heads);
    float* vec = is_q ? (d_q + head_idx * head_dim) 
                      : (d_k + (head_idx - num_q_heads) * head_dim);
    const float* norm_scale = is_q ? q_norm : k_norm;

    // 1. Compute RMS of head vector (32 threads handle 8 elements each = 256)
    float sum_sq = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float val = vec[tid + k * 32];
        sum_sq += val * val;
    }
    sum_sq = warp_reduce_sum(sum_sq);
    float head_sum = __shfl_sync(0xffffffff, sum_sq, 0);
    float inv_rms = rsqrtf((head_sum / static_cast<float>(head_dim)) + 1e-6f);

    // Apply norm scaling
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        uint32_t d = tid + k * 32;
        float w = norm_scale ? norm_scale[d] : 1.0f;
        vec[d] = vec[d] * inv_rms * w;
    }

    // 2. Apply NEOX RoPE to first rope_dim (64) elements: pair (tid, tid + 32)
    if (tid < 32) {
        float freq = powf(freq_base, -static_cast<float>(2 * tid) / static_cast<float>(rope_dim));
        float angle = static_cast<float>(pos) * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float v0 = vec[tid];
        float v1 = vec[tid + 32];
        vec[tid]      = v0 * cos_a - v1 * sin_a;
        vec[tid + 32] = v0 * sin_a + v1 * cos_a;
    }
}

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
    uint32_t tid = threadIdx.x; // 0..31

    const float* q_vec = d_q + (q_head * head_dim);
    float q_scale = 1.0f / sqrtf(static_cast<float>(head_dim)); // 1 / 16 = 0.0625f

    if (is_paged_enabled && d_block_table && d_paged_k && d_paged_v) {
        // Paged Radix KV Cache Path
        uint32_t write_block = pos / block_size;
        uint32_t write_offset = pos % block_size;
        uint32_t write_page = d_block_table[stream_slot * max_blocks_per_stream + write_block];
        size_t page_stride = static_cast<size_t>(block_size) * num_kv_heads * head_dim;
        size_t total_page_stride = 16ULL * page_stride;
        size_t write_kv_off = static_cast<size_t>(write_page) * total_page_stride +
                              static_cast<size_t>(attn_layer_idx) * page_stride +
                              static_cast<size_t>(write_offset) * num_kv_heads * head_dim +
                              static_cast<size_t>(kv_head) * head_dim;

        // Write new K and V (only 1 query head per KV group writes to prevent redundancy)
        if (q_head % (num_q_heads / num_kv_heads) == 0) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint32_t d = tid + k * 32;
                d_paged_k[write_kv_off + d] = d_k_new[kv_head * head_dim + d];
                d_paged_v[write_kv_off + d] = d_v_new[kv_head * head_dim + d];
            }
        }
        // Write new K and V asynchronously (head 0 of KV group)
        if (q_head % (num_q_heads / num_kv_heads) == 0) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint32_t d = tid + k * 32;
                d_paged_k[write_kv_off + d] = d_k_new[kv_head * head_dim + d];
                d_paged_v[write_kv_off + d] = d_v_new[kv_head * head_dim + d];
            }
        }

        // Numerically stable Online Softmax across all positions 0..pos
        float m_prev = -1e20f;
        float l_prev = 0.0f;
        float o_acc[8] = {0.0f};

        for (uint32_t p = 0; p <= pos; ++p) {
            float dot = 0.0f;
            if (p == pos) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    uint32_t d = tid + k * 32;
                    dot += q_vec[d] * d_k_new[kv_head * head_dim + d];
                }
            } else {
                uint32_t p_block = p / block_size;
                uint32_t p_offset = p % block_size;
                uint32_t p_page = d_block_table[stream_slot * max_blocks_per_stream + p_block];
                size_t k_offset = static_cast<size_t>(p_page) * total_page_stride +
                                  static_cast<size_t>(attn_layer_idx) * page_stride +
                                  static_cast<size_t>(p_offset) * num_kv_heads * head_dim +
                                  static_cast<size_t>(kv_head) * head_dim;
                const float* k_vec = d_paged_k + k_offset;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    uint32_t d = tid + k * 32;
                    dot += q_vec[d] * k_vec[d];
                }
            }
            dot = warp_reduce_sum(dot) * q_scale;
            float score_p = __shfl_sync(0xffffffff, dot, 0);

            float m_curr = fmaxf(m_prev, score_p);
            float alpha = expf(m_prev - m_curr);
            float beta = expf(score_p - m_curr);
            l_prev = alpha * l_prev + beta;
            m_prev = m_curr;

            if (p == pos) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    uint32_t d = tid + k * 32;
                    o_acc[k] = alpha * o_acc[k] + beta * d_v_new[kv_head * head_dim + d];
                }
            } else {
                uint32_t p_block = p / block_size;
                uint32_t p_offset = p % block_size;
                uint32_t p_page = d_block_table[stream_slot * max_blocks_per_stream + p_block];
                size_t v_offset = static_cast<size_t>(p_page) * total_page_stride +
                                  static_cast<size_t>(attn_layer_idx) * page_stride +
                                  static_cast<size_t>(p_offset) * num_kv_heads * head_dim +
                                  static_cast<size_t>(kv_head) * head_dim;
                const float* v_vec = d_paged_v + v_offset;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    uint32_t d = tid + k * 32;
                    o_acc[k] = alpha * o_acc[k] + beta * v_vec[d];
                }
            }
        }

        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 1.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t d = tid + k * 32;
            d_out[q_head * head_dim + d] = o_acc[k] * inv_l;
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

        // Write new K and V asynchronously (head 0 of KV group)
        if (q_head % (num_q_heads / num_kv_heads) == 0) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint32_t d = tid + k * 32;
                d_k_cache[pos_kv_offset + d] = d_k_new[kv_head * head_dim + d];
                d_v_cache[pos_kv_offset + d] = d_v_new[kv_head * head_dim + d];
            }
        }

        // Numerically stable Online Softmax across all positions 0..pos
        float m_prev = -1e20f;
        float l_prev = 0.0f;
        float o_acc[8] = {0.0f};

        for (uint32_t p = 0; p <= pos; ++p) {
            float dot = 0.0f;
            if (p == pos) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    uint32_t d = tid + k * 32;
                    dot += q_vec[d] * d_k_new[kv_head * head_dim + d];
                }
            } else {
                size_t k_offset = stream_base +
                                  static_cast<size_t>(attn_layer_idx) * layer_kv_stride +
                                  static_cast<size_t>(p) * num_kv_heads * head_dim +
                                  static_cast<size_t>(kv_head) * head_dim;
                const float* k_vec = d_k_cache + k_offset;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    uint32_t d = tid + k * 32;
                    dot += q_vec[d] * k_vec[d];
                }
            }
            dot = warp_reduce_sum(dot) * q_scale;
            float score_p = __shfl_sync(0xffffffff, dot, 0);

            float m_curr = fmaxf(m_prev, score_p);
            float alpha = expf(m_prev - m_curr);
            float beta = expf(score_p - m_curr);
            l_prev = alpha * l_prev + beta;
            m_prev = m_curr;

            if (p == pos) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    uint32_t d = tid + k * 32;
                    o_acc[k] = alpha * o_acc[k] + beta * d_v_new[kv_head * head_dim + d];
                }
            } else {
                size_t v_offset = stream_base +
                                  static_cast<size_t>(attn_layer_idx) * layer_kv_stride +
                                  static_cast<size_t>(p) * num_kv_heads * head_dim +
                                  static_cast<size_t>(kv_head) * head_dim;
                const float* v_vec = d_v_cache + v_offset;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    uint32_t d = tid + k * 32;
                    o_acc[k] = alpha * o_acc[k] + beta * v_vec[d];
                }
            }
        }

        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 1.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t d = tid + k * 32;
            d_out[q_head * head_dim + d] = o_acc[k] * inv_l;
        }
    }
}

// ----------------------------------------------------------------------------
// SSM 1D Causal Depthwise Convolution Kernel (10,240 channels, K=4, Multi-Stream)
// ----------------------------------------------------------------------------
__global__ void kernel_ssm_conv1d_step(
    const float* __restrict__ d_qkv,
    const float* __restrict__ d_ssm_conv1d,
    float* __restrict__ d_ssm_conv_state,
    float* __restrict__ d_conv_out,
    uint32_t layer_idx,
    uint32_t stream_slot
) {
    uint32_t c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= 10240) return;

    size_t state_base = (static_cast<size_t>(stream_slot) * 48ULL + layer_idx) * (4ULL * 10240ULL) + c;

    // Shift history buffer and push new input
    float h0 = d_ssm_conv_state[state_base + 1 * 10240];
    float h1 = d_ssm_conv_state[state_base + 2 * 10240];
    float h2 = d_ssm_conv_state[state_base + 3 * 10240];
    float h3 = d_qkv[c];

    d_ssm_conv_state[state_base + 0 * 10240] = h0;
    d_ssm_conv_state[state_base + 1 * 10240] = h1;
    d_ssm_conv_state[state_base + 2 * 10240] = h2;
    d_ssm_conv_state[state_base + 3 * 10240] = h3;

    if (d_ssm_conv1d != nullptr) {
        // GGUF tensor shape [4, 10240] has ne[0] = 4, so weights for channel c are at c * 4 + k
        float w0 = d_ssm_conv1d[c * 4 + 0];
        float w1 = d_ssm_conv1d[c * 4 + 1];
        float w2 = d_ssm_conv1d[c * 4 + 2];
        float w3 = d_ssm_conv1d[c * 4 + 3];

        float conv_val = w0 * h0 + w1 * h1 + w2 * h2 + w3 * h3;
        // SiLU non-linearity: x * sigmoid(x)
        d_conv_out[c] = conv_val / (1.0f + expf(-conv_val));
    } else {
        d_conv_out[c] = h3;
    }
}

// ----------------------------------------------------------------------------
// Gated DeltaNet L2 Normalization Kernel (16 Key & Query heads of dim 128)
// ----------------------------------------------------------------------------
__global__ void kernel_deltanet_l2_norm(
    float* __restrict__ d_q, // [16 * 128]
    float* __restrict__ d_k  // [16 * 128]
) {
    uint32_t head_idx = blockIdx.x; // 0..15
    if (head_idx >= 16) return;

    float* q_head = d_q + head_idx * 128;
    float* k_head = d_k + head_idx * 128;

    float q_sq = 0.0f;
    float k_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t idx = threadIdx.x + j * 32;
        float q_val = q_head[idx];
        float k_val = k_head[idx];
        q_sq += q_val * q_val;
        k_sq += k_val * k_val;
    }

    q_sq = warp_reduce_sum(q_sq);
    k_sq = warp_reduce_sum(k_sq);

    float q_inv = rsqrtf(__shfl_sync(0xffffffff, q_sq, 0) + 1e-6f);
    float k_inv = rsqrtf(__shfl_sync(0xffffffff, k_sq, 0) + 1e-6f);

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t idx = threadIdx.x + j * 32;
        q_head[idx] *= q_inv;
        k_head[idx] *= k_inv;
    }
}

// ----------------------------------------------------------------------------
// Gated DeltaNet Dynamic Alpha/Beta Projections and Decay Kernel (48 heads)
// ----------------------------------------------------------------------------
__global__ void kernel_deltanet_alpha_beta_step(
    const float* __restrict__ d_x_norm,      // [5120]
    const void* __restrict__ ssm_alpha,      // [48, 5120] (BF16)
    const void* __restrict__ ssm_beta,       // [48, 5120] (BF16)
    const float* __restrict__ ssm_dt,        // [48]
    const float* __restrict__ ssm_a,         // [48]
    float* __restrict__ d_decay,             // [48]
    float* __restrict__ d_beta               // [48]
) {
    uint32_t head = blockIdx.x; // 0..47
    if (head >= 48) return;

    uint32_t tid = threadIdx.x; // 0..127
    float acc_alpha = 0.0f;
    float acc_beta = 0.0f;

    const uint16_t* w_alpha = ssm_alpha ? (reinterpret_cast<const uint16_t*>(ssm_alpha) + head * 5120) : nullptr;
    const uint16_t* w_beta  = ssm_beta  ? (reinterpret_cast<const uint16_t*>(ssm_beta)  + head * 5120) : nullptr;

    if (w_alpha && w_beta) {
        for (uint32_t j = tid; j < 5120; j += blockDim.x) {
            float x_val = d_x_norm[j];
            uint32_t u_a = static_cast<uint32_t>(w_alpha[j]) << 16;
            uint32_t u_b = static_cast<uint32_t>(w_beta[j]) << 16;
            acc_alpha += x_val * (*reinterpret_cast<const float*>(&u_a));
            acc_beta  += x_val * (*reinterpret_cast<const float*>(&u_b));
        }
    }

    acc_alpha = warp_reduce_sum(acc_alpha);
    acc_beta  = warp_reduce_sum(acc_beta);

    __shared__ float s_alpha[4];
    __shared__ float s_beta[4];
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;

    if (lane == 0) {
        s_alpha[warp_id] = acc_alpha;
        s_beta[warp_id] = acc_beta;
    }
    __syncthreads();

    if (warp_id == 0) {
        float tot_alpha = (lane < 4) ? s_alpha[lane] : 0.0f;
        float tot_beta  = (lane < 4) ? s_beta[lane]  : 0.0f;
        tot_alpha = warp_reduce_sum(tot_alpha);
        tot_beta  = warp_reduce_sum(tot_beta);

        if (lane == 0) {
            float dt = ssm_dt ? ssm_dt[head] : 0.0f;
            float alpha_biased = tot_alpha + dt;
            float alpha_softplus = (alpha_biased > 20.0f) ? alpha_biased : log1pf(expf(alpha_biased));
            float a_val = ssm_a ? ssm_a[head] : -1.0f;
            float gate = alpha_softplus * a_val;
            d_decay[head] = expf(gate);

            // beta_sigmoid
            d_beta[head] = 1.0f / (1.0f + expf(-tot_beta));
        }
    }
}

// ----------------------------------------------------------------------------
// Gated DeltaNet Fast Register-Blocked Column Recurrence Kernel (48 heads x 128 cols)
// ----------------------------------------------------------------------------
__global__ void kernel_deltanet_recurrence_fast(
    const float* __restrict__ d_conv_q,      // [16 * 128] = [2048]
    const float* __restrict__ d_conv_k,      // [16 * 128] = [2048]
    const float* __restrict__ d_conv_v,      // [48 * 128] = [6144]
    const float* __restrict__ d_decay,       // [48]
    const float* __restrict__ d_beta,        // [48]
    float* __restrict__ d_ssm_state,         // [MAX_STREAMS * 48 * 48 * 16384]
    float* __restrict__ d_attn_core,         // [6144]
    uint32_t ssm_layer_idx,
    uint32_t stream_slot
) {
    uint32_t head = blockIdx.x; // 0..47
    if (head >= 48) return;

    // 8 blocks per head, each block handles 16 columns (16 * 8 = 128 columns)
    uint32_t block_col_base = blockIdx.y * 16;

    uint32_t warp_id = threadIdx.x / 32; // 0..3
    uint32_t lane = threadIdx.x % 32;    // 0..31
    uint32_t col_in_warp = lane / 8;     // 0..3
    uint32_t lane_in_col = lane % 8;     // 0..7

    uint32_t col = block_col_base + warp_id * 4 + col_in_warp;
    if (col >= 128) return;

    uint32_t qk_head = head % 16;
    const float* q = d_conv_q + qk_head * 128;
    const float* k = d_conv_k + qk_head * 128;
    const float* v = d_conv_v + head * 128;

    size_t state_stride = 48ULL * 48ULL * 16384ULL;
    size_t head_offset = static_cast<size_t>(stream_slot) * state_stride +
                         static_cast<size_t>(ssm_layer_idx) * (48ULL * 16384ULL) +
                         static_cast<size_t>(head) * 16384ULL;
    float* S = d_ssm_state + head_offset;

    float g_val = d_decay[head];
    float beta_val = d_beta[head];

    float s_shard[16];
    float k_reg[16];
    float q_reg[16];

    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        int row = r * 8 + lane_in_col;
        s_shard[r] = S[col * 128 + row];
        k_reg[r] = k[row];
        q_reg[r] = q[row];
    }

    // kv[col] = sum_row S[row, col] * k[row]
    float kv_shard = 0.0f;
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        kv_shard += s_shard[r] * k_reg[r];
    }
    // 8-lane reduction
    for (int offset = 4; offset > 0; offset /= 2) {
        kv_shard += __shfl_down_sync(0xffffffff, kv_shard, offset);
    }
    float kv_col = __shfl_sync(0xffffffff, kv_shard, (lane & ~7));

    // delta[col] = (v[col] - g_val * kv_col) * beta_val
    float delta_col = (v[col] - g_val * kv_col) * beta_val;

    // S[row, col] = g_val * S[row, col] + k[row] * delta_col
    // attn[col] = sum_row S[row, col] * q[row]
    float attn_partial = 0.0f;
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }
    for (int offset = 4; offset > 0; offset /= 2) {
        attn_partial += __shfl_down_sync(0xffffffff, attn_partial, offset);
    }

    if (lane_in_col == 0) {
        // scale = 1 / sqrt(128)
        constexpr float scale = 0.08838834764831845f;
        d_attn_core[head * 128 + col] = attn_partial * scale;
    }

    // Write updated state back
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        int row = r * 8 + lane_in_col;
        S[col * 128 + row] = s_shard[r];
    }
}

// ----------------------------------------------------------------------------
// Group RMSNorm & SiLU Gating Kernel for Gated DeltaNet
// ----------------------------------------------------------------------------
__global__ void kernel_gated_rmsnorm_silu(
    const float* __restrict__ d_attn_core, // [48 * 128] = 6144
    const float* __restrict__ ssm_norm,    // [128]
    const float* __restrict__ d_gate,      // [6144]
    float* __restrict__ d_out,             // [6144]
    float eps
) {
    uint32_t head = blockIdx.x; // 0..47
    if (head >= 48) return;

    uint32_t tid = threadIdx.x; // 0..127
    const float* in_vec = d_attn_core + head * 128;
    const float* gate_vec = d_gate + head * 128;

    // Permute from linear GDN value head layout {128, 16, 3}
    // to tiled output projection layout {128, 3, 16}:
    // head = k + 16 * v  -->  out_head = v + 3 * k
    uint32_t k = head % 16;
    uint32_t v = head / 16;
    uint32_t out_head = v + 3 * k;
    float* out_vec = d_out + out_head * 128;

    float val = in_vec[tid];
    float sq = val * val;
    sq = warp_reduce_sum(sq);

    __shared__ float s_warp_sq[4];
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) s_warp_sq[warp_id] = sq;
    __syncthreads();

    if (warp_id == 0) {
        float tot = (lane < 4) ? s_warp_sq[lane] : 0.0f;
        tot = warp_reduce_sum(tot);
        if (lane == 0) {
            s_warp_sq[0] = rsqrtf((tot / 128.0f) + eps);
        }
    }
    __syncthreads();

    float inv_rms = s_warp_sq[0];
    float w = ssm_norm ? ssm_norm[tid] : 1.0f;

    // SiLU gate: z * sigmoid(z)
    float z = gate_vec[tid];
    float silu_z = z / (1.0f + expf(-z));

    out_vec[tid] = (val * inv_rms * w) * silu_z;
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

    // Fallback: Check for XML format <function=name> ... <parameter=key>value</parameter> ... </function>
    if (result.name.empty()) {
        size_t fn_pos = raw_json.find("<function=");
        if (fn_pos != std::string::npos) {
            size_t fn_start = fn_pos + 10;
            size_t fn_end = raw_json.find('>', fn_start);
            if (fn_end != std::string::npos) {
                result.name = raw_json.substr(fn_start, fn_end - fn_start);
                while (!result.name.empty() && (isspace(result.name.back()) || result.name.back() == '\r' || result.name.back() == '\n')) {
                    result.name.pop_back();
                }
                while (!result.name.empty() && (isspace(result.name.front()) || result.name.front() == '\r' || result.name.front() == '\n')) {
                    result.name.erase(0, 1);
                }

                std::string args_json = "{";
                bool first_param = true;
                size_t p_pos = fn_end + 1;
                while ((p_pos = raw_json.find("<parameter=", p_pos)) != std::string::npos) {
                    size_t p_name_start = p_pos + 11;
                    size_t p_name_end = raw_json.find('>', p_name_start);
                    if (p_name_end == std::string::npos) break;
                    std::string p_name = raw_json.substr(p_name_start, p_name_end - p_name_start);

                    size_t v_start = p_name_end + 1;
                    size_t v_end = raw_json.find("</parameter>", v_start);
                    if (v_end == std::string::npos) {
                        v_end = raw_json.find("<parameter=", v_start);
                        if (v_end == std::string::npos) {
                            v_end = raw_json.find("</function>", v_start);
                            if (v_end == std::string::npos) v_end = raw_json.size();
                        }
                    }
                    std::string p_val = raw_json.substr(v_start, v_end - v_start);
                    while (!p_val.empty() && (isspace(p_val.back()) || p_val.back() == '\r' || p_val.back() == '\n')) {
                        p_val.pop_back();
                    }
                    while (!p_val.empty() && (isspace(p_val.front()) || p_val.front() == '\r' || p_val.front() == '\n')) {
                        p_val.erase(0, 1);
                    }

                    if (!first_param) args_json += ", ";
                    first_param = false;
                    args_json += "\"" + p_name + "\": ";
                    if (p_val == "true" || p_val == "false") {
                        args_json += p_val;
                    } else {
                        bool is_num = !p_val.empty();
                        for (char c : p_val) {
                            if (!isdigit(c) && c != '-' && c != '.') { is_num = false; break; }
                        }
                        if (is_num) {
                            args_json += p_val;
                        } else {
                            std::string escaped;
                            for (char c : p_val) {
                                if (c == '"') escaped += "\\\"";
                                else if (c == '\\') escaped += "\\\\";
                                else if (c == '\n') escaped += "\\n";
                                else if (c == '\r') escaped += "\\r";
                                else if (c == '\t') escaped += "\\t";
                                else escaped += c;
                            }
                            args_json += "\"" + escaped + "\"";
                        }
                    }
                    p_pos = v_end;
                }
                args_json += "}";
                result.arguments = args_json;
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

    size_t table_entries = static_cast<size_t>(max_streams_) * max_blocks_per_stream;
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
    for (uint32_t s = 0; s < max_streams_; ++s) {
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
    if (!radix_tree_ || stream_slot >= max_streams_ || prompt_tokens.empty()) return 0;
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
    if (stream_slot >= max_streams_ || !page_pool_ || num_blocks == 0) {
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
    if (stream_slot >= max_streams_ || !page_pool_ || logical_block >= paged_kv_.max_blocks_per_stream) {
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
    if (stream_slot >= max_streams_) return;
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
        if (d_q_heads_) cudaFree(d_q_heads_);
        if (d_attn_gate_) cudaFree(d_attn_gate_);
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
        if (d_ssm_conv_state_) cudaFree(d_ssm_conv_state_);
        if (d_conv_out_) cudaFree(d_conv_out_);
        if (d_grammar_mask_) cudaFree(d_grammar_mask_);
        if (d_top_tokens_) cudaFree(d_top_tokens_);
        if (d_top_logits_) cudaFree(d_top_logits_);
        if (h_top_tokens_) cudaFreeHost(h_top_tokens_);
        if (h_top_logits_) cudaFreeHost(h_top_logits_);
        if (d_signs_5120_) cudaFree(d_signs_5120_);
        if (d_signs_6144_) cudaFree(d_signs_6144_);
        if (d_signs_17408_) cudaFree(d_signs_17408_);
        if (d_decay_) cudaFree(d_decay_);
        if (d_beta_) cudaFree(d_beta_);
        if (d_attn_core_) cudaFree(d_attn_core_);
        disable_dynamic_paged_kv();
    }
}

bool NativeDecoder::initialize(
    const std::filesystem::path& model_path,
    uint32_t max_streams,
    uint32_t max_seq_len
) {
    if (initialized_) return true;

    max_streams_ = (max_streams > 0) ? max_streams : DEFAULT_MAX_STREAMS;
    max_seq_len_ = (max_seq_len > 0) ? max_seq_len : DEFAULT_MAX_SEQ_LEN;
    stream_seq_pos_.assign(max_streams_, 0);
    stream_allocated_pages_.clear();
    stream_allocated_pages_.resize(max_streams_);
    stream_radix_nodes_.clear();
    stream_radix_nodes_.resize(max_streams_);

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
            has_hadamard_ = true; // PrismML PTQ1_0 requires Hadamard outlier rotation
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
            const GGUFTensor* t_sa = model.find_tensor(p + "ssm_a");
            const GGUFTensor* t_salpha = model.find_tensor(p + "ssm_alpha.weight");
            const GGUFTensor* t_sbeta = model.find_tensor(p + "ssm_beta.weight");
            const GGUFTensor* t_sdt = model.find_tensor(p + "ssm_dt.bias");
            const GGUFTensor* t_snorm = model.find_tensor(p + "ssm_norm.weight");
            const GGUFTensor* t_so = model.find_tensor(p + "ssm_out.weight");

            if (t_sg) lay.ssm_gate = d_weights_ + t_sg->offset;
            if (t_sqkv) lay.ssm_qkv = d_weights_ + t_sqkv->offset;
            if (t_sconv) lay.ssm_conv1d = reinterpret_cast<const float*>(d_weights_ + t_sconv->offset);
            if (t_sa) lay.ssm_a = reinterpret_cast<const float*>(d_weights_ + t_sa->offset);
            if (t_salpha) lay.ssm_alpha = d_weights_ + t_salpha->offset;
            if (t_sbeta) lay.ssm_beta = d_weights_ + t_sbeta->offset;
            if (t_sdt) lay.ssm_dt = reinterpret_cast<const float*>(d_weights_ + t_sdt->offset);
            if (t_snorm) lay.ssm_norm = reinterpret_cast<const float*>(d_weights_ + t_snorm->offset);
            if (t_so) lay.ssm_out = d_weights_ + t_so->offset;
        }
    }

    // 4. Allocate Working Device Buffers
    CUDA_CHECK(cudaMalloc(&d_x_, embedding_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_norm_, embedding_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_, embedding_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_, 12288 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_heads_, 6144 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_gate_, 6144 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_out_, 6144 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qkv_, 10240 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_out_, 10240 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_decay_, 48 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta_, 48 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_core_, 6144 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate_, ffn_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up_, ffn_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act_, ffn_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_out_, embedding_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits_, vocab_size_ * sizeof(float)));

    // Multi-Stream KV Cache (max_streams_ * 16 layers, max_seq_len_ tokens, 4 heads, 256 dim)
    size_t kv_cache_elements = static_cast<size_t>(max_streams_) * 16ULL * max_seq_len_ * 4 * 256;
    CUDA_CHECK(cudaMalloc(&d_k_cache_, kv_cache_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_cache_, kv_cache_elements * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_k_cache_, 0, kv_cache_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_v_cache_, 0, kv_cache_elements * sizeof(float), stream));

    // Multi-Stream Gated DeltaNet Matrix State (max_streams_ * 48 layers * 48 heads * 128 * 128 floats)
    size_t ssm_state_elements = static_cast<size_t>(max_streams_) * 48ULL * 48ULL * 16384ULL;
    CUDA_CHECK(cudaMalloc(&d_ssm_state_, ssm_state_elements * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_ssm_state_, 0, ssm_state_elements * sizeof(float), stream));

    // Multi-Stream SSM Conv1D History Buffer (max_streams_ * 48 layers * 4 steps * 10240 floats)
    size_t conv_state_elements = static_cast<size_t>(max_streams_) * 48ULL * 4ULL * 10240ULL;
    CUDA_CHECK(cudaMalloc(&d_ssm_conv_state_, conv_state_elements * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_ssm_conv_state_, 0, conv_state_elements * sizeof(float), stream));

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
              << max_streams_ << " streams, Context: " << max_seq_len_ << " tokens). Vocab: "
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

    uint32_t ffn_blocks = (ffn_dim_ + 3) / 4;
    uint32_t x_blocks = (embedding_dim_ + 3) / 4;

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
            uint32_t q_blocks = (12288 + 3) / 4;
            uint32_t kv_blocks = (1024 + 3) / 4;

            kernel_gemv_tq1_0<<<q_blocks, 128, 0, stream>>>(d_x_norm_, lay.attn_q, d_q_, embedding_dim_, 12288);
            kernel_gemv_tq1_0<<<kv_blocks, 128, 0, stream>>>(d_x_norm_, lay.attn_k, d_k_, embedding_dim_, 1024);
            kernel_gemv_tq1_0<<<kv_blocks, 128, 0, stream>>>(d_x_norm_, lay.attn_v, d_v_, embedding_dim_, 1024);

            // De-interleave Q (24*256) and Gate (24*256) from interleaved d_q_ (12288)
            kernel_deinterleave_q_gate<<<24, 256, 0, stream>>>(
                d_q_,
                d_q_heads_,
                d_attn_gate_
            );

            // Apply per-head QK-RMSNorm and NEOX Rotary Position Embeddings (RoPE) to Q (24 heads * 256) and K (4 heads * 256)
            kernel_apply_qk_norm_and_rope<<<28, 32, 0, stream>>>(
                d_q_heads_,
                d_k_,
                lay.attn_q_norm,
                lay.attn_k_norm,
                head_count_,
                head_count_kv_,
                head_dim_,
                64,
                10000000.0f,
                pos
            );

            kernel_attention_decode<<<head_count_, 32, 0, stream>>>(
                d_q_heads_,
                d_k_,
                d_v_,
                d_k_cache_,
                d_v_cache_,
                d_attn_out_,
                attn_layer_counter,
                pos,
                max_seq_len_,
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

            // Apply learned sigmoid gate from d_attn_gate_ to attention output
            kernel_gate_attention_output<<<(6144 + 255) / 256, 256, 0, stream>>>(
                d_attn_out_,
                d_attn_gate_,
                6144
            );

            if (has_hadamard_) {
                kernel_hadamard_forward<<<6144 / 1024, 512, 0, stream>>>(
                    d_attn_out_, d_signs_6144_, 6144
                );
            }

            kernel_gemv_tq1_0<<<x_blocks, 128, 0, stream>>>(d_attn_out_, lay.attn_output, d_res_, 6144, embedding_dim_);
            kernel_add_residual<<<(embedding_dim_ + 255) / 256, 256, 0, stream>>>(d_x_, d_res_, embedding_dim_);

            attn_layer_counter++;
        } else {
            uint32_t qkv_blocks = (10240 + 3) / 4;
            uint32_t gate_blocks = (6144 + 3) / 4;

            kernel_deltanet_alpha_beta_step<<<48, 128, 0, stream>>>(
                d_x_norm_,
                lay.ssm_alpha,
                lay.ssm_beta,
                lay.ssm_dt,
                lay.ssm_a,
                d_decay_,
                d_beta_
            );

            kernel_gemv_tq1_0<<<gate_blocks, 128, 0, stream>>>(d_x_norm_, lay.ssm_gate, d_gate_, embedding_dim_, 6144);
            kernel_gemv_tq1_0<<<qkv_blocks, 128, 0, stream>>>(d_x_norm_, lay.ssm_qkv, d_qkv_, embedding_dim_, 10240);

            // 1D Causal Depthwise Convolution step (10240 channels, K=4, with SiLU)
            kernel_ssm_conv1d_step<<<40, 256, 0, stream>>>(
                d_qkv_,
                lay.ssm_conv1d,
                d_ssm_conv_state_,
                d_conv_out_,
                ssm_layer_counter,
                stream_slot
            );

            // L2-Normalize Q (0..2047) and K (2048..4095) for all 16 key/query heads
            kernel_deltanet_l2_norm<<<16, 32, 0, stream>>>(
                d_conv_out_,
                d_conv_out_ + 2048
            );

            // Gated DeltaNet Matrix Recurrence: 48 heads, 8 blocks per head (16 cols each)
            kernel_deltanet_recurrence_fast<<<dim3(48, 8), 128, 0, stream>>>(
                d_conv_out_,          // Q: 16 heads * 128
                d_conv_out_ + 2048,   // K: 16 heads * 128
                d_conv_out_ + 4096,   // V: 48 heads * 128
                d_decay_,
                d_beta_,
                d_ssm_state_,
                d_attn_core_,
                ssm_layer_counter,
                stream_slot
            );

            // Group RMSNorm & SiLU Gating: self.norm(core_attn_out, z)
            kernel_gated_rmsnorm_silu<<<48, 128, 0, stream>>>(
                d_attn_core_,
                lay.ssm_norm,
                d_gate_,
                d_attn_out_,
                rms_norm_eps_
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

        uint32_t vocab_blocks = (vocab_size_ + 3) / 4;
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
        CUDA_CHECK(cudaStreamSynchronize(stream));

    }
}

void NativeDecoder::release_stream_slot(uint32_t stream_slot) {
    if (stream_slot >= max_streams_) return;
    stream_seq_pos_[stream_slot] = 0;
    if (page_pool_) {
        release_stream_blocks(stream_slot);
    }
}

void NativeDecoder::generate(
    const std::vector<uint32_t>& prompt_tokens_in,
    int max_tokens,
    const TokenCallback& on_token,
    const SamplingConfig& sampling,
    uint32_t stream_id,
    bool reset_state,
    const uint32_t* d_grammar_bitmask,
    const ToolCallCallback& on_tool_call,
    const uint32_t* stop_tokens,
    uint32_t stop_token_count
) {
    if (!initialized_) return;

    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_);
    uint32_t stream_slot = stream_id % max_streams_;

    size_t stream_kv_stride = 16ULL * max_seq_len_ * 4 * 256;
    size_t stream_ssm_stride = 48ULL * 48ULL * 16384ULL;
    size_t stream_conv_stride = 48ULL * 4ULL * 10240ULL;

    if (reset_state) {
        stream_seq_pos_[stream_slot] = 0;
        CUDA_CHECK(cudaMemsetAsync(d_k_cache_ + stream_slot * stream_kv_stride, 0, stream_kv_stride * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_v_cache_ + stream_slot * stream_kv_stride, 0, stream_kv_stride * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_ssm_state_ + stream_slot * stream_ssm_stride, 0, stream_ssm_stride * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_ssm_conv_state_ + stream_slot * stream_conv_stride, 0, stream_conv_stride * sizeof(float), stream));
        if (page_pool_) {
            release_stream_blocks(stream_slot);
        }
    }

    std::vector<uint32_t> prompt_tokens = prompt_tokens_in;
    if (prompt_tokens.empty()) prompt_tokens.push_back(bos_token_id_);

    uint32_t seq_pos = stream_seq_pos_[stream_slot];

    // 1. Dynamic Grammar Bitmask Initial State Sync
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

        // Pre-allocate all prefill blocks in a single batched transfer
        uint32_t total_prefill_blocks = (prompt_tokens.size() + paged_kv_.block_size - 1) / paged_kv_.block_size;
        allocate_stream_blocks_batch(stream_slot, 0, total_prefill_blocks);
    }

    // Prefill Phase with KV-Cache & Prefix Reuse (Issue H22B)
    size_t start_token_idx = 0;
    if (!reset_state && seq_pos > 0 && prompt_tokens.size() > seq_pos) {
        // Optimization: The prefix tokens (0..seq_pos-1) are already resident in KV cache and SSM state
        start_token_idx = seq_pos;
    } else if (!reset_state && seq_pos > 0 && prompt_tokens.size() <= seq_pos) {
        // Prompt was trimmed or reset; reset seq_pos
        seq_pos = 0;
        stream_seq_pos_[stream_slot] = 0;
        CUDA_CHECK(cudaMemsetAsync(d_k_cache_ + stream_slot * stream_kv_stride, 0, stream_kv_stride * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_v_cache_ + stream_slot * stream_kv_stride, 0, stream_kv_stride * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_ssm_state_ + stream_slot * stream_ssm_stride, 0, stream_ssm_stride * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_ssm_conv_state_ + stream_slot * stream_conv_stride, 0, stream_conv_stride * sizeof(float), stream));
        if (page_pool_) {
            release_stream_blocks(stream_slot);
        }
    }

    for (size_t i = start_token_idx; i < prompt_tokens.size(); ++i) {
        if (page_pool_) {
            uint32_t logical_block = seq_pos / paged_kv_.block_size;
            allocate_stream_block(stream_slot, logical_block);
        }
        bool is_last = (i + 1 == prompt_tokens.size());
        forward_token(prompt_tokens[i], seq_pos, is_last, stream_slot, active_bitmask);
        seq_pos++;
        if (seq_pos >= max_seq_len_ - 1) break;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 2. Autoregressive Generation Loop
    int generated_count = 0;
    std::vector<uint32_t> generated_tokens;
    generated_tokens.reserve(max_tokens);

    std::string accumulated_text;

    while (generated_count < max_tokens && seq_pos < max_seq_len_ - 1) {
        uint32_t next_token = sample_next_token(sampling, generated_tokens);
        float logit = h_top_logits_[0];

        bool is_stop = (next_token == eos_token_id_ || next_token == bos_token_id_ || next_token == tokenizer_.im_end_id());
        if (!is_stop && stop_tokens != nullptr && stop_token_count > 0) {
            for (uint32_t s = 0; s < stop_token_count; ++s) {
                if (next_token == stop_tokens[s]) {
                    is_stop = true;
                    break;
                }
            }
        }

        if (is_stop) {
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
                ToolCall tc = parse_tool_call_json(accumulated_text, stream_id);
                if (!tc.name.empty()) {
                    on_tool_call(tc);
                    stream_seq_pos_[stream_slot] = seq_pos;
                    return;
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
        if (on_tool_call) {
            // 1. Tagged format: <tool_call> ... </tool_call>
            if (accumulated_text.find("</tool_call>") != std::string::npos) {
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

            // 2. Untagged/Raw JSON format: {"name": "...", "arguments": {...}}
            if (token_str.find('}') != std::string::npos) {
                size_t name_pos = accumulated_text.find("\"name\"");
                if (name_pos != std::string::npos) {
                    size_t start_obj = accumulated_text.rfind('{', name_pos);
                    if (start_obj != std::string::npos) {
                        int depth = 0;
                        bool in_str = false;
                        bool esc = false;
                        for (size_t k = start_obj; k < accumulated_text.size(); ++k) {
                            char c = accumulated_text[k];
                            if (in_str) {
                                if (esc) esc = false;
                                else if (c == '\\') esc = true;
                                else if (c == '"') in_str = false;
                            } else {
                                if (c == '"') in_str = true;
                                else if (c == '{') depth++;
                                else if (c == '}') {
                                    depth--;
                                    if (depth == 0) {
                                        std::string raw_obj = accumulated_text.substr(start_obj, k - start_obj + 1);
                                        ToolCall tc = parse_tool_call_json(raw_obj, stream_id);
                                        if (!tc.name.empty()) {
                                            on_tool_call(tc);
                                            stream_seq_pos_[stream_slot] = seq_pos;
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
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

        // Dynamically synchronize next step grammar bitmask
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
    const TokenCallback& on_token,
    const SamplingConfig& sampling,
    uint32_t stream_id,
    bool reset_state,
    const uint32_t* d_grammar_bitmask,
    const ToolCallCallback& on_tool_call
) {
    if (!initialized_) return;

    // Format prompt with ChatML if needed
    std::string formatted_prompt = prompt;
    if (reset_state && prompt.find("<|im_start|>") == std::string::npos) {
        formatted_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    }

    std::vector<uint32_t> prompt_tokens = tokenizer_.encode(formatted_prompt);
    if (prompt_tokens.empty()) prompt_tokens.push_back(bos_token_id_);

    generate(prompt_tokens, max_tokens, on_token, sampling, stream_id, reset_state,
             d_grammar_bitmask, on_tool_call, nullptr, 0);
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
    server.poll_and_reply_pings();
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

void NativeDecoder::generate(
    const std::vector<uint32_t>& prompt_tokens,
    int max_tokens,
    ipc::ShmServer& server,
    uint32_t stream_id,
    const SamplingConfig& sampling,
    bool reset_state,
    const uint32_t* d_grammar_bitmask,
    const uint32_t* stop_tokens,
    uint32_t stop_token_count
) {
    server.poll_and_reply_pings();
    generate(
        prompt_tokens,
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
        },
        stop_tokens,
        stop_token_count
    );
}

} // namespace hydrogen
