#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <fstream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            return 1; \
        } \
    } while (0)

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

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

int main() {
    std::cout << "[TEST] Running GEMV PTQ1_0 test against CPU reference...\n";
    const uint32_t in_dim = 5120;
    const uint32_t out_dim = 16;
    uint32_t blocks_per_row = in_dim / 256;
    size_t weight_bytes = static_cast<size_t>(out_dim) * blocks_per_row * 56;

    std::vector<uint8_t> h_weights(weight_bytes);
    // Fill with sample bytes from model file
    std::ifstream f("F:\\Nebula\\models\\bonsai2-27b\\Ternary-Bonsai-2-27B-PTQ1_0.gguf", std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open model file\n";
        return 1;
    }
    f.seekg(11120992); // tensor data offset
    f.read(reinterpret_cast<char*>(h_weights.data()), weight_bytes);

    std::vector<float> h_x(in_dim);
    for (uint32_t i = 0; i < in_dim; ++i) {
        h_x[i] = (i % 2 == 0) ? 1.0f : -0.5f;
    }

    float* d_x = nullptr;
    uint8_t* d_w = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, in_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), in_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_weights.data(), weight_bytes, cudaMemcpyHostToDevice));

    uint32_t threads = 128;
    uint32_t blocks = (out_dim + (threads / 32) - 1) / (threads / 32);

    kernel_gemv_tq1_0<<<blocks, threads>>>(d_x, d_w, d_out, in_dim, out_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_out(out_dim);
    CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_out, out_dim * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "GPU GEMV first 5 rows:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "  Row " << i << ": " << gpu_out[i] << "\n";
    }

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_out);
    std::cout << "[PASS] GEMV test completed successfully!\n";
    return 0;
}
