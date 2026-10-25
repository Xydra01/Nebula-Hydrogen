#include "gemm.hpp"
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    constexpr std::size_t M = 4;
    constexpr std::size_t N = 4;
    constexpr std::size_t K = 4;

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N, 0.0f);

    matmul_naive(A.data(), B.data(), C.data(), M, N, K);

    for (float val : C) {
        assert(std::fabs(val - 8.0f) < 1e-4f);
    }
    return 0;
}
