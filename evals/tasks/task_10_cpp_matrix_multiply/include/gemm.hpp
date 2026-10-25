#pragma once

#include <cstddef>
#include <vector>

void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t N,
    std::size_t K
);
