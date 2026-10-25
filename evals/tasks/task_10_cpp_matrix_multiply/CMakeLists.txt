cmake_minimum_required(VERSION 3.20)
project(task_10_gemm CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(gemm src/gemm.cpp)
target_include_directories(gemm PUBLIC include)

enable_testing()
add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE gemm)
add_test(NAME test_gemm COMMAND test_gemm)
