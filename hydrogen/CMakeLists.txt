cmake_minimum_required(VERSION 3.28)
project(hydrogen LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Enforce ISO C++20 for host and device compilation
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Target NVIDIA Ada Lovelace SM 8.9 (RTX 4070)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Common runtime output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler flags
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/std:c++20>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CXX>:/Oi>
        $<$<COMPILE_LANGUAGE:CXX>:/Gy>
        $<$<COMPILE_LANGUAGE:CXX>:/DNOMINMAX>
        $<$<COMPILE_LANGUAGE:CXX>:/DWIN32_LEAN_AND_MEAN>
        $<$<COMPILE_LANGUAGE:CXX>:/D_CRT_SECURE_NO_WARNINGS>
    )
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/permissive-,/W4>
    )
endif()

# Sanity Check Target (Issue H01)
add_executable(hydrogen_sanity_test src/sanity_check.cu)

target_include_directories(hydrogen_sanity_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(hydrogen_sanity_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Model Subsystem Library (Issue H02, H19)
add_library(hydrogen_model STATIC
    src/model/tensor_loader.cpp
    src/model/tokenizer.cpp
)

target_include_directories(hydrogen_model PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Model Inspection Utility (Issue H02)
add_executable(hydrogen_inspect_model
    src/tools/inspect_model.cpp
)

target_link_libraries(hydrogen_inspect_model PRIVATE
    hydrogen_model
)

# Standalone Tokenizer Unit Test (Issue H19)
add_executable(hydrogen_test_tokenizer
    src/tools/test_tokenizer.cpp
)

target_include_directories(hydrogen_test_tokenizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_test_tokenizer PRIVATE
    hydrogen_model
)

# IPC Subsystem Library (Issue H16)
add_library(hydrogen_ipc STATIC
    src/ipc/shm_server.cpp
)

target_include_directories(hydrogen_ipc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Forward Baseline Engine Library (Issues H03, H07, H08, H09, H19)
add_library(hydrogen_engine STATIC
    src/model/forward_baseline.cu
    src/attention/cascade_attention.cu
    src/model/marlin_gemm.cu
    src/attention/dispatch.cpp
    src/model/kernel_tiler.cu
    src/model/native_decoder.cu
)

target_include_directories(hydrogen_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_engine PUBLIC
    hydrogen_model
    hydrogen_ipc
    CUDA::cudart
)

set_target_properties(hydrogen_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Standalone Autoregressive Decoder Unit Test (Issue H19)
add_executable(hydrogen_test_autoregressive_decoder
    src/tools/test_autoregressive_decoder.cpp
)

target_include_directories(hydrogen_test_autoregressive_decoder PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_test_autoregressive_decoder PRIVATE
    hydrogen_engine
    hydrogen_ipc
    hydrogen_model
    CUDA::cudart
)

# Standalone Tool Grammar & Function Calling Test (Issue H20)
add_executable(hydrogen_test_tool_grammar_and_calling
    src/tools/test_tool_grammar_and_calling.cu
)

target_include_directories(hydrogen_test_tool_grammar_and_calling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_test_tool_grammar_and_calling PRIVATE
    hydrogen_engine
    hydrogen_ipc
    hydrogen_model
    hydrogen_grammar
    CUDA::cudart
)

# Standalone Dual Sub-Block PTQ1_0 & Walsh-Hadamard Transform Test (Issue H23)
add_executable(hydrogen_test_hadamard_and_tq1
    src/tools/test_hadamard_and_tq1.cu
)

target_include_directories(hydrogen_test_hadamard_and_tq1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_test_hadamard_and_tq1 PRIVATE
    CUDA::cudart
)

set_target_properties(hydrogen_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Forward Baseline Smoke Test Target (Issue H03)
add_executable(hydrogen_smoke_test
    src/smoke_test.cpp
)

target_include_directories(hydrogen_smoke_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_smoke_test PRIVATE
    hydrogen_engine
    hydrogen_model
)

# Memory Subsystem Library (Issues H04, H05, H06, H13, H14, H15)
add_library(hydrogen_memory STATIC
    src/memory/page_allocator.cpp
    src/memory/radix_tree.cpp
    src/memory/page_pool.cpp
    src/memory/host_slab.cpp
    src/memory/dma_swapper.cu
    src/memory/dma_guard.cpp
    src/memory/watermark_supervisor.cpp
)

target_include_directories(hydrogen_memory PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_memory PUBLIC
    CUDA::cudart
)

set_target_properties(hydrogen_memory PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Page Block Test Target (Issue H04)
add_executable(hydrogen_test_page_block
    src/tools/test_page_block.cpp
)

target_include_directories(hydrogen_test_page_block PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_test_page_block PRIVATE
    hydrogen_memory
)

# Radix Tree Test Target (Issue H05)
add_executable(hydrogen_test_radix_tree
    src/tools/test_radix_tree.cpp
)

target_include_directories(hydrogen_test_radix_tree PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_test_radix_tree PRIVATE
    hydrogen_memory
)

# Lock-Free Page Pool Stress Test Target (Issue H06)
add_executable(hydrogen_stress_test_page_pool
    src/tools/stress_test_page_pool.cpp
)

target_include_directories(hydrogen_stress_test_page_pool PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_stress_test_page_pool PRIVATE
    hydrogen_memory
)

# FlashInfer Cascade Attention Benchmark Target (Issue H07)
add_executable(hydrogen_bench_cascade_attention
    src/tools/bench_cascade_attention.cpp
)

target_include_directories(hydrogen_bench_cascade_attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_bench_cascade_attention PRIVATE
    hydrogen_engine
    hydrogen_memory
)

# Marlin / CUTLASS Quantized GEMM Benchmark Target (Issue H08)
add_executable(hydrogen_bench_marlin_gemm
    src/tools/bench_marlin_gemm.cpp
)

target_include_directories(hydrogen_bench_marlin_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_bench_marlin_gemm PRIVATE
    hydrogen_engine
    hydrogen_memory
    hydrogen_model
)

# Templated Dispatch & WDDM Clamping Benchmark Target (Issue H09)
add_executable(hydrogen_bench_dispatch_tiler
    src/tools/bench_dispatch_tiler.cpp
)

target_include_directories(hydrogen_bench_dispatch_tiler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_bench_dispatch_tiler PRIVATE
    hydrogen_engine
    hydrogen_memory
    hydrogen_model
)

# Split-Fuse Continuous Batcher Library (Issue H10)
add_library(hydrogen_batcher STATIC
    src/batcher/split_fuse.cpp
)

target_include_directories(hydrogen_batcher PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_batcher PUBLIC
    CUDA::cudart
)

# Split-Fuse Continuous Batching Benchmark Target (Issue H10)
add_executable(hydrogen_bench_split_fuse_batcher
    src/tools/bench_split_fuse_batcher.cpp
)

target_include_directories(hydrogen_bench_split_fuse_batcher PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_bench_split_fuse_batcher PRIVATE
    hydrogen_batcher
    hydrogen_engine
    hydrogen_memory
    hydrogen_model
    CUDA::cudart
)

# GPU-Fused FSM Tool Grammar Bitmask Sampler Library (Issue H11)
add_library(hydrogen_grammar STATIC
    src/grammar/fsm_compiler.cpp
    src/grammar/fsm_sampler.cu
)

target_include_directories(hydrogen_grammar PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_grammar PUBLIC
    CUDA::cudart
)

set_target_properties(hydrogen_grammar PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

target_link_libraries(hydrogen_engine PUBLIC
    hydrogen_grammar
    hydrogen_memory
)

# GPU-Fused FSM Tool Grammar Sampler Benchmark Target (Issue H11)
add_executable(hydrogen_bench_fsm_sampler
    src/tools/bench_fsm_sampler.cpp
)

target_include_directories(hydrogen_bench_fsm_sampler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_bench_fsm_sampler PRIVATE
    hydrogen_grammar
    CUDA::cudart
)

# Speculative Tool Dispatcher & Safety Barrier Library (Issue H12)
add_library(hydrogen_tools STATIC
    src/tools/speculative_dispatcher.cpp
)

target_include_directories(hydrogen_tools PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_tools PUBLIC
    CUDA::cudart
)

# Speculative Safety Verification Test Target (Issue H12)
add_executable(hydrogen_test_speculative_safety
    src/tools/test_speculative_safety.cpp
)

target_include_directories(hydrogen_test_speculative_safety PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_test_speculative_safety PRIVATE
    hydrogen_tools
    CUDA::cudart
)

# Tiered Host Pinned Slab Allocator Benchmark Target (Issue H13)
add_executable(hydrogen_bench_host_slab
    src/memory/bench_host_slab.cpp
)

target_include_directories(hydrogen_bench_host_slab PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_bench_host_slab PRIVATE
    hydrogen_memory
    CUDA::cudart
)

# PCIe 4.0 DMA Swap Stream Benchmark Target (Issue H14)
add_executable(hydrogen_bench_dma_swapper
    src/memory/bench_dma_swapper.cu
)

target_include_directories(hydrogen_bench_dma_swapper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_bench_dma_swapper PRIVATE
    hydrogen_memory
    CUDA::cudart
)

# 30s Zombie Process Auto-Reclaim & 500ms WDDM TDR Guard Test Target (Issue H15)
add_executable(hydrogen_test_deadlock_and_tdr
    src/tools/test_deadlock_and_tdr.cpp
)

target_include_directories(hydrogen_test_deadlock_and_tdr PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_test_deadlock_and_tdr PRIVATE
    hydrogen_memory
    CUDA::cudart
)


# Win32 Named Shared Memory IPC Benchmark Target (Issue H16)
add_executable(hydrogen_bench_shm_ipc
    src/tools/bench_shm_ipc.cpp
)

target_include_directories(hydrogen_bench_shm_ipc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_bench_shm_ipc PRIVATE
    hydrogen_ipc
)

# Standalone Hydrogen Inference Engine Daemon Server
add_executable(hydrogen_engine_bin
    src/server_main.cpp
)

target_include_directories(hydrogen_engine_bin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hydrogen_engine_bin PRIVATE
    hydrogen_engine
    hydrogen_ipc
    hydrogen_model
    CUDA::cudart
)

set_target_properties(hydrogen_engine_bin PROPERTIES
    OUTPUT_NAME "hydrogen_engine"
)





