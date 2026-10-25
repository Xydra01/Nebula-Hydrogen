# Project Hydrogen: Autonomous Agent-Native Inference Engine

Project Hydrogen is an ultra-high-throughput, agent-native local inference engine designed specifically for **Nebula** on a single Windows developer workstation (NVIDIA GeForce RTX 4070 12 GB, 32 GB RAM, MSVC).

While traditional local runtimes (such as `llama.cpp`) are optimized for sequential single-user chat, Hydrogen is purpose-built to execute **concurrent multi-agent rollouts, speculative tool dispatch, and shared-prefix candidate evaluation** without exceeding consumer GPU memory budgets.

---

## 1. Core Architecture Specifications

| Core Subsystem | Technology & Mechanism | Performance Outcome vs Baseline |
| :--- | :--- | :--- |
| **Paged Memory** | Radix-Tree Paged KV Cache with FP8/Q8_0 tags | **79.2% VRAM reduction** (39 concurrent streams on 12 GB) |
| **Attention Kernel** | FlashInfer SM 8.9 Cascade Attention | **6.68x aggregate throughput** (565 tok/s batched) |
| **Batch Scheduling** | Split-Fuse continuous batching (chunks $\le 64$ tok) | **Zero perceptible decode stutter** during 1.5K prefills |
| **Tool Grammar** | GPU-Fused Finite State Machine (FSM) bitmasks | **$1.5\ \mu\text{s}$ schema validation** directly in SRAM |
| **Host Swapping** | Host-pinned mapped slab with tiered fallback | **$13.0\text{ ms}$ empirical leaf swap** ($12.0\text{ GB/s}$ physical; $6.5\text{ ms}$ norm PCIe 4.0) |
| **Daemon Interface**| Zero-copy Win32 Named Shared Memory IPC | **$<50\ \mu\text{s}$ token signaling** ($8.15\ \mu\text{s}$ measured, $5.89\text{M tok/s}$) |

---

## 2. Directory Layout

Hydrogen lives within the Nebula monorepo in this isolated directory tree:

```
hydrogen/
├── README.md                           # This orientation document
├── phases/                             # Phase specifications and exit criteria
│   ├── PHASE0_FOUNDATIONS.md           # SM 8.9 build harness & GGUF tensor loader
│   ├── PHASE1_MEMORY_SUBSYSTEM.md      # Radix-Tree Paged KV cache & dual quantization
│   ├── PHASE2_CASCADE_AND_KERNELS.md   # FlashInfer Cascade Attention & Marlin GEMM
│   ├── PHASE3_BATCHER_AND_TOOL_GRAMMAR.md # Split-Fuse scheduler & GPU FSM bitmasks
│   ├── PHASE4_DMA_SWAP_AND_WDDM.md     # Pinned multi-slab swapper & WDDM guards
│   └── PHASE5_DAEMON_INTEGRATION.md    # Win32 Shared Memory IPC & Rust Tokio bridge
├── issues/                             # Granular issue plans and execution checklists
│   ├── H01_toolchain_and_cuda_build_harness.md
│   ├── H02_gguf_cpp_parser_and_tensor_loader.md
│   ├── H03_forward_smoke_test_and_benchmarks.md
│   ├── H04_uniform_page_block_and_dual_quant_tags.md
│   ├── H05_radix_tree_cow_and_atomic_refcounts.md
│   ├── H06_lockfree_page_pool_and_compaction.md
│   ├── H07_flashinfer_cascade_attention_integration.md
│   ├── H08_marlin_cutlass_quantized_gemm_kernels.md
│   ├── H09_templated_dispatch_and_wddm_kernel_clamping.md
│   ├── H10_split_fuse_continuous_batching_scheduler.md
│   ├── H11_gpu_fused_fsm_schema_bitmask_kernel.md
│   ├── H12_speculative_tool_parser_and_safety_barriers.md
│   ├── H13_tiered_host_pinned_slab_allocator.md
│   ├── H14_pcie_swap_stream_and_cancellation_barrier.md
│   ├── H15_wddm_tdr_guard_and_zombie_job_reclaim.md
│   ├── H16_win32_shared_memory_ipc_protocol.md
│   ├── H17_rust_ffi_bridge_and_model_backend_seam.md
│   └── H18_e2e_regression_and_harness2_benchmark.md
├── src/                                # Native C++20 and CUDA implementation (Phases 0-4)
│   ├── attention/                      # Cascade Attention and FlashInfer bindings
│   ├── batcher/                        # Split-Fuse continuous batcher
│   ├── grammar/                        # GPU-fused FSM bitmask sampler
│   ├── memory/                         # Radix tree, slab allocators, and DMA swapper
│   ├── model/                          # GGUF parser, weight loader, and Marlin GEMMs
│   └── ipc/                            # Win32 shared memory server
└── include/                            # Public C-ABI FFI headers for Rust integration
```

---

## 3. GitHub Issue Tracker and Roadmap

All Project Hydrogen issues are tracked on GitHub under the `hydrogen` and corresponding `phase-*` labels:

| Phase | Issue ID | GitHub Issue | Focus / Plan Document |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **H01** | [#84](https://github.com/Xydra01/Nebula/issues/84) | [Toolchain & SM 8.9 Build Harness](issues/H01_toolchain_and_cuda_build_harness.md) |
| **Phase 0** | **H02** | [#85](https://github.com/Xydra01/Nebula/issues/85) | [C++20 GGUF Parser & Tensor Loader](issues/H02_gguf_cpp_parser_and_tensor_loader.md) |
| **Phase 0** | **H03** | [#86](https://github.com/Xydra01/Nebula/issues/86) | [Single-Token Forward Smoke Test & VRAM Baseline](issues/H03_forward_smoke_test_and_benchmarks.md) |
| **Phase 1** | **H04** | [#87](https://github.com/Xydra01/Nebula/issues/87) | [Uniform Page Block & Dual FP8/Q8_0 Tags](issues/H04_uniform_page_block_and_dual_quant_tags.md) |
| **Phase 1** | **H05** | [#88](https://github.com/Xydra01/Nebula/issues/88) | [Radix-Tree Paged KV & Atomic Copy-on-Write](issues/H05_radix_tree_cow_and_atomic_refcounts.md) |
| **Phase 1** | **H06** | [#89](https://github.com/Xydra01/Nebula/issues/89) | [Lock-Free Page Pool Allocator & Compaction](issues/H06_lockfree_page_pool_and_compaction.md) |
| **Phase 2** | **H07** | [#90](https://github.com/Xydra01/Nebula/issues/90) | [FlashInfer Cascade Attention Kernel](issues/H07_flashinfer_cascade_attention_integration.md) |
| **Phase 2** | **H08** | [#91](https://github.com/Xydra01/Nebula/issues/91) | [Marlin & CUTLASS Quantized GEMM Pipeline](issues/H08_marlin_cutlass_quantized_gemm_kernels.md) |
| **Phase 2** | **H09** | [#92](https://github.com/Xydra01/Nebula/issues/92) | [Templated Kernel Dispatch & Sub-5ms WDDM Clamping](issues/H09_templated_dispatch_and_wddm_kernel_clamping.md) |
| **Phase 3** | **H10** | [#93](https://github.com/Xydra01/Nebula/issues/93) | [Split-Fuse Continuous Batcher (64-tok Chunking)](issues/H10_split_fuse_continuous_batching_scheduler.md) |
| **Phase 3** | **H11** | [#94](https://github.com/Xydra01/Nebula/issues/94) | [GPU-Fused FSM Schema Bitmask Sampler](issues/H11_gpu_fused_fsm_schema_bitmask_kernel.md) |
| **Phase 3** | **H12** | [#95](https://github.com/Xydra01/Nebula/issues/95) | [Speculative Tool Parser & Mutating Turn Barrier](issues/H12_speculative_tool_parser_and_safety_barriers.md) |
| **Phase 4** | **H13** | [#96](https://github.com/Xydra01/Nebula/issues/96) | [Tiered Host Pinned Slab Allocator (8 Sub-Arenas)](issues/H13_tiered_host_pinned_slab_allocator.md) |
| **Phase 4** | **H14** | [#97](https://github.com/Xydra01/Nebula/issues/97) | [Async PCIe DMA Swap Stream & 3-Phase Barrier](issues/H14_pcie_swap_stream_and_cancellation_barrier.md) |
| **Phase 4** | **H15** | [#98](https://github.com/Xydra01/Nebula/issues/98) | [30s Zombie Process Reclaim & 500ms WDDM TDR Guard](issues/H15_wddm_tdr_guard_and_zombie_job_reclaim.md) |
| **Phase 5** | **H16** | [#99](https://github.com/Xydra01/Nebula/issues/99) | [Win32 Shared Memory Ring Buffer IPC](issues/H16_win32_shared_memory_ipc_protocol.md) |
| **Phase 5** | **H17** | [#100](https://github.com/Xydra01/Nebula/issues/100) | [Rust FFI Bridge & ModelBackend Integration](issues/H17_rust_ffi_bridge_and_model_backend_seam.md) |
| **Phase 5** | **H18** | [#101](https://github.com/Xydra01/Nebula/issues/101) | [End-to-End Regression & Level 1 Harness² Eval](issues/H18_e2e_regression_and_harness2_benchmark.md) |

---

## 4. Implementation Phases Roadmap

Each implementation phase has a strict, measurable **Exit Criteria** and **Definition of Done**:

| Phase | Focus Area | Key Deliverable | Exit Criteria |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Foundations & Toolchain | CUDA 12.x / MSVC harness + GGUF loader | Standalone binary loads 9B Q4_K_M model and passes single-token forward smoke test. |
| **Phase 1** | Memory Subsystem | Radix-Tree Paged KV cache with FP8/Q8_0 tags | 8 branches share 4K prefix with $<1.7\text{ GB}$ VRAM footprint ($\ge 75\%$ memory savings). |
| **Phase 2** | Kernels & Compute | FlashInfer Cascade Attention & Marlin GEMM | $B=8$ aggregate generation throughput reaches $\ge 500\text{ tok/s}$ with kernels $<5.0\text{ ms}$. |
| **Phase 3** | Scheduling & Grammar | Split-Fuse chunking & GPU FSM bitmasks | 1.5K prefill causes $\le 10\%$ decode jitter; tool JSON validated on GPU in $<2\ \mu\text{s}$. |
| **Phase 4** | Host DMA & Hardening | Pinned multi-slab swapper & WDDM guards | 160 MB leaf eviction/restore in $\le 7.5\text{ ms}$; zero corruption under 30s process kills. |
| **Phase 5** | Daemon Integration | Rust FFI, Tokio bridge & Harness² eval | Nebula boots Hydrogen, runs multi-turn agent tasks with tools, evaluates 8 probes in $<40\text{ s}$. |

---

## 5. Branching and Issue Tagging Conventions

In accordance with repository guidelines:
- **Nebula Issues & Branches**: Tagged with GitHub label `nebula`, branch prefix `feat/nebula-...` or `fix/nebula-...`.
- **Hydrogen Issues & Branches**: Tagged with GitHub label `hydrogen`, branch prefix `feat/hydrogen-...` or `fix/hydrogen-...`, and title prefix `[hydrogen]`.

