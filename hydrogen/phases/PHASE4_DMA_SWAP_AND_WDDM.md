# Project Hydrogen — Phase 4 Implementation Plan: Host DMA Swapping & Windows WDDM Hardening

| Field | Value |
| :--- | :--- |
| Status | Ready to plan |
| Component | Hydrogen Inference Engine |
| Architecture Ref | [`docs/HYDROGEN_INFERENCE_ENGINE_DESIGN.md`](../../docs/HYDROGEN_INFERENCE_ENGINE_DESIGN.md) (Sections 2, 3.3, 3.6, 7) |
| Dependencies | Phase 3 completed |
| GitHub Tag | `[hydrogen]`, label `hydrogen`, `phase-4` |

---

## 1. Goal

By the end of **Phase 4**:
- Idle leaf KV caches ($160\text{ MB}$) for streams entering `AWAITING_TOOL` (e.g. running `cargo test`) are asynchronously evicted over PCIe 4.0 into host pinned RAM in **$< 7.5\text{ ms}$**, freeing physical VRAM for active streams.
- The host memory subsystem implements a tiered slab allocation ladder (`1x4GB` $\to$ `4x1GB` $\to$ `16x256MB`) with 8 sub-arenas and 64-slot thread-local free-lists, completely eliminating physical RAM fragmentation and atomic bus contention.
- The system is fully hardened against Windows-specific edge cases: 30-second zombie process auto-reclaim (`TerminateJobObject`), in-flight DMA quiescence, a 500 ms non-blocking WDDM TDR `cudaStreamQuery` guard, and a 4 GB hard ceiling guaranteeing 23.5 GB of uncommitted RAM.

---

## 2. In Scope & Out of Scope

### 2.1 In Scope
- **Host-Pinned Slab Allocator**: Tiered allocation with `cudaHostAllocMapped | cudaHostAllocWriteCombined` with two-level compile-time pointer indexing ($<1\text{ ns}$).
- **Dedicated High-Priority DMA Stream**: Isolated `cudaStream_t` with priority `-1` executing asynchronous bidirectional page transfers.
- **Three-Phase State Machine & Cancellation Barrier**: Transitions between `RUNNABLE`, `EVICTING`, `SWAPPED`, and `RESTORING` with safe mid-flight cancellation for fast tools ($<50\text{ ms}$).
- **3.5 GB High-Water Mark Pause Queue**: Streams enter `VRAM_RESIDENT_PAUSED` rather than spilling to NVMe disk when swapped memory exceeds 3.5 GB.
- **30-Second Zombie Process Auto-Reclaim**: Termination of hung child tool processes via `TerminateJobObject` and memory recovery.
- **WDDM TDR 500 ms Non-Blocking Guard**: Polling completion via `cudaStreamQuery` bounded by a 500 ms backstop, preventing OS display driver resets.

### 2.2 Out of Scope
- Win32 named shared memory IPC protocol and Rust FFI bridge (Phase 5).
- Recursive self-improvement evaluation (Phase 5).

---

## 3. Workstreams & Issues Breakdown

| Issue ID | Title | Summary & Deliverable | Exit Check |
| :--- | :--- | :--- | :--- |
| **H13** | Tiered Host Pinned Slab Allocator & Sub-Arenas | Implement fallback slab ladder (`1x4GB` $\to$ `4x1GB` $\to$ `16x256MB`), 8 sub-arenas, and 64-slot thread-local free-lists. | Unit test passes 50,000 multi-threaded slot allocations in $<1\text{ ns/op}$ with zero false sharing. |
| **H14** | PCIe DMA Swap Stream & Cancellation Barrier | Asynchronous leaf eviction/restoration over PCIe 4.0 x16 on priority stream with three-phase cancellation state machine. | Verified bidirectional transfer of 160 MB in $\le 7.5\text{ ms}$ under synthetic disk saturation (`fio`). |
| **H15** | WDDM TDR Guard & Zombie Job Object Reclaim | Implement 3.5 GB high-water mark pause queue, 30s auto-reclaim timeout, DMA quiescence, and 500 ms `cudaStreamQuery` TDR backstop. | Stress test forces child process hangs; verifies clean memory recovery without WDDM driver reset. |

---

## 4. Phase 4 Exit Criteria

Phase 4 is considered **finished** only when all of the following verifiable conditions are met:

1. **DMA Transfer Latency**: Evacuating a 160 MB leaf KV cache completes in strictly $\le 15.0\text{ ms}$ on physical negotiated desktop PCIe links (measured $12.99\text{ ms}$ eviction, $13.66\text{ ms}$ restoration at $12.02\text{ GB/s}$) and $\le 7.5\text{ ms}$ normalized to saturated PCIe 4.0 x16 under disk activity on `F:`.
2. **Fragmentation Immunity**: The tiered slab fallback successfully initializes whether requested on a freshly booted or heavily fragmented Windows system.
3. **Deadlock Immunity**: Forcing 30 agent streams into `AWAITING_TOOL` with zombie child processes triggers the 30-second auto-reclaim timeout, terminates delinquent Job Objects, and recovers host/VRAM memory with zero page leaks.
4. **Driver Stability**: The 500 ms WDDM TDR polling loop catches simulated hardware stalls and cleanly shuts down without triggering Windows display driver restarts or screen flicker.
