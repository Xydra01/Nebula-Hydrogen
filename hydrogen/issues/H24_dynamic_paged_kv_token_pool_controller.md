# Issue Plan: H24 — Dynamic Paged KV Token Pool & Elastic Context-per-Stream Controller

| Field | Value |
| :--- | :--- |
| Issue ID | H24 |
| Title | `[hydrogen] feat(memory): Dynamic Paged KV Token Pool & Elastic Context-per-Stream Controller` |
| Phase | Phase 6: Native Engine & Autonomous Agent Execution |
| Labels | `hydrogen`, `phase-6`, `core`, `memory`, `cuda` |
| Target Files | `hydrogen/include/page_pool.hpp`, `hydrogen/include/native_decoder.hpp`, `hydrogen/src/model/native_decoder.cu`, `hydrogen/src/server_main.cpp`, `crates/nebula-model/src/hydrogen.rs` |

---

## 1. Problem Statement

On a 12 GB GPU (RTX 4070), static $B \times L$ KV cache pre-allocation creates severe operational bottlenecks:
1. **PCIe Memory Spillover**: Pre-allocating contiguous buffers for 8 streams at 8,192 context requires 8.19 GB of KV cache and 1.15 GB of SSM state. Combined with 5.54 GB of model weights and ~2.2 GB consumed by Windows Desktop Window Manager (`dwm.exe`), total VRAM demands exceed 14.5 GB. Windows WDDM forcibly pages the excess into system RAM across the PCIe 4.0 x16 bus ("Shared GPU Memory"), dropping decode bandwidth from 504 GB/s down to ~25 GB/s and throttling token generation from 30+ t/s to ~3 t/s.
2. **Artificial Context Starvation**: In single-agent long-context scenarios (such as deep codebase search or extensive `<think>` reasoning traces reaching 16k–32k tokens), the model cannot exceed 8,192 tokens because the memory is locked into unused static stream slots.
3. **Rigid Resource Partitioning**: Workloads naturally vary between 1 deep-reasoning stream (needs 32k context) and multiple parallel tool-execution streams (need 2k–4k context each). Static allocation cannot adapt to these shifts.

### Architectural Advantage of Hydrogen's Hybrid Model
Hydrogen combines 48 Gated DeltaNet (SSM) layers and 16 Full Attention layers:
- **SSM State is O(1) with respect to context length**: Each stream requires a fixed $128 \times 128$ recurrence matrix per head ($\approx 144\text{ MiB}$ per stream), invariant to whether sequence length is 50 tokens or 64,000 tokens.
- **Only the 16 Attention layers scale with context**: Attention requires $16\text{ layers} \times 4\text{ heads} \times 256\text{ dim} \times 4\text{ bytes} \times 2\text{ (K+V)} = \mathbf{64\text{ KiB per token}}$.

---

## 2. Implementation Plan

### 1. Global Dynamic Paged KV Token Pool
- Replace the legacy static `d_k_cache_` and `d_v_cache_` flat arrays with a unified, shared `PagePool` of 16-token physical blocks (Issue H06).
- Sizing the shared physical token pool:
  $$\text{Pool Size} = 2,048\text{ blocks} \times 16\text{ tokens/block} = 32,768\text{ tokens total}$$
  $$\text{VRAM Footprint} = 32,768 \times 64\text{ KiB} = \mathbf{2,048\text{ MiB (2.0 GiB)}}$$
- Total GPU memory footprint with weights and pool:
  $$5.54\text{ GB (Weights)} + 2.00\text{ GB (Global Token Pool)} + 0.58\text{ GB (4 Stream SSM)} + 0.20\text{ GB (Working)} = \mathbf{\sim 8.32\text{ GB}}$$
  Leaving **~3.9 GB of true physical headroom** for Windows DWM and background applications with 0 MiB Shared GPU Memory paging.

### 2. Elastic Context-per-Stream Controller
- Streams allocate physical 16-token page IDs on demand via `allocate_stream_block()` as new tokens are emitted.
- Supported elastic profiles within the same 2.0 GB VRAM budget:
  - **1 Active Stream**: Scales up to **32,768 context** (full codebase ingest / deep reasoning).
  - **2 Concurrent Streams**: Scale up to **16,384 context** each.
  - **4 Concurrent Streams**: Scale up to **8,192 context** each.
  - **8 Concurrent Streams**: Scale up to **4,096 context** each.

### 3. WDDM Headroom Governor
- At startup and during stream allocation, query `cudaMemGetInfo` to verify physical VRAM availability.
- Enforce a strict OS safety buffer (e.g. 2.5 GB reserved for DWM and display composition).
- If physical free VRAM approaches the watermark, the controller denies additional stream expansion or queues the request rather than allowing WDDM to spill over PCIe.

### 4. Zero-Copy Block Eviction & Stream Lifecycle
- When a task completes (`MSG_RELEASE_STREAM` or EOS token emission), all physical pages allocated to the stream's block table are immediately released back to `PagePool::free_list_`.
- When combined with `RadixTree` prefix caching (Issue H05/H22), common workspace prompts (system prompt, tool schemas, repository maps) remain pinned in shared root pages without duplicate allocation.

---

## 3. Verification & Proof Plan

1. **Long-Context Test (32k Tokens)**:
   - Run a single stream up to 32,768 tokens.
   - Verify zero out-of-memory errors and verify tokens are correctly retrieved via paged attention lookup.
2. **Elastic Concurrency Test**:
   - Run 4 concurrent streams expanding to 8,192 tokens each.
   - Verify all 4 streams generate simultaneously without page corruption.
3. **WDDM Spillover Verification**:
   - Query `nebula resources` and `nvidia-smi` during generation.
   - Verify Shared GPU Memory usage remains strictly **0 MiB** and physical VRAM remains $\le 9.5\text{ GB}$.
4. **Throughput Benchmark**:
   - Verify decode speed maintains $\ge 25\text{ tokens/s}$ across the entire context window without PCIe paging stalls.

---

## 4. Acceptance Criteria

- [ ] Flat static `d_k_cache_` and `d_v_cache_` allocations are replaced by the dynamic `PagePool` allocator by default.
- [ ] Single-stream generation can dynamically scale up to 32,768 tokens without static pre-allocation waste.
- [ ] Multi-stream concurrent generation dynamically draws blocks from the global pool without cross-stream page leakage.
- [ ] Windows WDDM Shared GPU Memory allocation remains strictly 0 MiB under heavy generation.
- [ ] Physical pages are immediately recycled upon stream release.
