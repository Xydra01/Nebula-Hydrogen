# Issue Plan: H22-B — Multi-Stream Unified Batched GEMM (M=B) Scheduler

| Field | Value |
| :--- | :--- |
| Issue ID | H22-B |
| Title | `[hydrogen] feat(batcher): Multi-stream unified batched GEMM (M=B) scheduler (#H22-B)` |
| Phase | Phase 6: Native Engine & Autonomous Agent Execution |
| Labels | `hydrogen`, `phase-6`, `core`, `compute`, `performance` |
| Target Files | `hydrogen/src/batcher/split_fuse.cu`, `hydrogen/src/model/native_decoder.cu`, `hydrogen/include/batcher.hpp` |

---

## 1. Problem Statement

In the current implementation, `NativeDecoder::generate()` decodes streams sequentially. Single-stream decode step time is $44.25\text{ ms}$ ($22.6\text{ tok/s}$). If 8 parallel agent streams decode serially, total step time scales to $8 \times 44.25\text{ ms} = 354\text{ ms}$, dropping aggregate throughput to $22.6\text{ tok/s}$.

The root bottleneck is that each sequential stream must re-read the entire $5.9\text{ GB}$ model weight matrix from VRAM into GPU SRAM. To achieve $\ge 500\text{ tok/s}$ aggregate throughput on the RTX 4070, all $B \le 8$ active stream tokens must be packed into a unified batched GEMM matrix multiplication ($M=B$) so that weights are read exactly once per decoding step.

---

## 2. Implementation Plan

1. **Active Stream Token Packing**:
   - In `SplitFuseBatcher::step()`, gather the next token from every active decoding stream ($B \le 8$) into a contiguous device buffer `d_batch_tokens[B]`.
   - Forward embedding lookup gathers $B$ input vectors into an $[B \times d_{\text{model}}]$ activation tensor.

2. **Unified Batched GEMM Dispatch ($M=B$)**:
   - For all 64 layers (SSM and Attention):
     - Dispatch linear projections as $[B \times K] \times [K \times N] \rightarrow [B \times N]$ GEMMs using Marlin / CUTLASS.
     - Weight matrices are streamed across the $504\text{ GB/s}$ bus once, computing outputs for all $B$ streams in parallel.
   - For Attention layers:
     - Invoke FlashInfer Cascade Attention with batched query heads and paged KV table pointers per stream.
   - For SSM layers:
     - Execute batched state updates with stream-indexed state pointers.

3. **Demultiplexing & Parallel Logit Sampling**:
   - Compute final logits for all $B$ streams in a single $[B \times V]$ GEMV pass.
   - Launch GPU stochastic sampler with grid dimension $B$, producing next tokens for all streams in parallel.

---

## 3. Verification & Proof Plan

- Run `hydrogen_bench_parallel_streams.exe -m "F:\Nebula\models\bonsai2-27b\Ternary-Bonsai-2-27B-PTQ1_0.gguf" --streams 8 --tokens 64`.
- **Proof Metric 1**: Decode step duration for 8 concurrent streams is $\le 16.0\text{ ms}$ ($> 500\text{ tok/s}$ aggregate throughput).
- **Proof Metric 2**: Individual stream latency remains $\le 16.0\text{ ms}$ per token (no single stream starves).
- **Proof Metric 3**: Per-stream generated tokens exactly match deterministic single-stream reference outputs.

---

## 4. Acceptance Criteria

- [ ] Linear projections in `native_decoder.cu` operate on unified $[B \times d]$ matrices when $B > 1$.
- [ ] Model weights are read once per step regardless of batch size $B \le 8$.
- [ ] FlashInfer Cascade Attention kernel dispatches batched queries with stream-specific page table lookups.
- [ ] Aggregate throughput scales linearly from $22.6\text{ tok/s}$ ($B=1$) to $\ge 500\text{ tok/s}$ ($B=8$).
- [ ] Zero cross-stream token contamination or KV cache corruption.
