# Issue Plan: H21-B — Binary Zero-Copy IPC Request Framing & Stream-Lease Registry

| Field | Value |
| :--- | :--- |
| Issue ID | H21-B |
| Title | `[hydrogen] feat(ipc): Binary zero-copy token ring framing and stream-lease registry (#H21-B)` |
| Phase | Phase 6: Native Engine & Autonomous Agent Execution |
| Labels | `hydrogen`, `phase-6`, `core`, `ipc` |
| Target Files | `hydrogen/src/ipc/shm_server.cpp`, `hydrogen/src/server_main.cpp`, `crates/nebula-hydrogen/src/ipc.rs` |

---

## 1. Problem Statement

Two architectural flaws degrade IPC efficiency and stability under high agent task churn:
1. **Multi-Turn JSON Serialization Tax**: `extract_chatml_prompt_from_json()` in `server_main.cpp` scans raw JSON strings using manual CPU loops. For 16K–100K character contexts (large tool observations and conversation histories), string scanning takes $1\text{–}3\text{ ms}$ on the CPU, defeating the sub-$10\ \mu\text{s}$ advantage of shared memory.
2. **Stream Slot Collision**: In `NativeDecoder::generate()`, the stream slot is assigned via `stream_id % MAX_STREAMS`. If `nebula-daemon` spawns tasks with monotonically increasing stream IDs, slots wrap around after 8. If an earlier task is still resident or has un-evicted KV pages, its state is overwritten or corrupted.

---

## 2. Implementation Plan

1. **Binary Token Request Framing**:
   - Define `MessageType::MSG_GENERATE_TOKENS_REQUEST` in `shm_layout.hpp`.
   - Payload struct contains:
     - `uint32_t stream_id`
     - `uint32_t num_tokens`
     - `uint32_t max_tokens`
     - `float temperature`
     - `float top_p`
     - `uint32_t token_ids[MAX_PROMPT_TOKENS]`
   - Update `crates/nebula-hydrogen/src/ipc.rs` to write pre-tokenized `uint32_t` arrays directly into the shared memory request ring, bypassing all JSON string generation and CPU scanning.

2. **Stream-Lease Registry**:
   - Maintain an atomic bitmap in `ShmHeader`: `std::atomic<uint32_t> active_stream_mask`.
   - Expose explicit `acquire_stream_slot()` and `release_stream_slot(uint32_t slot_id)` methods.
   - When a slot is released, trigger immediate reclamation or eviction of its resident KV pages in the Radix Tree.
   - If all 8 slots are leased, return an immediate `MSG_ERROR_SLOTS_EXHAUSTED` backpressure frame so the daemon queues the task.

3. **Rust RAII `StreamLeaseGuard`**:
   - Implement `StreamLeaseGuard` in `crates/nebula-model/src/hydrogen.rs`.
   - Acquires a lease on task start and automatically releases the slot on `Drop`.

---

## 3. Verification & Proof Plan

- Run `hydrogen_bench_shm_ipc.exe --binary-framing --tokens 16384`.
- **Proof Metric 1**: IPC request ingestion latency for 16,384 tokens is $\le 10.0\ \mu\text{s}$ (sub-$15\ \mu\text{s}$ p99).
- **Proof Metric 2**: Churn test spawning 100 consecutive tasks across 8 slots shows 0 slot collisions and 0 memory leaks in the page pool.

---

## 4. Acceptance Criteria

- [ ] Binary token array frame defined in C++ and mirrored in Rust FFI.
- [ ] `crates/nebula-hydrogen` writes pre-tokenized arrays directly into SHM.
- [ ] Atomic stream-lease registry enforces exclusive ownership of stream slots $0..7$.
- [ ] Explicit KV cache recycling executed on stream lease release.
- [ ] Zero CPU string parsing during generation request ingestion.
