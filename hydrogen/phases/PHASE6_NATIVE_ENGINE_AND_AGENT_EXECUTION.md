# Project Hydrogen — Phase 6 Implementation Plan: Native Engine & Autonomous Agent Execution

| Field | Value |
| :--- | :--- |
| Status | Ready to implement |
| Component | Hydrogen Native Inference Engine & Nebula Daemon Core |
| Architecture Ref | [`docs/HYDROGEN_INFERENCE_ENGINE_DESIGN.md`](../../docs/HYDROGEN_INFERENCE_ENGINE_DESIGN.md) (Sections 1, 2, 3, 5, 7) |
| Dependencies | Phases 0–5 completed |
| GitHub Tag | `[hydrogen]`, label `hydrogen`, `phase-6` |

---

## 1. Goal

By the end of **Phase 6**:
- The standalone C++20/CUDA server binary (`hydrogen_engine.exe`) replaces the synthetic echo harness with a fully integrated, multi-stream autoregressive decoder running native GGUF weights on NVIDIA Ada Lovelace SM 8.9 (RTX 4070 12 GB).
- Nebula users can hold interactive conversational chat (`nebula chat`) running entirely on the native Hydrogen engine with proper ChatML templating, multi-turn context retention, and GPU stochastic sampling (temperature, top-p, min-p, repetition penalty).
- Nebula can execute autonomous coding tasks (`nebula task`) through the native engine, emitting schema-constrained tool calls (`fs.read`, `fs.write`, `shell.run`, `git.*`) with 0% JSON syntax failures via in-engine grammar bitmasks.
- The engine and IPC transport support **at least 8 parallel agent streams** decoding simultaneously in $< 3.5\text{ GB}$ of KV VRAM, utilizing Radix-Tree prefix deduplication, stream-isolated SSM recurrent state matrices, and a non-interfering IPC demuxer router.

---

## 2. In Scope & Out of Scope

### 2.1 In Scope
- **Production Autoregressive Decoder Integration**: Linking `native_decoder.cu` into CMake, wiring GGUF tensor weights into 64 hybrid layers (48 SSM + 16 Attention), and implementing stochastic sampling.
- **Robust GGUF Tokenizer & ChatML Formatter**: Dual BPE merge table and SentencePiece byte-fallback tokenization (`<0x00>`...`<0xFF>`), parsing multi-turn `ChatMessage` histories into Qwen/ChatML tokens.
- **In-Engine Tool Grammar & IPC Serialization**: Fused FSM bitmask schema constraints, `<tool_call>` extraction, and native `MessageType::ToolCall` frame serialization.
- **Multi-Stream IPC Demuxer Router**: Eliminating the frame-stealing bug in `crates/nebula-model/src/hydrogen.rs` with a dedicated reader task dispatching to per-stream channels.
- **Continuous Split-Fuse Batching & Stream-Isolated SSM Memory**: Orchestrating $B \ge 8$ streams with 64-token clamped prefill and private `[B_max, 48, ssm_inner]` recurrent state allocation.
- **Paged Radix-Tree KV Integration**: Hooking `PagePool` and `RadixTree` CoW into the decoding loop to deduplicate shared workspace system prompts.
- **End-to-End Verification**: Live validation of `nebula chat`, `nebula task`, and 8-stream parallel candidate probe execution under real model weights.

### 2.2 Out of Scope
- Distributed multi-GPU or remote cluster inference (Phase 7+).
- In-tree model fine-tuning or LoRA adapter training (Phase 8+).

---

## 3. Workstreams & Issues Breakdown

| Issue ID | Title | Summary & Deliverable | Exit Check |
| :--- | :--- | :--- | :--- |
| **H19** | Full Autoregressive Decoder Integration & BPE Tokenizer | Integrate `native_decoder.cu` into CMake; build real dual BPE / byte-fallback tokenizer; implement ChatML formatter; implement GPU stochastic sampler kernel (temperature, top-p, min-p, repetition penalty). | Coherent text generation matching llama.cpp perplexity within $\Delta\text{PPL} < 0.05$. Verified via `hydrogen_smoke_test -p "Write a Rust hello world"`. |
| **H20** | In-Engine Tool Grammar & Function Calling Parser | Integrate H11 GPU-Fused FSM bitmask schemas into decoder loop; emit `<tool_call>` tokens and serialize `MessageType::ToolCall` IPC frames; handle tool observation feedback turns without cache eviction. | 100 tool-call turns executed with 0% schema validation errors and 0 syntax failures. |
| **H21** | Multi-Stream IPC Demuxer & Concurrent Router | Fix frame-stealing bug in `crates/nebula-model/src/hydrogen.rs` with centralized background demuxer task routing to per-stream channels; update `ShmServer` to handle interleaved stream IDs. | 8 concurrent Tokio streams receive bit-exact token streams without frame drops, timeouts, or cross-stream bleeding. |
| **H22** | Batched Continuous Scheduler & Paged Radix KV | Integrate `SplitFuse` batcher (H10), `RadixTree` (H05), and `CascadeAttention` (H07) into decoder; segment SSM recurrent state matrix `[B, 48, 6144]` per stream; clamp prefills to $M \le 64$. | 8 parallel streams achieve aggregate throughput $\ge 500\text{ tokens/s}$ in $< 3.5\text{ GB}$ KV cache without WDDM driver reset. |
| **H23** | End-to-End Conversational Chat & Parallel Agent Task Eval | Wire `nebula chat` and `nebula task` to native Hydrogen engine; execute multi-turn autonomous coding task in isolated worktree; verify 8-probe parallel evaluation runs in $< 40\text{ s}$. | 1. `nebula chat` holds normal multi-turn conversation.<br>2. `nebula task` completes multi-step bug fix with tools.<br>3. 8-probe parallel eval finishes in $< 40\text{ s}$ with zero regressions. |

---

## 4. Mandatory Adversarial Engineering Invariants

Phase 6 must strictly enforce the following four architectural requirements:

1. **SSM Stream Isolation & Mathematical Completeness Invariant**:
   Ternary Bonsai-2 27B contains 48 State Space Model (SSM) recurrent layers. Unlike self-attention where tokens are keyed by position in a page table, SSMs maintain a continuous recurrent hidden state. In the multi-stream batcher, the SSM state buffer must be strictly indexed by `stream_slot * (48 * ssm_inner)`. Concurrent agent streams must never read or write across another stream's SSM slot. Furthermore, the decoder MUST implement the full mathematically exact recurrence: 1D causal depthwise convolution (`ssm_conv1d`), time-step projection ($\Delta t$), discretized state transition $A$, input projection $B$, output projection $C$, group norm (`ssm_norm`), QK RMSNorm (`attn_q_norm`, `attn_k_norm`), and Rotary Position Embeddings (RoPE). Toy recurrence stubs are strictly forbidden.
2. **Unified Multi-Stream Batched GEMM Invariant ($M=B$)**:
   The engine MUST NOT execute serial GEMV passes across concurrent streams (which takes $44.25\text{ ms} \times B$ per step). Multi-stream scheduling must assemble tokens into a unified batched GEMM ($M = B$) across all active streams in a single kernel dispatch, amortizing weight reading bandwidth across all streams to guarantee per-step execution time $T_{\text{step}} \le 16.0\text{ ms}$ ($\ge 500\text{ tokens/s}$ aggregate).
3. **Marlin Dequantization Tile Saturation Invariant**:
   To counteract memory bandwidth roll-off at higher batch sizes ($522.5\text{ GB/s}$ at $M=1$ dropping to $201.7\text{ GB/s}$ at $M=8$ due to register pressure and bank conflicts), the GEMM kernel must configure $128 \times 128$ macro-tiles with 2-stage double-buffered shared memory staging, sustaining $\ge 70\%$ bus saturation at $M=8$.
4. **Empirical Desktop PCIe Swap Budget**:
   Host memory swapping budgets must anchor to physical desktop PCIe link negotiation ($13.0\text{ ms}$ leaf eviction at $12.0\text{ GB/s}$) rather than theoretical Gen 4 line rates ($6.1\text{ ms}$), with cancellation barriers bounded by `MAX_SWAP_LATENCY_CEILING_PCIE3_MS` ($15.0\text{ ms}$).
5. **Dual-Mode Tokenizer Invariant**:
   The GGUF tokenizer must check for `tokenizer.ggml.merges`. If present, it executes rank-based BPE pair merges. If absent, it must fall back to dynamic programming maximum-scoring byte-fallback SentencePiece, translating unmapped bytes to `<0x00>`...`<0xFF>` tokens. ChatML control tokens (`<|im_start|>`, `<|im_end|>`, `<tool_call>`, `</tool_call>`) must be parsed as indivisible atomic tokens.
6. **Radix Prefix Sharing Mathematical Invariant**:
   On the RTX 4070 (12 GB VRAM), static weights consume 5.95 GB and context scratchpads consume 480 MB, leaving 5.85 GB for KV cache. Without prefix deduplication, 8 concurrent tasks with 8K context require $8 \times 8,192 \times 160\text{ KiB} = 10,240\text{ MB}$, causing an instant OOM crash. With Radix prefix deduplication sharing the 4K-token system and tool prompt, 8 streams require $4,096 \times 80\text{ KiB} + 8 \times 2,048 \times 80\text{ KiB} = 1,638.4\text{ MB}$ (only 28% of the pool). Radix tree sharing is mandatory.
7. **WDDM TDR Prefill Clamping Invariant**:
   Windows Display Driver Model enforces a 2,000 ms execution quantum (`TdrDelay`). Processing large tool observations ($>2,000$ tokens) in an unchunked GEMM pass risks triggering a driver reset. All prefills are strictly clamped to $M \le 64$ tokens per continuous step ($t_{\text{compute}} \approx 9.76\text{ ms} < 14.08\text{ ms}$), guaranteeing zero UI stutter and a $>130\times$ safety margin below the WDDM TDR ceiling.

---

## 5. Phase 6 Exit Criteria

Phase 6 is considered **finished** only when all of the following verifiable conditions are met:

1. **Conversational Chat**: `nebula chat` runs against `profile = "hydrogen"`, holding coherent, multi-turn conversations with accurate context retention and natural streaming latency ($< 15\text{ ms}$ inter-token latency).
2. **Autonomous Task Execution**: `nebula task` runs against `profile = "hydrogen"`, successfully planning, invoking tools (`fs.read`, `fs.write`, `shell.run`, `git.*`), and completing tasks in an isolated Windows Job Object sandbox.
3. **Concurrent Multi-Stream Scaling**: The engine processes 8 parallel agent streams simultaneously over Win32 shared memory with 0 frame drops, 0 cross-talk, $\ge 500\text{ tokens/s}$ aggregate throughput, and KV memory footprint strictly $< 3.5\text{ GB}$.
4. **Tool Reliability**: 100 consecutive tool invocations produce zero JSON syntax errors, verified by the daemon executor.
5. **Regression Parity**: Full workspace test suites pass (`cargo nextest run --workspace --locked` and `bench/` pytest suites) with zero regressions.
