# Issue Plan: H19-B — Exact Qwen 3.5 SSM Recurrence, RoPE, and QK-RMSNorm

| Field | Value |
| :--- | :--- |
| Issue ID | H19-B |
| Title | `[hydrogen] fix(decoder): Exact Qwen 3.5 SSM recurrence, depthwise conv1d, RoPE, and QK RMSNorm (#H19-B)` |
| Phase | Phase 6: Native Engine & Autonomous Agent Execution |
| Labels | `hydrogen`, `phase-6`, `core`, `blocker` |
| Target Files | `hydrogen/src/model/native_decoder.cu`, `hydrogen/include/native_decoder.hpp`, `hydrogen/CMakeLists.txt` |

---

## 1. Problem Statement

While single-token baseline GEMV passes and the Win32 IPC bridge function as verified in Phase 5, `NativeDecoder::generate()` across full 64 layers (48 SSM + 16 attention) produces semantically degenerate garbled subwords (e.g. `raelramereetíomaticadesewise...`) at $22.6\text{ tok/s}$.

An adversarial audit of `hydrogen/src/model/native_decoder.cu` reveals three critical mathematical gaps:
1. **Toy SSM Recurrence**: Line 545 implements `new_s = prev_s * 0.95f + u * 0.05f` rather than true discretized Qwen 3.5 state space equations. It completely bypasses 1D causal depthwise convolution (`ssm_conv1d`), time-step $\Delta t$ projection, input/output projections $B/C$, discretized state matrix $A$, and SSM group norm (`ssm_norm`).
2. **Missing Rotary Position Embeddings (RoPE)**: The 16 full-attention layers compute queries and keys without applying rotary positional frequency embeddings $\Theta_{i,m}$, causing attention weights to lose token distance and positional ordering.
3. **Unused QK-RMSNorm**: `attn_q_norm` and `attn_k_norm` weights are parsed from GGUF tensors into `NativeDecoder` memory, but are never evaluated in `forward_token()` before computing dot products.

Until these kernels are mathematically correct, end-to-end token generation from real weights remains degenerate.

---

## 2. Implementation Plan

1. **Exact 1D Causal Depthwise Convolution (`ssm_conv1d`)**:
   - Maintain a rolling conv state buffer of width $K=4$ tokens per SSM layer per stream.
   - For incoming projected vector $u$, compute causal 1D convolution:
     $$x_{\text{conv}} = \sum_{k=0}^{K-1} w_{\text{conv}}[k] \cdot \text{state}[K - 1 - k] + b_{\text{conv}}$$
   - Apply SiLU non-linearity: $\text{SiLU}(x) = x \cdot \sigma(x)$.

2. **Discretized SSM Recurrence Kernel**:
   - Project time-step $\Delta t = \text{softplus}(\text{Linear}(u) + \Delta t_{\text{bias}})$.
   - Discretize continuous state transition matrix $\bar{A} = \exp(\Delta t \cdot A)$ and input matrix $\bar{B} = \Delta t \cdot B$.
   - Update state vector:
     $$h_t = \bar{A} \odot h_{t-1} + \bar{B} \odot x$$
   - Compute output projection $y = C \cdot h_t + D \cdot x$.
   - Normalize output via per-group RMSNorm (`ssm_norm`).

3. **Rotary Position Embeddings (RoPE)**:
   - For attention layers, compute rotary angle frequencies for token position $m$:
     $$\theta_k = b^{-2k/d}, \quad \begin{pmatrix} q_{2k}' \\ q_{2k+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{pmatrix} \begin{pmatrix} q_{2k} \\ q_{2k+1} \end{pmatrix}$$
   - Apply identical rotation to keys $k$.

4. **QK-RMSNorm Scaling**:
   - Apply `attn_q_norm` and `attn_k_norm` directly after Q and K projections before computing attention scores:
     $$Q_{\text{norm}} = \text{RMSNorm}(Q, \gamma_q), \quad K_{\text{norm}} = \text{RMSNorm}(K, \gamma_k)$$

---

## 3. Verification & Proof Plan

- Run `hydrogen_test_autoregressive_decoder.exe -m "F:\Nebula\models\bonsai2-27b\Ternary-Bonsai-2-27B-PTQ1_0.gguf" -p "Write a Rust hello world" --tokens 64`.
- **Proof Metric 1**: Token stream decodes into syntactically valid Rust source code (`fn main() { println!("Hello, world!"); }`).
- **Proof Metric 2**: Greedy decoding matches deterministic reference output without non-ASCII corrupted byte sequences.
- **Proof Metric 3**: Single-stream step latency on RTX 4070 remains $\le 45\text{ ms}$ for 64-layer evaluation.

---

## 4. Acceptance Criteria

- [ ] `native_decoder.cu` compiles under MSVC `/std:c++20` and `nvcc` with zero warnings under `/W4`.
- [ ] 1D causal depthwise convolution correctly maintains $K=4$ history per stream without memory leaks.
- [ ] State transition $\bar{A}$, $\bar{B}$, $C$, and `ssm_norm` match Qwen 3.5 architectural specification.
- [ ] Rotary position embeddings applied correctly to $Q$ and $K$ heads.
- [ ] `attn_q_norm` and `attn_k_norm` evaluated prior to FlashInfer attention.
- [ ] Autoregressive test produces clean English/code tokens for standard prompts.
