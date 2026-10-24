// hydrogen/include/native_decoder.hpp
// Native Autoregressive Token Decoder & Compute Graph for Project Hydrogen
// SM 8.9 (NVIDIA Ada Lovelace RTX 4070) Native TQ1_0 Inference Engine

#pragma once

#include "tensor_loader.hpp"
#include "shm_ring_buffer.hpp"
#include "tokenizer.hpp"
#include "fsm_grammar.hpp"
#include "page_pool.hpp"
#include "radix_tree.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <memory>
#include <cstdint>
#include <functional>
#include <span>

namespace hydrogen {

enum class TurnRole {
    None,
    System,
    User,
    Assistant,
    Tool
};

struct PagedKvTable {
    const uint32_t* d_block_table = nullptr;
    uint32_t block_size = 16;
    uint32_t max_blocks_per_stream = 2048;
    float* d_paged_k = nullptr;
    float* d_paged_v = nullptr;
    bool is_paged_enabled = false;
    size_t total_pages = 4096;
};

struct SamplingConfig {
    float temperature = 0.7f;
    float top_p = 0.80f;
    float min_p = 0.0f;
    float repetition_penalty = 1.05f;
    uint32_t top_k = 40;
};

struct LayerTensors {
    // Norms
    const float* attn_norm = nullptr;
    const float* post_attn_norm = nullptr;

    // FFN
    const uint8_t* ffn_gate = nullptr;
    const uint8_t* ffn_up = nullptr;
    const uint8_t* ffn_down = nullptr;

    // Full Attention (if layer % 4 == 3)
    bool is_attention = false;
    const uint8_t* attn_q = nullptr;
    const uint8_t* attn_k = nullptr;
    const uint8_t* attn_v = nullptr;
    const float* attn_q_norm = nullptr;
    const float* attn_k_norm = nullptr;
    const uint8_t* attn_output = nullptr;

    // SSM (if layer % 4 != 3)
    const uint8_t* ssm_gate = nullptr;
    const uint8_t* ssm_qkv = nullptr;
    const float* ssm_conv1d = nullptr;
    const float* ssm_a = nullptr;
    const void* ssm_alpha = nullptr;
    const void* ssm_beta = nullptr;
    const float* ssm_dt = nullptr;
    const float* ssm_norm = nullptr;
    const uint8_t* ssm_out = nullptr;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

// Tool call JSON parser (<tool_call>{"name": ..., "arguments": ...}</tool_call>)
ToolCall parse_tool_call_json(const std::string& raw_json, uint32_t stream_id);

class NativeDecoder {
public:
    static constexpr uint32_t MAX_STREAMS = 8;
    static constexpr uint32_t MAX_SEQ_LEN = 8192;

    using TokenCallback = std::function<bool(uint32_t token_id, float logit, const std::string& text, bool is_final)>;
    using ToolCallCallback = std::function<bool(const ToolCall& tool_call)>;

    NativeDecoder();
    ~NativeDecoder();

    // Initialize decoder, allocate VRAM, load weights and pre-allocate activation buffers
    bool initialize(const std::filesystem::path& model_path);

    // Autoregressively generate tokens with streaming callback and optional tool call detection
    void generate(
        const std::string& prompt,
        int max_tokens,
        const TokenCallback& on_token,
        const SamplingConfig& sampling = SamplingConfig{},
        uint32_t stream_id = 0,
        bool reset_state = true,
        const uint32_t* d_grammar_bitmask = nullptr,
        const ToolCallCallback& on_tool_call = nullptr
    );

    // Autoregressively generate tokens and stream to daemon over Win32 Shared Memory IPC
    void generate(
        const std::string& prompt,
        int max_tokens,
        ipc::ShmServer& server,
        uint32_t stream_id,
        const SamplingConfig& sampling = SamplingConfig{},
        bool reset_state = true,
        const uint32_t* d_grammar_bitmask = nullptr
    );

    // Tokenizer & Detokenizer access
    Tokenizer& tokenizer() noexcept { return tokenizer_; }
    [[nodiscard]] const Tokenizer& tokenizer() const noexcept { return tokenizer_; }

    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }
    [[nodiscard]] uint32_t vocab_size() const noexcept { return vocab_size_; }
    [[nodiscard]] uint32_t layer_count() const noexcept { return layer_count_; }
    [[nodiscard]] uint32_t embedding_dim() const noexcept { return embedding_dim_; }
    [[nodiscard]] uint32_t current_seq_pos(uint32_t stream_slot) const noexcept {
        return (stream_slot < MAX_STREAMS) ? stream_seq_pos_[stream_slot] : 0;
    }

    // Dynamic Radix Page Pool KV Binding (Issue H21 / H15)
    void bind_paged_kv(
        const uint32_t* d_block_table,
        float* d_paged_k,
        float* d_paged_v,
        uint32_t block_size = 16,
        uint32_t max_blocks_per_stream = 2048
    );
    void unbind_paged_kv();
    [[nodiscard]] bool is_paged_kv_enabled() const noexcept { return paged_kv_.is_paged_enabled; }
    [[nodiscard]] const PagedKvTable& paged_kv_table() const noexcept { return paged_kv_; }

    // Dynamic Radix Page Pool Engine (Issue H22 / H23)
    bool enable_dynamic_paged_kv(size_t total_pages = 1024, uint32_t block_size = 16, uint32_t max_blocks_per_stream = 2048);
    void disable_dynamic_paged_kv();
    size_t register_shared_prefix(std::span<const int32_t> prefix_tokens);
    size_t attach_stream_to_prefix(uint32_t stream_slot, std::span<const int32_t> prompt_tokens);
    uint32_t allocate_stream_block(uint32_t stream_slot, uint32_t logical_block);
    size_t allocate_stream_blocks_batch(uint32_t stream_slot, uint32_t start_block, uint32_t num_blocks);
    void release_stream_blocks(uint32_t stream_slot);
    [[nodiscard]] std::shared_ptr<PagePool> page_pool() const noexcept { return page_pool_; }
    [[nodiscard]] std::shared_ptr<RadixTree> radix_tree() const noexcept { return radix_tree_; }
    [[nodiscard]] const uint32_t* device_fsm_bitmasks() const noexcept { return d_fsm_bitmasks_; }

    // Dynamic FSM Tool Grammar Bitmask Compiler & Step Synchronizer
    void set_grammar_schema(const grammar::ToolSchema& schema);
    void set_grammar_table(std::shared_ptr<grammar::FsmCompiledTable> table);
    void clear_grammar();
    [[nodiscard]] bool has_active_grammar() const noexcept { return active_fsm_table_ != nullptr; }
    [[nodiscard]] int current_grammar_state() const noexcept { return current_fsm_state_; }

    // Test hooks for standalone verification of GPU kernels and parser
    void test_apply_grammar_mask(float* d_logits, const uint32_t* d_bitmask);
    void test_extract_top_candidates(const float* d_logits, uint32_t* d_top_tokens, float* d_top_logits, uint32_t max_candidates);

private:
    bool initialized_ = false;
    ModelLoader loader_;
    Tokenizer tokenizer_;

    // Model Architecture Dimensions
    uint32_t vocab_size_ = 248320;
    uint32_t layer_count_ = 64;
    uint32_t embedding_dim_ = 5120;
    uint32_t ffn_dim_ = 17408;
    uint32_t head_count_ = 24;
    uint32_t head_count_kv_ = 4;
    uint32_t head_dim_ = 256;
    uint32_t ssm_inner_ = 6144;
    float rms_norm_eps_ = 1e-6f;
    uint32_t eos_token_id_ = 248046;
    uint32_t bos_token_id_ = 248044;

    // Device Memory Pointers
    uint8_t* d_weights_ = nullptr;
    size_t total_weight_bytes_ = 0;

    // Global Model Tensors
    const uint8_t* d_token_embd_ = nullptr;
    const float* d_output_norm_ = nullptr;
    const uint8_t* d_output_ = nullptr;

    // Per-layer Tensors (64 layers)
    std::vector<LayerTensors> layers_;

    // Activation Working Buffers (Device)
    float* d_x_ = nullptr;             // [5120]
    float* d_x_norm_ = nullptr;        // [5120]
    float* d_res_ = nullptr;           // [5120]
    float* d_q_ = nullptr;             // [12288]
    float* d_k_ = nullptr;             // [1024]
    float* d_v_ = nullptr;             // [1024]
    float* d_attn_out_ = nullptr;      // [6144]
    float* d_qkv_ = nullptr;           // [10240]
    float* d_gate_ = nullptr;          // [17408]
    float* d_up_ = nullptr;            // [17408]
    float* d_act_ = nullptr;           // [17408]
    float* d_ffn_out_ = nullptr;       // [5120]
    float* d_logits_ = nullptr;        // [248320]

    // Multi-Stream KV Cache & SSM States
    float* d_k_cache_ = nullptr;       // [MAX_STREAMS * 16 * MAX_SEQ_LEN * 4 * 256]
    float* d_v_cache_ = nullptr;       // [MAX_STREAMS * 16 * MAX_SEQ_LEN * 4 * 256]
    float* d_ssm_state_ = nullptr;     // [MAX_STREAMS * 48 * ssm_inner_]
    uint32_t stream_seq_pos_[MAX_STREAMS] = {0};

    // Grammar Mask Buffer (Device)
    uint32_t* d_grammar_mask_ = nullptr; // [(vocab_size + 31) / 32]

    // Dynamic Radix Paged KV Table & Engine (H22)
    PagedKvTable paged_kv_;
    std::shared_ptr<PagePool> page_pool_;
    std::shared_ptr<RadixTree> radix_tree_;
    uint32_t* d_dynamic_block_table_ = nullptr;
    float* d_dynamic_paged_k_ = nullptr;
    float* d_dynamic_paged_v_ = nullptr;
    std::vector<uint32_t> h_stream_block_table_;
    std::vector<uint32_t> stream_allocated_pages_[MAX_STREAMS];
    std::shared_ptr<RadixTreeNode> stream_radix_nodes_[MAX_STREAMS];

    // Dynamic FSM Tool Grammar State (Zero PCIe copy per step)
    std::shared_ptr<grammar::FsmCompiledTable> active_fsm_table_;
    int current_fsm_state_ = 0;
    uint32_t* d_fsm_bitmasks_ = nullptr;
    size_t fsm_bitmasks_allocated_words_ = 0;

    // Pinned Sampling Buffers (Top-K extraction)
    static constexpr size_t TOP_CANDIDATES = 128;
    uint32_t* d_top_tokens_ = nullptr;
    float* d_top_logits_ = nullptr;
    uint32_t* h_top_tokens_ = nullptr;
    float* h_top_logits_ = nullptr;

    // Hadamard Outlier Rotation Buffers (PrismML PTQ1_0)
    bool has_hadamard_ = false;
    int8_t* d_signs_5120_ = nullptr;
    int8_t* d_signs_6144_ = nullptr;
    int8_t* d_signs_17408_ = nullptr;

    // CUDA Stream
    void* cuda_stream_ = nullptr;

    // Internal Compute Steps
    void forward_token(
        uint32_t token_id,
        uint32_t pos,
        bool compute_logits,
        uint32_t stream_slot = 0,
        const uint32_t* d_bitmask = nullptr
    );

    uint32_t sample_next_token(
        const SamplingConfig& sampling,
        const std::vector<uint32_t>& generated_tokens
    );
};

} // namespace hydrogen
