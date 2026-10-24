// hydrogen/include/tokenizer.hpp
// Production GGUF BPE and SentencePiece Tokenizer for Project Hydrogen
// Supports Qwen/ChatML, GPT-2 byte encoding, and SentencePiece byte fallbacks.

#pragma once

#include "gguf_parser.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>

namespace hydrogen {

struct ChatMessageInput {
    std::string role;    // "system", "user", "assistant", "tool"
    std::string content;
};

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    // Load vocabulary, merges, and special tokens from parsed GGUF model metadata
    bool load(const GGUFModel& model);

    // Encode text into token IDs
    std::vector<uint32_t> encode(const std::string& text, bool add_special = false) const;

    // Decode token ID to string representation
    std::string decode_token(uint32_t token_id) const;

    // Decode a sequence of token IDs to full string
    std::string decode(const std::vector<uint32_t>& tokens) const;

    // Format a multi-turn conversation into standard Qwen ChatML prompt format
    std::string format_chatml(const std::vector<ChatMessageInput>& messages, bool add_generation_prompt = true) const;

    // Control tokens
    [[nodiscard]] uint32_t bos_id() const noexcept { return bos_token_id_; }
    [[nodiscard]] uint32_t eos_id() const noexcept { return eos_token_id_; }
    [[nodiscard]] uint32_t im_start_id() const noexcept { return im_start_token_id_; }
    [[nodiscard]] uint32_t im_end_id() const noexcept { return im_end_token_id_; }
    [[nodiscard]] uint32_t vocab_size() const noexcept { return static_cast<uint32_t>(vocab_strings_.size()); }
    [[nodiscard]] bool is_bpe() const noexcept { return is_bpe_; }

private:
    bool is_bpe_ = false;
    uint32_t vocab_size_ = 0;
    uint32_t bos_token_id_ = 248044;
    uint32_t eos_token_id_ = 248046;
    uint32_t pad_token_id_ = 248044;
    uint32_t im_start_token_id_ = 248045;
    uint32_t im_end_token_id_ = 248046;

    // Direct vocabulary mappings
    std::vector<std::string> vocab_strings_;
    std::vector<float> token_scores_;
    std::vector<int32_t> token_types_;
    std::unordered_map<std::string, uint32_t> token_to_id_;

    // BPE pair ranks: pair "tokenA tokenB" -> merge priority rank
    std::unordered_map<std::string, uint32_t> bpe_ranks_;

    // Byte tokens: 0..255 -> token_id
    uint32_t byte_tokens_[256];
    bool has_byte_tokens_ = false;

    // GPT-2 Byte-to-Unicode and Unicode-to-Byte mappings
    std::string byte_to_unicode_[256];
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    // Helper functions
    void init_byte_tokens();
    std::vector<std::string> bpe_tokenize_word(const std::string& word) const;
    std::vector<uint32_t> sentencepiece_tokenize_word(const std::string& word) const;
};

} // namespace hydrogen
