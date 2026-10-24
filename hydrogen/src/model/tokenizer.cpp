// hydrogen/src/model/tokenizer.cpp
// Production GGUF BPE and SentencePiece Tokenizer implementation for Project Hydrogen

#include "tokenizer.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <cstring>
#include <cctype>

namespace hydrogen {

namespace {

// Decode hex char ('0'-'9', 'a'-'f', 'A'-'F')
int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Convert byte to "<0xXX>" string
std::string byte_to_hex_token(uint8_t b) {
    static const char hex_digits[] = "0123456789ABCDEF";
    std::string s = "<0x00>";
    s[3] = hex_digits[(b >> 4) & 0x0F];
    s[4] = hex_digits[b & 0x0F];
    return s;
}

} // anonymous namespace

bool Tokenizer::load(const GGUFModel& model) {
    // 1. Load Tokens Array
    auto it_tokens = model.metadata.find("tokenizer.ggml.tokens");
    if (it_tokens == model.metadata.end()) {
        std::cerr << "[Tokenizer] Error: tokenizer.ggml.tokens metadata not found\n";
        return false;
    }

    const auto& arr_tokens = it_tokens->second.as_array();
    vocab_size_ = static_cast<uint32_t>(arr_tokens.size());
    vocab_strings_.reserve(vocab_size_);
    token_to_id_.reserve(vocab_size_);

    for (size_t i = 0; i < arr_tokens.size(); ++i) {
        std::string tok = arr_tokens[i].as_string();
        vocab_strings_.push_back(tok);
        token_to_id_[tok] = static_cast<uint32_t>(i);
    }

    // 2. Load Scores if present
    auto it_scores = model.metadata.find("tokenizer.ggml.scores");
    if (it_scores != model.metadata.end()) {
        const auto& arr_scores = it_scores->second.as_array();
        token_scores_.reserve(arr_scores.size());
        for (const auto& s : arr_scores) {
            token_scores_.push_back(s.as_float());
        }
    } else {
        token_scores_.assign(vocab_size_, 0.0f);
    }

    // 3. Load Token Types if present
    auto it_types = model.metadata.find("tokenizer.ggml.token_type");
    if (it_types != model.metadata.end()) {
        const auto& arr_types = it_types->second.as_array();
        token_types_.reserve(arr_types.size());
        for (const auto& t : arr_types) {
            token_types_.push_back(t.as_int32());
        }
    } else {
        token_types_.assign(vocab_size_, 1);
    }

    // 4. Load Special Tokens
    bos_token_id_ = model.get_uint32("tokenizer.ggml.bos_token_id", 248044);
    eos_token_id_ = model.get_uint32("tokenizer.ggml.eos_token_id", 248046);
    pad_token_id_ = model.get_uint32("tokenizer.ggml.padding_token_id", 248044);

    auto it_im_start = token_to_id_.find("<|im_start|>");
    if (it_im_start != token_to_id_.end()) {
        im_start_token_id_ = it_im_start->second;
    } else {
        im_start_token_id_ = 248045;
    }

    auto it_im_end = token_to_id_.find("<|im_end|>");
    if (it_im_end != token_to_id_.end()) {
        im_end_token_id_ = it_im_end->second;
    } else {
        im_end_token_id_ = 248046;
    }

    // 5. Load BPE Merges if present
    auto it_merges = model.metadata.find("tokenizer.ggml.merges");
    if (it_merges != model.metadata.end()) {
        is_bpe_ = true;
        const auto& arr_merges = it_merges->second.as_array();
        bpe_ranks_.reserve(arr_merges.size());
        for (size_t i = 0; i < arr_merges.size(); ++i) {
            bpe_ranks_[arr_merges[i].as_string()] = static_cast<uint32_t>(i);
        }
    } else {
        is_bpe_ = false;
    }

    init_byte_tokens();
    return true;
}

void Tokenizer::init_byte_tokens() {
    std::memset(byte_tokens_, 0, sizeof(byte_tokens_));
    has_byte_tokens_ = true;
    unicode_to_byte_.clear();

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool self_map = (b >= '!' && b <= '~') ||
                        (b >= 0xA1 && b <= 0xAC) ||
                        (b >= 0xAE && b <= 0xFF);
        std::string utf8_char;
        if (self_map) {
            if (b < 0x80) {
                utf8_char.push_back(static_cast<char>(b));
            } else {
                utf8_char.push_back(static_cast<char>(0xC0 | ((b >> 6) & 0x1F)));
                utf8_char.push_back(static_cast<char>(0x80 | (b & 0x3F)));
            }
        } else {
            int cp = 256 + n;
            utf8_char.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
            utf8_char.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            n++;
        }
        byte_to_unicode_[b] = utf8_char;
        unicode_to_byte_[utf8_char] = static_cast<uint8_t>(b);

        auto it = token_to_id_.find(utf8_char);
        if (it != token_to_id_.end()) {
            byte_tokens_[b] = it->second;
        } else {
            std::string hex_tok = byte_to_hex_token(static_cast<uint8_t>(b));
            auto it_hex = token_to_id_.find(hex_tok);
            byte_tokens_[b] = (it_hex != token_to_id_.end()) ? it_hex->second : 0;
        }
    }
}

std::vector<std::string> Tokenizer::bpe_tokenize_word(const std::string& word) const {
    if (word.empty()) return {};

    // Initial split into UTF-8 characters
    std::vector<std::string> parts;
    size_t i = 0;
    while (i < word.size()) {
        unsigned char c = static_cast<unsigned char>(word[i]);
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;

        if (i + len <= word.size()) {
            parts.push_back(word.substr(i, len));
            i += len;
        } else {
            parts.push_back(std::string(1, word[i]));
            i++;
        }
    }

    if (parts.size() <= 1) return parts;

    // Iterative BPE pair merge
    while (parts.size() > 1) {
        int best_pair_idx = -1;
        uint32_t lowest_rank = UINT32_MAX;

        for (size_t p = 0; p + 1 < parts.size(); ++p) {
            std::string pair_key = parts[p] + " " + parts[p + 1];
            auto it = bpe_ranks_.find(pair_key);
            if (it != bpe_ranks_.end()) {
                if (it->second < lowest_rank) {
                    lowest_rank = it->second;
                    best_pair_idx = static_cast<int>(p);
                }
            }
        }

        if (best_pair_idx < 0) {
            break; // No further valid merges found
        }

        // Merge pair at best_pair_idx
        std::vector<std::string> new_parts;
        new_parts.reserve(parts.size() - 1);
        for (int p = 0; p < static_cast<int>(parts.size()); ++p) {
            if (p == best_pair_idx) {
                new_parts.push_back(parts[p] + parts[p + 1]);
                p++; // skip merged neighbor
            } else {
                new_parts.push_back(parts[p]);
            }
        }
        parts = std::move(new_parts);
    }

    return parts;
}

std::vector<uint32_t> Tokenizer::sentencepiece_tokenize_word(const std::string& word) const {
    const size_t n = word.size();
    if (n == 0) return {};

    // Dynamic programming: find maximum-scoring segmentation
    std::vector<float> best_score(n + 1, -1e9f);
    std::vector<int> best_prev(n + 1, -1);
    std::vector<uint32_t> best_token(n + 1, 0);

    best_score[0] = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        if (best_score[i] < -1e8f) continue;

        size_t max_len = std::min<size_t>(n - i, 64);
        for (size_t len = 1; len <= max_len; ++len) {
            std::string sub = word.substr(i, len);
            auto it = token_to_id_.find(sub);
            if (it != token_to_id_.end()) {
                uint32_t tid = it->second;
                float score = (tid < token_scores_.size()) ? token_scores_[tid] : 0.0f;
                float total = best_score[i] + score;
                if (total > best_score[i + len]) {
                    best_score[i + len] = total;
                    best_prev[i + len] = static_cast<int>(i);
                    best_token[i + len] = tid;
                }
            }
        }

        // Byte fallback step for single byte
        uint8_t byte_val = static_cast<uint8_t>(word[i]);
        uint32_t byte_tid = byte_tokens_[byte_val];
        float byte_score = -100.0f; // Penalty for fallback
        if (best_score[i] + byte_score > best_score[i + 1]) {
            best_score[i + 1] = best_score[i] + byte_score;
            best_prev[i + 1] = static_cast<int>(i);
            best_token[i + 1] = byte_tid;
        }
    }

    // Backtrack path
    std::vector<uint32_t> result;
    int curr = static_cast<int>(n);
    while (curr > 0) {
        int prev = best_prev[curr];
        if (prev < 0) break;
        result.push_back(best_token[curr]);
        curr = prev;
    }

    std::reverse(result.begin(), result.end());
    return result;
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text, bool add_special) const {
    std::vector<uint32_t> tokens;
    if (text.empty()) return tokens;

    if (add_special && bos_token_id_ != 0) {
        tokens.push_back(bos_token_id_);
    }

    // Known special control tokens
    static const std::vector<std::string> special_tokens = {
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<|endoftext|>",
        "<|extra_0|>",
        "<tool_call>",
        "</tool_call>"
    };

    auto process_segment = [&](const std::string& segment) {
        if (segment.empty()) return;

        if (is_bpe_) {
            // Check exact string in vocabulary first
            auto it_exact = token_to_id_.find(segment);
            if (it_exact != token_to_id_.end()) {
                tokens.push_back(it_exact->second);
                return;
            }

            // Word-level pre-tokenization
            size_t pos = 0;
            const size_t n = segment.size();
            while (pos < n) {
                unsigned char c = static_cast<unsigned char>(segment[pos]);
                size_t start = pos;
                if (c == '\r' || c == '\n') {
                    while (pos < n && (segment[pos] == '\r' || segment[pos] == '\n')) {
                        pos++;
                    }
                } else if (c == ' ' || c == '\t') {
                    if (pos + 1 < n && std::isalpha(static_cast<unsigned char>(segment[pos + 1]))) {
                        pos++;
                        while (pos < n && std::isalpha(static_cast<unsigned char>(segment[pos]))) {
                            pos++;
                        }
                    } else {
                        while (pos < n && (segment[pos] == ' ' || segment[pos] == '\t')) {
                            pos++;
                        }
                    }
                } else if (std::isalpha(c)) {
                    while (pos < n && std::isalpha(static_cast<unsigned char>(segment[pos]))) {
                        pos++;
                    }
                } else if (std::isdigit(c)) {
                    pos++;
                } else {
                    pos++;
                }

                std::string chunk = segment.substr(start, pos - start);
                // Convert chunk's raw bytes to GPT-2 unicode mapping
                std::string chunk_mapped;
                chunk_mapped.reserve(chunk.size() * 2);
                for (unsigned char b : chunk) {
                    chunk_mapped += byte_to_unicode_[b];
                }

                auto it_word = token_to_id_.find(chunk_mapped);
                if (it_word != token_to_id_.end()) {
                    tokens.push_back(it_word->second);
                } else {
                    // Apply BPE pair merges
                    std::vector<std::string> subwords = bpe_tokenize_word(chunk_mapped);
                    for (const auto& sw : subwords) {
                        auto it_sw = token_to_id_.find(sw);
                        if (it_sw != token_to_id_.end()) {
                            tokens.push_back(it_sw->second);
                        } else {
                            // Single character fallback
                            for (unsigned char b : sw) {
                                tokens.push_back(byte_tokens_[b]);
                            }
                        }
                    }
                }
            }
        } else {
            // SentencePiece Viterbi DP segmentation
            std::vector<uint32_t> sp_tokens = sentencepiece_tokenize_word(segment);
            tokens.insert(tokens.end(), sp_tokens.begin(), sp_tokens.end());
        }
    };

    // Scan for earliest special token without std::regex
    size_t cursor = 0;
    while (cursor < text.size()) {
        size_t earliest_pos = std::string::npos;
        size_t match_len = 0;
        uint32_t matched_token_id = 0;

        for (const auto& st : special_tokens) {
            size_t p = text.find(st, cursor);
            if (p != std::string::npos && (earliest_pos == std::string::npos || p < earliest_pos)) {
                earliest_pos = p;
                match_len = st.size();
                auto it = token_to_id_.find(st);
                matched_token_id = (it != token_to_id_.end()) ? it->second : 0;
            }
        }

        if (earliest_pos == std::string::npos) {
            process_segment(text.substr(cursor));
            break;
        }

        if (earliest_pos > cursor) {
            process_segment(text.substr(cursor, earliest_pos - cursor));
        }

        if (matched_token_id != 0) {
            tokens.push_back(matched_token_id);
        }

        cursor = earliest_pos + match_len;
    }

    return tokens;
}

std::string Tokenizer::decode_token(uint32_t token_id) const {
    if (token_id >= vocab_strings_.size()) return "";

    const std::string& raw = vocab_strings_[token_id];

    // Check for byte token format "<0xXX>"
    if (raw.size() == 6 && raw[0] == '<' && raw[1] == '0' && raw[2] == 'x' && raw[5] == '>') {
        int h1 = hex_val(raw[3]);
        int h2 = hex_val(raw[4]);
        if (h1 >= 0 && h2 >= 0) {
            uint8_t byte_val = static_cast<uint8_t>((h1 << 4) | h2);
            return std::string(1, static_cast<char>(byte_val));
        }
    }

    std::string out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size();) {
        unsigned char c = static_cast<unsigned char>(raw[i]);
        // SentencePiece space " " (0xE2 0x96 0x81)
        if (c == 0xe2 && i + 2 < raw.size() &&
            static_cast<unsigned char>(raw[i + 1]) == 0x96 &&
            static_cast<unsigned char>(raw[i + 2]) == 0x81) {
            out += ' ';
            i += 3;
            continue;
        }

        // GPT-2 BPE unicode character mapping check
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;

        if (i + len <= raw.size()) {
            std::string sub = raw.substr(i, len);
            auto it = unicode_to_byte_.find(sub);
            if (it != unicode_to_byte_.end()) {
                out += static_cast<char>(it->second);
                i += len;
                continue;
            }
        }

        out += raw[i];
        i++;
    }

    return out;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string text;
    for (uint32_t tid : tokens) {
        // Skip special boundary tokens during natural text rendering
        if (tid == bos_token_id_ || tid == eos_token_id_ || tid == im_end_token_id_) {
            continue;
        }
        text += decode_token(tid);
    }
    return text;
}

std::string Tokenizer::format_chatml(
    const std::vector<ChatMessageInput>& messages,
    bool add_generation_prompt
) const {
    std::ostringstream oss;
    for (const auto& msg : messages) {
        oss << "<|im_start|>" << msg.role << "\n"
            << msg.content << "<|im_end|>\n";
    }
    if (add_generation_prompt) {
        oss << "<|im_start|>assistant\n";
    }
    return oss.str();
}

} // namespace hydrogen
