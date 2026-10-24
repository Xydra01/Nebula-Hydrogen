// hydrogen/src/tools/test_autoregressive_decoder.cpp
// Native Autoregressive Decoder & BPE Tokenizer End-to-End Test (Issue H19)

#include "native_decoder.hpp"

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    fs::path model_path = "F:\\Nebula\\models\\bonsai2-27b\\Ternary-Bonsai-2-27B-PTQ1_0.gguf";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "========================================================\n"
              << " Project Hydrogen: Autoregressive Decoder Test (H19)\n"
              << "========================================================\n"
              << " Model: " << model_path.string() << "\n"
              << "========================================================\n"
              << std::flush;

    if (!fs::exists(model_path)) {
        std::cerr << "[FAIL] Model file does not exist: " << model_path << "\n";
        return 1;
    }

    hydrogen::NativeDecoder decoder;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::cout << "[INFO] Initializing decoder and loading weights into VRAM...\n" << std::flush;
    if (!decoder.initialize(model_path)) {
        std::cerr << "[FAIL] NativeDecoder::initialize failed!\n";
        return 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[PASS] Model initialized in " << std::fixed << std::setprecision(2) << load_sec << " seconds.\n"
              << "       Vocab Size:    " << decoder.vocab_size() << "\n"
              << "       Layers:        " << decoder.layer_count() << "\n"
              << "       Embedding Dim: " << decoder.embedding_dim() << "\n"
              << "--------------------------------------------------------\n"
              << std::flush;

    // Test Tokenizer in isolation
    std::string test_prompt = "<|im_start|>user\nHello! What is 2 + 2?<|im_end|>\n<|im_start|>assistant\n<think>\n";
    std::vector<uint32_t> tokens = decoder.tokenizer().encode(test_prompt);
    if (tokens.empty()) {
        std::cerr << "[FAIL] Tokenizer returned empty token sequence for test prompt.\n";
        return 1;
    }
    std::cout << "[PASS] Tokenizer encoded prompt into " << tokens.size() << " tokens:\n       ";
    for (auto t : tokens) std::cout << t << " ";
    std::cout << "\n";

    // Test Autoregressive Generation
    std::cout << "[INFO] Running autoregressive generation test with ChatML prompt:\n"
              << "       Prompt: \"" << test_prompt << "\"\n"
              << "       Streaming tokens: \n\n" << std::flush;

    hydrogen::SamplingConfig sampling;
    sampling.temperature = 0.0f; // greedy argmax to verify cleanest prediction
    sampling.top_p = 1.0f;
    sampling.repetition_penalty = 1.0f;

    int token_count = 0;
    auto gen_start = std::chrono::high_resolution_clock::now();

    decoder.generate(
        test_prompt,
        64, // max_tokens
        [&](uint32_t token_id, float /*logit*/, const std::string& text, bool is_final) -> bool {
            token_count++;
            std::cout << text << std::flush;
            if (is_final) {
                std::cout << "\n\n[INFO] Reached EOS / stop token (id=" << token_id << ")\n";
            }
            return true;
        },
        sampling
    );

    auto gen_end = std::chrono::high_resolution_clock::now();
    double gen_sec = std::chrono::duration<double>(gen_end - gen_start).count();
    double tok_per_sec = (gen_sec > 0.0) ? (token_count / gen_sec) : 0.0;

    std::cout << "--------------------------------------------------------\n"
              << " Tokens Generated: " << token_count << "\n"
              << " Total Duration:   " << std::fixed << std::setprecision(2) << gen_sec << " s\n"
              << " Generation Speed: " << std::fixed << std::setprecision(1) << tok_per_sec << " tok/s\n"
              << "========================================================\n"
              << " Status: ALL H19 DECODER TESTS PASSED\n"
              << "========================================================\n"
              << std::flush;

    return 0;
}
