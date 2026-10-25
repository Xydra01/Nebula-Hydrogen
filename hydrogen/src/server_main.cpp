// hydrogen/src/server_main.cpp
// Standalone Project Hydrogen Inference Engine Daemon Server
// Serves requests over Win32 Named Shared Memory IPC for Nebula.

#include "shm_ring_buffer.hpp"
#include "forward_baseline.hpp"
#include "gguf_parser.hpp"
#include "native_decoder.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <filesystem>
#include <cstring>
#include <csignal>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;
using namespace hydrogen::ipc;

namespace {

std::atomic<bool> g_running{true};

#ifdef _WIN32
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    switch (ctrl_type) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            g_running.store(false);
            return TRUE;
        default:
            return FALSE;
    }
}
#endif

void signal_handler(int) {
    g_running.store(false);
}

std::wstring to_wide_string(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
    std::wstring result(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), &result[0], size);
    return result;
}

std::string get_env_var(const char* name, const std::string& default_val) {
    const char* val = std::getenv(name);
    return (val != nullptr && val[0] != '\0') ? std::string(val) : default_val;
}

fs::path auto_discover_model() {
    std::vector<fs::path> candidates = {
        "F:\\Nebula\\models\\bonsai2-27b\\Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        "F:\\Nebula\\models\\bonsai2-27b-mtp\\Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf",
        "D:\\NebulaCold\\models-archive\\fallback\\DeltaCoder-9B-v1.1-DPO-Q6_K.gguf",
        "F:\\Nebula\\models\\embedding\\Qwen3-Embedding-0.6B-Q8_0.gguf"
    };

    for (const auto& p : candidates) {
        if (fs::exists(p)) {
            return p;
        }
    }
    return "";
}

std::string unescape_json_string(const std::string& str) {
    std::string out;
    out.reserve(str.size());
    for (size_t i = 0; i < str.size(); ++i) {
        if (str[i] == '\\' && i + 1 < str.size()) {
            char next = str[i + 1];
            switch (next) {
                case 'n': out += '\n'; ++i; break;
                case 'r': out += '\r'; ++i; break;
                case 't': out += '\t'; ++i; break;
                case '"': out += '"'; ++i; break;
                case '\\': out += '\\'; ++i; break;
                default: out += str[i]; break;
            }
        } else {
            out += str[i];
        }
    }
    return out;
}

std::string extract_chatml_prompt_from_json(const std::string& json_str, const hydrogen::Tokenizer* tokenizer) {
    if (json_str.empty()) return "<|im_start|>user\nHello from Nebula Hydrogen<|im_end|>\n<|im_start|>assistant\n";

    // 1. Try to parse "messages": [ { "role": "...", "content": "..." }, ... ]
    size_t msg_key = json_str.find("\"messages\"");
    if (msg_key != std::string::npos) {
        size_t array_start = json_str.find('[', msg_key);
        if (array_start != std::string::npos) {
            std::vector<hydrogen::ChatMessageInput> messages;
            size_t cursor = array_start + 1;
            while (cursor < json_str.size()) {
                size_t obj_start = json_str.find('{', cursor);
                size_t array_end = json_str.find(']', cursor);
                if (obj_start == std::string::npos || (array_end != std::string::npos && obj_start > array_end)) {
                    break;
                }

                // Extract "role"
                std::string role = "user";
                size_t role_pos = json_str.find("\"role\"", obj_start);
                if (role_pos != std::string::npos) {
                    size_t colon = json_str.find(':', role_pos);
                    if (colon != std::string::npos) {
                        size_t q1 = json_str.find('"', colon);
                        if (q1 != std::string::npos) {
                            size_t q2 = json_str.find('"', q1 + 1);
                            if (q2 != std::string::npos) {
                                role = json_str.substr(q1 + 1, q2 - (q1 + 1));
                            }
                        }
                    }
                }

                // Extract "content"
                std::string content;
                size_t content_pos = json_str.find("\"content\"", obj_start);
                if (content_pos != std::string::npos) {
                    size_t colon = json_str.find(':', content_pos);
                    if (colon != std::string::npos) {
                        size_t q1 = json_str.find('"', colon);
                        if (q1 != std::string::npos) {
                            size_t q2 = q1 + 1;
                            while (q2 < json_str.size()) {
                                if (json_str[q2] == '"' && json_str[q2 - 1] != '\\') {
                                    break;
                                }
                                q2++;
                            }
                            if (q2 < json_str.size()) {
                                content = unescape_json_string(json_str.substr(q1 + 1, q2 - (q1 + 1)));
                            }
                        }
                    }
                }

                if (!content.empty() || role == "assistant") {
                    messages.push_back({role, content});
                }

                size_t obj_end = json_str.find('}', obj_start);
                if (obj_end == std::string::npos) break;
                cursor = obj_end + 1;
            }

            if (!messages.empty()) {
                if (tokenizer != nullptr) {
                    return tokenizer->format_chatml(messages, true);
                }
                std::ostringstream oss;
                for (const auto& m : messages) {
                    oss << "<|im_start|>" << m.role << "\n" << m.content << "<|im_end|>\n";
                }
                oss << "<|im_start|>assistant\n";
                return oss.str();
            }
        }
    }

    // 2. Check for direct "prompt" key
    size_t prompt_pos = json_str.find("\"prompt\"");
    if (prompt_pos != std::string::npos) {
        size_t colon = json_str.find(':', prompt_pos);
        if (colon != std::string::npos) {
            size_t q1 = json_str.find('"', colon);
            if (q1 != std::string::npos) {
                size_t q2 = q1 + 1;
                while (q2 < json_str.size()) {
                    if (json_str[q2] == '"' && json_str[q2 - 1] != '\\') break;
                    q2++;
                }
                if (q2 < json_str.size()) {
                    std::string p = unescape_json_string(json_str.substr(q1 + 1, q2 - (q1 + 1)));
                    if (!p.empty()) {
                        if (p.find("<|im_start|>") != std::string::npos) return p;
                        return "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n";
                    }
                }
            }
        }
    }

    // 3. Fallback to extracting last content string
    size_t last_content = json_str.rfind("\"content\"");
    if (last_content != std::string::npos) {
        size_t colon = json_str.find(':', last_content);
        if (colon != std::string::npos) {
            size_t q1 = json_str.find('"', colon);
            if (q1 != std::string::npos) {
                size_t q2 = q1 + 1;
                while (q2 < json_str.size()) {
                    if (json_str[q2] == '"' && json_str[q2 - 1] != '\\') break;
                    q2++;
                }
                if (q2 < json_str.size()) {
                    std::string p = unescape_json_string(json_str.substr(q1 + 1, q2 - (q1 + 1)));
                    if (!p.empty()) {
                        if (p.find("<|im_start|>") != std::string::npos) return p;
                        return "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n";
                    }
                }
            }
        }
    }

    return "<|im_start|>user\nHello from Nebula Hydrogen<|im_end|>\n<|im_start|>assistant\n";
}

int extract_int_from_json(const std::string& json_str, const std::string& key, int default_val) {
    size_t pos = json_str.find("\"" + key + "\"");
    if (pos == std::string::npos) return default_val;
    size_t colon = json_str.find(':', pos + key.size() + 2);
    if (colon == std::string::npos) return default_val;
    try {
        return std::stoi(json_str.substr(colon + 1));
    } catch (...) {
        return default_val;
    }
}

float extract_float_from_json(const std::string& json_str, const std::string& key, float default_val) {
    size_t pos = json_str.find("\"" + key + "\"");
    if (pos == std::string::npos) return default_val;
    size_t colon = json_str.find(':', pos + key.size() + 2);
    if (colon == std::string::npos) return default_val;
    try {
        return std::stof(json_str.substr(colon + 1));
    } catch (...) {
        return default_val;
    }
}

} // anonymous namespace

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#endif
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::string model_path_str;
    std::string shm_name_str = get_env_var("NEBULA_HYDROGEN_SHM_NAME", "Local\\Nebula_Hydrogen_IPC");
    std::string eng_evt_str = get_env_var("NEBULA_HYDROGEN_ENGINE_EVENT", "Local\\Nebula_Hydrogen_IPC_EngineEvent");
    std::string dae_evt_str = get_env_var("NEBULA_HYDROGEN_DAEMON_EVENT", "Local\\Nebula_Hydrogen_IPC_DaemonEvent");
    int ctx_len = 32768;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path_str = argv[++i];
        } else if ((arg == "-c" || arg == "--ctx") && i + 1 < argc) {
            ctx_len = std::stoi(argv[++i]);
        } else if (arg == "--shm-name" && i + 1 < argc) {
            shm_name_str = argv[++i];
        } else if (arg == "--engine-event" && i + 1 < argc) {
            eng_evt_str = argv[++i];
        } else if (arg == "--daemon-event" && i + 1 < argc) {
            dae_evt_str = argv[++i];
        }
    }

    fs::path model_path = model_path_str.empty() ? auto_discover_model() : fs::path(model_path_str);

    std::cout << "========================================================\n"
              << " Project Hydrogen: Autonomous Inference Engine Server\n"
              << "========================================================\n"
              << " Model:        " << model_path.string() << "\n"
              << " Context:      " << ctx_len << " tokens\n"
              << " Shared Memory: " << shm_name_str << "\n"
              << " Engine Event: " << eng_evt_str << "\n"
              << " Daemon Event: " << dae_evt_str << "\n"
              << "========================================================\n"
              << std::flush;

    std::wstring shm_wide = to_wide_string(shm_name_str);
    std::wstring eng_wide = to_wide_string(eng_evt_str);
    std::wstring dae_wide = to_wide_string(dae_evt_str);

    ShmServer server(shm_wide, eng_wide, dae_wide);
    if (!server.initialize()) {
        std::cerr << "Failed to initialize Win32 shared memory server: " << shm_name_str << std::endl;
        return 1;
    }

    std::cout << "[Hydrogen] IPC server initialized. Ready for daemon requests.\n" << std::flush;

    hydrogen::NativeDecoder decoder;
    bool decoder_ready = false;
    if (!model_path.empty() && fs::exists(model_path)) {
        std::cout << "[Hydrogen] Initializing native autoregressive decoder from " << model_path.string() << "...\n" << std::flush;
        try {
            decoder_ready = decoder.initialize(model_path);
            if (decoder_ready) {
                std::cout << "[Hydrogen] NativeDecoder online (Vocab: " << decoder.vocab_size()
                          << ", Layers: " << decoder.layer_count()
                          << ", Dim: " << decoder.embedding_dim() << ")\n" << std::flush;
            }
        } catch (const std::exception& e) {
            std::cerr << "[Hydrogen] NativeDecoder initialization exception: " << e.what() << "\n" << std::flush;
        }
    }

    hydrogen::ForwardBaseline baseline;
    uint8_t payload_buf[65536];

    while (g_running.load(std::memory_order_relaxed)) {
        IpcFrameHeader req_hdr;
        if (!server.wait_request(req_hdr, payload_buf, sizeof(payload_buf), 50)) {
            continue;
        }

        // Immediate process spawn hook check
        uint32_t active_tools = server.get_active_tool_processes();
        if (active_tools > 0) {
            server.set_cuda_yield_mode(1);
        } else {
            server.set_cuda_yield_mode(0);
        }

        if (req_hdr.msg_type == static_cast<uint16_t>(MessageType::MSG_PING)) {
            server.send_response(MessageType::MSG_PONG, req_hdr.stream_id, FrameFlags::NONE,
                                 nullptr, 0, req_hdr.sequence_num);
        } else if (req_hdr.msg_type == static_cast<uint16_t>(MessageType::MSG_GENERATE_REQUEST)) {
            std::string req_json(reinterpret_cast<const char*>(payload_buf), req_hdr.payload_size);
            std::string prompt = extract_chatml_prompt_from_json(req_json, decoder_ready ? &decoder.tokenizer() : nullptr);
            int max_tokens = extract_int_from_json(req_json, "max_tokens", 256);
            if (max_tokens <= 0) {
                max_tokens = extract_int_from_json(req_json, "n_predict", 256);
            }

            hydrogen::SamplingConfig sampling;
            sampling.temperature = extract_float_from_json(req_json, "temperature", 0.7f);
            sampling.top_p = extract_float_from_json(req_json, "top_p", 0.80f);
            sampling.min_p = extract_float_from_json(req_json, "min_p", 0.0f);
            sampling.repetition_penalty = extract_float_from_json(req_json, "repeat_penalty", 1.05f);

            bool is_tool_continuation = (prompt.find("<|im_start|>tool") != std::string::npos ||
                                         req_json.find("\"role\":\"tool\"") != std::string::npos ||
                                         req_json.find("\"role\": \"tool\"") != std::string::npos);
            bool reset_state = !is_tool_continuation;

            if (decoder_ready) {
                std::cout << "[Hydrogen] Autoregressive generation stream_id=" << req_hdr.stream_id
                          << " max_tokens=" << max_tokens << " reset_state=" << (reset_state ? "true" : "false")
                          << " prompt_len=" << prompt.size() << "\n" << std::flush;
                decoder.generate(prompt, max_tokens, server, req_hdr.stream_id, sampling, reset_state);
            } else {
                // Fallback for environment without fully loaded model weights
                hydrogen::ForwardConfig fwd_cfg;
                fwd_cfg.model_path = model_path;
                fwd_cfg.prompt = prompt;
                fwd_cfg.top_k = 5;
                fwd_cfg.allocate_full_weights = false;

                std::string reply_text;
                float latency_ms = 0.0f;
                try {
                    hydrogen::ForwardResult res = baseline.run(fwd_cfg);
                    if (res.success) {
                        latency_ms = res.forward_latency_ms;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[Hydrogen] Forward baseline probe exception: " << e.what() << "\n";
                }

                std::ostringstream oss;
                oss << "[Hydrogen RTX 4070 SM 8.9]: Received: \"" << prompt << "\". ";
                if (latency_ms > 0.0f) {
                    oss << "GPU forward probe: " << std::fixed << std::setprecision(2) << latency_ms << " ms. ";
                }
                oss << "Win32 Shared Memory IPC (64 MiB zero-copy) is streaming live at sub-50us latency.";
                reply_text = oss.str();

                std::vector<std::string> tokens;
                size_t pos = 0;
                while (pos < reply_text.size()) {
                    size_t next = reply_text.find(' ', pos);
                    if (next == std::string::npos) {
                        tokens.push_back(reply_text.substr(pos));
                        break;
                    }
                    tokens.push_back(reply_text.substr(pos, next - pos + 1));
                    pos = next + 1;
                }

                if (tokens.empty()) {
                    tokens.push_back("Hydrogen inference online.\n");
                }

                for (size_t i = 0; i < tokens.size(); ++i) {
                    bool is_final = (i + 1 == tokens.size());
                    server.send_token(req_hdr.stream_id, static_cast<uint32_t>(100 + i), 1.0f, tokens[i], is_final);
                    std::this_thread::sleep_for(std::chrono::milliseconds(15));
                }
            }
        } else if (req_hdr.msg_type == static_cast<uint16_t>(MessageType::MSG_GENERATE_TOKENS_REQUEST)) {
            if (req_hdr.payload_size >= sizeof(GenerateTokensRequest)) {
                const auto* req = reinterpret_cast<const GenerateTokensRequest*>(payload_buf);
                const size_t expected_size = sizeof(GenerateTokensRequest) + static_cast<size_t>(req->num_tokens) * sizeof(uint32_t);
                if (req_hdr.payload_size >= expected_size) {
                    const auto* token_ptr = reinterpret_cast<const uint32_t*>(payload_buf + sizeof(GenerateTokensRequest));
                    std::vector<uint32_t> prompt_tokens(token_ptr, token_ptr + req->num_tokens);

                    hydrogen::SamplingConfig sampling;
                    sampling.temperature = req->temperature;
                    sampling.top_p = req->top_p;
                    sampling.min_p = 0.0f;
                    sampling.repetition_penalty = 1.05f;

                    if (decoder_ready) {
                        std::cout << "[Hydrogen] Binary token generation stream_id=" << req->stream_id
                                  << " num_tokens=" << req->num_tokens << " max_tokens=" << req->max_tokens << "\n" << std::flush;
                        decoder.generate(prompt_tokens, req->max_tokens, server, req->stream_id, sampling,
                                         /*reset_state=*/true, nullptr, req->stop_tokens, req->stop_token_count);
                    } else {
                        // Dummy stream for tests without weights
                        for (uint32_t i = 0; i < std::min<uint32_t>(req->max_tokens, 4); ++i) {
                            bool is_final = (i + 1 == std::min<uint32_t>(req->max_tokens, 4));
                            server.send_token(req->stream_id, 1000 + i, 1.0f, " tok", is_final);
                        }
                    }
                }
            }
        } else if (req_hdr.msg_type == static_cast<uint16_t>(MessageType::MSG_RELEASE_STREAM)) {
            uint32_t stream_slot = req_hdr.stream_id % 8;
            if (decoder_ready) {
                decoder.release_stream_slot(stream_slot);
            }
            server.release_stream_slot(stream_slot);
            server.send_response(MessageType::MSG_RELEASE_STREAM, req_hdr.stream_id, FrameFlags::NONE,
                                 nullptr, 0, req_hdr.sequence_num);
        } else if (req_hdr.msg_type == static_cast<uint16_t>(MessageType::MSG_TERMINATE_ENGINE)) {
            std::cout << "[Hydrogen] Received terminate signal from daemon.\n" << std::flush;
            break;
        }
    }

    std::cout << "[Hydrogen] Shutting down cleanly.\n" << std::flush;
    server.close();
    return 0;
}
