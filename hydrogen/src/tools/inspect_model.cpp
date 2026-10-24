#include "tensor_loader.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

namespace {

std::string escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: hydrogen_inspect_model <path_to_gguf> [--json]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    bool json_output = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--json") {
            json_output = true;
        }
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        hydrogen::ModelLoader loader;
        loader.load(model_path);

        auto end_time = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        const auto& model = loader.model();
        const auto& mf = loader.mapped_file();

        // Calculate quantization breakdown
        std::map<std::string, size_t> quant_counts;
        for (const auto& t : model.tensors) {
            std::string type_name = hydrogen::ggml_type_name(t.type);
            quant_counts[type_name]++;
        }

        if (json_output) {
            std::cout << "{\n"
                      << "  \"status\": \"PASS\",\n"
                      << "  \"file_path\": \"" << escape_json(model_path) << "\",\n"
                      << "  \"file_size_bytes\": " << mf.size() << ",\n"
                      << "  \"header_size_bytes\": " << model.header_size_bytes << ",\n"
                      << "  \"tensor_data_offset\": " << model.tensor_data_offset << ",\n"
                      << "  \"version\": " << model.version << ",\n"
                      << "  \"tensor_count\": " << model.tensor_count << ",\n"
                      << "  \"metadata_kv_count\": " << model.kv_count << ",\n"
                      << "  \"alignment\": " << model.alignment << ",\n"
                      << "  \"architecture\": \"" << model.architecture() << "\",\n"
                      << "  \"layer_count\": " << model.layer_count() << ",\n"
                      << "  \"context_length\": " << model.context_length() << ",\n"
                      << "  \"embedding_length\": " << model.embedding_length() << ",\n"
                      << "  \"head_count\": " << model.head_count() << ",\n"
                      << "  \"head_count_kv\": " << model.head_count_kv() << ",\n"
                      << "  \"parse_duration_ms\": " << std::fixed << std::setprecision(3) << duration_ms << ",\n"
                      << "  \"quantization_breakdown\": {\n";

            bool first_q = true;
            for (const auto& [q_type, count] : quant_counts) {
                if (!first_q) std::cout << ",\n";
                std::cout << "    \"" << q_type << "\": " << count;
                first_q = false;
            }
            std::cout << "\n  },\n";

            // First 5 tensors
            std::cout << "  \"first_tensors\": [\n";
            size_t n_first = std::min<size_t>(5, model.tensors.size());
            for (size_t i = 0; i < n_first; ++i) {
                const auto& t = model.tensors[i];
                if (i > 0) std::cout << ",\n";
                std::cout << "    {\"name\": \"" << escape_json(t.name) << "\", \"type\": \""
                          << hydrogen::ggml_type_name(t.type) << "\", \"size_bytes\": "
                          << t.size_bytes << ", \"offset\": " << t.offset << "}";
            }
            std::cout << "\n  ]\n";

            std::cout << "}" << std::endl;
        } else {
            std::cout << "===================================================\n"
                      << "Project Hydrogen GGUF Model Inspection\n"
                      << "===================================================\n"
                      << "File:               " << model_path << "\n"
                      << "File Size:          " << (mf.size() / (1024.0 * 1024.0)) << " MiB ("
                      << mf.size() << " bytes)\n"
                      << "Header Size:        " << model.header_size_bytes << " bytes\n"
                      << "Tensor Data Offset: " << model.tensor_data_offset << " bytes\n"
                      << "GGUF Version:       " << model.version << "\n"
                      << "Alignment:          " << model.alignment << " bytes\n"
                      << "Parse Duration:     " << std::fixed << std::setprecision(3) << duration_ms << " ms\n"
                      << "---------------------------------------------------\n"
                      << "Architecture:       " << model.architecture() << "\n"
                      << "Layer Count:        " << model.layer_count() << "\n"
                      << "Context Length:     " << model.context_length() << "\n"
                      << "Embedding Length:   " << model.embedding_length() << "\n"
                      << "Attention Heads:    " << model.head_count() << " (KV: " << model.head_count_kv() << ")\n"
                      << "---------------------------------------------------\n"
                      << "Tensors:            " << model.tensor_count << "\n"
                      << "Metadata Entries:   " << model.kv_count << "\n"
                      << "Quantization Formats:\n";

            for (const auto& [q_type, count] : quant_counts) {
                std::cout << "  - " << std::left << std::setw(12) << q_type << ": " << count << " tensors\n";
            }

            std::cout << "---------------------------------------------------\n"
                      << "Metadata Entries (" << model.metadata.size() << "):\n";
            for (const auto& [k, v] : model.metadata) {
                if (v.type == hydrogen::GGUFType::ARRAY) {
                    std::cout << "  " << k << " = [ARRAY len=" << v.as_array().size() << "]\n";
                } else if (v.type == hydrogen::GGUFType::STRING) {
                    std::string s = v.as_string();
                    if (s.size() > 60) s = s.substr(0, 57) + "...";
                    std::cout << "  " << k << " = \"" << s << "\"\n";
                } else {
                    std::cout << "  " << k << " = " << v.as_string() << "\n";
                }
            }

            std::cout << "---------------------------------------------------\n"
                      << "Key Tensors (Global & Blk 0 & Blk 3):\n";
            for (const auto& t : model.tensors) {
                if (t.name.find("token_embd") != std::string::npos ||
                    t.name.find("output") != std::string::npos ||
                    t.name.find("blk.0.") != std::string::npos ||
                    t.name.find("blk.3.") != std::string::npos) {
                    std::cout << "  " << std::left << std::setw(35) << t.name
                              << " | Type: " << std::setw(8) << hydrogen::ggml_type_name(t.type)
                              << " | Shape: [";
                    for (size_t d = 0; d < t.shape.size(); ++d) {
                        std::cout << (d > 0 ? ", " : "") << t.shape[d];
                    }
                    std::cout << "] | Size: " << t.size_bytes << " B\n";
                }
            }
            std::cout << "===================================================" << std::endl;
        }

        return 0;
    } catch (const std::exception& ex) {
        if (json_output) {
            std::cout << "{\n"
                      << "  \"status\": \"FAIL\",\n"
                      << "  \"error\": \"" << ex.what() << "\"\n"
                      << "}" << std::endl;
        } else {
            std::cerr << "Error inspecting model '" << model_path << "': " << ex.what() << std::endl;
        }
        return 1;
    }
}
