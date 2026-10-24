#include "shm_ring_buffer.hpp"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <thread>

namespace hydrogen::ipc {

// ============================================================================
// ShmRingBuffer Implementation
// ============================================================================

ShmRingBuffer::ShmRingBuffer(ShmQueueHeader* queue_hdr, uint8_t* buffer_base, uint64_t capacity) {
    init(queue_hdr, buffer_base, capacity);
}

void ShmRingBuffer::init(ShmQueueHeader* queue_hdr, uint8_t* buffer_base, uint64_t capacity) {
    header_ = queue_hdr;
    buffer_ = buffer_base;
    capacity_ = capacity;
    if (header_ != nullptr) {
        header_->capacity_bytes = capacity;
        header_->mask = capacity - 1;
    }
}

uint64_t ShmRingBuffer::available_read_bytes() const noexcept {
    if (!header_) return 0;
    const uint64_t head = header_->head.load(std::memory_order_acquire);
    const uint64_t tail = header_->tail.load(std::memory_order_relaxed);
    return (head >= tail) ? (head - tail) : 0;
}

uint64_t ShmRingBuffer::available_write_bytes() const noexcept {
    if (!header_) return 0;
    const uint64_t used = available_read_bytes();
    return (capacity_ > used) ? (capacity_ - used) : 0;
}

bool ShmRingBuffer::is_empty() const noexcept {
    return available_read_bytes() == 0;
}

void ShmRingBuffer::reset() noexcept {
    if (header_) {
        header_->head.store(0, std::memory_order_release);
        header_->tail.store(0, std::memory_order_release);
        header_->total_bytes_written = 0;
        header_->total_bytes_read = 0;
        header_->total_frames_written = 0;
        header_->total_frames_read = 0;
    }
}

bool ShmRingBuffer::write_frame(MessageType msg_type,
                                uint32_t stream_id,
                                uint16_t flags,
                                const void* payload,
                                uint32_t payload_size,
                                uint64_t sequence_num) {
    if (!header_ || !buffer_ || capacity_ == 0) {
        return false;
    }

    const uint64_t total_frame_size = sizeof(IpcFrameHeader) + payload_size;
    if (total_frame_size > capacity_) {
        return false;
    }

    uint64_t current_head = header_->head.load(std::memory_order_relaxed);
    const uint64_t current_tail = header_->tail.load(std::memory_order_acquire);
    uint64_t used = current_head - current_tail;

    if (used + total_frame_size > capacity_) {
        return false; // Queue full
    }

    uint64_t head_offset = current_head % capacity_;

    // Handle ring buffer physical wrap-around:
    // If the frame would cross the end of the memory block, insert a WRAP marker
    // and write the contiguous frame at offset 0.
    if (head_offset + total_frame_size > capacity_) {
        const uint64_t remaining_end = capacity_ - head_offset;

        if (used + remaining_end + total_frame_size > capacity_) {
            return false; // Cannot fit wrap padding + frame
        }

        if (remaining_end >= sizeof(IpcFrameHeader)) {
            auto* wrap_hdr = reinterpret_cast<IpcFrameHeader*>(buffer_ + head_offset);
            wrap_hdr->magic = IPC_MAGIC;
            wrap_hdr->msg_type = static_cast<uint16_t>(MessageType::MSG_WRAP_MARKER);
            wrap_hdr->flags = FrameFlags::WRAP_AROUND;
            wrap_hdr->stream_id = 0;
            wrap_hdr->payload_size = 0;
            wrap_hdr->sequence_num = 0;
            wrap_hdr->timestamp_ns = 0;
        }

        current_head += remaining_end;
        head_offset = 0;
    }

    // Write frame header
    auto* hdr = reinterpret_cast<IpcFrameHeader*>(buffer_ + head_offset);
    hdr->magic = IPC_MAGIC;
    hdr->msg_type = static_cast<uint16_t>(msg_type);
    hdr->flags = flags;
    hdr->stream_id = stream_id;
    hdr->payload_size = payload_size;
    hdr->sequence_num = sequence_num ? sequence_num : seq_counter_++;

    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    hdr->timestamp_ns = static_cast<uint64_t>(now_ns);

    // Write payload
    if (payload_size > 0 && payload != nullptr) {
        std::memcpy(buffer_ + head_offset + sizeof(IpcFrameHeader), payload, payload_size);
    }

    // Commit write with release semantics
    header_->head.store(current_head + total_frame_size, std::memory_order_release);
    header_->total_bytes_written += total_frame_size;
    header_->total_frames_written++;

    return true;
}

bool ShmRingBuffer::read_frame(IpcFrameHeader& out_header,
                               void* out_payload,
                               size_t max_payload_len) {
    const uint8_t* payload_ptr = nullptr;
    if (!read_frame_zero_copy(out_header, payload_ptr)) {
        return false;
    }

    if (out_payload != nullptr && out_header.payload_size > 0 && payload_ptr != nullptr) {
        const size_t copy_bytes = std::min<size_t>(max_payload_len, out_header.payload_size);
        std::memcpy(out_payload, payload_ptr, copy_bytes);
    }

    return true;
}

bool ShmRingBuffer::read_frame_zero_copy(IpcFrameHeader& out_header,
                                         const uint8_t*& out_payload_ptr) {
    if (!header_ || !buffer_ || capacity_ == 0) {
        return false;
    }

    uint64_t current_tail = header_->tail.load(std::memory_order_relaxed);
    const uint64_t current_head = header_->head.load(std::memory_order_acquire);

    if (current_tail >= current_head) {
        return false; // Queue empty
    }

    uint64_t tail_offset = current_tail % capacity_;

    // Handle end-of-buffer wrap sentinel
    if (tail_offset + sizeof(IpcFrameHeader) > capacity_) {
        const uint64_t remaining_end = capacity_ - tail_offset;
        current_tail += remaining_end;
        header_->tail.store(current_tail, std::memory_order_release);
        tail_offset = 0;

        if (current_tail >= current_head) {
            return false;
        }
    }

    const auto* hdr = reinterpret_cast<const IpcFrameHeader*>(buffer_ + tail_offset);

    // If this is a wrap marker, advance tail to start of buffer
    if (hdr->msg_type == static_cast<uint16_t>(MessageType::MSG_WRAP_MARKER) ||
        (hdr->flags & FrameFlags::WRAP_AROUND)) {
        const uint64_t remaining_end = capacity_ - tail_offset;
        current_tail += remaining_end;
        header_->tail.store(current_tail, std::memory_order_release);
        tail_offset = 0;

        if (current_tail >= current_head) {
            return false;
        }
        hdr = reinterpret_cast<const IpcFrameHeader*>(buffer_ + tail_offset);
    }

    if (hdr->magic != IPC_MAGIC) {
        return false; // Frame corrupt
    }

    const uint64_t total_frame_size = sizeof(IpcFrameHeader) + hdr->payload_size;
    out_header = *hdr;
    out_payload_ptr = buffer_ + tail_offset + sizeof(IpcFrameHeader);

    // Advance tail with release semantics
    header_->tail.store(current_tail + total_frame_size, std::memory_order_release);
    header_->total_bytes_read += total_frame_size;
    header_->total_frames_read++;

    return true;
}

// ============================================================================
// ShmServer Implementation
// ============================================================================

ShmServer::ShmServer(const std::wstring& shm_name,
                     const std::wstring& engine_event_name,
                     const std::wstring& daemon_event_name,
                     size_t total_capacity)
    : shm_name_(shm_name),
      engine_event_name_(engine_event_name),
      daemon_event_name_(daemon_event_name),
      total_capacity_(total_capacity) {}

ShmServer::~ShmServer() {
    close();
}

bool ShmServer::initialize() {
    if (is_initialized_) {
        return true;
    }

#ifdef _WIN32
    h_map_file_ = CreateFileMappingW(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        static_cast<DWORD>(total_capacity_),
        shm_name_.c_str()
    );

    if (h_map_file_ == nullptr) {
        return false;
    }

    mapped_view_ = reinterpret_cast<uint8_t*>(
        MapViewOfFile(h_map_file_, FILE_MAP_ALL_ACCESS, 0, 0, total_capacity_)
    );

    if (mapped_view_ == nullptr) {
        CloseHandle(h_map_file_);
        h_map_file_ = nullptr;
        return false;
    }

    h_engine_event_ = CreateEventW(NULL, FALSE, FALSE, engine_event_name_.c_str());
    h_daemon_event_ = CreateEventW(NULL, FALSE, FALSE, daemon_event_name_.c_str());

    if (h_engine_event_ == nullptr || h_daemon_event_ == nullptr) {
        close();
        return false;
    }
#else
    // Fallback in-memory simulation for non-Windows platforms
    mapped_view_ = new uint8_t[total_capacity_];
    std::memset(mapped_view_, 0, total_capacity_);
#endif

    // Initialize or attach to global header
    global_header_ = reinterpret_cast<ShmGlobalHeader*>(mapped_view_);
    const bool already_initialized = (global_header_->magic == IPC_MAGIC && global_header_->version == IPC_VERSION);

    if (!already_initialized) {
        std::memset(global_header_, 0, sizeof(ShmGlobalHeader));
        global_header_->magic = IPC_MAGIC;
        global_header_->version = IPC_VERSION;
        global_header_->total_size_bytes = total_capacity_;

        const uint64_t req_offset = (sizeof(ShmGlobalHeader) + 63) & ~63ULL;
        const uint64_t resp_offset = req_offset + REQUEST_QUEUE_CAPACITY_BYTES;

        global_header_->request_queue_offset = req_offset;
        global_header_->response_queue_offset = resp_offset;

        request_ring_.init(&global_header_->request_queue,
                           mapped_view_ + req_offset,
                           REQUEST_QUEUE_CAPACITY_BYTES);

        response_ring_.init(&global_header_->response_queue,
                            mapped_view_ + resp_offset,
                            RESPONSE_QUEUE_CAPACITY_BYTES);

        request_ring_.reset();
        response_ring_.reset();
    } else {
        request_ring_.init(&global_header_->request_queue,
                           mapped_view_ + global_header_->request_queue_offset,
                           REQUEST_QUEUE_CAPACITY_BYTES);

        response_ring_.init(&global_header_->response_queue,
                            mapped_view_ + global_header_->response_queue_offset,
                            RESPONSE_QUEUE_CAPACITY_BYTES);
    }

#ifdef _WIN32
    global_header_->engine_pid = GetCurrentProcessId();
#else
    global_header_->engine_pid = 1001;
#endif

    is_initialized_ = true;
    return true;
}

void ShmServer::close() {
    if (!is_initialized_) {
        return;
    }

#ifdef _WIN32
    if (mapped_view_ != nullptr) {
        UnmapViewOfFile(mapped_view_);
        mapped_view_ = nullptr;
    }
    if (h_map_file_ != nullptr) {
        CloseHandle(h_map_file_);
        h_map_file_ = nullptr;
    }
    if (h_engine_event_ != nullptr) {
        CloseHandle(h_engine_event_);
        h_engine_event_ = nullptr;
    }
    if (h_daemon_event_ != nullptr) {
        CloseHandle(h_daemon_event_);
        h_daemon_event_ = nullptr;
    }
#else
    delete[] mapped_view_;
    mapped_view_ = nullptr;
#endif

    global_header_ = nullptr;
    is_initialized_ = false;
}

bool ShmServer::send_response(MessageType msg_type,
                              uint32_t stream_id,
                              uint16_t flags,
                              const void* payload,
                              uint32_t payload_size,
                              uint64_t sequence_num) {
    if (!is_initialized_) return false;
    std::lock_guard<std::mutex> lock(response_mutex_);

    // Check if queue was empty prior to this write
    const bool was_empty = response_ring_.is_empty();
    const bool is_urgent = (flags & FrameFlags::END_OF_STREAM) != 0 ||
                           msg_type == MessageType::MSG_TOOL_CALL ||
                           msg_type == MessageType::MSG_PONG;
    const bool is_heartbeat_forced = (unsignaled_tokens_ >= MAX_UNSIGNALED_TOKENS);

    const bool ok = response_ring_.write_frame(msg_type, stream_id, flags, payload, payload_size, sequence_num);
    if (ok) {
#ifdef _WIN32
        if (h_engine_event_) {
            // Adaptive signaling: only trigger OS kernel transition if consumer was idle
            // (empty queue), frame requires immediate delivery (EOS / ToolCall / Pong),
            // or heartbeat limit reached to prevent consumer starvation under high frequency.
            if (!adaptive_signaling_ || was_empty || is_urgent || is_heartbeat_forced) {
                SetEvent(h_engine_event_);
                total_events_signaled_++;
                unsignaled_tokens_ = 0;
            } else {
                total_signals_suppressed_++;
                unsignaled_tokens_++;
            }
        }
#endif
    }
    return ok;
}

bool ShmServer::send_token(uint32_t stream_id,
                           uint32_t token_id,
                           float logit,
                           const std::string& token_text,
                           bool is_final) {
    TokenPayload payload;
    std::memset(&payload, 0, sizeof(payload));
    payload.token_id = token_id;
    payload.logit = logit;
    payload.is_final = is_final ? 1 : 0;
    payload.text_len = static_cast<uint16_t>(std::min<size_t>(token_text.size(), sizeof(payload.text) - 1));

    if (payload.text_len > 0) {
        std::memcpy(payload.text, token_text.data(), payload.text_len);
    }
    payload.text[payload.text_len] = '\0';

    const uint16_t flags = is_final ? FrameFlags::END_OF_STREAM : FrameFlags::NONE;
    return send_response(MessageType::MSG_TOKEN_CHUNK, stream_id, flags, &payload, sizeof(payload));
}

bool ShmServer::send_tool_call(uint32_t stream_id,
                               const std::string& id,
                               const std::string& name,
                               const std::string& arguments) {
    if (!is_initialized_) return false;

    // Construct JSON payload matching nebula-model deserializer:
    // {"id": "...", "name": "...", "arguments": ...}
    std::string payload = "{\"id\":\"" + id + "\",\"name\":\"" + name + "\",\"arguments\":" + arguments + "}";
    return send_response(MessageType::MSG_TOOL_CALL, stream_id, FrameFlags::END_OF_STREAM,
                         payload.data(), static_cast<uint32_t>(payload.size()));
}

bool ShmServer::poll_request(IpcFrameHeader& out_header,
                             void* out_payload,
                             size_t max_payload_len) {
    if (!is_initialized_) return false;
    std::lock_guard<std::mutex> lock(request_mutex_);
    return request_ring_.read_frame(out_header, out_payload, max_payload_len);
}

bool ShmServer::wait_request(IpcFrameHeader& out_header,
                             void* out_payload,
                             size_t max_payload_len,
                             uint32_t timeout_ms) {
    if (!is_initialized_) return false;

    // Fast path: non-blocking poll with microsecond spin
    for (int spin = 0; spin < 100; ++spin) {
        if (poll_request(out_header, out_payload, max_payload_len)) {
            return true;
        }
    }

#ifdef _WIN32
    if (h_daemon_event_ != nullptr) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (true) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) break;
            const DWORD remaining_ms = static_cast<DWORD>(
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
            const DWORD wait_res = WaitForSingleObject(h_daemon_event_, std::max<DWORD>(1, remaining_ms));
            if (wait_res == WAIT_OBJECT_0) {
                if (poll_request(out_header, out_payload, max_payload_len)) {
                    return true;
                }
            } else if (wait_res == WAIT_TIMEOUT) {
                break;
            }
        }
    }
#else
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return poll_request(out_header, out_payload, max_payload_len);
#endif

    return false;
}

void ShmServer::set_active_tool_processes(uint32_t count) noexcept {
    if (global_header_) {
        global_header_->active_tool_processes.store(count, std::memory_order_release);
    }
}

uint32_t ShmServer::get_active_tool_processes() const noexcept {
    if (global_header_) {
        return global_header_->active_tool_processes.load(std::memory_order_acquire);
    }
    return 0;
}

void ShmServer::set_cuda_yield_mode(uint32_t mode) noexcept {
    if (global_header_) {
        global_header_->cuda_yield_mode.store(mode, std::memory_order_release);
    }
}

uint32_t ShmServer::get_cuda_yield_mode() const noexcept {
    if (global_header_) {
        return global_header_->cuda_yield_mode.load(std::memory_order_acquire);
    }
    return 0;
}

// ============================================================================
// ShmClient Implementation
// ============================================================================

ShmClient::ShmClient(const std::wstring& shm_name,
                     const std::wstring& engine_event_name,
                     const std::wstring& daemon_event_name)
    : shm_name_(shm_name),
      engine_event_name_(engine_event_name),
      daemon_event_name_(daemon_event_name) {}

ShmClient::~ShmClient() {
    disconnect();
}

bool ShmClient::connect(uint32_t retry_timeout_ms) {
    if (is_connected_) {
        return true;
    }

    const auto start = std::chrono::steady_clock::now();

#ifdef _WIN32
    while (true) {
        h_map_file_ = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, shm_name_.c_str());
        if (h_map_file_ != nullptr) {
            break;
        }

        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed >= retry_timeout_ms) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    mapped_view_ = reinterpret_cast<uint8_t*>(
        MapViewOfFile(h_map_file_, FILE_MAP_ALL_ACCESS, 0, 0, 0)
    );

    if (mapped_view_ == nullptr) {
        CloseHandle(h_map_file_);
        h_map_file_ = nullptr;
        return false;
    }

    h_engine_event_ = OpenEventW(SYNCHRONIZE | EVENT_MODIFY_STATE, FALSE, engine_event_name_.c_str());
    h_daemon_event_ = OpenEventW(SYNCHRONIZE | EVENT_MODIFY_STATE, FALSE, daemon_event_name_.c_str());

    if (h_engine_event_ == nullptr || h_daemon_event_ == nullptr) {
        disconnect();
        return false;
    }
#else
    return false;
#endif

    global_header_ = reinterpret_cast<ShmGlobalHeader*>(mapped_view_);
    if (global_header_->magic != IPC_MAGIC || global_header_->version != IPC_VERSION) {
        disconnect();
        return false;
    }

#ifdef _WIN32
    global_header_->daemon_pid = GetCurrentProcessId();
#endif

    request_ring_.init(&global_header_->request_queue,
                       mapped_view_ + global_header_->request_queue_offset,
                       REQUEST_QUEUE_CAPACITY_BYTES);

    response_ring_.init(&global_header_->response_queue,
                        mapped_view_ + global_header_->response_queue_offset,
                        RESPONSE_QUEUE_CAPACITY_BYTES);

    is_connected_ = true;
    return true;
}

void ShmClient::disconnect() {
    if (!is_connected_) {
        return;
    }

#ifdef _WIN32
    if (mapped_view_ != nullptr) {
        UnmapViewOfFile(mapped_view_);
        mapped_view_ = nullptr;
    }
    if (h_map_file_ != nullptr) {
        CloseHandle(h_map_file_);
        h_map_file_ = nullptr;
    }
    if (h_engine_event_ != nullptr) {
        CloseHandle(h_engine_event_);
        h_engine_event_ = nullptr;
    }
    if (h_daemon_event_ != nullptr) {
        CloseHandle(h_daemon_event_);
        h_daemon_event_ = nullptr;
    }
#endif

    global_header_ = nullptr;
    is_connected_ = false;
}

bool ShmClient::send_request(MessageType msg_type,
                             uint32_t stream_id,
                             uint16_t flags,
                             const void* payload,
                             uint32_t payload_size,
                             uint64_t sequence_num) {
    if (!is_connected_) return false;
    const bool ok = request_ring_.write_frame(msg_type, stream_id, flags, payload, payload_size, sequence_num);
    if (ok) {
#ifdef _WIN32
        if (h_daemon_event_) {
            SetEvent(h_daemon_event_);
        }
#endif
    }
    return ok;
}

bool ShmClient::poll_response(IpcFrameHeader& out_header,
                              void* out_payload,
                              size_t max_payload_len) {
    if (!is_connected_) return false;
    return response_ring_.read_frame(out_header, out_payload, max_payload_len);
}

bool ShmClient::wait_response(IpcFrameHeader& out_header,
                              void* out_payload,
                              size_t max_payload_len,
                              uint32_t timeout_ms) {
    if (!is_connected_) return false;

    // Fast path: non-blocking microsecond spin
    for (int spin = 0; spin < 100; ++spin) {
        if (poll_response(out_header, out_payload, max_payload_len)) {
            return true;
        }
    }

#ifdef _WIN32
    if (h_engine_event_ != nullptr) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (true) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) break;
            const DWORD remaining_ms = static_cast<DWORD>(
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
            const DWORD wait_res = WaitForSingleObject(h_engine_event_, std::max<DWORD>(1, remaining_ms));
            if (wait_res == WAIT_OBJECT_0) {
                if (poll_response(out_header, out_payload, max_payload_len)) {
                    return true;
                }
            } else if (wait_res == WAIT_TIMEOUT) {
                break;
            }
        }
    }
#endif

    return false;
}

bool ShmClient::wait_response_zero_copy(IpcFrameHeader& out_header,
                                        const uint8_t*& out_payload_ptr,
                                        uint32_t timeout_ms) {
    if (!is_connected_) return false;

    // Fast path
    for (int spin = 0; spin < 100; ++spin) {
        if (response_ring_.read_frame_zero_copy(out_header, out_payload_ptr)) {
            return true;
        }
    }

#ifdef _WIN32
    if (h_engine_event_ != nullptr) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (true) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) break;
            const DWORD remaining_ms = static_cast<DWORD>(
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
            const DWORD wait_res = WaitForSingleObject(h_engine_event_, std::max<DWORD>(1, remaining_ms));
            if (wait_res == WAIT_OBJECT_0) {
                if (response_ring_.read_frame_zero_copy(out_header, out_payload_ptr)) {
                    return true;
                }
            } else if (wait_res == WAIT_TIMEOUT) {
                break;
            }
        }
    }
#endif

    return false;
}

void ShmClient::notify_tool_process_spawn() noexcept {
    if (global_header_) {
        global_header_->active_tool_processes.fetch_add(1, std::memory_order_release);
    }
}

void ShmClient::notify_tool_process_complete() noexcept {
    if (global_header_) {
        global_header_->active_tool_processes.fetch_sub(1, std::memory_order_release);
    }
}

} // namespace hydrogen::ipc
