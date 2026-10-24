#pragma once

#include "hydrogen_ipc.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <functional>
#include <stdexcept>
#include <vector>

namespace hydrogen::ipc {

// ============================================================================
// Single-Producer Single-Consumer Zero-Copy Ring Buffer
// ============================================================================

class ShmRingBuffer {
public:
    ShmRingBuffer() = default;
    ShmRingBuffer(ShmQueueHeader* queue_hdr, uint8_t* buffer_base, uint64_t capacity);

    /**
     * @brief Writes a message frame into the ring buffer.
     * @param msg_type Type of message (MSG_TOKEN_CHUNK, MSG_GENERATE_REQUEST, etc.)
     * @param stream_id Associated stream ID
     * @param flags Bitwise frame flags (FrameFlags::*)
     * @param payload Pointer to binary payload
     * @param payload_size Byte length of payload
     * @param sequence_num Optional sequence number (0 to auto-generate)
     * @return True if written successfully, False if queue is full
     */
    bool write_frame(MessageType msg_type,
                     uint32_t stream_id,
                     uint16_t flags,
                     const void* payload,
                     uint32_t payload_size,
                     uint64_t sequence_num = 0);

    /**
     * @brief Reads the next available message frame from the ring buffer.
     * @param out_header Output header structure
     * @param out_payload Destination buffer for payload
     * @param max_payload_len Capacity of destination buffer
     * @return True if a frame was read, False if queue is empty
     */
    bool read_frame(IpcFrameHeader& out_header,
                    void* out_payload,
                    size_t max_payload_len);

    /**
     * @brief Zero-copy read: returns a direct pointer to contiguous payload in SHM.
     * @param out_header Output header structure
     * @param out_payload_ptr Output pointer directly into shared memory
     * @return True if frame read, False if empty
     */
    bool read_frame_zero_copy(IpcFrameHeader& out_header,
                              const uint8_t*& out_payload_ptr);

    [[nodiscard]] uint64_t available_read_bytes() const noexcept;
    [[nodiscard]] uint64_t available_write_bytes() const noexcept;
    [[nodiscard]] bool is_empty() const noexcept;

    void init(ShmQueueHeader* queue_hdr, uint8_t* buffer_base, uint64_t capacity);

    void reset() noexcept;

private:
    ShmQueueHeader* header_ = nullptr;
    uint8_t* buffer_ = nullptr;
    uint64_t capacity_ = 0;
    uint64_t seq_counter_ = 1;
};

// ============================================================================
// Shared Memory IPC Server (Hydrogen Inference Engine side)
// ============================================================================

class ShmServer {
public:
    explicit ShmServer(const std::wstring& shm_name = DEFAULT_SHM_NAME,
                       const std::wstring& engine_event_name = DEFAULT_ENGINE_EVENT_NAME,
                       const std::wstring& daemon_event_name = DEFAULT_DAEMON_EVENT_NAME,
                       size_t total_capacity = SHM_TOTAL_CAPACITY_BYTES);
    ~ShmServer();

    // Non-copyable, non-movable
    ShmServer(const ShmServer&) = delete;
    ShmServer& operator=(const ShmServer&) = delete;
    ShmServer(ShmServer&&) = delete;
    ShmServer& operator=(ShmServer&&) = delete;

    bool initialize();
    void close();

    [[nodiscard]] bool is_initialized() const noexcept { return is_initialized_; }

    // Stream responses to Daemon (Engine -> Daemon)
    bool send_response(MessageType msg_type,
                       uint32_t stream_id,
                       uint16_t flags,
                       const void* payload,
                       uint32_t payload_size,
                       uint64_t sequence_num = 0);

    bool send_token(uint32_t stream_id,
                    uint32_t token_id,
                    float logit,
                    const std::string& token_text,
                    bool is_final = false);

    bool send_tool_call(uint32_t stream_id,
                        const std::string& id,
                        const std::string& name,
                        const std::string& arguments);

    // Read requests from Daemon (Daemon -> Engine)
    bool poll_request(IpcFrameHeader& out_header,
                      void* out_payload,
                      size_t max_payload_len);

    bool wait_request(IpcFrameHeader& out_header,
                      void* out_payload,
                      size_t max_payload_len,
                      uint32_t timeout_ms = 1000);

    // Process Spawn Hook Coordination
    void set_active_tool_processes(uint32_t count) noexcept;
    [[nodiscard]] uint32_t get_active_tool_processes() const noexcept;
    void set_cuda_yield_mode(uint32_t mode) noexcept;
    [[nodiscard]] uint32_t get_cuda_yield_mode() const noexcept;

    // Adaptive Win32 Event Signaling (suppresses redundant syscalls when ring is non-empty)
    void set_adaptive_signaling(bool enable) noexcept { adaptive_signaling_ = enable; }
    [[nodiscard]] bool is_adaptive_signaling_enabled() const noexcept { return adaptive_signaling_; }
    [[nodiscard]] uint64_t total_events_signaled() const noexcept { return total_events_signaled_; }
    [[nodiscard]] uint64_t total_signals_suppressed() const noexcept { return total_signals_suppressed_; }

    [[nodiscard]] ShmGlobalHeader* raw_header() noexcept { return global_header_; }

private:
    std::wstring shm_name_;
    std::wstring engine_event_name_;
    std::wstring daemon_event_name_;
    size_t total_capacity_;

#ifdef _WIN32
    HANDLE h_map_file_ = nullptr;
    HANDLE h_engine_event_ = nullptr;
    HANDLE h_daemon_event_ = nullptr;
#endif

    uint8_t* mapped_view_ = nullptr;
    ShmGlobalHeader* global_header_ = nullptr;
    bool is_initialized_ = false;

    ShmRingBuffer request_ring_;  // Daemon -> Engine
    ShmRingBuffer response_ring_; // Engine -> Daemon

    mutable std::mutex response_mutex_;
    mutable std::mutex request_mutex_;

    bool adaptive_signaling_ = true;
    uint32_t unsignaled_tokens_ = 0;
    static constexpr uint32_t MAX_UNSIGNALED_TOKENS = 16;
    uint64_t total_events_signaled_ = 0;
    uint64_t total_signals_suppressed_ = 0;
};

// ============================================================================
// Shared Memory IPC Client (Nebula Daemon / Test Benchmark side)
// ============================================================================

class ShmClient {
public:
    explicit ShmClient(const std::wstring& shm_name = DEFAULT_SHM_NAME,
                       const std::wstring& engine_event_name = DEFAULT_ENGINE_EVENT_NAME,
                       const std::wstring& daemon_event_name = DEFAULT_DAEMON_EVENT_NAME);
    ~ShmClient();

    // Non-copyable, non-movable
    ShmClient(const ShmClient&) = delete;
    ShmClient& operator=(const ShmClient&) = delete;
    ShmClient(ShmClient&&) = delete;
    ShmClient& operator=(ShmClient&&) = delete;

    bool connect(uint32_t retry_timeout_ms = 5000);
    void disconnect();

    [[nodiscard]] bool is_connected() const noexcept { return is_connected_; }

    // Submit requests to Engine (Daemon -> Engine)
    bool send_request(MessageType msg_type,
                      uint32_t stream_id,
                      uint16_t flags,
                      const void* payload,
                      uint32_t payload_size,
                      uint64_t sequence_num = 0);

    // Read responses/tokens from Engine (Engine -> Daemon)
    bool poll_response(IpcFrameHeader& out_header,
                       void* out_payload,
                       size_t max_payload_len);

    bool wait_response(IpcFrameHeader& out_header,
                       void* out_payload,
                       size_t max_payload_len,
                       uint32_t timeout_ms = 1000);

    bool wait_response_zero_copy(IpcFrameHeader& out_header,
                                 const uint8_t*& out_payload_ptr,
                                 uint32_t timeout_ms = 1000);

    // Process Spawn Hook Coordination
    void notify_tool_process_spawn() noexcept;
    void notify_tool_process_complete() noexcept;

private:
    std::wstring shm_name_;
    std::wstring engine_event_name_;
    std::wstring daemon_event_name_;

#ifdef _WIN32
    HANDLE h_map_file_ = nullptr;
    HANDLE h_engine_event_ = nullptr;
    HANDLE h_daemon_event_ = nullptr;
#endif

    uint8_t* mapped_view_ = nullptr;
    ShmGlobalHeader* global_header_ = nullptr;
    bool is_connected_ = false;

    ShmRingBuffer request_ring_;  // Daemon -> Engine
    ShmRingBuffer response_ring_; // Engine -> Daemon
};

} // namespace hydrogen::ipc
