//! Win32 shared memory IPC: safe Rust mirror of `hydrogen_ipc.h`.
//!
//! Every field offset and size matches the C++ `#pragma pack(push, 8)` / `alignas(64)` layout
//! so the Rust daemon and the C++ engine share the same memory without serialization.

#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_ALL_ACCESS, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile,
};
use windows::Win32::System::Threading::{CreateEventW, OpenEventW, SetEvent, WaitForSingleObject};
use windows::core::PCWSTR;

use crate::{IPC_MAGIC, IPC_VERSION, SHM_TOTAL_CAPACITY_BYTES};

/// IPC protocol errors.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum IpcError {
    /// A Win32 API call failed.
    #[error("win32: {0}")]
    Win32(String),
    /// The shared memory header has a wrong magic or version.
    #[error(
        "protocol mismatch: expected magic 0x{expected_magic:08X} version {expected_version}, got magic 0x{actual_magic:08X} version {actual_version}"
    )]
    ProtocolMismatch {
        /// Expected magic.
        expected_magic: u32,
        /// Expected version.
        expected_version: u32,
        /// Actual magic.
        actual_magic: u32,
        /// Actual version.
        actual_version: u32,
    },
    /// The mapped region is too small.
    #[error("shared memory too small: need {need} bytes, got {got}")]
    TooSmall {
        /// Required size.
        need: usize,
        /// Actual size.
        got: usize,
    },
    /// A frame has invalid content.
    #[error("bad frame: {0}")]
    BadFrame(String),
    /// The ring buffer has no data available.
    #[error("ring buffer empty")]
    Empty,
    /// Wait timed out.
    #[error("wait timed out after {ms} ms")]
    Timeout {
        /// Milliseconds waited.
        ms: u32,
    },
}

// ---------------------------------------------------------------------------
// Protocol message types — mirrors `MessageType` enum in hydrogen_ipc.h
// ---------------------------------------------------------------------------

/// IPC message type discriminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum MessageType {
    /// No message.
    None = 0x0000,
    /// Ping (health check request).
    Ping = 0x0001,
    /// Pong (health check response).
    Pong = 0x0002,
    /// Wrap-around sentinel in the ring buffer.
    WrapMarker = 0x000F,
    /// Chat/generate request (daemon → engine).
    GenerateRequest = 0x0010,
    /// Pre-tokenized binary token request (daemon → engine).
    GenerateTokensRequest = 0x0011,
    /// Backpressure: all 8 stream slots are currently leased.
    ErrorSlotsExhausted = 0x0012,
    /// Explicit release of stream slot lease & KV cache.
    ReleaseStream = 0x0013,
    /// Token chunk (engine → daemon).
    TokenChunk = 0x0020,
    /// Tool call descriptor (engine → daemon).
    ToolCall = 0x0030,
    /// Tool process spawned notification (daemon → engine).
    ToolSpawn = 0x0031,
    /// Tool process completed notification (daemon → engine).
    ToolComplete = 0x0032,
    /// Cancel an in-flight stream (daemon → engine).
    CancelStream = 0x0040,
    /// Terminate the engine (daemon → engine).
    TerminateEngine = 0x00FF,
}

impl MessageType {
    /// Converts a raw `u16` to a `MessageType`, returning `None` for unknown values.
    #[must_use]
    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            0x0000 => Some(Self::None),
            0x0001 => Some(Self::Ping),
            0x0002 => Some(Self::Pong),
            0x000F => Some(Self::WrapMarker),
            0x0010 => Some(Self::GenerateRequest),
            0x0011 => Some(Self::GenerateTokensRequest),
            0x0012 => Some(Self::ErrorSlotsExhausted),
            0x0013 => Some(Self::ReleaseStream),
            0x0020 => Some(Self::TokenChunk),
            0x0030 => Some(Self::ToolCall),
            0x0031 => Some(Self::ToolSpawn),
            0x0032 => Some(Self::ToolComplete),
            0x0040 => Some(Self::CancelStream),
            0x00FF => Some(Self::TerminateEngine),
            _ => Option::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Frame flags
// ---------------------------------------------------------------------------

/// Bitmask flags on an IPC frame.
pub struct FrameFlags;

impl FrameFlags {
    /// No flags.
    pub const NONE: u16 = 0x0000;
    /// Last frame in a stream.
    pub const END_OF_STREAM: u16 = 0x0001;
    /// Priority / out-of-band frame.
    pub const URGENT: u16 = 0x0002;
    /// Ring buffer wrap-around marker.
    pub const WRAP_AROUND: u16 = 0x0004;
}

// ---------------------------------------------------------------------------
// Frame header — mirrors `IpcFrameHeader` (pack 8)
// ---------------------------------------------------------------------------

/// Binary frame header at the start of every IPC message.
///
/// Layout matches C++ `#pragma pack(push, 8)`: total 32 bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed(8))]
pub struct IpcFrameHeader {
    /// Protocol magic (`IPC_MAGIC`).
    pub magic: u32,
    /// [`MessageType`] discriminator.
    pub msg_type: u16,
    /// [`FrameFlags`] bitmask.
    pub flags: u16,
    /// Stream identifier (ties request to response tokens).
    pub stream_id: u32,
    /// Byte length of the payload immediately following this header.
    pub payload_size: u32,
    /// Monotonically increasing sequence number.
    pub sequence_num: u64,
    /// Monotonic nanosecond timestamp for latency measurement.
    pub timestamp_ns: u64,
}

/// Size of a frame header in bytes.
pub const FRAME_HEADER_SIZE: usize = std::mem::size_of::<IpcFrameHeader>();

// ---------------------------------------------------------------------------
// Token payload — mirrors `TokenPayload` (pack 8)
// ---------------------------------------------------------------------------

/// Compact inline token payload for zero-allocation streaming.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed(8))]
pub struct TokenPayload {
    /// Vocabulary token id.
    pub token_id: u32,
    /// Raw logit value.
    pub logit: f32,
    /// Byte length of the UTF-8 text in `text`.
    pub text_len: u16,
    /// 1 if this is the final token in the sequence.
    pub is_final: u8,
    /// Reserved for alignment.
    pub reserved: u8,
    /// Inline null-terminated text (up to 63 UTF-8 bytes + NUL).
    pub text: [u8; 64],
}

impl TokenPayload {
    /// The text content as a `&str`, clipped to `text_len`.
    ///
    /// Returns an empty string if the bytes are not valid UTF-8.
    #[must_use]
    pub fn text_str(&self) -> &str {
        let len = (self.text_len as usize).min(63);
        std::str::from_utf8(&self.text[..len]).unwrap_or("")
    }

    /// Safely parses a `TokenPayload` from raw payload bytes.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return None;
        }
        // SAFETY: The slice length is verified to be at least size_of::<TokenPayload>().
        let payload = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<Self>()) };
        Some(payload)
    }
}

// ---------------------------------------------------------------------------
// GenerateTokensRequest payload — mirrors `GenerateTokensRequest` (pack 8)
// ---------------------------------------------------------------------------

/// Pre-tokenized binary token request submitted with `GenerateTokensRequest`.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed(8))]
pub struct GenerateTokensRequest {
    /// Target stream slot ID.
    pub stream_id: u32,
    /// Number of pre-tokenized token IDs following this header.
    pub num_tokens: u32,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature (0.0 for greedy argmax).
    pub temperature: f32,
    /// Top-p nucleus sampling threshold.
    pub top_p: f32,
    /// Number of valid stop tokens in `stop_tokens`.
    pub stop_token_count: u32,
    /// Array of up to 16 stop token IDs.
    pub stop_tokens: [u32; 16],
}

/// Size of `GenerateTokensRequest` header in bytes.
pub const GENERATE_TOKENS_REQUEST_SIZE: usize = std::mem::size_of::<GenerateTokensRequest>();

// ---------------------------------------------------------------------------
// Queue header — mirrors `ShmQueueHeader` (alignas 64)
// ---------------------------------------------------------------------------

/// Lock-free SPSC ring buffer control block.
///
/// Field layout matches C++ `alignas(64)` on `head` and `tail`, with a 192-byte total size.
#[repr(C, align(64))]
pub struct ShmQueueHeader {
    /// Ring buffer capacity.
    pub capacity_bytes: u64,
    /// Bitmask for power-of-two wrapping.
    pub mask: u64,
    /// Padding to 64-byte boundary for `head`.
    _pad_head: [u8; 48],
    /// Write cursor (advanced by producer).
    pub head: AtomicU64,
    /// Padding to 64-byte boundary for `tail`.
    _pad_tail: [u8; 56],
    /// Read cursor (advanced by consumer).
    pub tail: AtomicU64,
    /// Total bytes written (diagnostic).
    pub total_bytes_written: u64,
    /// Total bytes read (diagnostic).
    pub total_bytes_read: u64,
    /// Total frames written (diagnostic).
    pub total_frames_written: u64,
    /// Total frames read (diagnostic).
    pub total_frames_read: u64,
    /// Trailing padding to 64-byte boundary.
    _pad_end: [u8; 24],
}

// ---------------------------------------------------------------------------
// Global header — mirrors `ShmGlobalHeader` (alignas 64)
// ---------------------------------------------------------------------------

/// Shared memory global header at byte 0 of the mapping.
///
/// Layout matches C++ `alignas(64)` on the struct itself and on
/// `active_tool_processes` / `cuda_yield_mode`, with a 576-byte total size.
#[repr(C, align(64))]
pub struct ShmGlobalHeader {
    /// Protocol magic.
    pub magic: u32,
    /// Protocol version.
    pub version: u32,
    /// Total mapped size.
    pub total_size_bytes: u64,
    /// Daemon PID.
    pub daemon_pid: u32,
    /// Engine PID.
    pub engine_pid: u32,
    /// Atomic bitmask of active stream leases (bits 0..7 for streams 0..7).
    pub active_stream_mask: AtomicU32,
    // 28 bytes used (24 + 4), pad to 64 for active_tool_processes alignment (36 bytes)
    _pad_atp: [u8; 36],
    /// Immediate process spawn hook: tool process counter.
    pub active_tool_processes: AtomicU32,
    // 4 bytes for AtomicU32, pad to 64 for cuda_yield_mode alignment
    _pad_cym: [u8; 60],
    /// Engine scheduling mode: 0 = spin/poll, 1 = `cudaDeviceScheduleYield`.
    pub cuda_yield_mode: AtomicU32,
    // 4 bytes padding to 8-byte boundary for uint64_t
    _pad_post_cym: [u8; 4],
    /// Byte offset of the request queue data region.
    pub request_queue_offset: u64,
    /// Byte offset of the response queue data region.
    pub response_queue_offset: u64,
    // 8 bytes (144..152) used, pad to 64-byte boundary for ShmQueueHeader (152..192)
    _pad_rq: [u8; 40],
    /// Request queue control block.
    pub request_queue: ShmQueueHeader,
    /// Response queue control block.
    pub response_queue: ShmQueueHeader,
}

// Compile-time layout guarantees matching C++ hydrogen_ipc.h
const _: () = {
    assert!(std::mem::size_of::<ShmQueueHeader>() == 192);
    assert!(std::mem::align_of::<ShmQueueHeader>() == 64);
    assert!(std::mem::size_of::<ShmGlobalHeader>() == 576);
    assert!(std::mem::align_of::<ShmGlobalHeader>() == 64);
};

// ---------------------------------------------------------------------------
// ShmHandle — the safe, owning wrapper
// ---------------------------------------------------------------------------

/// Owning handle to the Win32 named shared memory section.
///
/// On drop, the view is unmapped and the handles are closed.
pub struct ShmHandle {
    /// Base pointer to the mapped view.
    base: NonNull<u8>,
    /// File mapping handle.
    map_handle: HANDLE,
    /// Engine event (signaled by engine when data is ready).
    engine_event: HANDLE,
    /// Daemon event (signaled by daemon when a request is submitted).
    daemon_event: HANDLE,
    /// Mapped size.
    size: usize,
}

// The shared memory is process-local but may be sent across threads.
// SAFETY: The memory is accessed through atomic operations and frame-level locking.
unsafe impl Send for ShmHandle {}
// SAFETY: Concurrent reads are safe via atomics; writes go through the SPSC discipline.
unsafe impl Sync for ShmHandle {}

impl ShmHandle {
    /// Creates (or opens) the shared memory section and both Win32 events.
    ///
    /// When `create` is true, the section is created and the header is initialized.
    /// When false, an existing section is opened and the header is validated.
    ///
    /// # Errors
    /// [`IpcError::Win32`] on API failure; [`IpcError::ProtocolMismatch`] if the header is wrong.
    pub fn open(
        shm_name: &str,
        engine_event_name: &str,
        daemon_event_name: &str,
        create: bool,
    ) -> Result<Self, IpcError> {
        let size = SHM_TOTAL_CAPACITY_BYTES;

        let shm_wide = to_wide(shm_name);
        let engine_evt_wide = to_wide(engine_event_name);
        let daemon_evt_wide = to_wide(daemon_event_name);

        // SAFETY: CreateFileMappingW / MapViewOfFile are Win32 FFI calls. We pass valid
        // null-terminated wide strings and check return values for null/invalid handle.
        let (map_handle, base) = unsafe {
            let h = CreateFileMappingW(
                HANDLE(-1isize as _), // INVALID_HANDLE_VALUE → page-file backed
                None,                 // default security
                PAGE_READWRITE,
                (size >> 32) as u32,
                size as u32,
                PCWSTR(shm_wide.as_ptr()),
            )
            .map_err(|e| IpcError::Win32(format!("CreateFileMappingW: {e}")))?;

            let ptr = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
            if ptr.Value.is_null() {
                let _ = CloseHandle(h);
                return Err(IpcError::Win32("MapViewOfFile returned null".into()));
            }
            (h, NonNull::new_unchecked(ptr.Value.cast::<u8>()))
        };

        // SAFETY: CreateEventW / OpenEventW with valid wide strings.
        let engine_event = unsafe {
            if create {
                CreateEventW(None, false, false, PCWSTR(engine_evt_wide.as_ptr()))
                    .map_err(|e| IpcError::Win32(format!("CreateEventW(engine): {e}")))?
            } else {
                OpenEventW(
                    windows::Win32::System::Threading::SYNCHRONIZATION_SYNCHRONIZE
                        | windows::Win32::System::Threading::EVENT_MODIFY_STATE,
                    false,
                    PCWSTR(engine_evt_wide.as_ptr()),
                )
                .map_err(|e| IpcError::Win32(format!("OpenEventW(engine): {e}")))?
            }
        };

        let daemon_event = unsafe {
            if create {
                CreateEventW(None, false, false, PCWSTR(daemon_evt_wide.as_ptr()))
                    .map_err(|e| IpcError::Win32(format!("CreateEventW(daemon): {e}")))?
            } else {
                OpenEventW(
                    windows::Win32::System::Threading::SYNCHRONIZATION_SYNCHRONIZE
                        | windows::Win32::System::Threading::EVENT_MODIFY_STATE,
                    false,
                    PCWSTR(daemon_evt_wide.as_ptr()),
                )
                .map_err(|e| IpcError::Win32(format!("OpenEventW(daemon): {e}")))?
            }
        };

        let handle = Self {
            base,
            map_handle,
            engine_event,
            daemon_event,
            size,
        };

        if create {
            let mut handle = handle;
            handle.init_header();
            Ok(handle)
        } else {
            handle.validate_header()?;
            Ok(handle)
        }
    }

    /// Mutable reference to the global header during initialization.
    fn header_mut(&mut self) -> &mut ShmGlobalHeader {
        // SAFETY: Only called during init when we hold &mut self.
        unsafe { &mut *self.base.as_ptr().cast::<ShmGlobalHeader>() }
    }

    /// Writes the initial header (magic, version, queue offsets).
    fn init_header(&mut self) {
        let total_size = self.size as u64;
        let hdr = self.header_mut();
        hdr.magic = IPC_MAGIC;
        hdr.version = IPC_VERSION;
        hdr.total_size_bytes = total_size;
        hdr.daemon_pid = std::process::id();
        hdr.engine_pid = 0;
        hdr.active_tool_processes.store(0, Ordering::Release);
        hdr.cuda_yield_mode.store(0, Ordering::Release);

        let header_size = std::mem::size_of::<ShmGlobalHeader>();
        hdr.request_queue_offset = header_size as u64;
        hdr.response_queue_offset = (header_size + crate::REQUEST_QUEUE_CAPACITY_BYTES) as u64;

        let rq = &mut hdr.request_queue;
        rq.capacity_bytes = crate::REQUEST_QUEUE_CAPACITY_BYTES as u64;
        rq.mask = (crate::REQUEST_QUEUE_CAPACITY_BYTES - 1) as u64;
        rq.head.store(0, Ordering::Release);
        rq.tail.store(0, Ordering::Release);

        let rsq = &mut hdr.response_queue;
        rsq.capacity_bytes = crate::RESPONSE_QUEUE_CAPACITY_BYTES as u64;
        rsq.mask = (crate::RESPONSE_QUEUE_CAPACITY_BYTES - 1) as u64;
        rsq.head.store(0, Ordering::Release);
        rsq.tail.store(0, Ordering::Release);
    }

    /// Validates the header magic and version.
    fn validate_header(&self) -> Result<(), IpcError> {
        let hdr = self.header();
        if hdr.magic != IPC_MAGIC || hdr.version != IPC_VERSION {
            return Err(IpcError::ProtocolMismatch {
                expected_magic: IPC_MAGIC,
                expected_version: IPC_VERSION,
                actual_magic: hdr.magic,
                actual_version: hdr.version,
            });
        }
        Ok(())
    }

    /// The global header, shared between daemon and engine.
    #[must_use]
    pub fn header(&self) -> &ShmGlobalHeader {
        // SAFETY: The base pointer is valid for at least SHM_TOTAL_CAPACITY_BYTES which is
        // much larger than ShmGlobalHeader. The header is initialized before any read.
        unsafe { &*self.base.as_ptr().cast::<ShmGlobalHeader>() }
    }

    /// Atomically increments `active_tool_processes` with `Release` ordering.
    ///
    /// Called by the daemon when a tool process is spawned, so the engine can transition
    /// from `cudaDeviceScheduleSpin` to `cudaDeviceScheduleYield`.
    pub fn tool_process_spawned(&self) {
        self.header()
            .active_tool_processes
            .fetch_add(1, Ordering::Release);
        tracing::debug!(event = "hydrogen.tool_spawned", active = ?self.active_tool_processes());
    }

    /// Atomically decrements `active_tool_processes` with `Release` ordering.
    ///
    /// Called by the daemon when a tool process completes.
    pub fn tool_process_completed(&self) {
        self.header()
            .active_tool_processes
            .fetch_sub(1, Ordering::Release);
        tracing::debug!(event = "hydrogen.tool_completed", active = ?self.active_tool_processes());
    }

    /// Current tool process count (loaded with `Acquire` ordering).
    #[must_use]
    pub fn active_tool_processes(&self) -> u32 {
        self.header().active_tool_processes.load(Ordering::Acquire)
    }

    /// Current CUDA yield mode (loaded with `Acquire` ordering).
    #[must_use]
    pub fn cuda_yield_mode(&self) -> u32 {
        self.header().cuda_yield_mode.load(Ordering::Acquire)
    }

    /// Writes a frame into the request ring buffer (daemon → engine).
    ///
    /// # Errors
    /// [`IpcError::BadFrame`] if the payload is too large for the ring buffer.
    pub fn write_request_frame(
        &self,
        msg_type: MessageType,
        flags: u16,
        stream_id: u32,
        payload: &[u8],
    ) -> Result<(), IpcError> {
        let hdr = self.header();
        let total = FRAME_HEADER_SIZE + payload.len();

        let capacity = hdr.request_queue.capacity_bytes as usize;
        if total > capacity / 2 {
            return Err(IpcError::BadFrame(format!(
                "frame too large: {total} bytes exceeds half the {capacity}-byte request queue"
            )));
        }

        let head = hdr.request_queue.head.load(Ordering::Acquire);
        let seq = hdr.request_queue.total_frames_written;

        let frame = IpcFrameHeader {
            magic: IPC_MAGIC,
            msg_type: msg_type as u16,
            flags,
            stream_id,
            payload_size: payload.len() as u32,
            sequence_num: seq,
            timestamp_ns: monotonic_ns(),
        };

        let rq_base = self.queue_data_ptr(hdr.request_queue_offset as usize);
        let mask = hdr.request_queue.mask as usize;

        // SAFETY: Writing into the request queue region. The mask ensures we stay in bounds.
        // This is the single producer in the SPSC model (the daemon).
        unsafe {
            let header_bytes = std::slice::from_raw_parts(
                &frame as *const IpcFrameHeader as *const u8,
                FRAME_HEADER_SIZE,
            );
            self.ring_write(rq_base, mask, head as usize, header_bytes);
            if !payload.is_empty() {
                self.ring_write(rq_base, mask, head as usize + FRAME_HEADER_SIZE, payload);
            }
        }

        hdr.request_queue
            .head
            .store(head + total as u64, Ordering::Release);

        Ok(())
    }

    /// Signals the daemon event, waking the engine's wait loop.
    ///
    /// # Errors
    /// [`IpcError::Win32`] if `SetEvent` fails.
    pub fn signal_daemon_event(&self) -> Result<(), IpcError> {
        // SAFETY: daemon_event is a valid event handle opened/created during construction.
        unsafe {
            SetEvent(self.daemon_event)
                .map_err(|e| IpcError::Win32(format!("SetEvent(daemon): {e}")))
        }
    }

    /// Waits for the engine event (token/response ready) up to `timeout_ms` milliseconds.
    ///
    /// # Errors
    /// [`IpcError::Timeout`] if the wait exceeds `timeout_ms`.
    pub fn wait_engine_event(&self, timeout_ms: u32) -> Result<(), IpcError> {
        // SAFETY: engine_event is a valid handle.
        let result = unsafe { WaitForSingleObject(self.engine_event, timeout_ms) };
        if result == WAIT_OBJECT_0 {
            Ok(())
        } else {
            Err(IpcError::Timeout { ms: timeout_ms })
        }
    }

    /// Signals the engine event, waking the daemon's wait loop (engine side).
    ///
    /// # Errors
    /// [`IpcError::Win32`] if `SetEvent` fails.
    pub fn signal_engine_event(&self) -> Result<(), IpcError> {
        // SAFETY: engine_event is a valid event handle.
        unsafe {
            SetEvent(self.engine_event)
                .map_err(|e| IpcError::Win32(format!("SetEvent(engine): {e}")))
        }
    }

    /// Waits for the daemon event (request ready) up to `timeout_ms` milliseconds (engine side).
    ///
    /// # Errors
    /// [`IpcError::Timeout`] if the wait exceeds `timeout_ms`.
    pub fn wait_daemon_event(&self, timeout_ms: u32) -> Result<(), IpcError> {
        // SAFETY: daemon_event is a valid handle.
        let result = unsafe { WaitForSingleObject(self.daemon_event, timeout_ms) };
        if result == WAIT_OBJECT_0 {
            Ok(())
        } else {
            Err(IpcError::Timeout { ms: timeout_ms })
        }
    }

    /// Reads the next frame header from the response ring buffer, if available.
    ///
    /// Returns `None` if the ring buffer is empty.
    #[must_use]
    pub fn peek_response_frame(&self) -> Option<IpcFrameHeader> {
        let hdr = self.header();
        let head = hdr.response_queue.head.load(Ordering::Acquire);
        let tail = hdr.response_queue.tail.load(Ordering::Acquire);

        if head == tail {
            return Option::None;
        }

        let rsq_base = self.queue_data_ptr(hdr.response_queue_offset as usize);
        let mask = hdr.response_queue.mask as usize;

        // SAFETY: Reading from the response queue region; tail < head guarantees data exists.
        let frame = unsafe {
            let mut buf = [0u8; FRAME_HEADER_SIZE];
            self.ring_read(rsq_base, mask, tail as usize, &mut buf);
            std::ptr::read_unaligned(buf.as_ptr().cast::<IpcFrameHeader>())
        };

        Some(frame)
    }

    /// Reads the payload of the current response frame (after [`peek_response_frame`]).
    ///
    /// Advances the tail cursor past the entire frame (header + payload).
    ///
    /// # Errors
    /// [`IpcError::Empty`] if no frame is available.
    pub fn read_response_payload(&self, payload_size: usize) -> Result<Vec<u8>, IpcError> {
        let hdr = self.header();
        let head = hdr.response_queue.head.load(Ordering::Acquire);
        let tail = hdr.response_queue.tail.load(Ordering::Acquire);

        let total = FRAME_HEADER_SIZE + payload_size;
        if head.wrapping_sub(tail) < total as u64 {
            return Err(IpcError::Empty);
        }

        let rsq_base = self.queue_data_ptr(hdr.response_queue_offset as usize);
        let mask = hdr.response_queue.mask as usize;

        let mut buf = vec![0u8; payload_size];
        // SAFETY: The bounds check above ensures payload_size bytes are available.
        unsafe {
            self.ring_read(rsq_base, mask, tail as usize + FRAME_HEADER_SIZE, &mut buf);
        }

        hdr.response_queue
            .tail
            .store(tail + total as u64, Ordering::Release);

        Ok(buf)
    }

    /// Sends a `MSG_PING` and waits for `MSG_PONG` within `timeout_ms`.
    ///
    /// # Errors
    /// [`IpcError`] on write, signal, or timeout.
    pub fn ping(&self, timeout_ms: u32) -> Result<(), IpcError> {
        self.write_request_frame(MessageType::Ping, FrameFlags::NONE, 0, &[])?;
        self.signal_daemon_event()?;
        self.wait_engine_event(timeout_ms)?;

        if let Some(frame) = self
            .peek_response_frame()
            .filter(|f| f.msg_type == MessageType::Pong as u16)
        {
            self.read_response_payload(frame.payload_size as usize)?;
            return Ok(());
        }
        Err(IpcError::Timeout { ms: timeout_ms })
    }

    // --- private helpers ---

    /// Pointer to the start of a queue's data region.
    fn queue_data_ptr(&self, offset: usize) -> *mut u8 {
        // SAFETY: offset is within the mapped region.
        unsafe { self.base.as_ptr().add(offset) }
    }

    /// Byte-by-byte ring buffer write with wrapping via mask.
    ///
    /// # Safety
    /// `base` must point to a valid region of at least `mask + 1` bytes.
    unsafe fn ring_write(&self, base: *mut u8, mask: usize, start: usize, data: &[u8]) {
        for (i, &b) in data.iter().enumerate() {
            let offset = (start + i) & mask;
            // SAFETY: offset is bounded by mask which is < queue capacity.
            unsafe { base.add(offset).write(b) };
        }
    }

    /// Byte-by-byte ring buffer read with wrapping via mask.
    ///
    /// # Safety
    /// `base` must point to a valid region of at least `mask + 1` bytes.
    unsafe fn ring_read(&self, base: *mut u8, mask: usize, start: usize, buf: &mut [u8]) {
        for (i, b) in buf.iter_mut().enumerate() {
            let offset = (start + i) & mask;
            // SAFETY: offset is bounded by mask which is < queue capacity.
            *b = unsafe { base.add(offset).read() };
        }
    }

    /// The mapped size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for ShmHandle {
    fn drop(&mut self) {
        // SAFETY: Unmapping and closing handles we own.
        unsafe {
            let _ = UnmapViewOfFile(windows::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.base.as_ptr().cast(),
            });
            let _ = CloseHandle(self.map_handle);
            let _ = CloseHandle(self.engine_event);
            let _ = CloseHandle(self.daemon_event);
        }
    }
}

/// Monotonic nanosecond timestamp (for frame timing).
fn monotonic_ns() -> u64 {
    use std::time::Instant;

    // Use a thread-local origin to avoid the overhead of repeated `Instant::now()` drift
    // calculations. The exact epoch doesn't matter — only relative differences.
    thread_local! {
        static ORIGIN: Instant = Instant::now();
    }
    ORIGIN.with(|origin| {
        let elapsed = origin.elapsed();
        elapsed.as_nanos() as u64
    })
}

/// Converts a Rust `&str` to a null-terminated wide string for Win32 APIs.
fn to_wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

use std::sync::{Arc, RwLock};

static GLOBAL_SHM: RwLock<Option<Arc<ShmHandle>>> = RwLock::new(None);

/// Registers the active shared memory handle for global tool process tracking.
pub fn register_global_shm(handle: Arc<ShmHandle>) {
    let mut lock = GLOBAL_SHM
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *lock = Some(handle);
}

/// Unregisters the active shared memory handle.
pub fn unregister_global_shm() {
    let mut lock = GLOBAL_SHM
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *lock = None;
}

/// Atomically increments `active_tool_processes` across shared memory with `Release` ordering.
pub fn tool_process_spawned() {
    let lock = GLOBAL_SHM
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(shm) = lock.as_ref() {
        shm.tool_process_spawned();
    }
}

/// Atomically decrements `active_tool_processes` across shared memory with `Release` ordering.
pub fn tool_process_completed() {
    let lock = GLOBAL_SHM
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(shm) = lock.as_ref() {
        shm.tool_process_completed();
    }
}

/// RAII guard that increments `active_tool_processes` on creation and decrements on drop.
pub struct ToolSpawnGuard {
    _private: (),
}

impl ToolSpawnGuard {
    /// Increments the active tool process count with `Release` ordering.
    #[must_use]
    pub fn new() -> Self {
        tool_process_spawned();
        Self { _private: () }
    }
}

impl Default for ToolSpawnGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ToolSpawnGuard {
    fn drop(&mut self) {
        tool_process_completed();
    }
}

// ---------------------------------------------------------------------------
// Stream Lease Registry & RAII Guard
// ---------------------------------------------------------------------------

impl ShmHandle {
    /// Attempts to acquire an exclusive stream lease slot in 0..7.
    ///
    /// Returns `Some(slot_id)` if acquired, or `None` if all 8 slots are busy.
    pub fn acquire_stream_slot(&self) -> Option<u32> {
        let global_hdr = self.header();
        for _ in 0..100 {
            let current_mask = global_hdr.active_stream_mask.load(Ordering::Relaxed);
            let mut free_slot = None;
            for i in 0..8 {
                if (current_mask & (1 << i)) == 0 {
                    free_slot = Some(i);
                    break;
                }
            }
            let slot = free_slot?;
            let new_mask = current_mask | (1 << slot);
            if global_hdr
                .active_stream_mask
                .compare_exchange_weak(current_mask, new_mask, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return Some(slot);
            }
        }
        None
    }

    /// Releases a previously acquired stream slot.
    pub fn release_stream_slot(&self, slot: u32) {
        if slot < 8 {
            let global_hdr = self.header();
            global_hdr
                .active_stream_mask
                .fetch_and(!(1 << slot), Ordering::AcqRel);
            // Submit an explicit MSG_RELEASE_STREAM frame to prompt engine KV page cache eviction
            let _ =
                self.write_request_frame(MessageType::ReleaseStream, FrameFlags::NONE, slot, &[]);
        }
    }

    /// Checks if a stream slot is currently active.
    #[must_use]
    pub fn is_slot_active(&self, slot: u32) -> bool {
        if slot >= 8 {
            return false;
        }
        (self.header().active_stream_mask.load(Ordering::Relaxed) & (1 << slot)) != 0
    }

    /// Submits a pre-tokenized binary token generation request directly into the request queue.
    ///
    /// Copies directly into the shared memory ring buffer without any heap allocations.
    pub fn write_generate_tokens_request(
        &self,
        stream_id: u32,
        tokens: &[u32],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        stop_tokens: &[u32],
    ) -> Result<(), IpcError> {
        let mut req = GenerateTokensRequest {
            stream_id,
            num_tokens: tokens.len() as u32,
            max_tokens,
            temperature,
            top_p,
            stop_token_count: stop_tokens.len().min(16) as u32,
            stop_tokens: [0; 16],
        };
        for (i, &st) in stop_tokens.iter().take(16).enumerate() {
            req.stop_tokens[i] = st;
        }

        let hdr = self.header();
        let payload_size = GENERATE_TOKENS_REQUEST_SIZE + std::mem::size_of_val(tokens);
        let total = FRAME_HEADER_SIZE + payload_size;

        let capacity = hdr.request_queue.capacity_bytes as usize;
        if total > capacity / 2 {
            return Err(IpcError::BadFrame(format!(
                "frame too large: {total} bytes exceeds half the {capacity}-byte request queue"
            )));
        }

        let head = hdr.request_queue.head.load(Ordering::Acquire);
        let seq = hdr.request_queue.total_frames_written;

        let frame = IpcFrameHeader {
            magic: IPC_MAGIC,
            msg_type: MessageType::GenerateTokensRequest as u16,
            flags: FrameFlags::NONE,
            stream_id,
            payload_size: payload_size as u32,
            sequence_num: seq,
            timestamp_ns: monotonic_ns(),
        };

        let rq_base = self.queue_data_ptr(hdr.request_queue_offset as usize);
        let mask = hdr.request_queue.mask as usize;

        // SAFETY: Writing directly into mapped ring buffer under SPSC single-producer rules.
        unsafe {
            let frame_bytes =
                std::slice::from_raw_parts((&raw const frame).cast::<u8>(), FRAME_HEADER_SIZE);
            let req_bytes = std::slice::from_raw_parts(
                (&raw const req).cast::<u8>(),
                GENERATE_TOKENS_REQUEST_SIZE,
            );
            let token_bytes = std::slice::from_raw_parts(
                tokens.as_ptr().cast::<u8>(),
                std::mem::size_of_val(tokens),
            );

            self.ring_write(rq_base, mask, head as usize, frame_bytes);
            self.ring_write(rq_base, mask, head as usize + FRAME_HEADER_SIZE, req_bytes);
            self.ring_write(
                rq_base,
                mask,
                head as usize + FRAME_HEADER_SIZE + GENERATE_TOKENS_REQUEST_SIZE,
                token_bytes,
            );
        }

        hdr.request_queue
            .head
            .store(head + total as u64, Ordering::Release);

        Ok(())
    }
}

/// RAII guard that automatically releases an acquired stream lease slot on drop.
pub struct StreamLeaseGuard<'a> {
    shm: &'a ShmHandle,
    slot: u32,
}

impl<'a> StreamLeaseGuard<'a> {
    /// Acquires a new stream lease from the shared memory header.
    #[must_use]
    pub fn acquire(shm: &'a ShmHandle) -> Option<Self> {
        shm.acquire_stream_slot().map(|slot| Self { shm, slot })
    }

    /// The acquired slot ID (0..7).
    #[must_use]
    pub fn slot(&self) -> u32 {
        self.slot
    }
}

impl Drop for StreamLeaseGuard<'_> {
    fn drop(&mut self) {
        self.shm.release_stream_slot(self.slot);
    }
}
