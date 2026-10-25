//! Safe Rust FFI bindings to the Hydrogen inference engine's Win32 shared memory IPC.
//!
//! This crate mirrors the C++ protocol defined in `hydrogen/include/hydrogen_ipc.h`, exposing
//! safe, strongly-typed Rust equivalents of the shared memory header, frame layouts, and ring
//! buffer control blocks. The Win32 shared memory handle ([`ShmHandle`]) opens or creates the
//! named section, maps the view, validates the magic/version header, and provides safe accessors
//! for writing request frames, reading response frames, and manipulating the
//! `active_tool_processes` atomic counter with correct memory ordering.
//!
//! All `unsafe` FFI code is confined to documented `// SAFETY:` blocks in [`ShmHandle`].

#[cfg(windows)]
mod ipc;

#[cfg(windows)]
pub use ipc::{
    FRAME_HEADER_SIZE, FrameFlags, GENERATE_TOKENS_REQUEST_SIZE, GenerateTokensRequest, IpcError,
    IpcFrameHeader, MessageType, ShmGlobalHeader, ShmHandle, ShmQueueHeader, StreamLeaseGuard,
    TokenPayload, ToolSpawnGuard, register_global_shm, tool_process_completed,
    tool_process_spawned, unregister_global_shm,
};

/// IPC protocol magic: `"HYDR"` (0x4859_4452).
pub const IPC_MAGIC: u32 = 0x4859_4452;

/// IPC protocol version.
pub const IPC_VERSION: u32 = 1;

/// Total shared memory backing size: 64 MiB.
pub const SHM_TOTAL_CAPACITY_BYTES: usize = 64 * 1024 * 1024;

/// Daemon → Engine request queue: 16 MiB.
pub const REQUEST_QUEUE_CAPACITY_BYTES: usize = 16 * 1024 * 1024;

/// Engine → Daemon response queue: 48 MiB.
pub const RESPONSE_QUEUE_CAPACITY_BYTES: usize = 48 * 1024 * 1024;

/// Default shared memory object name.
pub const DEFAULT_SHM_NAME: &str = r"Local\Nebula_Hydrogen_IPC";

/// Engine event name (signaled when response/token is ready).
pub const DEFAULT_ENGINE_EVENT_NAME: &str = r"Local\Nebula_Hydrogen_IPC_EngineEvent";

/// Daemon event name (signaled when request/command is submitted).
pub const DEFAULT_DAEMON_EVENT_NAME: &str = r"Local\Nebula_Hydrogen_IPC_DaemonEvent";
