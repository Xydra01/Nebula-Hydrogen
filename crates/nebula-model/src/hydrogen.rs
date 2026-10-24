//! The Hydrogen inference engine backend, communicating via Win32 named shared memory IPC.
//!
//! [`HydrogenBackend`] implements [`ModelBackend`](crate::backend::ModelBackend) by writing
//! request frames into the daemon→engine ring buffer and reading token/tool-call response
//! frames from the engine→daemon ring buffer. The shared memory protocol is defined in
//! `hydrogen/include/hydrogen_ipc.h` and mirrored in [`nebula_hydrogen`].
//!
//! # Memory ordering
//!
//! The `active_tool_processes` counter in the global shared memory header is incremented
//! with `Release` ordering when a tool process spawns (by the daemon's
//! [`ShmHandle::tool_process_spawned`](nebula_hydrogen::ShmHandle::tool_process_spawned))
//! and loaded with `Acquire` ordering in the engine's dispatch loop. This pairing prevents
//! MSVC `/O2` from reordering the store past subsequent kernel dispatches, ensuring the
//! engine transitions to `cudaDeviceScheduleYield` before the next batch.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use async_trait::async_trait;
use nebula_hydrogen::{
    DEFAULT_DAEMON_EVENT_NAME, DEFAULT_ENGINE_EVENT_NAME, DEFAULT_SHM_NAME, FrameFlags,
    MessageType, ShmHandle,
};
use nebula_proto::{ReasoningEffort, StopReason};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::ModelError;
use crate::backend::{Activity, ModelBackend};
use crate::types::{BackendHealth, ChatRequest, ChatStream, StreamItem, ToolCall};

/// Timeout for health/ping probes (ms).
const HEALTH_TIMEOUT_MS: u32 = 2_000;

type StreamSender = mpsc::Sender<Result<StreamItem, ModelError>>;
type RouterTable = Arc<Mutex<HashMap<u32, StreamSender>>>;

/// The Hydrogen inference engine backend, connected via Win32 named shared memory.
///
/// Unlike [`LlamaServerBackend`](crate::backend::LlamaServerBackend) which communicates over
/// HTTP SSE, this backend uses a zero-copy 64 MiB shared memory ring buffer with Win32
/// event signaling for sub-50 µs token delivery latency.
pub struct HydrogenBackend {
    /// The shared memory handle (owned, maps the 64 MiB section).
    shm: Arc<ShmHandle>,
    /// Activity tracking (in-flight count + cancellation).
    activity: Activity,
    /// Next stream id for request framing.
    next_stream_id: std::sync::atomic::AtomicU32,
    /// Mutex serializing request frame writes to the shared request ring.
    request_lock: Arc<Mutex<()>>,
    /// Routing table mapping active stream IDs to their channel senders.
    router_table: RouterTable,
    /// Cancellation token for stopping the background demuxer thread.
    router_cancel: CancellationToken,
    /// Dedicated background OS thread demuxing response frames.
    router_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Notifier for engine health check (Pong) frames.
    pong_notify: Arc<tokio::sync::Notify>,
}

impl HydrogenBackend {
    /// Opens (or creates) the shared memory IPC section and both Win32 events.
    ///
    /// When `create` is true, the section is created and the header initialized (daemon-side).
    /// When false, an existing section created by the engine is opened.
    ///
    /// # Errors
    /// [`ModelError::Transport`] on Win32 API failure or protocol mismatch.
    pub fn open(create: bool) -> Result<Self, ModelError> {
        let shm_name = std::env::var("NEBULA_HYDROGEN_SHM_NAME")
            .unwrap_or_else(|_| DEFAULT_SHM_NAME.to_string());
        let eng_evt = std::env::var("NEBULA_HYDROGEN_ENGINE_EVENT")
            .unwrap_or_else(|_| DEFAULT_ENGINE_EVENT_NAME.to_string());
        let dae_evt = std::env::var("NEBULA_HYDROGEN_DAEMON_EVENT")
            .unwrap_or_else(|_| DEFAULT_DAEMON_EVENT_NAME.to_string());
        Self::open_named(&shm_name, &eng_evt, &dae_evt, create)
    }

    /// Opens with custom shared memory and event names (useful for testing).
    ///
    /// # Errors
    /// [`ModelError::Transport`] on Win32 API failure or protocol mismatch.
    pub fn open_named(
        shm_name: &str,
        engine_event_name: &str,
        daemon_event_name: &str,
        create: bool,
    ) -> Result<Self, ModelError> {
        let shm = ShmHandle::open(shm_name, engine_event_name, daemon_event_name, create)
            .map_err(|e| ModelError::Transport(format!("hydrogen IPC: {e}")))?;

        let shm = Arc::new(shm);
        nebula_hydrogen::register_global_shm(Arc::clone(&shm));

        let router_table: RouterTable = Arc::new(Mutex::new(HashMap::new()));
        let router_cancel = CancellationToken::new();
        let pong_notify = Arc::new(tokio::sync::Notify::new());

        let shm_for_router = Arc::clone(&shm);
        let table_for_router = Arc::clone(&router_table);
        let cancel_for_router = router_cancel.clone();
        let notify_for_router = Arc::clone(&pong_notify);

        let router_thread = std::thread::Builder::new()
            .name("hydrogen-demuxer".into())
            .spawn(move || {
                Self::demuxer_loop(
                    &shm_for_router,
                    &table_for_router,
                    &cancel_for_router,
                    &notify_for_router,
                );
            })
            .map_err(|e| ModelError::Transport(format!("spawn demuxer thread: {e}")))?;

        Ok(Self {
            shm,
            activity: Activity::new(),
            next_stream_id: std::sync::atomic::AtomicU32::new(1),
            request_lock: Arc::new(Mutex::new(())),
            router_table,
            router_cancel,
            router_thread: Mutex::new(Some(router_thread)),
            pong_notify,
        })
    }

    /// The shared memory handle, for direct access to `active_tool_processes`.
    #[must_use]
    pub fn shm(&self) -> &Arc<ShmHandle> {
        &self.shm
    }

    /// The request-tracking handle shared with the supervisor.
    #[must_use]
    pub fn activity(&self) -> &Activity {
        &self.activity
    }

    /// Allocates a unique stream id for a new request.
    fn alloc_stream_id(&self) -> u32 {
        self.next_stream_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Builds the JSON payload for a `MSG_GENERATE_REQUEST` frame from a [`ChatRequest`].
    fn build_generate_payload(req: &ChatRequest) -> Vec<u8> {
        let effort: u8 = match req.reasoning {
            ReasoningEffort::None => 0,
            ReasoningEffort::Low => 1,
            ReasoningEffort::Medium => 2,
            ReasoningEffort::Xhigh => 3,
        };
        let body = json!({
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "reasoning_effort": effort,
            "tools": req.tools,
            "response_schema": req.response_schema,
        });
        serde_json::to_vec(&body).unwrap_or_default()
    }

    /// Centralized background demuxer loop: continuously reads response frames
    /// from the response ring buffer and dispatches to registered per-stream channels.
    fn demuxer_loop(
        shm: &ShmHandle,
        router_table: &RouterTable,
        cancel: &CancellationToken,
        pong_notify: &tokio::sync::Notify,
    ) {
        while !cancel.is_cancelled() {
            let mut drained_any = false;

            // Drain all available frames from the response ring buffer until empty.
            while let Some(frame) = shm.peek_response_frame() {
                drained_any = true;
                let payload = match shm.read_response_payload(frame.payload_size as usize) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::error!(event = "hydrogen.read_payload_error", error = %e);
                        break;
                    }
                };

                let Some(msg_type) = MessageType::from_u16(frame.msg_type) else {
                    tracing::warn!(
                        event = "hydrogen.unknown_msg",
                        msg_type = frame.msg_type,
                        "skipping unknown message type"
                    );
                    continue;
                };

                if msg_type == MessageType::Pong {
                    pong_notify.notify_one();
                    continue;
                }

                let item_opt = match msg_type {
                    MessageType::TokenChunk => {
                        if let Some(token) = nebula_hydrogen::TokenPayload::from_bytes(&payload) {
                            let text = token.text_str().to_owned();
                            let is_final = token.is_final != 0;
                            Some((
                                if text.is_empty() {
                                    None
                                } else {
                                    Some(StreamItem::Token {
                                        text,
                                        reasoning: false,
                                    })
                                },
                                is_final,
                            ))
                        } else {
                            None
                        }
                    }
                    MessageType::ToolCall => {
                        if let Ok(tc) =
                            std::str::from_utf8(&payload).map_err(|_| ()).and_then(|t| {
                                serde_json::from_str::<serde_json::Value>(t).map_err(|_| ())
                            })
                        {
                            let tool_call = ToolCall {
                                id: tc
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_owned(),
                                name: tc
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_owned(),
                                arguments: tc
                                    .get("arguments")
                                    .and_then(|v| v.as_str().map(str::to_owned))
                                    .or_else(|| tc.get("arguments").map(ToString::to_string))
                                    .unwrap_or_default(),
                            };
                            Some((Some(StreamItem::ToolCall(tool_call)), true))
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                let is_eos = (frame.flags & FrameFlags::END_OF_STREAM) != 0;
                let mut remove = false;

                let mut table = router_table.lock().unwrap_or_else(PoisonError::into_inner);
                if let Some(tx) = table.get(&frame.stream_id) {
                    let mut stream_done = false;
                    if let Some((item_content, item_is_final)) = item_opt {
                        if let Some(item) = item_content {
                            remove = tx.blocking_send(Ok(item)).is_err();
                        }
                        if item_is_final && !remove {
                            let _ = tx.blocking_send(Ok(StreamItem::Done(StopReason::Stop)));
                            stream_done = true;
                            remove = true;
                        }
                    }

                    if is_eos && !stream_done && !remove {
                        let _ = tx.blocking_send(Ok(StreamItem::Done(StopReason::Stop)));
                        remove = true;
                    }
                }

                if remove {
                    table.remove(&frame.stream_id);
                }
            }

            if drained_any {
                continue;
            }

            // Buffer was empty: wait for engine event with 50ms timeout to regularly check cancellation.
            match shm.wait_engine_event(50) {
                Ok(()) => {}
                Err(nebula_hydrogen::IpcError::Timeout { .. }) => continue,
                Err(e) => {
                    tracing::warn!(event = "hydrogen.wait_engine_event_error", error = %e);
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
            }
        }
    }
}

impl std::fmt::Debug for HydrogenBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HydrogenBackend")
            .field("shm_size", &self.shm.size())
            .field("active_tool_processes", &self.shm.active_tool_processes())
            .finish()
    }
}

impl Drop for HydrogenBackend {
    fn drop(&mut self) {
        self.router_cancel.cancel();
        let _ = self.shm.signal_engine_event();
        if let Some(handle) = self
            .router_thread
            .lock()
            .ok()
            .and_then(|mut lock| lock.take())
        {
            let _ = handle.join();
        }
        nebula_hydrogen::unregister_global_shm();
    }
}

#[async_trait]
impl ModelBackend for HydrogenBackend {
    fn activity(&self) -> &Activity {
        &self.activity
    }

    #[tracing::instrument(level = "debug", skip_all)]
    async fn chat(&self, req: ChatRequest) -> Result<ChatStream, ModelError> {
        if self.activity.is_cancelled() {
            return Err(ModelError::Unavailable(
                "hydrogen backend is stopping".into(),
            ));
        }

        let stream_id = self.alloc_stream_id();
        let payload = Self::build_generate_payload(&req);

        let (tx, rx) = mpsc::channel(1024);

        // Register stream in router table before queueing request.
        {
            let mut table = self
                .router_table
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            table.insert(stream_id, tx);
        }

        // Write request frame under request_lock serialization.
        {
            let _lock = self
                .request_lock
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            self.shm
                .write_request_frame(
                    MessageType::GenerateRequest,
                    FrameFlags::NONE,
                    stream_id,
                    &payload,
                )
                .map_err(|e| {
                    let mut table = self
                        .router_table
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner);
                    table.remove(&stream_id);
                    ModelError::Transport(format!("write request: {e}"))
                })?;

            self.shm.signal_daemon_event().map_err(|e| {
                let mut table = self
                    .router_table
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                table.remove(&stream_id);
                ModelError::Transport(format!("signal: {e}"))
            })?;
        }

        let _guard = self.activity.enter();
        let router_table = Arc::clone(&self.router_table);
        let cancel = self.activity.cancel_token();
        tokio::spawn(async move {
            let _g = _guard;
            cancel.cancelled().await;
            let sender = {
                let mut table = router_table.lock().unwrap_or_else(PoisonError::into_inner);
                table.remove(&stream_id)
            };
            if let Some(tx) = sender {
                let _ = tx.send(Ok(StreamItem::Done(StopReason::Cancelled))).await;
            }
        });

        Ok(ChatStream::new(rx))
    }

    async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, ModelError> {
        // Hydrogen does not support embeddings (those use the CPU embedding server).
        Err(ModelError::Unavailable(
            "hydrogen backend does not support embeddings".into(),
        ))
    }

    async fn tokenize(&self, _text: &str) -> Result<Vec<u32>, ModelError> {
        // Tokenization is handled by the engine internally; not exposed over IPC yet.
        Err(ModelError::Unavailable(
            "hydrogen backend does not support tokenize over IPC".into(),
        ))
    }

    #[tracing::instrument(level = "debug", skip_all)]
    async fn health(&self) -> Result<BackendHealth, ModelError> {
        let notified = self.pong_notify.notified();

        {
            let _lock = self
                .request_lock
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            self.shm
                .write_request_frame(MessageType::Ping, FrameFlags::NONE, 0, &[])
                .map_err(|e| ModelError::Transport(format!("write ping: {e}")))?;
            self.shm
                .signal_daemon_event()
                .map_err(|e| ModelError::Transport(format!("signal: {e}")))?;
        }

        match tokio::time::timeout(
            tokio::time::Duration::from_millis(u64::from(HEALTH_TIMEOUT_MS)),
            notified,
        )
        .await
        {
            Ok(()) => Ok(BackendHealth::Ok),
            Err(_) => Err(ModelError::Transport(format!(
                "hydrogen ping: wait timed out after {HEALTH_TIMEOUT_MS} ms"
            ))),
        }
    }
}
