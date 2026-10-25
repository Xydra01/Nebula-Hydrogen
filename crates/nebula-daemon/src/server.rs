//! The named-pipe server: one task per connection, NDJSON JSON-RPC in and out.

use std::sync::{Arc, PoisonError};
use std::time::Duration;

use futures_util::StreamExt;
use nebula_model::{ChatRequest, ModelError, StreamItem};
use nebula_proto::{
    ChatCancelParams, ChatDone, ChatError, ChatId, ChatStartParams, ChatStarted, ChatToken,
    DaemonStatus, DoctorReport, Empty, Event, LogEvent, LogsSubscribeParams, Message, Method,
    ModelSetProfileParams, ModelState, Notification, PROTO_VERSION, ReasoningEffort, Request,
    RequestId, Response, RpcError, TaskCancelParams, TaskCreateParams, TaskId, TaskStarted,
    TaskStopAllParams, TaskStopAllResponse, ToolCallOutcome, ToolInfo, ToolList, ToolsCallParams,
    TraceId, Usage, error_code,
};

use nebula_tools::ToolError;
use serde::Serialize;
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::net::windows::named_pipe::NamedPipeServer;
use tokio::sync::{broadcast, mpsc};
use tokio_util::codec::{FramedRead, LinesCodec, LinesCodecError};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::checks;
use crate::executor;
use crate::win::PipeSecurity;
use crate::{RunningTaskState, Shared};

/// Longest accepted request line.
const MAX_LINE: usize = 16 * 1024 * 1024;
/// Generation cap when `chat.start` doesn't set one.
const DEFAULT_MAX_TOKENS: u32 = 4096;
/// Outgoing messages buffered per connection.
const OUTBOX: usize = 1024;

type Outbox = mpsc::Sender<Message>;

/// The pipe and its next unconnected instance.
pub(crate) struct Listener {
    security: PipeSecurity,
    path: String,
    next: NamedPipeServer,
}

impl Listener {
    /// Creates the first instance; fails if another process owns the pipe name.
    pub(crate) fn bind(path: &str) -> std::io::Result<Self> {
        let security = PipeSecurity::current_user_only()?;
        let next = security.create(path, true)?;
        Ok(Self {
            security,
            path: path.to_owned(),
            next,
        })
    }

    async fn accept(&mut self) -> std::io::Result<NamedPipeServer> {
        if let Err(e) = self.next.connect().await {
            self.next = self.security.create(&self.path, false)?;
            return Err(e);
        }
        let fresh = self.security.create(&self.path, false)?;
        Ok(std::mem::replace(&mut self.next, fresh))
    }
}

pub(crate) async fn accept_loop(shared: Arc<Shared>, mut listener: Listener) {
    loop {
        tokio::select! {
            () = shared.shutdown.cancelled() => break,
            r = listener.accept() => match r {
                Ok(pipe) => {
                    shared.tasks.spawn(connection(Arc::clone(&shared), pipe));
                }
                Err(e) => {
                    tracing::warn!(event = "daemon.accept_failed", error = %e);
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            },
        }
    }
}

fn note(event: Event) -> Message {
    Message::Notification(Notification::new(event))
}

fn reply<T: Serialize>(id: RequestId, result: Result<T, RpcError>) -> Message {
    let resp = match result {
        Ok(v) => Response::ok(id.clone(), &v).unwrap_or_else(|e| {
            Response::err(id, RpcError::new(error_code::INTERNAL_ERROR, e.to_string()))
        }),
        Err(e) => Response::err(id, e),
    };
    Message::Response(resp)
}

/// The `id` of a line that failed to decode, if it has a usable one.
fn raw_id(line: &str) -> RequestId {
    let v: Value = serde_json::from_str(line).unwrap_or(Value::Null);
    match &v["id"] {
        Value::Number(n) => RequestId::Num(n.as_u64().unwrap_or(0)),
        Value::String(s) => RequestId::Str(s.clone()),
        _ => RequestId::Num(0),
    }
}

async fn connection(shared: Arc<Shared>, pipe: NamedPipeServer) {
    let (rd, mut wr) = tokio::io::split(pipe);
    let conn = shared.shutdown.child_token();
    let (tx, mut rx) = mpsc::channel::<Message>(OUTBOX);
    let writer = {
        let conn = conn.clone();
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                let Ok(line) = msg.encode() else { continue };
                if wr.write_all(line.as_bytes()).await.is_err() {
                    conn.cancel();
                    break;
                }
            }
            let _ = wr.flush().await;
        })
    };
    forward_model_events(shared.chat.subscribe(), tx.clone(), conn.clone());
    if let Some(e) = &shared.embedding {
        forward_model_events(e.subscribe(), tx.clone(), conn.clone());
    }
    forward_game_mode_events(shared.game_mode.subscribe(), tx.clone(), conn.clone());
    forward_task_events(shared.task_events.subscribe(), tx.clone(), conn.clone());

    let mut lines = FramedRead::new(rd, LinesCodec::new_with_max_length(MAX_LINE));
    loop {
        let next = tokio::select! {
            () = conn.cancelled() => break,
            next = lines.next() => next,
        };
        match next {
            None | Some(Err(LinesCodecError::Io(_))) => break,
            Some(Err(LinesCodecError::MaxLineLengthExceeded)) => {
                let err = RpcError::new(error_code::INVALID_REQUEST, "request line too long");
                let _ = tx.send(reply::<Empty>(RequestId::Num(0), Err(err))).await;
                break;
            }
            Some(Ok(line)) if line.trim().is_empty() => {}
            Some(Ok(line)) => match Message::decode(&line) {
                Ok(Message::Request(req)) => {
                    let (s, tx, conn) = (Arc::clone(&shared), tx.clone(), conn.clone());
                    shared.tasks.spawn(handle(s, req, tx, conn));
                }
                Ok(_) => {
                    let err = RpcError::new(
                        error_code::INVALID_REQUEST,
                        "the daemon only accepts requests",
                    );
                    let _ = tx.send(reply::<Empty>(raw_id(&line), Err(err))).await;
                }
                Err(e) => {
                    let _ = tx
                        .send(reply::<Empty>(raw_id(&line), Err(e.to_rpc_error())))
                        .await;
                }
            },
        }
    }
    conn.cancel();
    drop(tx);
    let _ = writer.await;
}

fn forward_model_events(
    mut rx: broadcast::Receiver<nebula_proto::ModelStateChanged>,
    tx: Outbox,
    conn: CancellationToken,
) {
    tokio::spawn(async move {
        loop {
            let ev = tokio::select! {
                () = conn.cancelled() => break,
                ev = rx.recv() => ev,
            };
            match ev {
                Ok(ev) => {
                    if tx.send(note(Event::ModelStateChanged(ev))).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });
}

fn forward_game_mode_events(
    mut rx: broadcast::Receiver<nebula_proto::GameModeChanged>,
    tx: Outbox,
    conn: CancellationToken,
) {
    tokio::spawn(async move {
        loop {
            let ev = tokio::select! {
                () = conn.cancelled() => break,
                ev = rx.recv() => ev,
            };
            match ev {
                Ok(ev) => {
                    if tx.send(note(Event::GameModeChanged(ev))).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });
}

fn forward_task_events(
    mut rx: broadcast::Receiver<nebula_proto::Event>,
    tx: Outbox,
    conn: CancellationToken,
) {
    tokio::spawn(async move {
        loop {
            let ev = tokio::select! {
                () = conn.cancelled() => break,
                ev = rx.recv() => ev,
            };
            match ev {
                Ok(ev) => {
                    if tx.send(note(ev)).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });
}

fn model_err(e: &ModelError) -> RpcError {
    let code = match e {
        ModelError::UnknownProfile(_) => error_code::NOT_FOUND,
        ModelError::Unavailable(_)
        | ModelError::Failed(_)
        | ModelError::Preflight(_)
        | ModelError::Launch(_) => error_code::MODEL_UNAVAILABLE,
        _ => error_code::INTERNAL_ERROR,
    };
    RpcError::new(code, e.to_string())
}

fn cancelled() -> RpcError {
    RpcError::new(error_code::CANCELLED, "chat cancelled")
}

fn to_value<T: Serialize>(v: &T) -> Result<Value, RpcError> {
    serde_json::to_value(v).map_err(|e| RpcError::new(error_code::INTERNAL_ERROR, e.to_string()))
}

async fn handle(shared: Arc<Shared>, req: Request, tx: Outbox, conn: CancellationToken) {
    let id = req.id.clone();
    if shared.shutdown.is_cancelled() {
        let err = RpcError::new(error_code::SHUTTING_DOWN, "the daemon is shutting down");
        let _ = tx.send(reply::<Empty>(id, Err(err))).await;
        return;
    }
    let result: Result<Value, RpcError> = match req.call {
        Method::DaemonStatus(Empty {}) => to_value(&DaemonStatus {
            version: env!("CARGO_PKG_VERSION").to_owned(),
            proto_version: PROTO_VERSION,
            pid: std::process::id(),
            uptime_s: shared.started.elapsed().as_secs(),
            model: shared.chat.status(),
        }),
        Method::DaemonShutdown(Empty {}) => {
            let _ = tx.send(reply(id, Ok(Empty {}))).await;
            shared.shutdown.cancel();
            return;
        }
        Method::ChatStart(p) => {
            start_chat(&shared, id, p, req.trace_id, tx, &conn).await;
            return;
        }
        Method::ChatCancel(ChatCancelParams { chat_id }) => {
            let token = shared
                .chats
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(&chat_id)
                .cloned();
            match token {
                Some(t) => {
                    t.cancel();
                    to_value(&Empty {})
                }
                None => Err(RpcError::new(
                    error_code::NOT_FOUND,
                    format!("no running chat {chat_id}"),
                )),
            }
        }
        Method::TaskCreate(p) => {
            start_task(&shared, id, p, req.trace_id, tx, &conn).await;
            return;
        }
        Method::TaskCancel(TaskCancelParams { task_id }) => {
            let (token, job) = {
                let mut running = shared
                    .executor_tasks
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                if let Some(s) = running.remove(&task_id) {
                    (Some(s.cancel), s.task_job)
                } else {
                    (None, None)
                }
            };
            match token {
                Some(t) => {
                    t.cancel();
                    if let Some(j) = job {
                        let _ = j.terminate();
                    }
                    shared
                        .finished_tasks
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .insert(task_id);
                    to_value(&Empty {})
                }
                None => {
                    let finished = shared
                        .finished_tasks
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner);
                    if finished.contains(&task_id) {
                        to_value(&Empty {})
                    } else {
                        Err(RpcError::new(
                            error_code::NOT_FOUND,
                            format!("no task {task_id}"),
                        ))
                    }
                }
            }
        }
        Method::TaskStopAll(TaskStopAllParams { reason }) => {
            let stopped = {
                let mut running = shared
                    .executor_tasks
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                let count = u32::try_from(running.len()).unwrap_or(u32::MAX);
                let mut finished = shared
                    .finished_tasks
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                for (task_id, state) in running.drain() {
                    state.cancel.cancel();
                    if let Some(j) = state.task_job {
                        let _ = j.terminate();
                    }
                    finished.insert(task_id);
                }
                count
            };
            tracing::warn!(event = "task.stop_all", stopped = stopped, reason = ?reason);
            to_value(&TaskStopAllResponse { stopped })
        }

        Method::GameModeSet(nebula_proto::GameModeSetParams { enabled }) => {
            match shared.game_mode.set_manual(&shared.chat, enabled).await {
                Ok(res) => to_value(&res),
                Err(e) => Err(RpcError::new(error_code::INTERNAL_ERROR, e)),
            }
        }
        Method::GameModeStatus(Empty {}) => to_value(&shared.game_mode.status()),

        Method::NotifyStatus(Empty {}) => {
            let cfg = shared.notifications.config();
            to_value(&nebula_proto::NotifyStatusResponse {
                enabled: cfg.enabled,
                server_url: cfg.server_url.clone(),
                topic: cfg.topic.clone(),
                tailscale_only: cfg.tailscale_only,
            })
        }
        Method::NotifyTest(nebula_proto::NotifyTestParams { message }) => {
            match shared.notifications.notify_test(message.as_deref()).await {
                Ok(()) => to_value(&nebula_proto::NotifyTestResponse {
                    success: true,
                    message: format!(
                        "Notification sent to topic '{}'",
                        shared.notifications.config().topic
                    ),
                }),
                Err(e) => to_value(&nebula_proto::NotifyTestResponse {
                    success: false,
                    message: e.to_string(),
                }),
            }
        }
        Method::ApprovalRespond(p) => {
            let trace_str = p.trace_id.clone();
            let decision = if p.approved { "approved" } else { "denied" };
            tracing::info!(
                event = "permit.decision",
                decision = decision,
                command = %p.command,
                tier = %p.tier,
                trace_id = %trace_str,
                reason = ?p.reason,
            );
            let _ = shared.approval_responses.send(p.clone());
            if p.approved {
                let tier = match p.tier.to_lowercase().as_str() {
                    "read" => nebula_sandbox::tier::Tier::Read,
                    "sandbox" => nebula_sandbox::tier::Tier::Sandbox,
                    "workspace" => nebula_sandbox::tier::Tier::Workspace,
                    _ => nebula_sandbox::tier::Tier::System,
                };
                let grant_id = nebula_sandbox::approval::GrantId::new(ulid::Ulid::generate().0);
                let normalized_cmd = nebula_sandbox::rules::normalize_command(&p.command);
                let scope = nebula_sandbox::approval::ApprovalScope {
                    command: normalized_cmd.clone(),
                    tier,
                    trace_id: trace_str,
                };
                let approval = nebula_sandbox::approval::Approval::grant(grant_id, scope);
                shared.approvals.insert(approval);

                let grant_id_task =
                    nebula_sandbox::approval::GrantId::new(ulid::Ulid::generate().0);
                let scope_task = nebula_sandbox::approval::ApprovalScope {
                    command: normalized_cmd,
                    tier,
                    trace_id: "task".into(),
                };
                let approval_task =
                    nebula_sandbox::approval::Approval::grant(grant_id_task, scope_task);
                shared.approvals.insert(approval_task);
                to_value(&nebula_proto::ApprovalRespondResponse {
                    success: true,
                    message: format!("Granted approval for command '{}'", p.command),
                })
            } else {
                to_value(&nebula_proto::ApprovalRespondResponse {
                    success: true,
                    message: format!("Denied approval for command '{}'", p.command),
                })
            }
        }

        Method::ModelStatus(Empty {}) => to_value(&shared.chat.status()),
        Method::ModelSetProfile(ModelSetProfileParams { profile }) => {
            match check_chat_profile(&shared, &profile) {
                Err(e) => Err(e),
                Ok(()) => {
                    let _guard = shared.switch.lock().await;
                    shared
                        .chat
                        .set_profile(&profile)
                        .await
                        .map_err(|e| model_err(&e))
                        .and_then(|s| to_value(&s))
                }
            }
        }
        Method::ResourcesSnapshot(Empty {}) => shared.latest_snapshot().map_or_else(
            || {
                Err(RpcError::new(
                    error_code::INTERNAL_ERROR,
                    "no resource snapshot yet",
                ))
            },
            |s| to_value(&s),
        ),
        Method::LogsSubscribe(p) => {
            subscribe_logs(&shared, p, tx.clone(), conn);
            to_value(&Empty {})
        }
        Method::DoctorRun(Empty {}) => run_doctor(&shared).await.and_then(|r| to_value(&r)),
        Method::ToolsList(Empty {}) => {
            let tools = shared
                .tool_host
                .list_tools()
                .into_iter()
                .map(|t| ToolInfo {
                    server: t.server,
                    name: t.name,
                    description: t.description,
                    input_schema: t.input_schema,
                })
                .collect();
            to_value(&ToolList { tools })
        }
        Method::ToolsCall(ToolsCallParams {
            tool,
            arguments,
            trace_id,
        }) => {
            let trace = trace_id.or(req.trace_id).unwrap_or_default();
            let span = tracing::info_span!("tools.call", trace_id = %trace, tool = %tool);
            shared
                .tool_host
                .call(&tool, arguments, Some(&trace.to_string()))
                .instrument(span)
                .await
                .map_err(|e| tool_err(&e))
                .and_then(|r| {
                    to_value(&ToolCallOutcome {
                        content: r.content,
                        is_error: r.is_error,
                    })
                })
        }
        Method::LanguageList(Empty {}) => {
            let languages = shared.languages.list();
            to_value(&nebula_proto::LanguageListResponse { languages })
        }
        Method::LanguageInfo(nebula_proto::LanguageInfoParams { language }) => {
            match shared.languages.get(&language) {
                Some(module) => {
                    let metadata = module.metadata();
                    let tools = module
                        .tools()
                        .into_iter()
                        .map(|t| ToolInfo {
                            server: format!("language_{}", module.id()),
                            name: t.name().to_string(),
                            description: t.description().map(ToOwned::to_owned),
                            input_schema: t.input_schema(),
                        })
                        .collect();
                    to_value(&nebula_proto::LanguageInfoResponse { metadata, tools })
                }
                None => Err(RpcError::new(
                    error_code::NOT_FOUND,
                    format!("unknown language module: {language}"),
                )),
            }
        }
    };
    let _ = tx.send(reply(id, result)).await;
}

fn tool_err(e: &ToolError) -> RpcError {
    let code = match e {
        ToolError::UnknownServer(_) | ToolError::UnknownTool(_) => error_code::NOT_FOUND,
        ToolError::InvalidArguments { .. } | ToolError::OutputTooLarge { .. } => {
            error_code::INVALID_PARAMS
        }
        ToolError::Timeout { .. } => error_code::CANCELLED,
        ToolError::Launch { .. } | ToolError::Unavailable { .. } => error_code::MODEL_UNAVAILABLE,
        ToolError::Protocol { .. } | ToolError::DuplicateTool { .. } => error_code::INTERNAL_ERROR,
    };
    RpcError::new(code, e.to_string())
}

fn check_chat_profile(shared: &Shared, name: &str) -> Result<(), RpcError> {
    match shared.config.model.profiles.get(name) {
        None => Err(RpcError::new(
            error_code::NOT_FOUND,
            format!("unknown model profile {name:?}"),
        )),
        Some(p) if p.embedding => Err(RpcError::new(
            error_code::INVALID_PARAMS,
            format!("{name} is an embedding profile, not a chat profile"),
        )),
        Some(_) => Ok(()),
    }
}

async fn start_chat(
    shared: &Arc<Shared>,
    id: RequestId,
    params: ChatStartParams,
    trace: Option<TraceId>,
    tx: Outbox,
    conn: &CancellationToken,
) {
    if let Some(Err(e)) = params
        .profile
        .as_deref()
        .map(|p| check_chat_profile(shared, p))
    {
        let _ = tx.send(reply::<Empty>(id, Err(e))).await;
        return;
    }
    let chat_id = ChatId::new();
    let trace_id = trace.unwrap_or_default();
    let token = conn.child_token();
    shared
        .chats
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(chat_id, token.clone());
    // The reply goes out before the task starts, so it precedes every chat event.
    let _ = tx
        .send(reply(id, Ok(ChatStarted { chat_id, trace_id })))
        .await;
    let span = tracing::info_span!("chat", trace_id = %trace_id, chat_id = %chat_id);
    let s = Arc::clone(shared);
    shared.tasks.spawn(
        async move {
            if let Err(e) = run_chat(&s, chat_id, params, trace_id, &tx, &token).await {
                tracing::info!(event = "chat.failed", code = e.code, message = %e.message);
                let _ = tx
                    .send(note(Event::ChatError(ChatError {
                        chat_id,
                        code: e.code,
                        message: e.message,
                    })))
                    .await;
            }
            s.chats
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&chat_id);
        }
        .instrument(span),
    );
}

async fn start_task(
    shared: &Arc<Shared>,
    id: RequestId,
    params: TaskCreateParams,
    trace: Option<TraceId>,
    tx: Outbox,
    conn: &CancellationToken,
) {
    let profile = params.profile.as_deref().or_else(|| {
        let p = shared.config.executor.default_profile.as_str();
        if p.is_empty() { None } else { Some(p) }
    });
    if let Some(Err(e)) = profile.map(|p| check_chat_profile(shared, p)) {
        let _ = tx.send(reply::<Empty>(id, Err(e))).await;
        return;
    }

    let task_id = TaskId::new();
    let trace_id = trace.unwrap_or_default();
    let wt_task_id = nebula_sandbox::worktree::TaskId::new(task_id.to_string());

    // Create the per-task worktree BEFORE TaskStarted (Requirement 1.1, 1.3)
    let worktree_root = match shared
        .worktrees
        .create(&wt_task_id, &params.repo, params.start_point.as_deref())
        .await
    {
        Ok(root) => root,
        Err(e) => {
            let _ = tx.send(reply::<Empty>(id, Err(worktree_err(&e)))).await;
            return;
        }
    };

    let token = conn.child_token();
    shared
        .executor_tasks
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(
            task_id,
            RunningTaskState {
                cancel: token.clone(),
                task_job: None,
            },
        );

    // Reply with TaskStarted before any task.* event is emitted
    let _ = tx
        .send(reply(id, Ok(TaskStarted { task_id, trace_id })))
        .await;

    let span = tracing::info_span!("task", trace_id = %trace_id, task_id = %task_id);
    let s = Arc::clone(shared);
    shared.tasks.spawn(
        async move {
            executor::run_task(
                Arc::clone(&s),
                task_id,
                trace_id,
                params,
                worktree_root,
                token,
            )
            .await;

            s.executor_tasks
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&task_id);
            s.finished_tasks
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(task_id);
        }
        .instrument(span),
    );
}

fn worktree_err(e: &nebula_sandbox::worktree::WorktreeError) -> RpcError {
    use nebula_sandbox::worktree::WorktreeError;
    let code = match e {
        WorktreeError::NotARepository(_)
        | WorktreeError::UnresolvableRef(_)
        | WorktreeError::BranchExists(_)
        | WorktreeError::RetiredDrive(_)
        | WorktreeError::TaskAlreadyActive(_) => error_code::INVALID_PARAMS,
        WorktreeError::UnknownTask(_) => error_code::NOT_FOUND,
        _ => error_code::INTERNAL_ERROR,
    };
    RpcError::new(code, e.to_string())
}

pub(crate) async fn ensure_profile(
    shared: &Shared,
    want: Option<&str>,
    token: &CancellationToken,
) -> Result<(), RpcError> {
    let status = shared.chat.status();
    let target = want.unwrap_or(&status.profile).to_owned();
    if status.profile == target && matches!(status.state, ModelState::Ready | ModelState::Busy) {
        return Ok(());
    }
    let _guard = tokio::select! {
        () = token.cancelled() => return Err(cancelled()),
        g = shared.switch.lock() => g,
    };
    tokio::select! {
        () = token.cancelled() => Err(cancelled()),
        r = shared.chat.set_profile(&target) => r.map(|_| ()).map_err(|e| model_err(&e)),
    }
}

async fn run_chat(
    shared: &Shared,
    chat_id: ChatId,
    params: ChatStartParams,
    trace_id: TraceId,
    tx: &Outbox,
    token: &CancellationToken,
) -> Result<(), RpcError> {
    ensure_profile(shared, params.profile.as_deref(), token).await?;
    let backend = shared.chat.backend().map_err(|e| model_err(&e))?;
    let mut req = ChatRequest::new(params.messages);
    req.response_schema = params.response_schema;
    req.max_tokens = params.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    req.reasoning = params.reasoning.unwrap_or(ReasoningEffort::None);
    req.trace_id = Some(trace_id);
    let mut stream = tokio::select! {
        () = token.cancelled() => return Err(cancelled()),
        r = backend.chat(req) => r.map_err(|e| model_err(&e))?,
    };
    let mut usage = Usage::default();
    loop {
        let item = tokio::select! {
            () = token.cancelled() => return Err(cancelled()),
            item = stream.next() => item,
        };
        let msg = match item {
            None => {
                return Err(RpcError::new(
                    error_code::INTERNAL_ERROR,
                    "stream ended without a finish reason",
                ));
            }
            Some(Err(e)) => return Err(model_err(&e)),
            Some(Ok(StreamItem::Token { text, reasoning })) => {
                let (text, reasoning) = if reasoning {
                    (String::new(), text)
                } else {
                    (text, String::new())
                };
                note(Event::ChatToken(ChatToken {
                    chat_id,
                    text,
                    reasoning,
                    streams: None,
                }))
            }
            Some(Ok(StreamItem::ToolCall(_))) => continue,
            Some(Ok(StreamItem::Usage(u))) => {
                usage = u;
                continue;
            }
            Some(Ok(StreamItem::Done(stop_reason))) => {
                let _ = tx
                    .send(note(Event::ChatDone(ChatDone {
                        chat_id,
                        stop_reason,
                        usage,
                    })))
                    .await;
                return Ok(());
            }
        };
        if tx.send(msg).await.is_err() {
            return Err(cancelled());
        }
    }
}

fn wants(ev: &LogEvent, p: &LogsSubscribeParams) -> bool {
    p.min_level.is_none_or(|l| ev.level >= l)
        && p.trace_id.is_none_or(|t| ev.trace_id == Some(t))
        && p.target_prefix
            .as_deref()
            .is_none_or(|pre| ev.target.starts_with(pre))
}

fn subscribe_logs(shared: &Shared, p: LogsSubscribeParams, tx: Outbox, conn: CancellationToken) {
    let mut rx = shared.telemetry.subscribe();
    tokio::spawn(async move {
        loop {
            let ev = tokio::select! {
                () = conn.cancelled() => break,
                ev = rx.recv() => ev,
            };
            match ev {
                Ok(ev) if wants(&ev, &p) => {
                    if tx.send(note(Event::LogEvent(ev))).await.is_err() {
                        break;
                    }
                }
                Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });
}

pub(crate) async fn run_doctor(shared: &Arc<Shared>) -> Result<DoctorReport, RpcError> {
    let mut all = vec![checks::model_check("model.chat", &shared.chat.status())];
    if let Some(e) = &shared.embedding {
        all.push(checks::model_check("model.embedding", &e.status()));
    }
    all.push(shared.notifications.check_health().await);
    let s = Arc::clone(shared);
    let rest = tokio::task::spawn_blocking(move || {
        let mut out = (s.local_checks)(&s.config).checks;
        if s.artifact_checks {
            out.extend(checks::runtime_checks(&s.config));
            let mut profiles = vec![
                s.chat.status().profile,
                s.config.daemon.load_on_start.clone(),
            ];
            if let Some(e) = &s.embedding {
                profiles.push(e.status().profile);
            }
            profiles.retain(|p| !p.is_empty());
            profiles.dedup();
            let names: Vec<&str> = profiles.iter().map(String::as_str).collect();
            let cache = s.config.paths.state.join("model-hashes.json");
            out.extend(checks::model_hash_checks(&s.config, &names, &cache));
        }
        out
    })
    .await
    .map_err(|e| RpcError::new(error_code::INTERNAL_ERROR, e.to_string()))?;
    all.extend(rest);
    Ok(DoctorReport::from_checks(all))
}
