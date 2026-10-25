//! Client-to-daemon methods (the `method` and `params` of a [`crate::Request`]).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ids::{ChatId, TaskId, TraceId};
use crate::types::{ChatMessage, LogLevel, ModelStatus, ReasoningEffort};

/// Parameters for methods that take none. Serialized as `{}`; JSON-RPC callers must still send
/// `"params": {}`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Empty {}

/// Every Phase 0 method.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum Method {
    /// Daemon version, uptime and model state. Result: [`DaemonStatus`].
    #[serde(rename = "daemon.status")]
    DaemonStatus(Empty),
    /// Graceful shutdown: stop accepting clients, cancel chats, stop the model, flush logs.
    /// Result: [`Empty`].
    #[serde(rename = "daemon.shutdown")]
    DaemonShutdown(Empty),
    /// Start a streaming chat. Result: [`ChatStarted`]; tokens follow as `chat.token` events.
    #[serde(rename = "chat.start")]
    ChatStart(ChatStartParams),
    /// Cancel a running chat. Result: [`Empty`].
    #[serde(rename = "chat.cancel")]
    ChatCancel(ChatCancelParams),
    /// Result: [`ModelStatus`].
    #[serde(rename = "model.status")]
    ModelStatus(Empty),
    /// Switch the model profile (drains in-flight requests first). Result: [`ModelStatus`].
    #[serde(rename = "model.set_profile")]
    ModelSetProfile(ModelSetProfileParams),
    /// Result: [`crate::ResourceSnapshot`].
    #[serde(rename = "resources.snapshot")]
    ResourcesSnapshot(Empty),
    /// Stream log events to this connection as `log.event`. Result: [`Empty`].
    #[serde(rename = "logs.subscribe")]
    LogsSubscribe(LogsSubscribeParams),
    /// Result: [`crate::DoctorReport`].
    #[serde(rename = "doctor.run")]
    DoctorRun(Empty),
    /// List every tool across all configured MCP servers. Result: [`ToolList`].
    #[serde(rename = "tools.list")]
    ToolsList(Empty),
    /// Call a tool by name. Result: [`ToolCallOutcome`].
    #[serde(rename = "tools.call")]
    ToolsCall(ToolsCallParams),
    /// Start an autonomous executor task. Result: [`TaskStarted`]; progress follows as `task.*` events.
    #[serde(rename = "task.create")]
    TaskCreate(TaskCreateParams),
    /// Cancel a running executor task. Result: [`Empty`].
    #[serde(rename = "task.cancel")]
    TaskCancel(TaskCancelParams),
    /// Cancel all running executor tasks immediately. Result: [`TaskStopAllResponse`].
    #[serde(rename = "task.stop_all")]
    TaskStopAll(TaskStopAllParams),
    /// Manually enable or disable game mode. Result: [`GameModeStatusResponse`].
    #[serde(rename = "game_mode.set")]
    GameModeSet(GameModeSetParams),
    /// Query current game mode state. Result: [`GameModeStatusResponse`].
    #[serde(rename = "game_mode.status")]
    GameModeStatus(Empty),
    /// Query push notification status. Result: [`NotifyStatusResponse`].
    #[serde(rename = "notify.status")]
    NotifyStatus(Empty),
    /// Send a test notification. Result: [`NotifyTestResponse`].
    #[serde(rename = "notify.test")]
    NotifyTest(NotifyTestParams),
    /// Respond to an approval request (approve or deny). Result: [`ApprovalRespondResponse`].
    #[serde(rename = "approval.respond")]
    ApprovalRespond(ApprovalRespondParams),
    /// List all registered language modules. Result: [`LanguageListResponse`].
    #[serde(rename = "language.list")]
    LanguageList(Empty),
    /// Get metadata and tools for a specific language module. Result: [`LanguageInfoResponse`].
    #[serde(rename = "language.info")]
    LanguageInfo(LanguageInfoParams),
}

impl Method {
    /// Every method name, in declaration order.
    pub const NAMES: [&'static str; 21] = [
        "daemon.status",
        "daemon.shutdown",
        "chat.start",
        "chat.cancel",
        "model.status",
        "model.set_profile",
        "resources.snapshot",
        "logs.subscribe",
        "doctor.run",
        "tools.list",
        "tools.call",
        "task.create",
        "task.cancel",
        "task.stop_all",
        "game_mode.set",
        "game_mode.status",
        "notify.status",
        "notify.test",
        "approval.respond",
        "language.list",
        "language.info",
    ];

    /// The JSON-RPC method name, e.g. `chat.start`.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::DaemonStatus(_) => "daemon.status",
            Self::DaemonShutdown(_) => "daemon.shutdown",
            Self::ChatStart(_) => "chat.start",
            Self::ChatCancel(_) => "chat.cancel",
            Self::ModelStatus(_) => "model.status",
            Self::ModelSetProfile(_) => "model.set_profile",
            Self::ResourcesSnapshot(_) => "resources.snapshot",
            Self::LogsSubscribe(_) => "logs.subscribe",
            Self::DoctorRun(_) => "doctor.run",
            Self::ToolsList(_) => "tools.list",
            Self::ToolsCall(_) => "tools.call",
            Self::TaskCreate(_) => "task.create",
            Self::TaskCancel(_) => "task.cancel",
            Self::TaskStopAll(_) => "task.stop_all",
            Self::GameModeSet(_) => "game_mode.set",
            Self::GameModeStatus(_) => "game_mode.status",
            Self::NotifyStatus(_) => "notify.status",
            Self::NotifyTest(_) => "notify.test",
            Self::ApprovalRespond(_) => "approval.respond",
            Self::LanguageList(_) => "language.list",
            Self::LanguageInfo(_) => "language.info",
        }
    }
}

/// Parameters of `chat.start`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatStartParams {
    /// Conversation so far, ending with the user's message.
    pub messages: Vec<ChatMessage>,
    /// Profile to use; the active one if absent. A different profile triggers a switch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    /// JSON Schema the reply must match (llama-server `json_schema` response format).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,
    /// Generation limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Reasoning effort; `none` if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningEffort>,
}

/// Result of `chat.start`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatStarted {
    /// Identifies the chat in later events and in `chat.cancel`.
    pub chat_id: ChatId,
    /// Trace covering the chat's log events and model call.
    pub trace_id: TraceId,
}

/// Parameters of `chat.cancel`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCancelParams {
    /// Chat to cancel.
    pub chat_id: ChatId,
}

/// Parameters of `model.set_profile`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSetProfileParams {
    /// Profile name from the config.
    pub profile: String,
}

/// Parameters of `logs.subscribe`. All filters are optional and combine with AND.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogsSubscribeParams {
    /// Only events at this level or above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_level: Option<LogLevel>,
    /// Only events in this trace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<TraceId>,
    /// Only events whose target starts with this prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_prefix: Option<String>,
}

/// Result of `daemon.status`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonStatus {
    /// Daemon version (crate version).
    pub version: String,
    /// Protocol version the daemon speaks.
    pub proto_version: u32,
    /// Daemon process ID.
    pub pid: u32,
    /// Seconds since start.
    pub uptime_s: u64,
    /// Model server state.
    pub model: ModelStatus,
}

/// Parameters of `tools.call`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolsCallParams {
    /// Tool to call (unique across servers).
    pub tool: String,
    /// Arguments object, validated against the tool's input schema before the call.
    #[serde(default)]
    pub arguments: Value,
    /// Trace to attach the tool call to; the daemon starts a new one if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<TraceId>,
}

/// Result of `tools.list`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolList {
    /// Every available tool, sorted by name.
    pub tools: Vec<ToolInfo>,
}

/// One advertised tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolInfo {
    /// The MCP server (manifest name) that exposes it.
    pub server: String,
    /// Tool name.
    pub name: String,
    /// Description, if the server gave one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the tool's arguments.
    pub input_schema: Value,
}

/// Result of `tools.call`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCallOutcome {
    /// Content blocks the tool returned (MCP `content`).
    pub content: Value,
    /// Whether the tool reported a tool-level error (MCP `isError`).
    #[serde(default)]
    pub is_error: bool,
}

/// Parameters of `task.create`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCreateParams {
    /// What the executor should accomplish.
    pub goal: String,
    /// Repository the task worktree will be cut from.
    pub repo: PathBuf,
    /// Starting commit, branch, or ref; defaults to repo HEAD if None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_point: Option<String>,
    /// Chat profile to use; defaults to active profile if None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    /// Maximum tool calls allowed for this task; defaults to config if None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    /// Maximum total model tokens allowed across all turns; defaults to config if None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Wall-clock timeout in seconds; defaults to config if None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_time_secs: Option<u64>,
    /// Whether to open a GitHub PR upon task completion; defaults to config if None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub create_pr: Option<bool>,
    /// Target language module identifier (e.g. "python", "cpp", "sql").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Optional language-specific configuration options.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language_config: Option<Value>,
    /// Optional observable verification command target (e.g. "test.run" or tagged "shell.run" command).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_target: Option<String>,
    /// Number of concurrent streams assigned to this task (defaults to 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streams: Option<u32>,
}

impl TaskCreateParams {
    /// Creates a minimal `TaskCreateParams` with required goal and repo.
    #[must_use]
    pub fn new(goal: impl Into<String>, repo: impl Into<PathBuf>) -> Self {
        Self {
            goal: goal.into(),
            repo: repo.into(),
            start_point: None,
            profile: None,
            max_tool_calls: None,
            max_tokens: None,
            wall_time_secs: None,
            create_pr: None,
            language: None,
            language_config: None,
            verification_target: None,
            streams: None,
        }
    }
}

/// Initial response to `task.create`, returned before any `task.*` events stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskStarted {
    /// The newly created task.
    pub task_id: TaskId,
    /// Trace ID for all events produced by this task.
    pub trace_id: TraceId,
}

/// Parameters of `task.cancel`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCancelParams {
    /// The task to cancel.
    pub task_id: TaskId,
}

/// Parameters of `task.stop_all`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskStopAllParams {
    /// Optional reason or comment for the global stop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Result of `task.stop_all`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskStopAllResponse {
    /// Number of tasks stopped.
    pub stopped: u32,
}

/// Parameters of `game_mode.set`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameModeSetParams {
    /// Enable (`true`) or disable (`false`) game mode manually.
    pub enabled: bool,
}

/// Result of `game_mode.set` and `game_mode.status`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameModeStatusResponse {
    /// Whether game mode is currently active.
    pub active: bool,
    /// Manual override state, if any (`Some(true)` or `Some(false)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manual_override: Option<bool>,
    /// Why game mode is active, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Process name that triggered game mode, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process_name: Option<String>,
}

/// Parameters of `notify.test`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NotifyTestParams {
    /// Optional test message override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Result of `notify.status`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NotifyStatusResponse {
    /// Whether push notifications are enabled.
    pub enabled: bool,
    /// Configured ntfy server URL.
    pub server_url: String,
    /// Configured ntfy topic.
    pub topic: String,
    /// Whether destination is confined to Tailscale only.
    pub tailscale_only: bool,
}

/// Result of `notify.test`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NotifyTestResponse {
    /// Whether the test notification was sent successfully.
    pub success: bool,
    /// Detail message.
    pub message: String,
}

/// Parameters of `approval.respond`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovalRespondParams {
    /// Trace ID of the request being approved or denied.
    pub trace_id: String,
    /// Whether the action is approved.
    pub approved: bool,
    /// Command identity being approved.
    pub command: String,
    /// Permission tier of the command.
    pub tier: String,
    /// Optional human explanation or reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Result of `approval.respond`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovalRespondResponse {
    /// Whether the response was recorded successfully.
    pub success: bool,
    /// Status message.
    pub message: String,
}

/// Metadata describing a registered language module.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageMetadata {
    /// Unique identifier (e.g. "python", "cpp", "sql", "rust").
    pub id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Short summary of module capabilities.
    pub description: String,
    /// Supported runtime versions (e.g. ["3.10", "3.11", "3.12"]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supported_versions: Vec<String>,
    /// Default version if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_version: Option<String>,
}

/// Result of `language.list`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageListResponse {
    /// Every registered language module, sorted by ID.
    pub languages: Vec<LanguageMetadata>,
}

/// Parameters of `language.info`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageInfoParams {
    /// Language module ID (e.g. "python").
    pub language: String,
}

/// Result of `language.info`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageInfoResponse {
    /// Metadata for the language module.
    pub metadata: LanguageMetadata,
    /// Tools exposed by this language module.
    pub tools: Vec<ToolInfo>,
}
