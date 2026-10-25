//! Daemon-to-client events (the `method` and `params` of a [`crate::Notification`]).

use serde::{Deserialize, Serialize};

use crate::ids::{ChatId, StepId, TaskId};
use crate::types::{LogEvent, ModelState, ResourceSnapshot, StopReason, Usage};

/// Every Phase 0 and Phase 1 event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum Event {
    /// A piece of streamed output.
    #[serde(rename = "chat.token")]
    ChatToken(ChatToken),
    /// A chat finished.
    #[serde(rename = "chat.done")]
    ChatDone(ChatDone),
    /// A chat failed.
    #[serde(rename = "chat.error")]
    ChatError(ChatError),
    /// The model server changed state.
    #[serde(rename = "model.state_changed")]
    ModelStateChanged(ModelStateChanged),
    /// A periodic resource sample.
    #[serde(rename = "resources.snapshot")]
    ResourcesSnapshot(ResourceSnapshot),
    /// A log event, for connections that called `logs.subscribe`.
    #[serde(rename = "log.event")]
    LogEvent(LogEvent),
    /// An executor step completed.
    #[serde(rename = "task.step")]
    TaskStep(TaskStep),
    /// An executor task finished.
    #[serde(rename = "task.done")]
    TaskDone(TaskDone),
    /// Game mode state changed (entered or exited).
    #[serde(rename = "game_mode.changed")]
    GameModeChanged(GameModeChanged),
    /// An approval is needed for an above-threshold action.
    #[serde(rename = "approval.needed")]
    ApprovalNeeded(ApprovalNeeded),
}

impl Event {
    /// The JSON-RPC method name, e.g. `chat.token`.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::ChatToken(_) => "chat.token",
            Self::ChatDone(_) => "chat.done",
            Self::ChatError(_) => "chat.error",
            Self::ModelStateChanged(_) => "model.state_changed",
            Self::ResourcesSnapshot(_) => "resources.snapshot",
            Self::LogEvent(_) => "log.event",
            Self::TaskStep(_) => "task.step",
            Self::TaskDone(_) => "task.done",
            Self::GameModeChanged(_) => "game_mode.changed",
            Self::ApprovalNeeded(_) => "approval.needed",
        }
    }
}

/// Streamed output. Reasoning and answer text arrive separately so clients can hide reasoning.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatToken {
    /// Chat the text belongs to.
    pub chat_id: ChatId,
    /// Answer text.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
    /// Reasoning text.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning: String,
    /// Optional number of concurrent streams assigned to the generating task/chat.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streams: Option<u32>,
}

/// End of a chat.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatDone {
    /// Finished chat.
    pub chat_id: ChatId,
    /// Why it stopped.
    pub stop_reason: StopReason,
    /// Token counts and timings.
    pub usage: Usage,
}

/// A failed chat.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatError {
    /// Failed chat.
    pub chat_id: ChatId,
    /// JSON-RPC style error code (see [`crate::error_code`]).
    pub code: i64,
    /// Human-readable message.
    pub message: String,
}

/// A model server state transition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelStateChanged {
    /// Profile the server runs.
    pub profile: String,
    /// Previous state.
    pub from: ModelState,
    /// New state.
    pub to: ModelState,
    /// Why, for failures and restarts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Final outcome of an executor task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// The model completed the goal without requesting further tools.
    Completed,
    /// A configured budget or limit was reached.
    LimitReached,
    /// The task was cancelled by client request or daemon shutdown.
    Cancelled,
    /// The task encountered an unrecoverable failure.
    Failed,
}

/// A progress step emitted during an executor task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskStep {
    /// Task this step belongs to.
    pub task_id: TaskId,
    /// Unique identifier for this step.
    pub step_id: StepId,
    /// 0-indexed step iteration.
    pub step_index: u32,
    /// Names of tools called during this step, if any.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<String>,
    /// Optional number of concurrent streams assigned to this task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streams: Option<u32>,
}

/// End of an executor task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskDone {
    /// Finished task.
    pub task_id: TaskId,
    /// Terminal status.
    pub status: TaskStatus,
    /// Human-readable summary of the outcome or reason for stopping.
    pub summary: String,
}

/// Notification emitted when game mode state changes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameModeChanged {
    /// Whether game mode is now active.
    pub active: bool,
    /// Reason for the change, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Notification emitted when an approval is required.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovalNeeded {
    /// Command base name requiring approval.
    pub command: String,
    /// Permission tier (e.g. Workspace or System).
    pub tier: String,
    /// Matched rule id, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_id: Option<String>,
    /// Trace ID of the command or task.
    pub trace_id: String,
    /// Human-readable explanation.
    pub summary: String,
}
