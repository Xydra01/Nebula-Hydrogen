//! Application state and event handling for the Nebula TUI.

use std::collections::VecDeque;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use nebula_proto::{
    ApprovalNeeded, DaemonStatus, Event, GameModeStatusResponse, ResourceSnapshot, TaskId,
    TaskStatus,
};

/// Maximum tool calls retained in memory for the TUI list.
const MAX_TOOL_CALLS: usize = 100;

/// Active pane with keyboard focus.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActivePane {
    /// Tasks list pane.
    #[default]
    Tasks,
    /// Live streaming output pane.
    Streaming,
    /// Tool calls list pane.
    ToolCalls,
}

impl ActivePane {
    /// Cycles to the next pane.
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Tasks => Self::Streaming,
            Self::Streaming => Self::ToolCalls,
            Self::ToolCalls => Self::Tasks,
        }
    }

    /// Cycles to the previous pane.
    #[must_use]
    pub const fn prev(self) -> Self {
        match self {
            Self::Tasks => Self::ToolCalls,
            Self::Streaming => Self::Tasks,
            Self::ToolCalls => Self::Streaming,
        }
    }
}

/// Current execution or completion status of a task for the TUI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuiTaskStatus {
    /// In progress.
    Running,
    /// Successfully completed.
    Completed,
    /// Voluntarily completed via task.finish.
    Finished,
    /// Reached budget or limit.
    LimitReached,
    /// Committing worktree changes.
    Committing,
    /// Cancelled.
    Cancelled,
    /// Failed with error.
    Failed,
}

impl From<TaskStatus> for TuiTaskStatus {
    fn from(status: TaskStatus) -> Self {
        match status {
            TaskStatus::Completed => Self::Completed,
            TaskStatus::LimitReached => Self::LimitReached,
            TaskStatus::Cancelled => Self::Cancelled,
            TaskStatus::Failed => Self::Failed,
        }
    }
}

/// A structured compiler or build diagnostic parsed from tool/process output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticItem {
    /// Source file path.
    pub file: String,
    /// 1-based line number.
    pub line: u32,
    /// 1-based column number.
    pub col: u32,
    /// Diagnostic severity (e.g. "error", "warning").
    pub severity: String,
    /// Optional error/warning code (e.g. "C2065", "E0308").
    pub code: Option<String>,
    /// Diagnostic message text.
    pub message: String,
}

/// A task item displayed in the tasks list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskItem {
    /// Task identifier.
    pub id: TaskId,
    /// Current task status.
    pub status: TuiTaskStatus,
    /// Short summary or goal.
    pub summary: String,
    /// Number of tool steps completed.
    pub step_count: usize,
    /// Active language module identifier.
    pub language: Option<String>,
    /// Whether the language MCP process is healthy.
    pub mcp_healthy: bool,
}

/// A recorded tool call step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCallItem {
    /// Task ID.
    pub task_id: TaskId,
    /// Step index.
    pub step_index: u32,
    /// Tool name and argument summary.
    pub summary: String,
    /// Status or result.
    pub status: String,
}

/// Action to dispatch from the UI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TuiAction {
    /// Send an approval response (true = approved, false = denied).
    ApprovalResponse {
        /// Approval request being answered.
        request: ApprovalNeeded,
        /// Approved flag.
        approved: bool,
    },
    /// Refresh resource snapshot.
    RefreshResources,
    /// Quit application.
    Quit,
}

/// Application state for the Nebula TUI.
#[derive(Debug)]
pub struct TuiApp {
    /// Recent tasks.
    pub tasks: Vec<TaskItem>,
    /// Selected index in tasks list.
    pub selected_task: usize,
    /// Streaming model output text.
    pub streaming_text: String,
    /// Vertical scroll offset for streaming text.
    pub streaming_scroll: usize,
    /// Whether streaming text auto-scrolls to the end on new tokens.
    pub auto_scroll: bool,
    /// Cached line count for streaming text to avoid O(N) line scans on every token.
    pub streaming_line_count: usize,
    /// Recent tool calls.
    pub tool_calls: VecDeque<ToolCallItem>,
    /// Selected index in tool calls.
    pub selected_tool_call: usize,
    /// Parsed compiler and build diagnostics.
    pub diagnostics: Vec<DiagnosticItem>,
    /// Whether structured diagnostic viewer pane is open.
    pub show_diagnostics: bool,
    /// Whether language picker modal is open.
    pub language_picker_open: bool,
    /// Available languages in the picker.
    pub available_languages: Vec<String>,
    /// Selected index in language picker.
    pub selected_language_idx: usize,
    /// User-selected active language override.
    pub active_language: Option<String>,
    /// Latest resource snapshot.
    pub resources: Option<ResourceSnapshot>,
    /// Latest daemon status.
    pub daemon_status: Option<DaemonStatus>,
    /// Active game mode status.
    pub game_mode: Option<GameModeStatusResponse>,
    /// Pending approval modal, if any.
    pub pending_approval: Option<ApprovalNeeded>,
    /// Whether approval modal is open.
    pub approval_modal_open: bool,
    /// Whether details modal is open.
    pub show_details: bool,
    /// Currently focused pane.
    pub active_pane: ActivePane,
    /// Whether application should exit.
    pub should_quit: bool,
    /// Pending actions to dispatch to the daemon.
    pub pending_actions: VecDeque<TuiAction>,
}

impl Default for TuiApp {
    fn default() -> Self {
        Self {
            tasks: Vec::new(),
            selected_task: 0,
            streaming_text: String::new(),
            streaming_scroll: 0,
            auto_scroll: true,
            streaming_line_count: 1,
            tool_calls: VecDeque::new(),
            selected_tool_call: 0,
            diagnostics: Vec::new(),
            show_diagnostics: false,
            language_picker_open: false,
            available_languages: vec![
                "Auto-detect".into(),
                "rust".into(),
                "python".into(),
                "cpp".into(),
                "sql".into(),
            ],
            selected_language_idx: 0,
            active_language: None,
            resources: None,
            daemon_status: None,
            game_mode: None,
            pending_approval: None,
            approval_modal_open: false,
            show_details: false,
            active_pane: ActivePane::Tasks,
            should_quit: false,
            pending_actions: VecDeque::new(),
        }
    }
}

impl TuiApp {
    /// Creates a fresh TUI state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of currently active executor tasks/streams.
    #[must_use]
    pub fn active_streams(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| matches!(t.status, TuiTaskStatus::Running | TuiTaskStatus::Committing))
            .count()
    }

    /// Returns the maximum concurrency streams supported by the engine/supervisor.
    #[must_use]
    pub const fn max_streams(&self) -> usize {
        4
    }

    /// Appends new text to the streaming buffer and updates line count incrementally.
    pub fn append_streaming_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let new_lines = text.bytes().filter(|&b| b == b'\n').count();
        self.streaming_line_count += new_lines;
        self.streaming_text.push_str(text);
        if self.auto_scroll {
            self.streaming_scroll = self.streaming_line_count;
        }
    }

    /// Syncs `streaming_line_count` from `streaming_text` if modified externally.
    pub fn sync_streaming_line_count(&mut self) {
        self.streaming_line_count = self.streaming_text.lines().count().max(1);
    }

    /// Handles keyboard input.
    pub fn handle_key(&mut self, key: KeyEvent) {
        // Language picker intercepts keys
        if self.language_picker_open {
            match key.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.selected_language_idx > 0 {
                        self.selected_language_idx -= 1;
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.selected_language_idx + 1 < self.available_languages.len() {
                        self.selected_language_idx += 1;
                    }
                }
                KeyCode::Enter => {
                    self.active_language = if self.selected_language_idx == 0 {
                        None
                    } else {
                        Some(self.available_languages[self.selected_language_idx].clone())
                    };
                    self.language_picker_open = false;
                }
                KeyCode::Esc | KeyCode::Char('l' | 'L') => {
                    self.language_picker_open = false;
                }
                _ => {}
            }
            return;
        }

        // Modal mode intercepts keys first
        if self.approval_modal_open {
            match key.code {
                KeyCode::Char('a' | 'A') => {
                    if let Some(req) = self.pending_approval.take() {
                        self.pending_actions.push_back(TuiAction::ApprovalResponse {
                            request: req,
                            approved: true,
                        });
                    }
                    self.approval_modal_open = false;
                    self.show_details = false;
                }
                KeyCode::Char('d' | 'D') | KeyCode::Esc => {
                    if let Some(req) = self.pending_approval.take() {
                        self.pending_actions.push_back(TuiAction::ApprovalResponse {
                            request: req,
                            approved: false,
                        });
                    }
                    self.approval_modal_open = false;
                    self.show_details = false;
                }
                KeyCode::Char('v' | 'V') => {
                    self.show_details = !self.show_details;
                }
                _ => {}
            }
            return;
        }

        // Global keybindings
        match (key.code, key.modifiers) {
            (KeyCode::Char('c'), KeyModifiers::CONTROL) | (KeyCode::Char('q'), _) => {
                self.should_quit = true;
                self.pending_actions.push_back(TuiAction::Quit);
            }
            (KeyCode::Tab, KeyModifiers::SHIFT) | (KeyCode::BackTab, _) => {
                self.active_pane = self.active_pane.prev();
            }
            (KeyCode::Tab, _) => {
                self.active_pane = self.active_pane.next();
            }
            (KeyCode::Char('1'), _) => {
                self.active_pane = ActivePane::Tasks;
            }
            (KeyCode::Char('2'), _) => {
                self.active_pane = ActivePane::Streaming;
            }
            (KeyCode::Char('3'), _) => {
                self.active_pane = ActivePane::ToolCalls;
            }
            (KeyCode::Char('r'), _) => {
                self.pending_actions.push_back(TuiAction::RefreshResources);
            }
            (KeyCode::Char('l' | 'L'), _) => {
                self.language_picker_open = true;
            }
            (KeyCode::Char('g' | 'G'), _) => {
                self.show_diagnostics = !self.show_diagnostics;
            }
            // Pane-specific navigation
            (KeyCode::Up | KeyCode::Char('k'), _) => match self.active_pane {
                ActivePane::Tasks => {
                    if self.selected_task > 0 {
                        self.selected_task -= 1;
                    }
                }
                ActivePane::Streaming => {
                    if self.auto_scroll {
                        self.streaming_scroll = self.streaming_line_count;
                        self.auto_scroll = false;
                    }
                    if self.streaming_scroll > 0 {
                        self.streaming_scroll -= 1;
                    }
                }
                ActivePane::ToolCalls => {
                    if self.selected_tool_call > 0 {
                        self.selected_tool_call -= 1;
                    }
                }
            },
            (KeyCode::Down | KeyCode::Char('j'), _) => match self.active_pane {
                ActivePane::Tasks => {
                    if !self.tasks.is_empty() && self.selected_task + 1 < self.tasks.len() {
                        self.selected_task += 1;
                    }
                }
                ActivePane::Streaming => {
                    self.streaming_scroll += 1;
                    if self.streaming_scroll >= self.streaming_line_count {
                        self.auto_scroll = true;
                    }
                }
                ActivePane::ToolCalls => {
                    if !self.tool_calls.is_empty()
                        && self.selected_tool_call + 1 < self.tool_calls.len()
                    {
                        self.selected_tool_call += 1;
                    }
                }
            },
            (KeyCode::PageUp, _) if self.active_pane == ActivePane::Streaming => {
                if self.auto_scroll {
                    self.streaming_scroll = self.streaming_line_count;
                    self.auto_scroll = false;
                }
                self.streaming_scroll = self.streaming_scroll.saturating_sub(10);
            }
            (KeyCode::PageDown, _) if self.active_pane == ActivePane::Streaming => {
                self.streaming_scroll = self.streaming_scroll.saturating_add(10);
                if self.streaming_scroll >= self.streaming_line_count {
                    self.auto_scroll = true;
                }
            }
            (KeyCode::Home, _) if self.active_pane == ActivePane::Streaming => {
                self.streaming_scroll = 0;
                self.auto_scroll = false;
            }
            (KeyCode::End, _) if self.active_pane == ActivePane::Streaming => {
                self.streaming_scroll = self.streaming_line_count;
                self.auto_scroll = true;
            }
            _ => {}
        }
    }

    /// Handles events arriving from the daemon pipe.
    pub fn handle_event(&mut self, event: Event) {
        match event {
            Event::ChatToken(t) => {
                let task_id = TaskId::from_ulid(t.chat_id.ulid());
                if !self.tasks.iter().any(|task| task.id == task_id) {
                    self.tasks.push(TaskItem {
                        id: task_id,
                        status: TuiTaskStatus::Running,
                        summary: format!("Task {task_id}"),
                        step_count: 0,
                        language: self.active_language.clone(),
                        mcp_healthy: true,
                    });
                }
                if !t.reasoning.is_empty() {
                    self.append_streaming_text(&t.reasoning);
                }
                if !t.text.is_empty() {
                    if t.text.contains("[ERROR]")
                        && let Some(tc) =
                            self.tool_calls.iter_mut().find(|tc| tc.status == "running")
                    {
                        tc.status = "error".into();
                    } else if t.text.contains("[OK]")
                        && let Some(tc) =
                            self.tool_calls.iter_mut().find(|tc| tc.status == "running")
                    {
                        tc.status = "ok".into();
                    }
                    extract_diagnostics(&t.text, &mut self.diagnostics);
                    self.append_streaming_text(&t.text);
                }
            }
            Event::TaskStep(step) => {
                // Update task or insert
                if let Some(task) = self.tasks.iter_mut().find(|t| t.id == step.task_id) {
                    task.step_count = step.step_index as usize;
                } else {
                    self.tasks.push(TaskItem {
                        id: step.task_id,
                        status: TuiTaskStatus::Running,
                        summary: format!("Task {}", step.task_id),
                        step_count: step.step_index as usize,
                        language: self.active_language.clone(),
                        mcp_healthy: true,
                    });
                }

                // Record tool calls with initial 'running' status
                for call in &step.tool_calls {
                    self.tool_calls.push_back(ToolCallItem {
                        task_id: step.task_id,
                        step_index: step.step_index,
                        summary: call.clone(),
                        status: "running".into(),
                    });
                    if self.tool_calls.len() > MAX_TOOL_CALLS {
                        self.tool_calls.pop_front();
                    }
                }

                let tools_desc = if step.tool_calls.is_empty() {
                    "thinking".to_string()
                } else {
                    step.tool_calls.join(", ")
                };
                let prefix =
                    if self.streaming_text.is_empty() || self.streaming_text.ends_with('\n') {
                        ""
                    } else {
                        "\n"
                    };
                let banner = format!("{prefix}─── Step {} [{tools_desc}] ───\n", step.step_index);
                self.append_streaming_text(&banner);
            }
            Event::TaskDone(done) => {
                let resolved_status = if done.summary.contains("task.finish") {
                    TuiTaskStatus::Finished
                } else if done.summary.contains("limit_reached")
                    || done.status == nebula_proto::TaskStatus::LimitReached
                {
                    TuiTaskStatus::LimitReached
                } else if done.summary.contains("committing") {
                    TuiTaskStatus::Committing
                } else {
                    done.status.into()
                };

                if let Some(task) = self.tasks.iter_mut().find(|t| t.id == done.task_id) {
                    task.status = resolved_status;
                    task.summary = done.summary.clone();
                } else {
                    self.tasks.push(TaskItem {
                        id: done.task_id,
                        status: resolved_status,
                        summary: done.summary.clone(),
                        step_count: 0,
                        language: self.active_language.clone(),
                        mcp_healthy: true,
                    });
                }

                // Resolve any tool calls still marked running
                for tc in self
                    .tool_calls
                    .iter_mut()
                    .filter(|tc| tc.status == "running")
                {
                    tc.status = match done.status {
                        nebula_proto::TaskStatus::Completed => "ok".into(),
                        _ => "error".into(),
                    };
                }

                let prefix =
                    if self.streaming_text.is_empty() || self.streaming_text.ends_with('\n') {
                        ""
                    } else {
                        "\n"
                    };
                let banner = format!(
                    "{prefix}═══ Task Done [{:?}] ═══\n{}\n",
                    done.status, done.summary
                );
                self.append_streaming_text(&banner);
            }
            Event::ApprovalNeeded(req) => {
                let prefix =
                    if self.streaming_text.is_empty() || self.streaming_text.ends_with('\n') {
                        ""
                    } else {
                        "\n"
                    };
                let banner = format!(
                    "{prefix}⚠ APPROVAL NEEDED: command '{}' ({})\n",
                    req.command, req.tier
                );
                self.append_streaming_text(&banner);
                self.pending_approval = Some(req);
                self.approval_modal_open = true;
            }
            Event::ResourcesSnapshot(snap) => {
                self.update_resources(snap);
            }
            Event::ModelStateChanged(ch) => {
                if let Some(ds) = &mut self.daemon_status {
                    ds.model.state = ch.to;
                }
            }
            Event::GameModeChanged(gm) => {
                self.game_mode = Some(GameModeStatusResponse {
                    active: gm.active,
                    manual_override: None,
                    reason: gm.reason,
                    process_name: None,
                });
            }
            _ => {}
        }
    }

    /// Updates resource snapshot.
    pub fn update_resources(&mut self, snap: ResourceSnapshot) {
        self.resources = Some(snap);
    }

    /// Updates daemon status.
    pub fn update_daemon_status(&mut self, status: DaemonStatus) {
        self.daemon_status = Some(status);
    }
}

/// Helper to parse compiler and build diagnostics from streamed text.
fn extract_diagnostics(text: &str, out: &mut Vec<DiagnosticItem>) {
    for line in text.lines() {
        let trimmed = line.trim();
        // MSVC pattern: path(line,col): error/warning Cxxxx: msg
        if let Some(paren_open) = trimmed.find('(')
            && let Some(paren_close) = trimmed[paren_open..].find(')')
        {
            let close_idx = paren_open + paren_close;
            let file = trimmed[..paren_open].trim();
            let coords = &trimmed[paren_open + 1..close_idx];
            let rest = trimmed[close_idx + 1..].trim_start_matches(':').trim();
            if let Some((l_str, c_str)) = coords.split_once(',')
                && let (Ok(line_no), Ok(col_no)) =
                    (l_str.trim().parse::<u32>(), c_str.trim().parse::<u32>())
            {
                let severity = if rest.starts_with("error") {
                    "error"
                } else if rest.starts_with("warning") {
                    "warning"
                } else {
                    "info"
                };
                if severity != "info" {
                    out.push(DiagnosticItem {
                        file: file.to_string(),
                        line: line_no,
                        col: col_no,
                        severity: severity.to_string(),
                        code: None,
                        message: rest.to_string(),
                    });
                    continue;
                }
            }
        }
        // Clang/GCC/Rust pattern: path:line:col: error/warning: msg
        let parts: Vec<&str> = trimmed.splitn(5, ':').collect();
        if parts.len() >= 4
            && let (Ok(line_no), Ok(col_no)) = (
                parts[1].trim().parse::<u32>(),
                parts[2].trim().parse::<u32>(),
            )
        {
            let rest = parts[3..].join(":").trim().to_string();
            let severity = if rest.starts_with("error") {
                "error"
            } else if rest.starts_with("warning") {
                "warning"
            } else {
                "info"
            };
            if severity != "info" {
                out.push(DiagnosticItem {
                    file: parts[0].trim().to_string(),
                    line: line_no,
                    col: col_no,
                    severity: severity.to_string(),
                    code: None,
                    message: rest,
                });
            }
        }
    }
}
