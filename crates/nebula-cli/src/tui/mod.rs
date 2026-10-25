//! Terminal user interface client for Nebula over named pipe (ADR-007, issue #31).

pub mod app;
pub mod event;
pub mod terminal;
pub mod ui;

pub use app::{
    ActivePane, DiagnosticItem, TaskItem, ToolCallItem, TuiAction, TuiApp, TuiTaskStatus,
};
pub use event::run_tui;
pub use ui::render;
