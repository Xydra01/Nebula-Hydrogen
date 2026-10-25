#![allow(clippy::unwrap_used, clippy::panic, missing_docs)]

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use nebula_cli::tui::{ActivePane, TaskItem, ToolCallItem, TuiAction, TuiApp, TuiTaskStatus};
use nebula_proto::{
    ApprovalNeeded, DaemonStatus, DiskUsage, GameModeStatusResponse, GpuProcess, ModelState,
    ModelStatus, ResourceSnapshot, TaskId,
};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use time::macros::datetime;

fn sample_resource_snapshot() -> ResourceSnapshot {
    let gb = 1024 * 1024 * 1024;
    ResourceSnapshot {
        taken_at: datetime!(2026-10-04 12:00:00 UTC),
        vram_used_mib: 9_300,
        vram_total_mib: 12_282,
        gpu_util_pct: Some(87),
        ram_used_mib: 20_480,
        ram_total_mib: 32_607,
        commit_used_mib: 37_000,
        commit_limit_mib: 65_375,
        cpu_pct: 12.4,
        disks: vec![
            DiskUsage {
                mount: r"F:\".into(),
                free_bytes: 700 * gb,
                total_bytes: 1863 * gb,
            },
            DiskUsage {
                mount: r"D:\".into(),
                free_bytes: 40 * gb,
                total_bytes: 931 * gb,
            },
        ],
        gpu_processes: vec![
            GpuProcess {
                pid: 4242,
                name: "llama-server.exe".into(),
                vram_mib: 8_900,
            },
            GpuProcess {
                pid: 77,
                name: "dwm.exe".into(),
                vram_mib: 210,
            },
        ],
    }
}

fn sample_daemon_status() -> DaemonStatus {
    DaemonStatus {
        version: "0.1.0".into(),
        proto_version: 2,
        pid: 1234,
        uptime_s: 3725,
        model: ModelStatus {
            profile: "code".into(),
            state: ModelState::Ready,
            since: datetime!(2026-10-04 11:00:00 UTC),
            restarts: 0,
            last_error: None,
        },
    }
}

#[test]
fn test_tui_empty_dashboard_layout() {
    let app = TuiApp::new();
    let backend = TestBackend::new(120, 32);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|f| nebula_cli::tui::render(&app, f)).unwrap();
    insta::assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_populated_dashboard_layout() {
    let mut app = TuiApp::new();
    app.daemon_status = Some(sample_daemon_status());
    app.resources = Some(sample_resource_snapshot());
    app.game_mode = Some(GameModeStatusResponse {
        active: false,
        manual_override: None,
        reason: None,
        process_name: None,
    });

    let task_id = "01J9ZQ3V5W8X0Y2Z4A6B8C0D2E".parse::<TaskId>().unwrap();
    app.tasks.push(TaskItem {
        id: task_id,
        status: TuiTaskStatus::Running,
        summary: "Refactor database migrations to use atomic transactions".into(),
        step_count: 3,
        language: Some("rust".into()),
        mcp_healthy: true,
    });

    app.streaming_text = "Analyzing schema definitions in crates/nebula-db/src/migrations.rs...\nFound 4 migrations requiring transactional boundaries.".into();

    app.tool_calls.push_back(ToolCallItem {
        task_id,
        step_index: 0,
        summary: "fs_read: crates/nebula-db/src/migrations.rs".into(),
        status: "ok".into(),
    });
    app.tool_calls.push_back(ToolCallItem {
        task_id,
        step_index: 1,
        summary: "git_diff: HEAD~1".into(),
        status: "ok".into(),
    });

    let backend = TestBackend::new(120, 32);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|f| nebula_cli::tui::render(&app, f)).unwrap();
    insta::assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_approval_modal_layout() {
    let mut app = TuiApp::new();
    app.daemon_status = Some(sample_daemon_status());
    app.resources = Some(sample_resource_snapshot());

    app.pending_approval = Some(ApprovalNeeded {
        command: "git push --force origin main".into(),
        tier: "Tier 3 (System)".into(),
        rule_id: Some("protected-branches".into()),
        trace_id: "trace-abc-123".into(),
        summary: "Force push to protected branch 'main' requires Tier 3 operator confirmation"
            .into(),
    });
    app.approval_modal_open = true;

    let backend = TestBackend::new(120, 32);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|f| nebula_cli::tui::render(&app, f)).unwrap();
    insta::assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_approval_modal_with_details_layout() {
    let mut app = TuiApp::new();
    app.daemon_status = Some(sample_daemon_status());
    app.resources = Some(sample_resource_snapshot());

    app.pending_approval = Some(ApprovalNeeded {
        command: "git push --force origin main".into(),
        tier: "Tier 3 (System)".into(),
        rule_id: Some("protected-branches".into()),
        trace_id: "trace-abc-123".into(),
        summary: "Force push to protected branch 'main' requires Tier 3 operator confirmation"
            .into(),
    });
    app.approval_modal_open = true;
    app.show_details = true;

    let backend = TestBackend::new(120, 32);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|f| nebula_cli::tui::render(&app, f)).unwrap();
    insta::assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_keyboard_navigation_and_actions() {
    let mut app = TuiApp::new();
    assert_eq!(app.active_pane, ActivePane::Tasks);

    // Tab switches pane
    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(app.active_pane, ActivePane::Streaming);

    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(app.active_pane, ActivePane::ToolCalls);

    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(app.active_pane, ActivePane::Tasks);

    // BackTab switches reverse
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT));
    assert_eq!(app.active_pane, ActivePane::ToolCalls);

    // Number keys switch directly
    app.handle_key(KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE));
    assert_eq!(app.active_pane, ActivePane::Tasks);

    app.handle_key(KeyEvent::new(KeyCode::Char('2'), KeyModifiers::NONE));
    assert_eq!(app.active_pane, ActivePane::Streaming);

    app.handle_key(KeyEvent::new(KeyCode::Char('3'), KeyModifiers::NONE));
    assert_eq!(app.active_pane, ActivePane::ToolCalls);

    // Approval modal interactions
    let approval = ApprovalNeeded {
        command: "reboot".into(),
        tier: "Tier 3".into(),
        rule_id: None,
        trace_id: "tr-123".into(),
        summary: "Reboot requested".into(),
    };
    app.pending_approval = Some(approval.clone());
    app.approval_modal_open = true;

    // Press 'a' to approve
    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
    assert!(!app.approval_modal_open);
    assert_eq!(
        app.pending_actions.pop_front(),
        Some(TuiAction::ApprovalResponse {
            request: approval.clone(),
            approved: true,
        })
    );

    // Modal with 'd' to deny
    app.pending_approval = Some(approval.clone());
    app.approval_modal_open = true;
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE));
    assert!(!app.approval_modal_open);
    assert_eq!(
        app.pending_actions.pop_front(),
        Some(TuiAction::ApprovalResponse {
            request: approval,
            approved: false,
        })
    );

    // Quit action
    app.handle_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE));
    assert!(app.should_quit);
}

#[test]
fn test_tui_game_mode_banner_layout() {
    let mut app = TuiApp::new();
    app.daemon_status = Some(sample_daemon_status());
    app.resources = Some(sample_resource_snapshot());
    app.game_mode = Some(GameModeStatusResponse {
        active: true,
        manual_override: None,
        reason: Some("DirectX 12 Game Running".into()),
        process_name: Some("Cyberpunk2077.exe".into()),
    });

    let backend = TestBackend::new(120, 32);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|f| nebula_cli::tui::render(&app, f)).unwrap();
    insta::assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_language_picker_layout() {
    let mut app = TuiApp::new();
    app.daemon_status = Some(sample_daemon_status());
    app.resources = Some(sample_resource_snapshot());
    app.language_picker_open = true;

    let backend = TestBackend::new(120, 32);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|f| nebula_cli::tui::render(&app, f)).unwrap();
    insta::assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_diagnostics_modal_layout() {
    let mut app = TuiApp::new();
    app.daemon_status = Some(sample_daemon_status());
    app.resources = Some(sample_resource_snapshot());
    app.show_diagnostics = true;
    app.diagnostics.push(nebula_cli::tui::DiagnosticItem {
        file: "src/main.cpp".into(),
        line: 42,
        col: 15,
        severity: "error".into(),
        code: Some("C2065".into()),
        message: "'undeclared_identifier': undeclared identifier".into(),
    });

    let backend = TestBackend::new(120, 32);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|f| nebula_cli::tui::render(&app, f)).unwrap();
    insta::assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_language_and_diagnostics_key_handling() {
    let mut app = TuiApp::new();
    assert!(!app.language_picker_open);
    assert!(!app.show_diagnostics);

    // 'l' opens language picker
    app.handle_key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::NONE));
    assert!(app.language_picker_open);

    // down arrow navigates picker
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(app.selected_language_idx, 1);

    // enter selects language
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert!(!app.language_picker_open);
    assert_eq!(app.active_language, Some("rust".into()));

    // 'g' toggles diagnostics
    app.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE));
    assert!(app.show_diagnostics);

    app.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE));
    assert!(!app.show_diagnostics);
}
