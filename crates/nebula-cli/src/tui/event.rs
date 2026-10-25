//! Event loop coordination for the Nebula TUI.

use std::time::Duration;

use anyhow::Result;
use nebula_daemon::client::Client;
use nebula_proto::{
    ApprovalRespondParams, ApprovalRespondResponse, DaemonStatus, Empty, Method, ResourceSnapshot,
};

use super::app::{TuiAction, TuiApp};
use super::terminal::TerminalGuard;
use super::ui;

/// Runs the interactive terminal UI connected to the Nebula daemon pipe.
///
/// # Errors
/// Returns an error if the daemon cannot be reached or the terminal fails to initialize.
pub async fn run_tui(pipe_path: &str) -> Result<()> {
    // Connect event stream client and command client to the daemon
    let mut cmd_client = Client::connect(pipe_path, Duration::from_secs(5)).await?;
    let mut event_client = Client::connect(pipe_path, Duration::from_secs(5)).await?;

    let mut app = TuiApp::new();

    // Fetch initial status and resources
    if let Ok(status) = cmd_client
        .call::<DaemonStatus>(Method::DaemonStatus(Empty {}))
        .await
    {
        app.update_daemon_status(status);
    }
    if let Ok(snap) = cmd_client
        .call::<ResourceSnapshot>(Method::ResourcesSnapshot(Empty {}))
        .await
    {
        app.update_resources(snap);
    }

    // Spawn background pipe event listener
    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(512);
    tokio::spawn(async move {
        while let Ok(event) = event_client.next_event().await {
            if event_tx.send(event).await.is_err() {
                break;
            }
        }
    });

    // Spawn background periodic resource poller
    let (res_tx, mut res_rx) = tokio::sync::mpsc::channel(4);
    let poll_pipe = pipe_path.to_string();
    tokio::spawn(async move {
        if let Ok(mut poll_client) = Client::connect(&poll_pipe, Duration::from_secs(5)).await {
            let mut timer = tokio::time::interval(Duration::from_secs(2));
            loop {
                timer.tick().await;
                if let Ok(snap) = poll_client
                    .call::<ResourceSnapshot>(Method::ResourcesSnapshot(Empty {}))
                    .await
                    && res_tx.send(snap).await.is_err()
                {
                    break;
                }
            }
        }
    });

    // Spawn background crossterm keyboard reader
    let (key_tx, mut key_rx) = tokio::sync::mpsc::channel(32);
    tokio::task::spawn_blocking(move || {
        while let Ok(event) = crossterm::event::read() {
            if let crossterm::event::Event::Key(key) = event
                && key.kind == crossterm::event::KeyEventKind::Press
                && key_tx.blocking_send(key).is_err()
            {
                break;
            }
        }
    });

    // Initialize terminal with RAII cleanup guard
    let mut guard = TerminalGuard::init()?;
    let terminal = guard.terminal_mut();

    let mut tick_timer = tokio::time::interval(Duration::from_millis(50));

    loop {
        terminal.draw(|f| ui::render(&app, f))?;

        tokio::select! {
            _ = tick_timer.tick() => {
                // Periodic frame tick
            }
            Some(snap) = res_rx.recv() => {
                app.update_resources(snap);
            }
            Some(key) = key_rx.recv() => {
                app.handle_key(key);
                while let Some(action) = app.pending_actions.pop_front() {
                    match action {
                        TuiAction::ApprovalResponse { request, approved } => {
                            let params = ApprovalRespondParams {
                                trace_id: request.trace_id,
                                approved,
                                command: request.command,
                                tier: request.tier,
                                reason: Some(if approved {
                                    "Approved via TUI modal".into()
                                } else {
                                    "Denied via TUI modal".into()
                                }),
                            };
                            let _ = cmd_client
                                .call::<ApprovalRespondResponse>(Method::ApprovalRespond(params))
                                .await;
                        }
                        TuiAction::RefreshResources => {
                            if let Ok(snap) = cmd_client
                                .call::<ResourceSnapshot>(Method::ResourcesSnapshot(Empty {}))
                                .await
                            {
                                app.update_resources(snap);
                            }
                        }
                        TuiAction::Quit => {
                            return Ok(());
                        }
                    }
                }
                if app.should_quit {
                    return Ok(());
                }
            }
            Some(event) = event_rx.recv() => {
                app.handle_event(event);
            }
        }
    }
}
