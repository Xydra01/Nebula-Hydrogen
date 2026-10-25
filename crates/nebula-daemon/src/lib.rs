//! The Nebula daemon (PHASE0_PLAN 6.6): a single instance per machine that owns the model
//! servers, samples resources, and serves JSON-RPC over a named pipe that only the current
//! user can open.
//!
//! [`start`] brings everything up and returns a [`Daemon`] handle; [`run`] also waits for
//! `daemon.shutdown` or Ctrl-C and shuts down gracefully. `main.rs` only wires the real
//! dependencies. Windows only (named pipes, Credential Manager).

mod builtin_providers;
pub mod checks;
pub mod circuit_breaker;
pub mod client;
pub mod executor;
pub mod game_mode;
pub mod github;
pub mod language;
pub mod notifications;
mod server;
pub mod win;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use nebula_config::NebulaConfig;
use nebula_model::{Launcher, ModelManager, Preflight, SupervisorConfig};
use nebula_proto::{ChatId, DoctorReport, TaskId};
use nebula_resources::{Sampler, SamplerConfig, Sources};
use nebula_telemetry::Telemetry;
use nebula_tools::ToolHost;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

/// Name of the single-instance mutex.
pub const INSTANCE_MUTEX: &str = r"Global\NebulaDaemon";

/// Startup failures.
#[derive(Debug, thiserror::Error)]
pub enum DaemonError {
    /// Another daemon holds the instance mutex.
    #[error("another Nebula daemon is already running")]
    AlreadyRunning,
    /// The pipe couldn't be created.
    #[error("named pipe {path}: {source}")]
    Pipe {
        /// Pipe path.
        path: String,
        /// Cause.
        source: std::io::Error,
    },
    /// A model manager couldn't start.
    #[error(transparent)]
    Model(#[from] nebula_model::ModelError),
    /// Any other startup problem.
    #[error("{0}")]
    Other(String),
}

/// Runs the checks that don't need the daemon (PowerShell, NVML, files).
pub type LocalChecks = dyn Fn(&NebulaConfig) -> DoctorReport + Send + Sync;

/// What the daemon runs on. `main.rs` passes the real ones; tests pass fakes.
pub struct Deps {
    /// Installed telemetry (for `logs.subscribe` and the blob store).
    pub telemetry: Telemetry,
    /// Starts model servers.
    pub launcher: Arc<dyn Launcher>,
    /// The MCP tool host, already started (its external servers launched). The caller builds it
    /// so `start` stays synchronous, matching how the model managers are constructed. `start`
    /// then registers the in-process built-in tools on it (wired to the live sampler) before
    /// wrapping it in `Arc` and sharing it, so registration — which needs `&mut ToolHost` — runs
    /// while the host is still exclusively owned.
    pub tool_host: ToolHost,
    /// Resource readers; `None` disables the sampler.
    pub sources: Option<Sources>,
    /// Runs before every model launch.
    pub preflight: Option<Arc<dyn Preflight>>,
    /// Restart and timeout policy for both model servers.
    pub supervisor: SupervisorConfig,
    /// Single-instance mutex name ([`INSTANCE_MUTEX`] in production).
    pub instance_name: String,
    /// Local doctor checks.
    pub local_checks: Arc<LocalChecks>,
    /// Whether `doctor.run` also checks runtime versions and model hashes.
    pub artifact_checks: bool,
    /// Optional foreground window detector seam (for tests).
    pub foreground_detector: Option<Arc<dyn win::ForegroundDetector>>,
}

pub(crate) struct RunningTaskState {
    pub(crate) cancel: CancellationToken,
    pub(crate) task_job: Option<Arc<nebula_sandbox::task_job::TaskJob>>,
}

pub(crate) struct Shared {
    pub(crate) config: NebulaConfig,
    pub(crate) telemetry: Telemetry,
    pub(crate) chat: ModelManager,
    pub(crate) embedding: Option<ModelManager>,
    pub(crate) tool_host: Arc<ToolHost>,
    pub(crate) sampler: Mutex<Option<Sampler>>,
    pub(crate) started: Instant,
    pub(crate) chats: Mutex<HashMap<ChatId, CancellationToken>>,
    pub(crate) executor_tasks: Mutex<HashMap<TaskId, RunningTaskState>>,
    pub(crate) finished_tasks: Mutex<HashSet<TaskId>>,
    pub(crate) shutdown: CancellationToken,
    pub(crate) tasks: TaskTracker,
    pub(crate) switch: tokio::sync::Mutex<()>,
    pub(crate) local_checks: Arc<LocalChecks>,
    pub(crate) artifact_checks: bool,
    pub(crate) worktrees: Arc<nebula_sandbox::worktree::manager::WorktreeManager>,
    pub(crate) game_mode: Arc<game_mode::GameModeCoordinator>,
    pub(crate) notifications: Arc<notifications::NotificationService>,
    pub(crate) approvals: Arc<nebula_sandbox::approval::ApprovalStore>,
    pub(crate) task_events: tokio::sync::broadcast::Sender<nebula_proto::Event>,
    pub(crate) approval_responses:
        tokio::sync::broadcast::Sender<nebula_proto::ApprovalRespondParams>,
    pub(crate) languages: Arc<language::LanguageRegistry>,
}

impl Shared {
    pub(crate) fn latest_snapshot(&self) -> Option<nebula_proto::ResourceSnapshot> {
        self.sampler
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .and_then(Sampler::latest)
    }

    pub(crate) fn emit_event(&self, event: nebula_proto::Event) {
        let _ = self.task_events.send(event);
    }
}

/// A running daemon.
pub struct Daemon {
    shared: Arc<Shared>,
    accept: tokio::task::JoinHandle<()>,
    _instance: win::InstanceGuard,
}

/// Brings the daemon up: instance lock, sampler, model managers (the startup profile loads
/// in the background), pipe server, then `daemon.ready`. Must run inside a Tokio runtime.
///
/// # Errors
/// [`DaemonError::AlreadyRunning`] for a second instance, or a startup failure.
pub fn start(config: NebulaConfig, deps: Deps) -> Result<Daemon, DaemonError> {
    let instance = win::acquire_instance(&deps.instance_name)
        .map_err(|e| DaemonError::Other(format!("instance mutex: {e}")))?
        .ok_or(DaemonError::AlreadyRunning)?;

    let sampler = match deps.sources {
        Some(sources) => {
            match Sampler::start(sources, SamplerConfig::from_config(&config.resources)) {
                Ok(s) => Some(s),
                Err(e) => {
                    tracing::warn!(event = "daemon.sampler_failed", error = %e);
                    None
                }
            }
        }
        None => None,
    };

    let approvals = Arc::new(nebula_sandbox::approval::ApprovalStore::new());

    // Register the in-process built-in tools on the host now, while it is still exclusively
    // owned (`register_builtin` takes `&mut self`) and the sampler exists, so the built-in
    // `ResourceProvider` reads the same source as `resources.snapshot` (Requirement 6.4).
    let mut tool_host = deps.tool_host;
    builtin_providers::register_builtins(
        &mut tool_host,
        &config,
        sampler.as_ref(),
        Arc::clone(&approvals),
    )?;

    // Construct the per-task worktree manager once and reconcile any worktrees a crash left behind
    // (issue #29, Req 6/9.4). Recovery is spawned, not awaited: `start` runs synchronously on the
    // live Tokio runtime, so blocking on it here would panic. The manager is retained for the
    // future executor (issue #32), which drives create/finalize/remove around each task.
    let worktree_manager =
        builtin_providers::spawn_worktree_recovery(&config, Arc::clone(&approvals));

    let blobs = deps.telemetry.blobs().cloned();
    let chat = ModelManager::spawn_with(
        config.model.clone(),
        deps.supervisor.clone(),
        Arc::clone(&deps.launcher),
        blobs.clone(),
        deps.preflight.clone(),
    )?;
    let embedding = if config.daemon.embedding_profile.is_empty() {
        None
    } else {
        let mut model = config.model.clone();
        model
            .default_profile
            .clone_from(&config.daemon.embedding_profile);
        Some(ModelManager::spawn_with(
            model,
            deps.supervisor,
            deps.launcher,
            blobs,
            deps.preflight,
        )?)
    };

    let detector: Arc<dyn win::ForegroundDetector> = deps
        .foreground_detector
        .unwrap_or_else(|| Arc::new(win::NullForegroundDetector));
    let game_mode = Arc::new(game_mode::GameModeCoordinator::new(
        config.game_mode.clone(),
        detector,
    ));

    let notifications = Arc::new(notifications::NotificationService::new(
        config.notifications.clone(),
        Arc::clone(deps.telemetry.redactor()),
    ));

    let pipe_path = config.daemon.pipe_path();
    let (task_events, _) = tokio::sync::broadcast::channel(256);
    let (approval_responses, _) = tokio::sync::broadcast::channel(32);
    let shared = Arc::new(Shared {
        config,
        telemetry: deps.telemetry,
        chat: chat.clone(),
        embedding,
        tool_host: Arc::new(tool_host),
        sampler: Mutex::new(sampler),
        started: Instant::now(),
        chats: Mutex::new(HashMap::new()),
        executor_tasks: Mutex::new(HashMap::new()),
        finished_tasks: Mutex::new(HashSet::new()),
        shutdown: CancellationToken::new(),
        tasks: TaskTracker::new(),
        switch: tokio::sync::Mutex::new(()),
        local_checks: deps.local_checks,
        artifact_checks: deps.artifact_checks,
        worktrees: worktree_manager,
        game_mode: Arc::clone(&game_mode),
        notifications: Arc::clone(&notifications),
        approvals: Arc::clone(&approvals),
        task_events,
        approval_responses,
        languages: Arc::new(language::LanguageRegistry::with_defaults()),
    });

    // Background monitor for game mode detection and GPU yielding
    game_mode.set_language_registry(Arc::clone(&shared.languages));
    let gm = Arc::clone(&game_mode);
    let chat_gm = chat;
    let shared_for_gm = Arc::clone(&shared);
    tokio::spawn(async move {
        let interval = Duration::from_secs(gm.config.poll_interval_s.max(1));
        let mut ticker = tokio::time::interval(interval);
        while !shared_for_gm.shutdown.is_cancelled() {
            tokio::select! {
                _ = ticker.tick() => {
                    let snapshot = shared_for_gm.latest_snapshot();
                    let procs = snapshot
                        .as_ref()
                        .map(|s| s.gpu_processes.clone())
                        .unwrap_or_default();
                    let gpu_util = snapshot.and_then(|s| s.gpu_util_pct);
                    gm.check_and_update(&chat_gm, &procs, gpu_util).await;
                }
                () = shared_for_gm.shutdown.cancelled() => break,
            }
        }
    });

    // Background monitor for periodic doctor health alerts
    if shared.config.notifications.enabled {
        let notif_service = Arc::clone(&notifications);
        let shared_for_doc = Arc::clone(&shared);
        tokio::spawn(async move {
            let interval = Duration::from_secs(
                shared_for_doc
                    .config
                    .notifications
                    .doctor_alert_interval_mins
                    .max(1)
                    * 60,
            );
            let mut ticker = tokio::time::interval(interval);
            ticker.tick().await; // skip initial immediate tick
            while !shared_for_doc.shutdown.is_cancelled() {
                tokio::select! {
                    _ = ticker.tick() => {
                        if let Ok(report) = server::run_doctor(&shared_for_doc).await {
                            for check in report.checks {
                                if check.status == nebula_proto::CheckStatus::Fail {
                                    notif_service.notify_doctor_fail(&check.name, &check.detail);
                                }
                            }
                        }
                    }
                    () = shared_for_doc.shutdown.cancelled() => break,
                }
            }
        });
    }

    let listener = server::Listener::bind(&pipe_path).map_err(|source| DaemonError::Pipe {
        path: pipe_path.clone(),
        source,
    })?;
    let accept = tokio::spawn(server::accept_loop(Arc::clone(&shared), listener));

    load_startup_profiles(&shared);
    tracing::info!(
        event = "daemon.ready",
        pipe = %pipe_path,
        version = env!("CARGO_PKG_VERSION"),
        pid = std::process::id(),
        load_on_start = %shared.config.daemon.load_on_start,
    );
    Ok(Daemon {
        shared,
        accept,
        _instance: instance,
    })
}

fn load_startup_profiles(shared: &Arc<Shared>) {
    let chat_profile = shared.config.daemon.load_on_start.clone();
    if !chat_profile.is_empty() {
        let s = Arc::clone(shared);
        tokio::spawn(async move {
            let _guard = s.switch.lock().await;
            if let Err(e) = s.chat.set_profile(&chat_profile).await {
                tracing::warn!(event = "daemon.startup_load_failed", profile = %chat_profile, error = %e);
            }
        });
    }
    if let Some(emb) = shared.embedding.clone() {
        let profile = shared.config.daemon.embedding_profile.clone();
        tokio::spawn(async move {
            if let Err(e) = emb.set_profile(&profile).await {
                tracing::warn!(event = "daemon.startup_load_failed", profile = %profile, error = %e);
            }
        });
    }
}

impl Daemon {
    /// Asks the daemon to shut down (same as `daemon.shutdown`).
    pub fn request_shutdown(&self) {
        self.shared.shutdown.cancel();
    }

    /// Reference to the daemon's language module registry.
    #[must_use]
    pub fn languages(&self) -> &Arc<language::LanguageRegistry> {
        &self.shared.languages
    }

    /// Resolves once shutdown has been requested.
    pub async fn shutdown_requested(&self) {
        self.shared.shutdown.cancelled().await;
    }

    /// Shuts down: stop accepting clients, cancel chats, stop both models (waiting for the
    /// processes to exit), stop the sampler, and give connections a moment to flush.
    pub async fn shutdown(self) {
        self.shared.shutdown.cancel();
        let _ = self.accept.await;
        let chats: Vec<CancellationToken> = self
            .shared
            .chats
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .drain()
            .map(|(_, t)| t)
            .collect();
        for t in chats {
            t.cancel();
        }
        let tasks: Vec<CancellationToken> = self
            .shared
            .executor_tasks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .drain()
            .map(|(_, s)| s.cancel)
            .collect();
        for t in tasks {
            t.cancel();
        }
        let chat = self.shared.chat.stop();
        let emb = async {
            if let Some(e) = &self.shared.embedding {
                let _ = e.stop().await;
            }
        };
        let (_, ()) = tokio::join!(chat, emb);
        self.shared.tool_host.stop().await;
        drop(
            self.shared
                .sampler
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take(),
        );
        self.shared.tasks.close();
        let _ = tokio::time::timeout(Duration::from_secs(3), self.shared.tasks.wait()).await;
        tracing::info!(
            event = "daemon.stopped",
            uptime_s = self.shared.started.elapsed().as_secs()
        );
    }
}

/// [`start`], then wait for `daemon.shutdown` or Ctrl-C, then [`Daemon::shutdown`].
///
/// # Errors
/// See [`start`].
pub async fn run(config: NebulaConfig, deps: Deps) -> Result<(), DaemonError> {
    let daemon = start(config, deps)?;
    tokio::select! {
        () = daemon.shutdown_requested() => tracing::info!(event = "daemon.shutdown_requested", by = "client"),
        r = tokio::signal::ctrl_c() => {
            if let Err(e) = r {
                tracing::warn!(event = "daemon.signal_error", error = %e);
            }
            tracing::info!(event = "daemon.shutdown_requested", by = "ctrl_c");
        }
    }
    daemon.shutdown().await;
    Ok(())
}

/// Reads the given Credential Manager targets and registers their values with the
/// redactor. Returns how many were found.
#[must_use]
pub fn register_secrets(telemetry: &Telemetry, targets: &[String]) -> usize {
    let mut found = 0;
    for t in targets {
        match win::read_credential(t) {
            Some(v) if !v.is_empty() => {
                telemetry.redactor().register(&v);
                found += 1;
            }
            _ => tracing::info!(event = "daemon.secret_missing", target = %t),
        }
    }
    found
}
