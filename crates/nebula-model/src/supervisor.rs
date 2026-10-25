//! Keeps one llama-server running: launch, health loop, restart with backoff, profile
//! switches, stop and unload.

use std::collections::VecDeque;
use std::net::{Ipv4Addr, TcpListener};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nebula_proto::{ModelState, ModelStateChanged, ModelStatus};
use nebula_telemetry::BlobStore;
use time::OffsetDateTime;
use tokio::sync::{broadcast, mpsc, oneshot, watch};

use crate::ModelError;
use crate::backend::{Activity, LlamaServerBackend, ModelBackend};
use crate::config::{ModelConfig, ModelProfile};
use crate::launcher::{LaunchSpec, Launcher, ServerProcess};
use crate::types::BackendHealth;

/// Timing and retry policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SupervisorConfig {
    /// How often `/health` is polled.
    pub health_interval: Duration,
    /// How long a launch may take to become healthy (model load).
    pub startup_timeout: Duration,
    /// Consecutive failed health checks on a ready server before it is restarted.
    pub hang_threshold: u32,
    /// First restart delay; doubles per consecutive failure.
    pub backoff_base: Duration,
    /// Restart delay cap.
    pub backoff_max: Duration,
    /// Window for counting failures.
    pub failure_window: Duration,
    /// Failures within the window that put the server in `Failed`.
    pub max_failures: usize,
    /// How long a profile switch or stop waits for in-flight requests before cancelling them.
    pub drain_timeout: Duration,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            health_interval: Duration::from_secs(2),
            startup_timeout: Duration::from_secs(600),
            hang_threshold: 3,
            backoff_base: Duration::from_secs(1),
            backoff_max: Duration::from_secs(60),
            failure_window: Duration::from_secs(600),
            max_failures: 5,
            drain_timeout: Duration::from_secs(30),
        }
    }
}

/// A check run before each launch. Returning `Err(reason)` refuses the launch.
pub trait Preflight: Send + Sync {
    /// Decides whether `profile` may be launched now.
    ///
    /// # Errors
    /// A human-readable reason for refusing.
    fn check(&self, name: &str, profile: &ModelProfile) -> Result<(), String>;
}

type Reply = oneshot::Sender<Result<ModelStatus, ModelError>>;

enum Command {
    SetProfile(String, Reply),
    Park(ModelState, oneshot::Sender<ModelStatus>),
}

/// Handle to a supervised model server. Cheap to clone; the server stops when the last
/// handle is dropped.
#[derive(Clone, Debug)]
pub struct ModelManager {
    cmd: mpsc::Sender<Command>,
    status: watch::Receiver<ModelStatus>,
    backend: watch::Receiver<Option<Arc<dyn ModelBackend>>>,
    pid: watch::Receiver<Option<u32>>,
    events: broadcast::Sender<ModelStateChanged>,
}

impl ModelManager {
    /// Starts the supervisor task in `Stopped`. Call [`ModelManager::set_profile`] to load a
    /// model. Must be called inside a Tokio runtime.
    ///
    /// # Errors
    /// [`ModelError::Config`] / [`ModelError::UnknownProfile`] if the config is inconsistent.
    pub fn spawn(
        config: ModelConfig,
        policy: SupervisorConfig,
        launcher: Arc<dyn Launcher>,
        blobs: Option<BlobStore>,
    ) -> Result<Self, ModelError> {
        Self::spawn_with(config, policy, launcher, blobs, None)
    }

    /// Like [`ModelManager::spawn`], with a check that runs before every launch (e.g.
    /// commit-charge headroom). A refused launch leaves the server `Stopped` with the reason
    /// in `last_error`; it is not retried.
    ///
    /// # Errors
    /// See [`ModelManager::spawn`].
    pub fn spawn_with(
        config: ModelConfig,
        policy: SupervisorConfig,
        launcher: Arc<dyn Launcher>,
        blobs: Option<BlobStore>,
        preflight: Option<Arc<dyn Preflight>>,
    ) -> Result<Self, ModelError> {
        config.validate()?;
        let (cmd_tx, cmd_rx) = mpsc::channel(16);
        let initial = ModelStatus {
            profile: config.default_profile.clone(),
            state: ModelState::Stopped,
            since: OffsetDateTime::now_utc(),
            restarts: 0,
            last_error: None,
        };
        let (status_tx, status_rx) = watch::channel(initial);
        let (backend_tx, backend_rx) = watch::channel(None);
        let (pid_tx, pid_rx) = watch::channel(None);
        let (events, _) = broadcast::channel(64);
        let actor = Actor {
            profile: config.default_profile.clone(),
            config,
            policy,
            launcher,
            blobs,
            preflight,
            state: ModelState::Stopped,
            since: OffsetDateTime::now_utc(),
            last_error: None,
            proc: None,
            backend: None,
            started_at: Instant::now(),
            health_fails: 0,
            failures: VecDeque::new(),
            attempt: 0,
            restart_at: None,
            waiters: Vec::new(),
            status_tx,
            backend_tx,
            pid_tx,
            events: events.clone(),
        };
        tokio::spawn(actor.run(cmd_rx));
        Ok(Self {
            cmd: cmd_tx,
            status: status_rx,
            backend: backend_rx,
            pid: pid_rx,
            events,
        })
    }

    /// Current status.
    #[must_use]
    pub fn status(&self) -> ModelStatus {
        self.status.borrow().clone()
    }

    /// OS process id of the running server, for per-process GPU and memory accounting.
    #[must_use]
    pub fn pid(&self) -> Option<u32> {
        *self.pid.borrow()
    }

    /// The backend, if the server is `Ready` or `Busy`.
    ///
    /// # Errors
    /// [`ModelError::Unavailable`] in any other state.
    pub fn backend(&self) -> Result<Arc<dyn ModelBackend>, ModelError> {
        self.backend.borrow().clone().ok_or_else(|| {
            let s = self.status();
            ModelError::Unavailable(format!("model {} is {:?}", s.profile, s.state))
        })
    }

    /// State-change events.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<ModelStateChanged> {
        self.events.subscribe()
    }

    /// A watch on the status, for waiting on transitions.
    #[must_use]
    pub fn watch(&self) -> watch::Receiver<ModelStatus> {
        self.status.clone()
    }

    /// Loads `profile` (switching from the current one if needed) and waits until it is
    /// `Ready`, or fails.
    ///
    /// # Errors
    /// [`ModelError::UnknownProfile`], [`ModelError::Failed`] if it can't be kept running, or
    /// [`ModelError::Unavailable`] if a later switch or stop superseded this one.
    pub async fn set_profile(&self, profile: &str) -> Result<ModelStatus, ModelError> {
        let (tx, rx) = oneshot::channel();
        self.cmd
            .send(Command::SetProfile(profile.to_owned(), tx))
            .await
            .map_err(|_| ModelError::Unavailable("supervisor stopped".into()))?;
        rx.await
            .map_err(|_| ModelError::Unavailable("supervisor stopped".into()))?
    }

    /// Stops the server (draining in-flight requests first).
    ///
    /// # Errors
    /// [`ModelError::Unavailable`] if the supervisor task is gone.
    pub async fn stop(&self) -> Result<ModelStatus, ModelError> {
        self.park(ModelState::Stopped).await
    }

    /// Unloads the server to free VRAM (game mode).
    ///
    /// # Errors
    /// [`ModelError::Unavailable`] if the supervisor task is gone.
    pub async fn unload(&self) -> Result<ModelStatus, ModelError> {
        self.park(ModelState::Unloaded).await
    }

    async fn park(&self, state: ModelState) -> Result<ModelStatus, ModelError> {
        let (tx, rx) = oneshot::channel();
        self.cmd
            .send(Command::Park(state, tx))
            .await
            .map_err(|_| ModelError::Unavailable("supervisor stopped".into()))?;
        rx.await
            .map_err(|_| ModelError::Unavailable("supervisor stopped".into()))
    }
}

fn free_port() -> Result<u16, ModelError> {
    let l = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .map_err(|e| ModelError::Launch(format!("no free port: {e}")))?;
    l.local_addr()
        .map(|a| a.port())
        .map_err(|e| ModelError::Launch(format!("no free port: {e}")))
}

fn api_key() -> Result<String, ModelError> {
    let mut buf = [0u8; 32];
    getrandom::fill(&mut buf).map_err(|e| ModelError::Launch(format!("no randomness: {e}")))?;
    Ok(hex::encode(buf))
}

struct Actor {
    config: ModelConfig,
    policy: SupervisorConfig,
    launcher: Arc<dyn Launcher>,
    blobs: Option<BlobStore>,
    preflight: Option<Arc<dyn Preflight>>,
    profile: String,
    state: ModelState,
    since: OffsetDateTime,
    last_error: Option<String>,
    proc: Option<Box<dyn ServerProcess>>,
    backend: Option<Arc<dyn ModelBackend>>,
    started_at: Instant,
    health_fails: u32,
    failures: VecDeque<Instant>,
    attempt: u32,
    restart_at: Option<Instant>,
    waiters: Vec<Reply>,
    status_tx: watch::Sender<ModelStatus>,
    backend_tx: watch::Sender<Option<Arc<dyn ModelBackend>>>,
    pid_tx: watch::Sender<Option<u32>>,
    events: broadcast::Sender<ModelStateChanged>,
}

impl Actor {
    async fn run(mut self, mut rx: mpsc::Receiver<Command>) {
        let mut tick = tokio::time::interval(self.policy.health_interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                cmd = rx.recv() => match cmd {
                    Some(Command::SetProfile(name, reply)) => self.switch(name, reply).await,
                    Some(Command::Park(state, reply)) => {
                        self.park(state).await;
                        let _ = reply.send(self.status());
                    }
                    None => break,
                },
                _ = tick.tick() => self.on_tick().await,
            }
        }
        self.teardown().await;
    }

    fn status(&self) -> ModelStatus {
        ModelStatus {
            profile: self.profile.clone(),
            state: self.state,
            since: self.since,
            restarts: u32::try_from(self.failures.len()).unwrap_or(u32::MAX),
            last_error: self.last_error.clone(),
        }
    }

    fn set_state(&mut self, to: ModelState, reason: Option<String>) {
        let from = self.state;
        if from == to {
            // Restart count or last error may still have changed.
            self.status_tx.send_replace(self.status());
            return;
        }
        self.state = to;
        self.since = OffsetDateTime::now_utc();
        tracing::info!(
            event = "model.state_changed",
            profile = %self.profile,
            from = ?from,
            to = ?to,
            reason = reason.as_deref(),
        );
        self.status_tx.send_replace(self.status());
        let _ = self.events.send(ModelStateChanged {
            profile: self.profile.clone(),
            from,
            to,
            reason,
        });
    }

    fn resolve_waiters(&mut self, result: &Result<ModelStatus, ModelError>) {
        for w in self.waiters.drain(..) {
            let r = match result {
                Ok(s) => Ok(s.clone()),
                Err(e) => Err(e.clone()),
            };
            let _ = w.send(r);
        }
    }

    async fn switch(&mut self, name: String, reply: Reply) {
        if let Err(e) = self.config.profile(&name) {
            let _ = reply.send(Err(e));
            return;
        }
        if name == self.profile && matches!(self.state, ModelState::Ready | ModelState::Busy) {
            let _ = reply.send(Ok(self.status()));
            return;
        }
        if name != self.profile {
            tracing::info!(event = "model.profile_switch", from = %self.profile, to = %name);
            let superseded = Err(ModelError::Unavailable(format!("superseded by {name}")));
            self.resolve_waiters(&superseded);
        }
        self.drain().await;
        self.teardown().await;
        self.profile = name;
        self.failures.clear();
        self.attempt = 0;
        self.last_error = None;
        self.waiters.push(reply);
        self.launch().await;
    }

    async fn park(&mut self, state: ModelState) {
        self.set_state(state, None);
        self.drain().await;
        self.teardown().await;
        self.restart_at = None;
        let parked = Err(ModelError::Unavailable(format!("model {state:?}")));
        self.resolve_waiters(&parked);
    }

    /// Waits for in-flight requests up to the drain timeout, then cancels the rest.
    async fn drain(&mut self) {
        let Some(backend) = &self.backend else { return };
        let activity = backend.activity().clone();
        let deadline = Instant::now() + self.policy.drain_timeout;
        while activity.in_flight() > 0 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        if activity.in_flight() > 0 {
            tracing::warn!(
                event = "model.drain_timeout",
                in_flight = activity.in_flight(),
                profile = %self.profile,
            );
        }
        activity.cancel_all();
    }

    async fn teardown(&mut self) {
        self.backend_tx.send_replace(None);
        if let Some(b) = self.backend.take() {
            b.activity().cancel_all();
        }
        if let Some(mut p) = self.proc.take() {
            p.stop().await;
        }
        self.pid_tx.send_replace(None);
    }

    async fn launch(&mut self) {
        self.restart_at = None;
        if let Err(reason) = self.run_preflight() {
            tracing::warn!(event = "model.preflight_refused", profile = %self.profile, reason = %reason);
            self.last_error = Some(reason.clone());
            let refused = Err(ModelError::Preflight(reason.clone()));
            self.resolve_waiters(&refused);
            self.set_state(ModelState::Stopped, Some(reason));
            return;
        }
        match self.try_launch().await {
            Ok(()) => {
                self.started_at = Instant::now();
                self.health_fails = 0;
                self.set_state(ModelState::Starting, None);
            }
            Err(e) => self.fail(format!("launch failed: {e}")).await,
        }
    }

    fn run_preflight(&self) -> Result<(), String> {
        let Some(preflight) = &self.preflight else {
            return Ok(());
        };
        let profile = self
            .config
            .profile(&self.profile)
            .map_err(|e| e.to_string())?;
        preflight.check(&self.profile, profile)
    }

    async fn try_launch(&mut self) -> Result<(), ModelError> {
        let profile = self.config.profile(&self.profile)?.clone();
        let program = self.config.runtime_for(&profile)?.clone();
        let port = free_port()?;
        let key = api_key()?;
        let mut env = profile.env();
        env.push(("LLAMA_API_KEY".into(), key.clone()));
        let spec = LaunchSpec {
            program,
            args: profile.args(port),
            env,
            port,
        };
        let proc = self.launcher.launch(spec).await?;
        let is_hydrogen = match profile.backend {
            Some(crate::config::ModelBackendKind::Hydrogen) => true,
            Some(crate::config::ModelBackendKind::LlamaServer) => false,
            None => self.config.backend == crate::config::ModelBackendKind::Hydrogen,
        };
        let backend: Arc<dyn ModelBackend> = if is_hydrogen {
            tracing::info!(event = "model.launch_hydrogen", profile = %self.profile, pid = proc.pid());
            let b = crate::hydrogen::HydrogenBackend::open(true)?;
            Arc::new(b)
        } else {
            tracing::info!(event = "model.launch", profile = %self.profile, port, pid = proc.pid());
            let b = LlamaServerBackend::new(
                format!("http://127.0.0.1:{port}"),
                Some(key),
                self.profile.clone(),
                profile,
                self.blobs.clone(),
                Activity::new(),
            )?;
            Arc::new(b)
        };
        self.pid_tx.send_replace(proc.pid());
        self.proc = Some(proc);
        self.backend = Some(backend);
        Ok(())
    }

    async fn on_tick(&mut self) {
        match self.state {
            ModelState::Restarting => {
                if self.restart_at.is_some_and(|t| Instant::now() >= t) {
                    self.launch().await;
                }
            }
            ModelState::Starting | ModelState::Ready | ModelState::Busy => self.check().await,
            ModelState::Stopped | ModelState::Failed | ModelState::Unloaded => {}
        }
    }

    async fn check(&mut self) {
        let Some(proc) = &mut self.proc else { return };
        if let Some(exit) = proc.try_exit() {
            self.fail(format!("llama-server exited ({exit})")).await;
            return;
        }
        let Some(backend) = self.backend.clone() else {
            return;
        };
        let health = backend.health().await;
        let starting = self.state == ModelState::Starting;
        match health {
            Ok(BackendHealth::Ok) => {
                self.health_fails = 0;
                if starting {
                    self.attempt = 0;
                    self.backend_tx.send_replace(Some(Arc::clone(&backend)));
                    self.set_state(ModelState::Ready, None);
                    let ok = Ok(self.status());
                    self.resolve_waiters(&ok);
                } else {
                    let busy = backend.activity().in_flight() > 0;
                    self.set_state(
                        if busy {
                            ModelState::Busy
                        } else {
                            ModelState::Ready
                        },
                        None,
                    );
                }
            }
            _ if starting => {
                if self.started_at.elapsed() > self.policy.startup_timeout {
                    let secs = self.policy.startup_timeout.as_secs();
                    self.fail(format!("not healthy after {secs} s")).await;
                }
            }
            other => {
                self.health_fails += 1;
                if self.health_fails >= self.policy.hang_threshold {
                    let why = match other {
                        Ok(_) => "reports loading".to_owned(),
                        Err(e) => e.to_string(),
                    };
                    self.fail(format!("unresponsive: {why}")).await;
                }
            }
        }
    }

    async fn fail(&mut self, reason: String) {
        let output = self
            .proc
            .as_ref()
            .map(|p| p.recent_output().join("\n"))
            .unwrap_or_default();
        self.teardown().await;
        let now = Instant::now();
        self.failures.push_back(now);
        while self
            .failures
            .front()
            .is_some_and(|t| now.duration_since(*t) > self.policy.failure_window)
        {
            self.failures.pop_front();
        }
        let output_blob = self
            .blobs
            .as_ref()
            .filter(|_| !output.is_empty())
            .and_then(|b| b.put(output.as_bytes()).ok())
            .map(|r| r.to_string());
        tracing::warn!(
            event = "model.failure",
            profile = %self.profile,
            reason = %reason,
            failures = self.failures.len(),
            output_blob = output_blob.as_deref(),
        );
        self.last_error = Some(reason.clone());
        if self.failures.len() >= self.policy.max_failures {
            self.restart_at = None;
            self.set_state(ModelState::Failed, Some(reason.clone()));
            let failed = Err(ModelError::Failed(reason));
            self.resolve_waiters(&failed);
        } else {
            let factor = 1u32.checked_shl(self.attempt).unwrap_or(u32::MAX);
            let delay = self
                .policy
                .backoff_base
                .saturating_mul(factor)
                .min(self.policy.backoff_max);
            self.attempt = self.attempt.saturating_add(1);
            self.restart_at = Some(now + delay);
            self.set_state(ModelState::Restarting, Some(reason));
        }
    }
}
