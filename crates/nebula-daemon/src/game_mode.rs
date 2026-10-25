//! Game mode detection, task pause/resume coordination, and GPU yielding.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use nebula_config::GameModeConfig;
use nebula_model::ModelManager;
use nebula_proto::{GameModeChanged, GameModeStatusResponse, GpuProcess};
use tokio::sync::{broadcast, watch};
use tokio_util::sync::CancellationToken;

use crate::win::{ForegroundDetector, ForegroundWindowInfo};

/// Manages game mode detection, manual override, model unloading, and task pause/resume.
pub struct GameModeCoordinator {
    pub(crate) config: GameModeConfig,
    active_tx: watch::Sender<bool>,
    active_rx: watch::Receiver<bool>,
    manual_override: Mutex<Option<bool>>,
    current_reason: Mutex<Option<String>>,
    current_process: Mutex<Option<String>>,
    saved_profile: Mutex<Option<String>>,
    cooldown_deadline: Mutex<Option<Instant>>,
    events_tx: broadcast::Sender<GameModeChanged>,
    detector: Arc<dyn ForegroundDetector>,
    languages: Mutex<Option<Arc<crate::language::LanguageRegistry>>>,
}

impl GameModeCoordinator {
    /// Creates a new coordinator.
    #[must_use]
    pub fn new(config: GameModeConfig, detector: Arc<dyn ForegroundDetector>) -> Self {
        let (active_tx, active_rx) = watch::channel(false);
        let (events_tx, _) = broadcast::channel(32);
        Self {
            config,
            active_tx,
            active_rx,
            manual_override: Mutex::new(None),
            current_reason: Mutex::new(None),
            current_process: Mutex::new(None),
            saved_profile: Mutex::new(None),
            cooldown_deadline: Mutex::new(None),
            events_tx,
            detector,
            languages: Mutex::new(None),
        }
    }

    /// Attaches the language registry for coordinating pause/resume hooks (Issue #35).
    pub fn set_language_registry(&self, registry: Arc<crate::language::LanguageRegistry>) {
        *self
            .languages
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(registry);
    }

    /// Subscribes to game mode change events.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<GameModeChanged> {
        self.events_tx.subscribe()
    }

    /// Whether game mode is currently active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        *self.active_rx.borrow()
    }

    /// Current status snapshot.
    #[must_use]
    pub fn status(&self) -> GameModeStatusResponse {
        GameModeStatusResponse {
            active: self.is_active(),
            manual_override: *self
                .manual_override
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
            reason: self
                .current_reason
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone(),
            process_name: self
                .current_process
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone(),
        }
    }

    /// Waits until game mode is inactive, or returns an error if cancelled.
    pub async fn wait_if_active(&self, cancel: &CancellationToken) -> Result<(), String> {
        if !self.is_active() {
            return Ok(());
        }
        let mut rx = self.active_rx.clone();
        while *rx.borrow() {
            tokio::select! {
                () = cancel.cancelled() => return Err("task cancelled while waiting for game mode to exit".into()),
                res = rx.changed() => {
                    if res.is_err() {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    /// Manually sets game mode on or off.
    pub async fn set_manual(
        &self,
        chat: &ModelManager,
        enabled: bool,
    ) -> Result<GameModeStatusResponse, String> {
        *self
            .manual_override
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(enabled);
        if enabled {
            self.activate(chat, "manual override".into(), None).await;
        } else {
            self.deactivate(chat, "manual override cleared".into())
                .await;
        }
        Ok(self.status())
    }

    /// Evaluates current system conditions (foreground window, GPU processes, and compute utilization).
    pub async fn check_and_update(
        &self,
        chat: &ModelManager,
        gpu_processes: &[GpuProcess],
        gpu_util_pct: Option<u8>,
    ) {
        if !self.config.enabled {
            return;
        }

        // 1. If manual override is set, it takes precedence.
        let manual = *self
            .manual_override
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(forced) = manual {
            if forced && !self.is_active() {
                self.activate(chat, "manual override".into(), None).await;
            } else if !forced && self.is_active() {
                self.deactivate(chat, "manual override cleared".into())
                    .await;
            }
            return;
        }

        // 2. Evaluate auto-detection triggers.
        let detected = self.detect_game_condition(gpu_processes, gpu_util_pct);

        if let Some((reason, proc_name)) = detected {
            // Reset cooldown because game is actively running
            *self
                .cooldown_deadline
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = None;
            if !self.is_active() {
                self.activate(chat, reason, Some(proc_name)).await;
            }
        } else if self.is_active() {
            // Game condition cleared; check or start cooldown
            let should_deactivate = {
                let mut deadline_lock = self
                    .cooldown_deadline
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                if let Some(deadline) = *deadline_lock {
                    if Instant::now() >= deadline {
                        *deadline_lock = None;
                        true
                    } else {
                        false
                    }
                } else {
                    *deadline_lock =
                        Some(Instant::now() + Duration::from_secs(self.config.cooldown_secs));
                    false
                }
            };
            if should_deactivate {
                self.deactivate(chat, "game exited".into()).await;
            }
        }
    }

    fn detect_game_condition(
        &self,
        gpu_processes: &[GpuProcess],
        gpu_util_pct: Option<u8>,
    ) -> Option<(String, String)> {
        // A. Fullscreen foreground window
        let fg_opt = self.detector.detect();
        if let Some(ref fg) = fg_opt
            && fg.is_fullscreen
        {
            let proc_name = get_process_name(fg.pid, gpu_processes);
            let name_to_check = if proc_name.is_empty() {
                "unknown".to_string()
            } else {
                proc_name
            };
            if !self.is_allowlisted(&name_to_check) {
                return Some((
                    format!("fullscreen window ({name_to_check})"),
                    name_to_check,
                ));
            }
        }

        // B. Known game processes
        for gp in gpu_processes {
            if self
                .config
                .known_processes
                .iter()
                .any(|k| k.eq_ignore_ascii_case(&gp.name))
            {
                return Some((
                    format!("game process running ({})", gp.name),
                    gp.name.clone(),
                ));
            }
        }

        // C. High VRAM consumption correlated with GPU compute utilization (> 15%) or foreground focus
        for gp in gpu_processes {
            if gp.vram_mib >= self.config.vram_threshold_mib && !self.is_allowlisted(&gp.name) {
                let is_foreground = fg_opt.as_ref().is_some_and(|fg| {
                    if fg.pid == gp.pid {
                        return true;
                    }
                    let fg_name = get_process_name(fg.pid, gpu_processes);
                    !fg_name.is_empty() && fg_name.eq_ignore_ascii_case(&gp.name)
                });
                let is_compute_active = gpu_util_pct.unwrap_or(0) > 15;

                if is_foreground || is_compute_active {
                    let reason_detail = if is_foreground {
                        "foreground focus"
                    } else {
                        "active compute"
                    };
                    return Some((
                        format!(
                            "high VRAM usage ({} MiB by {}, correlated with {})",
                            gp.vram_mib, gp.name, reason_detail
                        ),
                        gp.name.clone(),
                    ));
                }
            }
        }

        None
    }

    fn is_allowlisted(&self, name: &str) -> bool {
        // Nebula's own processes are never games
        if name.eq_ignore_ascii_case("llama-server.exe")
            || name.eq_ignore_ascii_case("llama-server")
            || name.eq_ignore_ascii_case("hydrogen_engine.exe")
            || name.eq_ignore_ascii_case("hydrogen_engine")
            || name.eq_ignore_ascii_case("nebula-daemon.exe")
            || name.eq_ignore_ascii_case("nebula-daemon")
            || name.eq_ignore_ascii_case("nebula.exe")
            || name.eq_ignore_ascii_case("nebula")
        {
            return true;
        }

        self.config
            .desktop_allowlist
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(name))
    }

    async fn activate(&self, chat: &ModelManager, reason: String, process: Option<String>) {
        let current_profile = chat.status().profile;
        if !current_profile.is_empty() {
            *self
                .saved_profile
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Some(current_profile);
        }
        *self
            .current_reason
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(reason.clone());
        *self
            .current_process
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = process;

        // Notify registered language modules to pause MCP sessions (Issue #35, #73)
        let langs = self
            .languages
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(registry) = langs {
            registry.on_pause().await;
        }

        // Unload model to release VRAM before marking game mode active
        let _ = chat.unload().await;

        let _ = self.active_tx.send(true);

        tracing::info!(
            event = "game_mode.entered",
            reason = %reason,
        );
        let _ = self.events_tx.send(GameModeChanged {
            active: true,
            reason: Some(reason),
        });
    }

    async fn deactivate(&self, chat: &ModelManager, reason: String) {
        *self
            .current_reason
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = None;
        *self
            .current_process
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = None;

        let saved = self
            .saved_profile
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();

        // Reload the previous model profile if one was saved
        if let Some(profile) = saved {
            let _ = chat.set_profile(&profile).await;
        }

        // Notify registered language modules that inference has resumed (Issue #35, #73)
        let langs = self
            .languages
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(registry) = langs {
            registry.on_resume().await;
        }

        let _ = self.active_tx.send(false);

        tracing::info!(
            event = "game_mode.exited",
            reason = %reason,
        );
        let _ = self.events_tx.send(GameModeChanged {
            active: false,
            reason: Some(reason),
        });
    }
}

fn get_process_name(pid: u32, gpu_processes: &[GpuProcess]) -> String {
    if let Some(gp) = gpu_processes.iter().find(|p| p.pid == pid) {
        return gp.name.clone();
    }
    let mut sys = sysinfo::System::new();
    let sys_pid = sysinfo::Pid::from_u32(pid);
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[sys_pid]), true);
    sys.process(sys_pid)
        .map(|p| p.name().to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Mock detector for unit and integration tests.
#[derive(Clone, Default)]
pub struct MockForegroundDetector {
    info: Arc<Mutex<Option<ForegroundWindowInfo>>>,
}

impl MockForegroundDetector {
    /// Creates a new mock detector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the mock foreground window information.
    pub fn set(&self, info: Option<ForegroundWindowInfo>) {
        *self.info.lock().unwrap_or_else(PoisonError::into_inner) = info;
    }
}

impl ForegroundDetector for MockForegroundDetector {
    fn detect(&self) -> Option<ForegroundWindowInfo> {
        *self.info.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nebula_proto::ModelState;
    use nebula_telemetry::Telemetry;

    fn test_telemetry() -> Telemetry {
        use std::sync::OnceLock;
        static T: OnceLock<Telemetry> = OnceLock::new();
        T.get_or_init(|| {
            nebula_telemetry::init(&nebula_telemetry::TelemetryConfig::default()).unwrap()
        })
        .clone()
    }

    fn test_supervisor() -> nebula_model::SupervisorConfig {
        nebula_model::SupervisorConfig {
            health_interval: Duration::from_millis(50),
            startup_timeout: Duration::from_secs(5),
            hang_threshold: 3,
            backoff_base: Duration::from_millis(50),
            backoff_max: Duration::from_millis(200),
            failure_window: Duration::from_secs(60),
            max_failures: 5,
            drain_timeout: Duration::from_millis(500),
        }
    }

    #[tokio::test]
    async fn test_game_mode_trips_on_fullscreen_game() {
        let mock_det = Arc::new(MockForegroundDetector::new());
        mock_det.set(Some(ForegroundWindowInfo {
            pid: 4321,
            is_fullscreen: true,
        }));

        let config = GameModeConfig {
            enabled: true,
            cooldown_secs: 0,
            ..Default::default()
        };
        let coord = GameModeCoordinator::new(config, mock_det);

        let fake_procs = vec![GpuProcess {
            pid: 4321,
            name: "Cyberpunk2077.exe".into(),
            vram_mib: 4000,
        }];

        let launcher =
            nebula_model::testing::FakeLauncher::new(nebula_model::testing::FakeState::new());
        let blobs = test_telemetry().blobs().cloned();
        let chat = ModelManager::spawn_with(
            nebula_model::testing::config(),
            test_supervisor(),
            Arc::clone(&launcher) as _,
            blobs,
            None,
        )
        .unwrap();

        assert!(!coord.is_active());
        coord.check_and_update(&chat, &fake_procs, None).await;
        assert!(coord.is_active());
        assert_eq!(chat.status().state, ModelState::Unloaded);

        let status = coord.status();
        assert!(status.active);
        assert!(status.reason.unwrap().contains("fullscreen window"));
    }

    #[tokio::test]
    async fn test_game_mode_ignores_fullscreen_allowlisted_app() {
        let mock_det = Arc::new(MockForegroundDetector::new());
        mock_det.set(Some(ForegroundWindowInfo {
            pid: 1111,
            is_fullscreen: true,
        }));

        let config = GameModeConfig {
            enabled: true,
            ..Default::default()
        };
        let coord = GameModeCoordinator::new(config, mock_det);

        let fake_procs = vec![GpuProcess {
            pid: 1111,
            name: "firefox.exe".into(),
            vram_mib: 1200,
        }];

        let launcher =
            nebula_model::testing::FakeLauncher::new(nebula_model::testing::FakeState::new());
        let blobs = test_telemetry().blobs().cloned();
        let chat = ModelManager::spawn_with(
            nebula_model::testing::config(),
            test_supervisor(),
            Arc::clone(&launcher) as _,
            blobs,
            None,
        )
        .unwrap();

        coord.check_and_update(&chat, &fake_procs, None).await;
        assert!(!coord.is_active());
        assert_ne!(chat.status().state, ModelState::Unloaded);
    }

    #[tokio::test]
    async fn test_game_mode_ignores_dwm_compositor_allocation() {
        let mock_det = Arc::new(MockForegroundDetector::new());
        mock_det.set(None);

        let config = GameModeConfig {
            enabled: true,
            vram_threshold_mib: 2048,
            cooldown_secs: 0,
            ..Default::default()
        };
        let coord = GameModeCoordinator::new(config, mock_det);

        let fake_procs = vec![GpuProcess {
            pid: 100,
            name: "dwm.exe".into(),
            vram_mib: 3200, // 4K multi-monitor desktop composition buffer
        }];

        let launcher =
            nebula_model::testing::FakeLauncher::new(nebula_model::testing::FakeState::new());
        let blobs = test_telemetry().blobs().cloned();
        let chat = ModelManager::spawn_with(
            nebula_model::testing::config(),
            test_supervisor(),
            Arc::clone(&launcher) as _,
            blobs,
            None,
        )
        .unwrap();

        coord.check_and_update(&chat, &fake_procs, Some(45)).await;
        assert!(!coord.is_active());
        assert_ne!(chat.status().state, ModelState::Unloaded);
    }

    #[tokio::test]
    async fn test_game_mode_ignores_idle_tray_launcher() {
        let mock_det = Arc::new(MockForegroundDetector::new());
        mock_det.set(None);

        let config = GameModeConfig {
            enabled: true,
            vram_threshold_mib: 2048,
            cooldown_secs: 0,
            ..Default::default()
        };
        let coord = GameModeCoordinator::new(config, mock_det);

        let fake_procs = vec![GpuProcess {
            pid: 500,
            name: "steam.exe".into(),
            vram_mib: 450,
        }];

        let launcher =
            nebula_model::testing::FakeLauncher::new(nebula_model::testing::FakeState::new());
        let blobs = test_telemetry().blobs().cloned();
        let chat = ModelManager::spawn_with(
            nebula_model::testing::config(),
            test_supervisor(),
            Arc::clone(&launcher) as _,
            blobs,
            None,
        )
        .unwrap();

        coord.check_and_update(&chat, &fake_procs, Some(0)).await;
        assert!(!coord.is_active());
        assert_ne!(chat.status().state, ModelState::Unloaded);
    }

    #[tokio::test]
    async fn test_game_mode_ignores_background_vram_without_compute_or_focus() {
        let mock_det = Arc::new(MockForegroundDetector::new());
        mock_det.set(None); // Not in foreground

        let config = GameModeConfig {
            enabled: true,
            vram_threshold_mib: 2048,
            cooldown_secs: 0,
            ..Default::default()
        };
        let coord = GameModeCoordinator::new(config, mock_det);

        let fake_procs = vec![GpuProcess {
            pid: 9999,
            name: "creative_app.exe".into(),
            vram_mib: 3500,
        }];

        let launcher =
            nebula_model::testing::FakeLauncher::new(nebula_model::testing::FakeState::new());
        let blobs = test_telemetry().blobs().cloned();
        let chat = ModelManager::spawn_with(
            nebula_model::testing::config(),
            test_supervisor(),
            Arc::clone(&launcher) as _,
            blobs,
            None,
        )
        .unwrap();

        // 5% GPU compute utilization (<= 15%) in background does NOT trip game mode
        coord.check_and_update(&chat, &fake_procs, Some(5)).await;
        assert!(!coord.is_active());
        assert_ne!(chat.status().state, ModelState::Unloaded);
    }

    #[tokio::test]
    async fn test_game_mode_trips_on_high_vram_correlated_with_compute() {
        let mock_det = Arc::new(MockForegroundDetector::new());
        mock_det.set(None);

        let config = GameModeConfig {
            enabled: true,
            vram_threshold_mib: 2048,
            cooldown_secs: 0,
            ..Default::default()
        };
        let coord = GameModeCoordinator::new(config, mock_det);

        let fake_procs = vec![GpuProcess {
            pid: 9999,
            name: "heavy_render.exe".into(),
            vram_mib: 3500,
        }];

        let launcher =
            nebula_model::testing::FakeLauncher::new(nebula_model::testing::FakeState::new());
        let blobs = test_telemetry().blobs().cloned();
        let chat = ModelManager::spawn_with(
            nebula_model::testing::config(),
            test_supervisor(),
            Arc::clone(&launcher) as _,
            blobs,
            None,
        )
        .unwrap();

        // 35% GPU compute utilization (> 15%) trips game mode
        coord.check_and_update(&chat, &fake_procs, Some(35)).await;
        assert!(coord.is_active());
        assert_eq!(chat.status().state, ModelState::Unloaded);
    }
}
