//! Typed Nebula configuration.
//!
//! The shipped defaults (`config/default.toml`) are embedded in the binary. A local override
//! file, if present, is merged over them table by table: tables merge recursively, and any
//! other value (including arrays) replaces the default. Unknown keys are an error.

use std::path::{Path, PathBuf};

use nebula_model::ModelConfig;
use nebula_sandbox::worktree::config::WorktreeConfig;
use nebula_telemetry::TelemetryConfig;
use nebula_tools::ToolHostConfig;
use serde::Deserialize;

/// The shipped defaults.
pub const DEFAULT_TOML: &str = include_str!("../../../config/default.toml");

/// Environment variable naming an override file, instead of [`DEFAULT_OVERRIDE_PATH`].
pub const OVERRIDE_ENV: &str = "NEBULA_CONFIG";

/// Where the local override lives unless [`OVERRIDE_ENV`] says otherwise.
pub const DEFAULT_OVERRIDE_PATH: &str = r"F:\Nebula\config\nebula.toml";

/// Config loading failures.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// A file could not be read.
    #[error("reading {path}: {source}")]
    Io {
        /// File.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
    /// TOML syntax or a type/unknown-key error.
    #[error("{origin}: {message}")]
    Parse {
        /// Which document failed.
        origin: String,
        /// Parser message.
        message: String,
    },
    /// The config parsed but is inconsistent or unsafe.
    #[error("invalid config: {0}")]
    Invalid(String),
}

/// Where Nebula keeps its data.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Paths {
    /// Root of hot data, e.g. `F:\Nebula`.
    pub data_root: PathBuf,
    /// Daily JSONL logs and `blobs\`.
    pub logs: PathBuf,
    /// Daemon state (doctor history, hash cache, SMART JSON, ...).
    pub state: PathBuf,
    /// Local config: the override file and `rclone.conf`. Backed up with `state`.
    pub config: PathBuf,
    /// Log archive on the cold drive.
    pub archive: PathBuf,
    /// Local copies of backups (PHASE0_PLAN 7.1).
    pub backups_local: PathBuf,
}

/// The `[daemon]` section.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonConfig {
    /// Pipe name; the full path is `\\.\pipe\<name>`.
    pub pipe_name: String,
    /// Chat profile loaded at startup; empty for none.
    #[serde(default)]
    pub load_on_start: String,
    /// Embedding profile run next to the chat model; empty to disable.
    #[serde(default)]
    pub embedding_profile: String,
    /// Credential Manager targets whose values are registered with the redactor.
    #[serde(default)]
    pub secrets: Vec<String>,
}

impl DaemonConfig {
    /// Full pipe path.
    #[must_use]
    pub fn pipe_path(&self) -> String {
        format!(r"\\.\pipe\{}", self.pipe_name)
    }
}

/// Disk-guard thresholds for one volume (PHASE0_PLAN 3.2).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VolumeGuard {
    /// Mount point, e.g. `F:\`.
    pub mount: String,
    /// Warn below this many GB free.
    pub warn_gb: u64,
    /// Block new work (or archive writes) below this.
    #[serde(default)]
    pub block_gb: Option<u64>,
    /// Pause running work below this.
    #[serde(default)]
    pub pause_gb: Option<u64>,
}

/// The `[resources]` section.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourcesConfig {
    /// Snapshot interval.
    pub sample_interval_ms: u64,
    /// One snapshot is logged every this many seconds.
    pub log_interval_s: u64,
    /// Commit charge kept free on top of a profile's estimate.
    pub commit_margin_mib: u64,
    /// Doctor warns when free VRAM is below this.
    pub vram_headroom_warn_mib: u64,
    /// Hot log budget on the data drive.
    pub log_budget_gb: u64,
    /// Log archive cap on the cold drive.
    pub archive_cap_gb: u64,
    /// Drive letter that must not be used, e.g. `C:`.
    pub retired_drive: String,
    /// Disk-guard thresholds.
    #[serde(default)]
    pub volumes: Vec<VolumeGuard>,
}

/// The `[backup]` section (PHASE0_PLAN 7.1).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackupConfig {
    /// The rclone executable.
    pub rclone: PathBuf,
    /// rclone's config file, encrypted with the `nebula/rclone_config_pass` credential.
    pub rclone_config: PathBuf,
    /// The cloud remote that holds the OAuth sign-in, e.g. `gdrive:`.
    pub auth_remote: String,
    /// The encrypted remote backups are written to, e.g. `gdrive-crypt:`.
    pub remote: String,
    /// How long a sign-in lasts (7 for a Google app in Testing); 0 if it doesn't expire.
    pub token_lifetime_days: u64,
    /// Doctor warns this many days before the sign-in expires.
    pub token_warn_days: u64,
    /// Cloud retention: the newest this many backups...
    pub keep_recent: usize,
    /// ...plus the newest of each of this many days...
    pub keep_daily: usize,
    /// ...ISO weeks...
    pub keep_weekly: usize,
    /// ...and months.
    pub keep_monthly: usize,
    /// Local copies in `paths.backups_local` are kept this many days.
    pub local_keep_days: u64,
    /// A retention pass that would delete more than this many files...
    pub max_delete_files: usize,
    /// ...or more than this many MiB stops and asks (`--allow-large-delete`).
    pub max_delete_mib: u64,
}

/// The `[sandbox]` section: command-classification rules for the shell tool.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SandboxConfig {
    /// Optional path to an external command-classification rules table (TOML). Empty (the
    /// default) resolves to the table embedded in the binary; editing that embedded table
    /// requires a rebuild and reinstall (the lock-file gotcha in AGENTS.md). Loading an
    /// external table from a non-empty path is not yet wired.
    #[serde(default)]
    pub rules_table_path: String,
}

/// Which committed-memory cap a task's Job Object enforces (issue #30).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryScopeConfig {
    /// Cap each member process individually.
    Process,
    /// Cap the task's whole process tree together (the default).
    #[default]
    Job,
}

/// The `[tasks]` section: optional per-task resource caps enforced via the Windows Job Object that
/// supervises a task's processes (issue #30).
///
/// Both caps are **off by default** in Phase 1 (`0` = no cap): the machine's VRAM and commit charge
/// are already tight (AGENTS.md), so a wrong default could kill a legitimate build. The executor
/// (issue #32) may override these per task; the memory cap is range-checked against installed
/// physical memory when a task's job is created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TasksConfig {
    /// Committed-memory cap per task, in mebibytes. `0` (the default) means no cap.
    #[serde(default)]
    pub memory_limit_mib: u64,
    /// Whether `memory_limit_mib` applies per-process or across the whole task (the default).
    #[serde(default)]
    pub memory_limit_scope: MemoryScopeConfig,
    /// Hard CPU-rate cap per task, as a percentage `1..=100`. `0` (the default) means no cap.
    #[serde(default)]
    pub cpu_rate_percent: u8,
}

/// The `[executor]` section: limits and defaults for autonomous executor tasks (issue #32).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutorConfig {
    /// Maximum tool calls per task before stopping cleanly as `limit_reached`.
    #[serde(default = "default_max_tool_calls")]
    pub max_tool_calls: u32,
    /// Maximum cumulative model tokens per task across all turns before stopping as `limit_reached`.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Wall-clock timeout in seconds per task before stopping cleanly as `limit_reached`.
    #[serde(default = "default_wall_time_secs")]
    pub wall_time_secs: u64,
    /// Default chat profile for executor tasks; empty uses the active profile.
    #[serde(default)]
    pub default_profile: String,
}

const fn default_max_tool_calls() -> u32 {
    50
}
const fn default_max_tokens() -> u32 {
    65536
}
const fn default_wall_time_secs() -> u64 {
    600
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_tool_calls: default_max_tool_calls(),
            max_tokens: default_max_tokens(),
            wall_time_secs: default_wall_time_secs(),
            default_profile: String::new(),
        }
    }
}

/// The `[github]` section: automated bot branch push and PR creation (issue #36).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct GitHubConfig {
    /// Whether to automatically push the task branch and open a PR upon successful task completion.
    #[serde(default)]
    pub enabled: bool,
    /// Target repository in "owner/repo" format.
    #[serde(default = "default_github_repo")]
    pub repo: String,
    /// Base branch to target for PRs. Pushes to this branch are refused in code.
    #[serde(default = "default_github_base_branch")]
    pub base_branch: String,
    /// Expected branch prefix for task branches (e.g. "nebula/"). Push refuses branches without this prefix.
    #[serde(default = "default_github_branch_prefix")]
    pub branch_prefix: String,
    /// Credential Manager target containing the GitHub bot token.
    #[serde(default = "default_github_token_target")]
    pub token_target: String,
    /// GitHub bot username for user-agent, author attribution, and PR search.
    #[serde(default = "default_github_bot_username")]
    pub bot_username: String,
}

fn default_github_repo() -> String {
    "Xydra01/Nebula".to_owned()
}
fn default_github_base_branch() -> String {
    "main".to_owned()
}
fn default_github_branch_prefix() -> String {
    "nebula/".to_owned()
}
fn default_github_token_target() -> String {
    "nebula/github_bot_token".to_owned()
}
fn default_github_bot_username() -> String {
    "Nebula-dev-bot".to_owned()
}

impl Default for GitHubConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            repo: default_github_repo(),
            base_branch: default_github_base_branch(),
            branch_prefix: default_github_branch_prefix(),
            token_target: default_github_token_target(),
            bot_username: default_github_bot_username(),
        }
    }
}

/// The whole configuration.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NebulaConfig {
    /// Data locations.
    pub paths: Paths,
    /// Off-site backups.
    pub backup: BackupConfig,
    /// Logging; `log_dir` defaults to `paths.logs`.
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    /// Daemon settings.
    pub daemon: DaemonConfig,
    /// Resource sampling and thresholds.
    pub resources: ResourcesConfig,
    /// Model runtimes and profiles.
    pub model: ModelConfig,
    /// MCP tool servers.
    #[serde(default)]
    pub tools: ToolHostConfig,
    /// Command-classification rules for the shell tool.
    #[serde(default)]
    pub sandbox: SandboxConfig,
    /// Per-task git worktrees (issue #29): where task worktrees live and how task branches are
    /// named.
    #[serde(default)]
    pub worktree: WorktreeConfig,
    /// Optional per-task resource caps (issue #30), enforced via the task's Job Object.
    #[serde(default)]
    pub tasks: TasksConfig,
    /// Autonomous single executor loop bounds and defaults (issue #32).
    #[serde(default)]
    pub executor: ExecutorConfig,
    /// GitHub flow: bot branch push and PR creation (issue #36).
    #[serde(default)]
    pub github: GitHubConfig,
    /// Circuit breakers: safety stops and runaway detection (issue #34).
    #[serde(default)]
    pub circuit_breakers: CircuitBreakersConfig,
    /// Game mode detection and GPU yielding (issue #35).
    #[serde(default)]
    pub game_mode: GameModeConfig,
    /// Push notifications over Tailscale via ntfy (issue #37).
    #[serde(default)]
    pub notifications: NotificationsConfig,
}

/// The `[circuit_breakers]` section: thresholds for halting tasks upon repetitive failures or runaways (issue #34).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct CircuitBreakersConfig {
    /// Maximum consecutive identical tool failures before tripping breaker.
    #[serde(default = "default_max_repeated_tool_failures")]
    pub max_repeated_tool_failures: u32,
    /// Maximum edits to the same file in a single task before tripping breaker.
    #[serde(default = "default_max_file_edits")]
    pub max_file_edits: u32,
    /// Maximum tokens in a single step before tripping runaway token breaker.
    #[serde(default = "default_runaway_step_tokens")]
    pub runaway_step_tokens: u32,
}

const fn default_max_repeated_tool_failures() -> u32 {
    3
}
const fn default_max_file_edits() -> u32 {
    5
}
const fn default_runaway_step_tokens() -> u32 {
    16384
}

impl Default for CircuitBreakersConfig {
    fn default() -> Self {
        Self {
            max_repeated_tool_failures: default_max_repeated_tool_failures(),
            max_file_edits: default_max_file_edits(),
            runaway_step_tokens: default_runaway_step_tokens(),
        }
    }
}

/// The `[game_mode]` section: detection and GPU yielding (issue #35, design Section 2.6).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct GameModeConfig {
    /// Whether automatic game-mode monitoring is enabled.
    #[serde(default = "default_game_mode_enabled")]
    pub enabled: bool,
    /// Poll interval in seconds for checking foreground window and GPU processes.
    #[serde(default = "default_game_mode_poll_interval_s")]
    pub poll_interval_s: u64,
    /// VRAM threshold in MiB held by a non-Nebula process to trigger game mode.
    #[serde(default = "default_game_mode_vram_threshold_mib")]
    pub vram_threshold_mib: u64,
    /// Resume cooldown in seconds after a game exits before reloading model and resuming tasks.
    #[serde(default = "default_game_mode_cooldown_secs")]
    pub cooldown_secs: u64,
    /// Known game processes or launcher executables.
    #[serde(default = "default_game_mode_known_processes")]
    pub known_processes: Vec<String>,
    /// Allowlist of normal desktop apps that should never trigger game mode.
    #[serde(default = "default_game_mode_desktop_allowlist")]
    pub desktop_allowlist: Vec<String>,
}

const fn default_game_mode_enabled() -> bool {
    true
}
const fn default_game_mode_poll_interval_s() -> u64 {
    2
}
const fn default_game_mode_vram_threshold_mib() -> u64 {
    2048
}
const fn default_game_mode_cooldown_secs() -> u64 {
    10
}
fn default_game_mode_known_processes() -> Vec<String> {
    Vec::new()
}
fn default_game_mode_desktop_allowlist() -> Vec<String> {
    vec![
        "dwm.exe".into(),
        "explorer.exe".into(),
        "cursor.exe".into(),
        "code.exe".into(),
        "firefox.exe".into(),
        "chrome.exe".into(),
        "msedge.exe".into(),
        "discord.exe".into(),
        "slack.exe".into(),
        "windowsterminal.exe".into(),
        "powershell.exe".into(),
        "cmd.exe".into(),
        "taskmgr.exe".into(),
        "llama-server.exe".into(),
        "hydrogen_engine.exe".into(),
    ]
}

impl Default for GameModeConfig {
    fn default() -> Self {
        Self {
            enabled: default_game_mode_enabled(),
            poll_interval_s: default_game_mode_poll_interval_s(),
            vram_threshold_mib: default_game_mode_vram_threshold_mib(),
            cooldown_secs: default_game_mode_cooldown_secs(),
            known_processes: default_game_mode_known_processes(),
            desktop_allowlist: default_game_mode_desktop_allowlist(),
        }
    }
}

/// The `[notifications]` section: remote push notifications over Tailscale via ntfy (issue #37, design Section 4.5).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct NotificationsConfig {
    /// Whether push notifications are enabled.
    #[serde(default = "default_notifications_enabled")]
    pub enabled: bool,
    /// ntfy server URL (e.g. "http://100.x.y.z:8080" or "http://127.0.0.1:8080").
    #[serde(default = "default_notifications_server_url")]
    pub server_url: String,
    /// ntfy topic name. Private topic on your tailnet-only ntfy server.
    #[serde(default = "default_notifications_topic")]
    pub topic: String,
    /// Credential Manager target containing an optional authentication token for ntfy.
    #[serde(default = "default_notifications_token_target")]
    pub token_target: String,
    /// Restricts destination server_url to Tailscale IPs (100.64.0.0/10, fd7a:115c:a1e0::/48),
    /// MagicDNS hostnames (*.ts.net), or local loopback. Rejects public internet servers.
    #[serde(default = "default_notifications_tailscale_only")]
    pub tailscale_only: bool,
    /// Request timeout in seconds for publishing notifications.
    #[serde(default = "default_notifications_timeout_secs")]
    pub timeout_secs: u64,
    /// Base URL for the tailnet-only approval page (e.g. "http://ej-pc.tailnet:8080/approve").
    #[serde(default)]
    pub approval_url_base: String,
    /// Minimum interval in minutes between periodic doctor health checks/alerts.
    #[serde(default = "default_notifications_doctor_alert_interval_mins")]
    pub doctor_alert_interval_mins: u64,
}

const fn default_notifications_enabled() -> bool {
    false
}
fn default_notifications_server_url() -> String {
    "http://127.0.0.1:8080".into()
}
fn default_notifications_topic() -> String {
    "nebula-alerts".into()
}
fn default_notifications_token_target() -> String {
    "nebula/ntfy_token".into()
}
const fn default_notifications_tailscale_only() -> bool {
    true
}
const fn default_notifications_timeout_secs() -> u64 {
    5
}
const fn default_notifications_doctor_alert_interval_mins() -> u64 {
    60
}

impl Default for NotificationsConfig {
    fn default() -> Self {
        Self {
            enabled: default_notifications_enabled(),
            server_url: default_notifications_server_url(),
            topic: default_notifications_topic(),
            token_target: default_notifications_token_target(),
            tailscale_only: default_notifications_tailscale_only(),
            timeout_secs: default_notifications_timeout_secs(),
            approval_url_base: String::new(),
            doctor_alert_interval_mins: default_notifications_doctor_alert_interval_mins(),
        }
    }
}

fn parse_table(text: &str, origin: &str) -> Result<toml::Table, ConfigError> {
    text.parse::<toml::Table>().map_err(|e| ConfigError::Parse {
        origin: origin.to_owned(),
        message: e.to_string(),
    })
}

/// Merges `over` into `base`: tables recursively, everything else replaced.
pub fn merge(base: &mut toml::Table, over: toml::Table) {
    for (k, v) in over {
        match (base.get_mut(&k), v) {
            (Some(toml::Value::Table(b)), toml::Value::Table(o)) => merge(b, o),
            (_, v) => {
                base.insert(k, v);
            }
        }
    }
}

impl NebulaConfig {
    /// Parses the defaults merged with an optional override document, then validates.
    ///
    /// # Errors
    /// [`ConfigError::Parse`] or [`ConfigError::Invalid`].
    pub fn from_toml(override_toml: Option<&str>) -> Result<Self, ConfigError> {
        let mut table = parse_table(DEFAULT_TOML, "config/default.toml")?;
        if let Some(text) = override_toml {
            merge(&mut table, parse_table(text, "override")?);
        }
        let mut cfg: Self = table
            .try_into()
            .map_err(|e: toml::de::Error| ConfigError::Parse {
                origin: "merged config".into(),
                message: e.to_string(),
            })?;
        if cfg.telemetry.log_dir.is_none() {
            cfg.telemetry.log_dir = Some(cfg.paths.logs.clone());
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// The override file path: `$NEBULA_CONFIG` or [`DEFAULT_OVERRIDE_PATH`].
    #[must_use]
    pub fn override_path() -> PathBuf {
        std::env::var_os(OVERRIDE_ENV).map_or_else(|| DEFAULT_OVERRIDE_PATH.into(), PathBuf::from)
    }

    /// Loads defaults plus the override at `path`, if that file exists.
    ///
    /// # Errors
    /// I/O errors other than a missing file, parse errors, or validation errors.
    pub fn load_from(path: &Path) -> Result<Self, ConfigError> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_toml(Some(&text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::from_toml(None),
            Err(source) => Err(ConfigError::Io {
                path: path.to_owned(),
                source,
            }),
        }
    }

    /// Loads defaults plus the override at [`NebulaConfig::override_path`].
    ///
    /// # Errors
    /// See [`NebulaConfig::load_from`].
    pub fn load() -> Result<Self, ConfigError> {
        Self::load_from(&Self::override_path())
    }

    /// Every configured path, labelled, for drive checks.
    #[must_use]
    pub fn all_paths(&self) -> Vec<(String, PathBuf)> {
        let mut out = vec![
            ("paths.data_root".to_owned(), self.paths.data_root.clone()),
            ("paths.logs".to_owned(), self.paths.logs.clone()),
            ("paths.state".to_owned(), self.paths.state.clone()),
            ("paths.config".to_owned(), self.paths.config.clone()),
            ("paths.archive".to_owned(), self.paths.archive.clone()),
            (
                "paths.backups_local".to_owned(),
                self.paths.backups_local.clone(),
            ),
            (
                "backup.rclone_config".to_owned(),
                self.backup.rclone_config.clone(),
            ),
        ];
        if let Some(d) = &self.telemetry.log_dir {
            out.push(("telemetry.log_dir".to_owned(), d.clone()));
        }
        for (name, p) in &self.model.runtimes {
            out.push((format!("model.runtimes.{name}"), p.clone()));
        }
        for (name, p) in &self.model.profiles {
            out.push((format!("model.profiles.{name}.model"), p.model.clone()));
            if let Some(b) = &p.kv_bias {
                out.push((format!("model.profiles.{name}.kv_bias"), b.clone()));
            }
        }
        out.extend(self.tools.command_paths());
        out.push((
            "tools.builtin.worktree_root".to_owned(),
            self.tools.builtin.worktree_root.clone(),
        ));
        out.push((
            "worktree.worktrees_dir".to_owned(),
            self.worktree.worktrees_dir.clone(),
        ));
        out
    }

    /// Configured paths on the retired drive.
    #[must_use]
    pub fn paths_on_retired_drive(&self) -> Vec<(String, PathBuf)> {
        let drive = self.resources.retired_drive.to_ascii_uppercase();
        self.all_paths()
            .into_iter()
            .filter(|(_, p)| {
                p.to_string_lossy()
                    .get(..drive.len())
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case(&drive))
            })
            .collect()
    }

    /// Checks profiles and runtimes, the default/embedding profile names, and that nothing
    /// points at the retired drive.
    ///
    /// # Errors
    /// [`ConfigError::Invalid`] describing the first problem.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.model
            .validate()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        for (what, name) in [
            ("daemon.load_on_start", &self.daemon.load_on_start),
            ("daemon.embedding_profile", &self.daemon.embedding_profile),
        ] {
            if !name.is_empty() && !self.model.profiles.contains_key(name) {
                return Err(ConfigError::Invalid(format!(
                    "{what} names unknown profile {name:?}"
                )));
            }
        }
        if let Some((name, p)) = self.paths_on_retired_drive().into_iter().next() {
            return Err(ConfigError::Invalid(format!(
                "{name} = {} is on the retired drive {}",
                p.display(),
                self.resources.retired_drive
            )));
        }
        if self.daemon.pipe_name.is_empty() || self.daemon.pipe_name.contains('\\') {
            return Err(ConfigError::Invalid(
                "daemon.pipe_name must be a bare name".into(),
            ));
        }
        if self.tasks.cpu_rate_percent > 100 {
            return Err(ConfigError::Invalid(format!(
                "tasks.cpu_rate_percent = {} must be 0 (no cap) or 1..=100",
                self.tasks.cpu_rate_percent
            )));
        }
        if self.executor.max_tool_calls == 0 {
            return Err(ConfigError::Invalid(
                "executor.max_tool_calls must be at least 1".into(),
            ));
        }
        if self.executor.max_tokens == 0 {
            return Err(ConfigError::Invalid(
                "executor.max_tokens must be at least 1".into(),
            ));
        }
        if self.executor.wall_time_secs == 0 {
            return Err(ConfigError::Invalid(
                "executor.wall_time_secs must be at least 1".into(),
            ));
        }
        if !self.executor.default_profile.is_empty()
            && !self
                .model
                .profiles
                .contains_key(&self.executor.default_profile)
        {
            return Err(ConfigError::Invalid(format!(
                "executor.default_profile names unknown profile {:?}",
                self.executor.default_profile
            )));
        }
        if self.notifications.enabled {
            if self.notifications.server_url.is_empty() {
                return Err(ConfigError::Invalid(
                    "notifications.server_url must not be empty when notifications are enabled"
                        .into(),
                ));
            }
            if self.notifications.topic.is_empty() {
                return Err(ConfigError::Invalid(
                    "notifications.topic must not be empty when notifications are enabled".into(),
                ));
            }
            if self.notifications.timeout_secs == 0 {
                return Err(ConfigError::Invalid(
                    "notifications.timeout_secs must be at least 1".into(),
                ));
            }
        }
        Ok(())
    }
}
