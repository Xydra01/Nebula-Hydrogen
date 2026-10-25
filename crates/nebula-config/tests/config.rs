#![allow(clippy::unwrap_used, clippy::panic, missing_docs)]

use std::path::PathBuf;

use nebula_config::{ConfigError, NebulaConfig};

#[test]
fn defaults_load_and_validate() {
    let cfg = NebulaConfig::from_toml(None).unwrap();
    assert_eq!(cfg.paths.data_root, PathBuf::from(r"F:\Nebula"));
    assert_eq!(
        cfg.telemetry.log_dir,
        Some(PathBuf::from(r"F:\Nebula\logs"))
    );
    assert_eq!(cfg.daemon.pipe_path(), r"\\.\pipe\nebula");
    assert_eq!(cfg.daemon.load_on_start, "standard");
    assert_eq!(cfg.resources.volumes.len(), 2);
    assert_eq!(cfg.resources.volumes[0].pause_gb, Some(12));
    assert_eq!(
        cfg.model.profile("standard").unwrap().commit_estimate_mib,
        Some(10700)
    );
    assert_eq!(cfg.sandbox.rules_table_path, "");
    assert!(cfg.paths_on_retired_drive().is_empty());
}

#[test]
fn override_merges_tables_and_replaces_values() {
    let cfg = NebulaConfig::from_toml(Some(
        r#"
        [daemon]
        load_on_start = ""
        [resources]
        commit_margin_mib = 2048
        [model.profiles.standard]
        ctx = 16384
        "#,
    ))
    .unwrap();
    assert_eq!(cfg.daemon.load_on_start, "");
    assert_eq!(cfg.daemon.pipe_name, "nebula");
    assert_eq!(cfg.resources.commit_margin_mib, 2048);
    assert_eq!(cfg.resources.log_interval_s, 30);
    let standard = cfg.model.profile("standard").unwrap();
    assert_eq!(standard.ctx, 16384);
    assert_eq!(standard.kv_type, "q4_0");
}

#[test]
fn unknown_keys_are_rejected() {
    let err = NebulaConfig::from_toml(Some("[daemon]\npipe_nmae = 'x'\n")).unwrap_err();
    assert!(matches!(err, ConfigError::Parse { .. }), "{err}");
}

#[test]
fn unknown_sandbox_key_is_rejected() {
    // `SandboxConfig` is `#[serde(deny_unknown_fields)]`, so an unknown key under `[sandbox]`
    // trips the parse rather than being silently ignored (Req 1.1, config convention).
    let err = NebulaConfig::from_toml(Some("[sandbox]\nbogus_key = true\n")).unwrap_err();
    assert!(matches!(&err, ConfigError::Parse { .. }), "{err}");
}

#[test]
fn sandbox_rules_table_path_override_round_trips() {
    // A non-empty override replaces the embedded-table sentinel verbatim; the resolver (not
    // this config layer) decides whether the empty string means "use the embedded table".
    // A TOML single-quoted literal string does not process escapes, so the backslashes in the
    // path survive verbatim into the parsed value.
    let cfg = NebulaConfig::from_toml(Some(
        "[sandbox]\nrules_table_path = 'F:\\Nebula\\rules.toml'\n",
    ))
    .unwrap();
    assert_eq!(cfg.sandbox.rules_table_path, r"F:\Nebula\rules.toml");
}

#[test]
fn paths_on_the_retired_drive_are_rejected() {
    let err = NebulaConfig::from_toml(Some("[paths]\nstate = 'c:\\Nebula\\state'\n")).unwrap_err();
    assert!(
        matches!(&err, ConfigError::Invalid(m) if m.contains("paths.state")),
        "{err}"
    );
    let err = NebulaConfig::from_toml(Some(
        "[model.runtimes]\nllama-prism = 'C:\\llama\\llama-server.exe'\n",
    ))
    .unwrap_err();
    assert!(matches!(err, ConfigError::Invalid(_)), "{err}");
}

#[test]
fn unknown_startup_profile_is_rejected() {
    let err = NebulaConfig::from_toml(Some("[daemon]\nload_on_start = 'quality'\n")).unwrap_err();
    assert!(matches!(err, ConfigError::Invalid(_)), "{err}");
}

#[test]
fn missing_override_file_means_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = NebulaConfig::load_from(&dir.path().join("absent.toml")).unwrap();
    assert_eq!(cfg, NebulaConfig::from_toml(None).unwrap());
    let path = dir.path().join("nebula.toml");
    std::fs::write(&path, "[telemetry]\nfilter = 'debug'\n").unwrap();
    assert_eq!(
        NebulaConfig::load_from(&path).unwrap().telemetry.filter,
        "debug"
    );
}

#[test]
fn all_paths_includes_the_builtin_worktree_root() {
    let cfg = NebulaConfig::from_toml(None).unwrap();
    let entry = cfg
        .all_paths()
        .into_iter()
        .find(|(label, _)| label == "tools.builtin.worktree_root")
        .expect("all_paths should label the built-in worktree root");
    assert_eq!(entry.1, cfg.tools.builtin.worktree_root);
    assert_eq!(entry.1, PathBuf::from("."));
}

#[test]
fn all_paths_includes_the_worktrees_dir() {
    // The retired-drive doctor check walks `all_paths()`, so the per-task `worktrees_dir` must be
    // listed there (git-worktree-per-task task 1.2; Req 8.3).
    let cfg = NebulaConfig::from_toml(None).unwrap();
    let entry = cfg
        .all_paths()
        .into_iter()
        .find(|(label, _)| label == "worktree.worktrees_dir")
        .expect("all_paths should label the worktrees dir");
    assert_eq!(entry.1, cfg.worktree.worktrees_dir);
    assert_eq!(entry.1, PathBuf::from(r"F:\Nebula\worktrees"));
}

#[test]
fn executor_defaults_load_and_validate() {
    let cfg = NebulaConfig::from_toml(None).unwrap();
    assert_eq!(cfg.executor.max_tool_calls, 50);
    assert_eq!(cfg.executor.max_tokens, 65536);
    assert_eq!(cfg.executor.wall_time_secs, 600);
    assert_eq!(cfg.executor.default_profile, "");
}

#[test]
fn executor_overrides_merge_and_validate() {
    let cfg = NebulaConfig::from_toml(Some(
        r#"
        [executor]
        max_tool_calls = 25
        max_tokens = 20000
        wall_time_secs = 300
        default_profile = "standard"
        "#,
    ))
    .unwrap();
    assert_eq!(cfg.executor.max_tool_calls, 25);
    assert_eq!(cfg.executor.max_tokens, 20000);
    assert_eq!(cfg.executor.wall_time_secs, 300);
    assert_eq!(cfg.executor.default_profile, "standard");
}

#[test]
fn unknown_executor_key_is_rejected() {
    let err = NebulaConfig::from_toml(Some("[executor]\nbogus_limit = 10\n")).unwrap_err();
    assert!(matches!(err, ConfigError::Parse { .. }), "{err}");
}

#[test]
fn zero_executor_limits_are_rejected() {
    let err = NebulaConfig::from_toml(Some("[executor]\nmax_tool_calls = 0\n")).unwrap_err();
    assert!(
        matches!(&err, ConfigError::Invalid(m) if m.contains("executor.max_tool_calls")),
        "{err}"
    );

    let err = NebulaConfig::from_toml(Some("[executor]\nmax_tokens = 0\n")).unwrap_err();
    assert!(
        matches!(&err, ConfigError::Invalid(m) if m.contains("executor.max_tokens")),
        "{err}"
    );

    let err = NebulaConfig::from_toml(Some("[executor]\nwall_time_secs = 0\n")).unwrap_err();
    assert!(
        matches!(&err, ConfigError::Invalid(m) if m.contains("executor.wall_time_secs")),
        "{err}"
    );
}

#[test]
fn unknown_executor_default_profile_is_rejected() {
    let err =
        NebulaConfig::from_toml(Some("[executor]\ndefault_profile = 'nonexistent'\n")).unwrap_err();
    assert!(
        matches!(&err, ConfigError::Invalid(m) if m.contains("executor.default_profile")),
        "{err}"
    );
}

#[test]
fn github_defaults_load_and_validate() {
    let cfg = NebulaConfig::from_toml(None).unwrap();
    assert!(!cfg.github.enabled);
    assert_eq!(cfg.github.repo, "Xydra01/Nebula");
    assert_eq!(cfg.github.base_branch, "main");
    assert_eq!(cfg.github.branch_prefix, "nebula/");
    assert_eq!(cfg.github.token_target, "nebula/github_bot_token");
    assert_eq!(cfg.github.bot_username, "Nebula-dev-bot");
}

#[test]
fn github_overrides_merge_and_validate() {
    let toml = r#"
[github]
enabled = true
repo = "custom-owner/custom-repo"
base_branch = "develop"
branch_prefix = "task/"
bot_username = "CustomBot"
"#;
    let cfg = NebulaConfig::from_toml(Some(toml)).unwrap();
    assert!(cfg.github.enabled);
    assert_eq!(cfg.github.repo, "custom-owner/custom-repo");
    assert_eq!(cfg.github.base_branch, "develop");
    assert_eq!(cfg.github.branch_prefix, "task/");
    assert_eq!(cfg.github.bot_username, "CustomBot");
}

#[test]
fn unknown_github_key_is_rejected() {
    let err = NebulaConfig::from_toml(Some("[github]\ninvalid_key = true\n")).unwrap_err();
    assert!(matches!(err, ConfigError::Parse { .. }), "{err}");
}

#[test]
fn circuit_breakers_defaults_load_and_validate() {
    let cfg = NebulaConfig::from_toml(None).unwrap();
    assert_eq!(cfg.circuit_breakers.max_repeated_tool_failures, 3);
    assert_eq!(cfg.circuit_breakers.max_file_edits, 5);
    assert_eq!(cfg.circuit_breakers.runaway_step_tokens, 16384);
}

#[test]
fn circuit_breakers_overrides_merge_and_validate() {
    let toml = r#"
[circuit_breakers]
max_repeated_tool_failures = 2
max_file_edits = 10
runaway_step_tokens = 8192
"#;
    let cfg = NebulaConfig::from_toml(Some(toml)).unwrap();
    assert_eq!(cfg.circuit_breakers.max_repeated_tool_failures, 2);
    assert_eq!(cfg.circuit_breakers.max_file_edits, 10);
    assert_eq!(cfg.circuit_breakers.runaway_step_tokens, 8192);
}

#[test]
fn unknown_circuit_breakers_key_is_rejected() {
    let err =
        NebulaConfig::from_toml(Some("[circuit_breakers]\ninvalid_field = 123\n")).unwrap_err();
    assert!(matches!(err, ConfigError::Parse { .. }), "{err}");
}

#[test]
fn game_mode_defaults_load_and_validate() {
    let cfg = NebulaConfig::from_toml(None).unwrap();
    assert!(cfg.game_mode.enabled);
    assert_eq!(cfg.game_mode.poll_interval_s, 2);
    assert_eq!(cfg.game_mode.vram_threshold_mib, 2048);
    assert_eq!(cfg.game_mode.cooldown_secs, 10);
    assert!(cfg.game_mode.known_processes.is_empty());
    assert!(cfg.game_mode.desktop_allowlist.contains(&"dwm.exe".into()));
    assert!(
        cfg.game_mode
            .desktop_allowlist
            .contains(&"firefox.exe".into())
    );
    assert!(
        cfg.game_mode
            .desktop_allowlist
            .contains(&"cursor.exe".into())
    );
}

#[test]
fn game_mode_overrides_merge_and_validate() {
    let toml = r#"
[game_mode]
enabled = false
poll_interval_s = 5
vram_threshold_mib = 4096
cooldown_secs = 30
known_processes = ["game.exe"]
desktop_allowlist = ["app.exe"]
"#;
    let cfg = NebulaConfig::from_toml(Some(toml)).unwrap();
    assert!(!cfg.game_mode.enabled);
    assert_eq!(cfg.game_mode.poll_interval_s, 5);
    assert_eq!(cfg.game_mode.vram_threshold_mib, 4096);
    assert_eq!(cfg.game_mode.cooldown_secs, 30);
    assert_eq!(cfg.game_mode.known_processes, vec!["game.exe"]);
    assert_eq!(cfg.game_mode.desktop_allowlist, vec!["app.exe"]);
}

#[test]
fn unknown_game_mode_key_is_rejected() {
    let err = NebulaConfig::from_toml(Some("[game_mode]\ninvalid_field = true\n")).unwrap_err();
    assert!(matches!(err, ConfigError::Parse { .. }), "{err}");
}

#[test]
fn notifications_defaults_load_and_validate() {
    let cfg = NebulaConfig::from_toml(None).unwrap();
    assert!(!cfg.notifications.enabled);
    assert_eq!(cfg.notifications.server_url, "http://127.0.0.1:8080");
    assert_eq!(cfg.notifications.topic, "nebula-alerts");
    assert_eq!(cfg.notifications.token_target, "nebula/ntfy_token");
    assert!(cfg.notifications.tailscale_only);
    assert_eq!(cfg.notifications.timeout_secs, 5);
    assert!(cfg.notifications.approval_url_base.is_empty());
    assert_eq!(cfg.notifications.doctor_alert_interval_mins, 60);
    assert!(
        cfg.daemon
            .secrets
            .contains(&"nebula/ntfy_token".to_string())
    );
}

#[test]
fn notifications_overrides_merge_and_validate() {
    let toml = r#"
[notifications]
enabled = true
server_url = "http://100.64.1.2:8080"
topic = "my-private-topic"
token_target = "nebula/custom_token"
tailscale_only = false
timeout_secs = 10
approval_url_base = "http://100.64.1.2:8080/approve"
doctor_alert_interval_mins = 30
"#;
    let cfg = NebulaConfig::from_toml(Some(toml)).unwrap();
    assert!(cfg.notifications.enabled);
    assert_eq!(cfg.notifications.server_url, "http://100.64.1.2:8080");
    assert_eq!(cfg.notifications.topic, "my-private-topic");
    assert_eq!(cfg.notifications.token_target, "nebula/custom_token");
    assert!(!cfg.notifications.tailscale_only);
    assert_eq!(cfg.notifications.timeout_secs, 10);
    assert_eq!(
        cfg.notifications.approval_url_base,
        "http://100.64.1.2:8080/approve"
    );
    assert_eq!(cfg.notifications.doctor_alert_interval_mins, 30);
}

#[test]
fn unknown_notifications_key_is_rejected() {
    let err = NebulaConfig::from_toml(Some("[notifications]\ninvalid_field = true\n")).unwrap_err();
    assert!(matches!(err, ConfigError::Parse { .. }), "{err}");
}

#[test]
fn notifications_validation_fails_on_empty_fields_when_enabled() {
    let toml_empty_url = "[notifications]\nenabled = true\nserver_url = \"\"\n";
    let err = NebulaConfig::from_toml(Some(toml_empty_url)).unwrap_err();
    assert!(matches!(err, ConfigError::Invalid(_)), "{err}");

    let toml_empty_topic = "[notifications]\nenabled = true\ntopic = \"\"\n";
    let err = NebulaConfig::from_toml(Some(toml_empty_topic)).unwrap_err();
    assert!(matches!(err, ConfigError::Invalid(_)), "{err}");

    let toml_zero_timeout = "[notifications]\nenabled = true\ntimeout_secs = 0\n";
    let err = NebulaConfig::from_toml(Some(toml_zero_timeout)).unwrap_err();
    assert!(matches!(err, ConfigError::Invalid(_)), "{err}");
}
