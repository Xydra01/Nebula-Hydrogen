#![allow(clippy::unwrap_used, clippy::panic, missing_docs)]

use clap::Parser;
use nebula_cli::format::{self, Style};
use nebula_cli::{
    BackupAction, ChatArgs, Cli, Command, DaemonAction, GameModeAction, LangAction, Level,
    LogsAction, ModelAction, NotifyAction, Reasoning, TaskAction, TaskRunArgs, doctor_exit_code,
};
use nebula_config::NebulaConfig;
use nebula_proto::{
    CheckStatus, DaemonStatus, DiskUsage, DoctorCheck, DoctorReport, GameModeStatusResponse,
    GpuProcess, ModelState, ModelStatus, NotifyStatusResponse, ResourceSnapshot, StepId,
    StopReason, TaskDone, TaskId, TaskStatus, TaskStep, Usage,
};
use time::macros::datetime;

fn parse(args: &[&str]) -> Cli {
    Cli::try_parse_from(std::iter::once("nebula").chain(args.iter().copied())).unwrap()
}

#[test]
fn parses_every_command() {
    assert!(matches!(
        parse(&["daemon", "start"]).command,
        Command::Daemon {
            action: DaemonAction::Start
        }
    ));
    assert!(matches!(
        parse(&["daemon", "stop"]).command,
        Command::Daemon {
            action: DaemonAction::Stop
        }
    ));
    let cli = parse(&["daemon", "status", "--json"]);
    assert!(cli.json);
    assert!(matches!(
        cli.command,
        Command::Daemon {
            action: DaemonAction::Status
        }
    ));

    let Command::Chat(ChatArgs {
        profile,
        schema,
        reasoning,
        max_tokens,
    }) = parse(&[
        "--json",
        "chat",
        "--profile",
        "long",
        "--schema",
        "s.json",
        "--reasoning",
        "xhigh",
        "--max-tokens",
        "100",
    ])
    .command
    else {
        panic!("not chat")
    };
    assert_eq!(profile.as_deref(), Some("long"));
    assert_eq!(schema.unwrap().to_str(), Some("s.json"));
    assert_eq!(reasoning, Some(Reasoning::Xhigh));
    assert_eq!(max_tokens, Some(100));

    let Command::Model {
        action: ModelAction::Profile { name },
    } = parse(&["model", "profile", "standard"]).command
    else {
        panic!("not model profile")
    };
    assert_eq!(name, "standard");
    assert!(matches!(
        parse(&["model", "status"]).command,
        Command::Model {
            action: ModelAction::Status
        }
    ));

    let trace = "01J9ZQ3V5W8X0Y2Z4A6B8C0D2E";
    let Command::Logs {
        action:
            LogsAction::Tail {
                level,
                trace: t,
                target,
            },
    } = parse(&["logs", "tail", "--level", "warn", "--trace", trace]).command
    else {
        panic!("not logs tail")
    };
    assert_eq!(level, Some(Level::Warn));
    assert_eq!(t.unwrap().to_string(), trace);
    assert!(target.is_none());

    assert!(matches!(
        parse(&["backup", "reauth", "--record-only"]).command,
        Command::Backup {
            action: BackupAction::Reauth { record_only: true }
        }
    ));
    assert!(matches!(
        parse(&["backup", "reauth"]).command,
        Command::Backup {
            action: BackupAction::Reauth { record_only: false }
        }
    ));

    assert!(matches!(parse(&["resources"]).command, Command::Resources));
    assert!(matches!(parse(&["doctor"]).command, Command::Doctor));

    let Command::Task {
        action:
            TaskAction::Run(TaskRunArgs {
                goal,
                repo,
                start_point,
                profile,
                max_tool_calls,
                max_tokens,
                wall_time_secs,
                pr,
                no_pr,
                language,
                python_version: _,
                verification_target,
                streams: _,
            }),
    } = parse(&[
        "task",
        "run",
        "fix failing test",
        "--repo",
        "F:\\repo",
        "--start-point",
        "main",
        "--profile",
        "standard",
        "--max-tool-calls",
        "10",
        "--max-tokens",
        "5000",
        "--wall-time-secs",
        "60",
    ])
    .command
    else {
        panic!("not task run");
    };
    assert_eq!(goal, "fix failing test");
    assert_eq!(repo.unwrap().to_str(), Some("F:\\repo"));
    assert_eq!(start_point.as_deref(), Some("main"));
    assert_eq!(profile.as_deref(), Some("standard"));
    assert_eq!(max_tool_calls, Some(10));
    assert_eq!(max_tokens, Some(5000));
    assert_eq!(wall_time_secs, Some(60));
    assert!(!pr);
    assert!(!no_pr);
    assert_eq!(language, None);
    assert_eq!(verification_target, None);

    let Command::Task {
        action:
            TaskAction::Run(TaskRunArgs {
                verification_target,
                ..
            }),
    } = parse(&[
        "task",
        "run",
        "verified task",
        "--verification-target",
        "pytest",
    ])
    .command
    else {
        panic!("not task run with verification-target");
    };
    assert_eq!(verification_target.as_deref(), Some("pytest"));

    let Command::Task {
        action: TaskAction::Run(TaskRunArgs { language, .. }),
    } = parse(&["task", "run", "python task", "--lang", "python"]).command
    else {
        panic!("not task run with lang");
    };
    assert_eq!(language.as_deref(), Some("python"));

    let Command::Task {
        action: TaskAction::Run(TaskRunArgs { pr, no_pr, .. }),
    } = parse(&["task", "run", "pr test", "--pr"]).command
    else {
        panic!("not task run");
    };
    assert!(pr);
    assert!(!no_pr);

    let task_id_str = "01J9ZQ3V5W8X0Y2Z4A6B8C0D2E";
    let Command::Task {
        action: TaskAction::Cancel { task_id },
    } = parse(&["task", "cancel", task_id_str]).command
    else {
        panic!("not task cancel");
    };
    assert_eq!(task_id.to_string(), task_id_str);

    let Command::Stop { reason } = parse(&["stop"]).command else {
        panic!("not stop");
    };
    assert_eq!(reason, None);

    let Command::Stop { reason } = parse(&["stop", "--reason", "emergency"]).command else {
        panic!("not stop with reason");
    };
    assert_eq!(reason.as_deref(), Some("emergency"));

    let Command::Task {
        action: TaskAction::StopAll { reason },
    } = parse(&["task", "stop-all"]).command
    else {
        panic!("not task stop-all");
    };
    assert_eq!(reason, None);

    assert!(matches!(
        parse(&["game-mode"]).command,
        Command::GameMode { action: None }
    ));
    assert!(matches!(
        parse(&["game-mode", "status"]).command,
        Command::GameMode {
            action: Some(GameModeAction::Status)
        }
    ));
    assert!(matches!(
        parse(&["game-mode", "on"]).command,
        Command::GameMode {
            action: Some(GameModeAction::On)
        }
    ));
    assert!(matches!(
        parse(&["game-mode", "off"]).command,
        Command::GameMode {
            action: Some(GameModeAction::Off)
        }
    ));
    assert!(matches!(parse(&["resume"]).command, Command::Resume));
    assert!(matches!(
        parse(&["notify"]).command,
        Command::Notify { action: None }
    ));
    assert!(matches!(
        parse(&["notify", "status"]).command,
        Command::Notify {
            action: Some(NotifyAction::Status)
        }
    ));
    let Command::Notify {
        action: Some(NotifyAction::Test { message }),
    } = parse(&["notify", "test", "--message", "hello world"]).command
    else {
        panic!("not notify test");
    };
    assert_eq!(message.as_deref(), Some("hello world"));
    let Command::Notify {
        action: Some(NotifyAction::Test { message }),
    } = parse(&["notify", "test"]).command
    else {
        panic!("not notify test");
    };
    assert!(message.is_none());

    assert!(matches!(
        parse(&["lang"]).command,
        Command::Lang { action: None }
    ));
    assert!(matches!(
        parse(&["lang", "list"]).command,
        Command::Lang {
            action: Some(LangAction::List)
        }
    ));
    assert!(matches!(
        parse(&["language", "list"]).command,
        Command::Lang {
            action: Some(LangAction::List)
        }
    ));
    let Command::Lang {
        action: Some(LangAction::Info { language }),
    } = parse(&["lang", "info", "python"]).command
    else {
        panic!("not lang info");
    };
    assert_eq!(language, "python");

    assert!(matches!(parse(&["tui"]).command, Command::Tui));
}

#[test]
fn rejects_bad_arguments() {
    for args in [
        &["nebula"][..],
        &["nebula", "chat", "--reasoning", "high"],
        &["nebula", "logs", "tail", "--trace", "nope"],
        &["nebula", "model", "profile"],
        &["nebula", "daemon", "restart"],
        &["nebula", "task", "cancel", "invalid-id"],
    ] {
        assert!(Cli::try_parse_from(args).is_err(), "{args:?} should fail");
    }
}

#[test]
fn doctor_exit_codes() {
    assert_eq!(doctor_exit_code(CheckStatus::Ok), 0);
    assert_eq!(doctor_exit_code(CheckStatus::Warn), 1);
    assert_eq!(doctor_exit_code(CheckStatus::Fail), 2);
}

#[test]
fn style_plain_has_no_escapes() {
    assert_eq!(Style::PLAIN.bad("x"), "x");
    assert_eq!(Style { color: true }.bad("x"), "\x1b[31mx\x1b[0m");
}

fn model() -> ModelStatus {
    ModelStatus {
        profile: "standard".into(),
        state: ModelState::Ready,
        since: datetime!(2026-10-04 12:00:00 UTC),
        restarts: 1,
        last_error: Some("llama-server exited with code 1".into()),
    }
}

#[test]
fn snapshot_daemon_status() {
    let s = DaemonStatus {
        version: "0.1.0".into(),
        proto_version: 2,
        pid: 1234,
        uptime_s: 3725,
        model: model(),
    };
    insta::assert_snapshot!(format::daemon_status(
        &s,
        datetime!(2026-10-04 12:05:03 UTC),
        Style::PLAIN
    ));
}

#[test]
fn snapshot_resources() {
    let gb = 1024 * 1024 * 1024;
    let snap = ResourceSnapshot {
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
    };
    let cfg = NebulaConfig::from_toml(None).unwrap();
    insta::assert_snapshot!(format::resources(
        &snap,
        &cfg.resources.volumes,
        Style::PLAIN
    ));
}

#[test]
fn snapshot_doctor() {
    let report = DoctorReport::from_checks(vec![
        DoctorCheck {
            name: "daemon".into(),
            status: CheckStatus::Ok,
            detail: "running".into(),
        },
        DoctorCheck {
            name: "disk.F:".into(),
            status: CheckStatus::Warn,
            detail: "90 GB free (warn < 100 GB)".into(),
        },
        DoctorCheck {
            name: "backup".into(),
            status: CheckStatus::Fail,
            detail: "never ran".into(),
        },
    ]);
    insta::assert_snapshot!(format::doctor(&report, Style::PLAIN));
}

#[test]
fn snapshot_usage_lines() {
    let u = Usage {
        prompt_n: 120,
        prompt_ms: 60.0,
        predicted_n: 50,
        predicted_ms: 1000.0,
    };
    let lines = [
        format::usage_line(&u, StopReason::Stop, Style::PLAIN),
        format::usage_line(&u, StopReason::Length, Style::PLAIN),
        format::usage_line(&Usage::default(), StopReason::Cancelled, Style::PLAIN),
    ];
    insta::assert_snapshot!(lines.join("\n"));
}

#[test]
fn durations() {
    assert_eq!(format::duration(42), "42s");
    assert_eq!(format::duration(303), "5m 03s");
    assert_eq!(format::duration(7380), "2h 03m");
    assert_eq!(format::duration(273_600), "3d 04h");
}

#[test]
fn snapshot_task_formatting() {
    let task_id = "01J9ZQ3V5W8X0Y2Z4A6B8C0D2E".parse::<TaskId>().unwrap();
    let step_id = "01J9ZQ3V5W8X0Y2Z4A6B8C0D2F".parse::<StepId>().unwrap();
    let step = TaskStep {
        task_id,
        step_id,
        step_index: 1,
        tool_calls: vec!["fs.read".into(), "shell.execute".into()],
    };
    let done = TaskDone {
        task_id,
        status: TaskStatus::Completed,
        summary: "Fixed failing test in tests/test_calc.py".into(),
    };
    let lines = [
        format::task_step(&step, Style::PLAIN),
        format::task_done(&done, Style::PLAIN),
    ];
    insta::assert_snapshot!(lines.join("\n"));
}

#[test]
fn snapshot_game_mode_formatting() {
    let inactive = GameModeStatusResponse {
        active: false,
        manual_override: None,
        reason: None,
        process_name: None,
    };
    let active = GameModeStatusResponse {
        active: true,
        manual_override: Some(true),
        reason: Some("fullscreen window (Cyberpunk2077.exe)".into()),
        process_name: Some("Cyberpunk2077.exe".into()),
    };
    let lines = [
        format::game_mode_status(&inactive, Style::PLAIN),
        format::game_mode_status(&active, Style::PLAIN),
    ];
    insta::assert_snapshot!(lines.join("---\n"));
}

#[test]
fn snapshot_notify_formatting() {
    let disabled = NotifyStatusResponse {
        enabled: false,
        server_url: String::new(),
        topic: String::new(),
        tailscale_only: true,
    };
    let enabled = NotifyStatusResponse {
        enabled: true,
        server_url: "http://100.64.0.1:8080".into(),
        topic: "alerts".into(),
        tailscale_only: true,
    };
    let lines = [
        format::notify_status(&disabled, Style::PLAIN),
        format::notify_status(&enabled, Style::PLAIN),
    ];
    insta::assert_snapshot!(lines.join("---\n"));
}

#[test]
fn snapshot_language_formatting() {
    let list = nebula_proto::LanguageListResponse {
        languages: vec![
            nebula_proto::LanguageMetadata {
                id: "polyglot".into(),
                display_name: "Generic / Polyglot".into(),
                description: "Standard environment with basic tools".into(),
                supported_versions: Vec::new(),
                default_version: None,
            },
            nebula_proto::LanguageMetadata {
                id: "python".into(),
                display_name: "Python (uv-managed)".into(),
                description: "Python development with uv virtual environments".into(),
                supported_versions: vec!["3.11".into(), "3.12".into()],
                default_version: Some("3.12".into()),
            },
        ],
    };
    let info = nebula_proto::LanguageInfoResponse {
        metadata: nebula_proto::LanguageMetadata {
            id: "python".into(),
            display_name: "Python (uv-managed)".into(),
            description: "Python development with uv virtual environments".into(),
            supported_versions: vec!["3.11".into(), "3.12".into()],
            default_version: Some("3.12".into()),
        },
        tools: vec![nebula_proto::ToolInfo {
            server: "language_python".into(),
            name: "python.run".into(),
            description: Some("Run a script in the venv".into()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "script": { "type": "string" } },
                "required": ["script"]
            }),
        }],
    };
    let lines = [
        format::language_list(&list, Style::PLAIN),
        format::language_info(&info, Style::PLAIN),
    ];
    insta::assert_snapshot!(lines.join("---\n"));
}
