//! Round-trip and golden-fixture tests for every message. A changed snapshot means a protocol
//! change: bump `PROTO_VERSION` if it is incompatible.
#![allow(clippy::unwrap_used, clippy::panic, missing_docs)]

use insta::assert_json_snapshot;
use nebula_proto::*;
use serde_json::{Value, json};
use time::macros::datetime;
use ulid::Ulid;

fn ulid(s: &str) -> Ulid {
    Ulid::from_string(s).unwrap()
}
fn trace() -> TraceId {
    TraceId::from_ulid(ulid("01K6P3Z8M4Q7X2V9B5N1C0D3EF"))
}
fn span() -> SpanId {
    SpanId::from_ulid(ulid("01K6P3Z8M4Q7X2V9B5N1C0D3EG"))
}
fn chat_id() -> ChatId {
    ChatId::from_ulid(ulid("01K6P3Z8M4Q7X2V9B5N1C0D3EH"))
}
fn task_id() -> TaskId {
    TaskId::from_ulid(ulid("01K6P3Z8M4Q7X2V9B5N1C0D3EJ"))
}
fn step_id() -> StepId {
    StepId::from_ulid(ulid("01K6P3Z8M4Q7X2V9B5N1C0D3EK"))
}

fn model_status() -> ModelStatus {
    ModelStatus {
        profile: "standard".into(),
        state: ModelState::Ready,
        since: datetime!(2026-10-04 07:00:00 UTC),
        restarts: 0,
        last_error: None,
    }
}

fn snapshot() -> ResourceSnapshot {
    ResourceSnapshot {
        taken_at: datetime!(2026-10-04 07:00:02 UTC),
        vram_used_mib: 10_257,
        vram_total_mib: 12_282,
        gpu_util_pct: Some(97),
        ram_used_mib: 19_800,
        ram_total_mib: 32_683,
        commit_used_mib: 44_400,
        commit_limit_mib: 65_400,
        cpu_pct: 12.5,
        disks: vec![
            DiskUsage {
                mount: "F:\\".into(),
                free_bytes: 69_000_000_000,
                total_bytes: 1_000_000_000_000,
            },
            DiskUsage {
                mount: "D:\\".into(),
                free_bytes: 172_000_000_000,
                total_bytes: 1_000_000_000_000,
            },
        ],
        gpu_processes: vec![GpuProcess {
            pid: 4242,
            name: "llama-server.exe".into(),
            vram_mib: 8_700,
        }],
    }
}

/// Encodes, checks the golden JSON, decodes, and checks the round trip.
fn check(name: &str, msg: &Message) {
    let line = msg.encode().unwrap();
    assert!(line.ends_with('\n'));
    assert_eq!(
        line.matches('\n').count(),
        1,
        "exactly one newline per message"
    );
    let value: Value = serde_json::from_str(&line).unwrap();
    assert_json_snapshot!(name, value);
    assert_eq!(&Message::decode(&line).unwrap(), msg);
}

fn request(id: u64, call: Method) -> Message {
    Message::Request(Request::new(id, call))
}

#[test]
fn requests() {
    let mut chat = Request::new(
        3,
        Method::ChatStart(ChatStartParams {
            messages: vec![
                ChatMessage {
                    role: Role::System,
                    content: "You are Nebula.".into(),
                },
                ChatMessage {
                    role: Role::User,
                    content: "What is 17 * 23?".into(),
                },
            ],
            profile: Some("standard".into()),
            response_schema: Some(
                json!({"type": "object", "properties": {"n": {"type": "integer"}}}),
            ),
            max_tokens: Some(256),
            reasoning: Some(ReasoningEffort::None),
        }),
    );
    chat.trace_id = Some(trace());

    let cases: Vec<(&str, Message)> = vec![
        (
            "req_daemon_status",
            request(1, Method::DaemonStatus(Empty {})),
        ),
        (
            "req_daemon_shutdown",
            request(2, Method::DaemonShutdown(Empty {})),
        ),
        ("req_chat_start", Message::Request(chat)),
        (
            "req_chat_cancel",
            request(
                4,
                Method::ChatCancel(ChatCancelParams { chat_id: chat_id() }),
            ),
        ),
        (
            "req_model_status",
            request(5, Method::ModelStatus(Empty {})),
        ),
        (
            "req_model_set_profile",
            request(
                6,
                Method::ModelSetProfile(ModelSetProfileParams {
                    profile: "long".into(),
                }),
            ),
        ),
        (
            "req_resources_snapshot",
            request(7, Method::ResourcesSnapshot(Empty {})),
        ),
        (
            "req_logs_subscribe",
            request(
                8,
                Method::LogsSubscribe(LogsSubscribeParams {
                    min_level: Some(LogLevel::Warn),
                    trace_id: Some(trace()),
                    target_prefix: Some("nebula_model".into()),
                }),
            ),
        ),
        ("req_doctor_run", request(9, Method::DoctorRun(Empty {}))),
        ("req_tools_list", request(10, Method::ToolsList(Empty {}))),
        (
            "req_tools_call",
            request(
                11,
                Method::ToolsCall(ToolsCallParams {
                    tool: "read_file".into(),
                    arguments: json!({ "path": "src/main.rs" }),
                    trace_id: Some(trace()),
                }),
            ),
        ),
        (
            "req_task_create",
            request(
                12,
                Method::TaskCreate(TaskCreateParams {
                    goal: "fix failing test in test_calc.py".into(),
                    repo: "F:\\repo".into(),
                    start_point: Some("main".into()),
                    profile: Some("standard".into()),
                    max_tool_calls: Some(15),
                    max_tokens: Some(40960),
                    wall_time_secs: Some(600),
                    create_pr: None,
                    language: None,
                    language_config: None,
                    verification_target: None,
                    streams: None,
                }),
            ),
        ),
        (
            "req_task_cancel",
            request(
                13,
                Method::TaskCancel(TaskCancelParams { task_id: task_id() }),
            ),
        ),
        (
            "req_task_stop_all",
            request(
                14,
                Method::TaskStopAll(TaskStopAllParams {
                    reason: Some("emergency stop".into()),
                }),
            ),
        ),
        (
            "req_game_mode_set",
            request(15, Method::GameModeSet(GameModeSetParams { enabled: true })),
        ),
        (
            "req_game_mode_status",
            request(16, Method::GameModeStatus(Empty {})),
        ),
        (
            "req_notify_status",
            request(17, Method::NotifyStatus(Empty {})),
        ),
        (
            "req_notify_test",
            request(
                18,
                Method::NotifyTest(NotifyTestParams {
                    message: Some("test notification".into()),
                }),
            ),
        ),
        (
            "req_approval_respond",
            request(
                19,
                Method::ApprovalRespond(ApprovalRespondParams {
                    trace_id: "01J9ZQ3V5W8X0Y2Z4A6B8C0D2E".into(),
                    approved: true,
                    command: "git push origin main".into(),
                    tier: "System".into(),
                    reason: Some("User approved via TUI".into()),
                }),
            ),
        ),
        (
            "req_language_list",
            request(20, Method::LanguageList(Empty {})),
        ),
        (
            "req_language_info",
            request(
                21,
                Method::LanguageInfo(LanguageInfoParams {
                    language: "python".into(),
                }),
            ),
        ),
    ];
    let mut seen = Vec::new();
    for (name, msg) in &cases {
        check(name, msg);
        if let Message::Request(r) = msg {
            seen.push(r.call.name());
        }
    }
    assert_eq!(
        seen,
        Method::NAMES,
        "every method has a fixture, in declaration order"
    );
}

#[test]
fn responses() {
    let id = RequestId::Num;
    let cases: Vec<(&str, Response)> = vec![
        (
            "res_daemon_status",
            Response::ok(
                id(1),
                &DaemonStatus {
                    version: "0.0.1".into(),
                    proto_version: PROTO_VERSION,
                    pid: 1234,
                    uptime_s: 3600,
                    model: model_status(),
                },
            )
            .unwrap(),
        ),
        ("res_empty", Response::ok(id(2), &Empty {}).unwrap()),
        (
            "res_chat_started",
            Response::ok(
                id(3),
                &ChatStarted {
                    chat_id: chat_id(),
                    trace_id: trace(),
                },
            )
            .unwrap(),
        ),
        (
            "res_model_status_failed",
            Response::ok(
                id(5),
                &ModelStatus {
                    state: ModelState::Failed,
                    restarts: 5,
                    last_error: Some("llama-server exited with code 3221225477".into()),
                    ..model_status()
                },
            )
            .unwrap(),
        ),
        (
            "res_resources_snapshot",
            Response::ok(id(7), &snapshot()).unwrap(),
        ),
        (
            "res_doctor_report",
            Response::ok(
                id(9),
                &DoctorReport::from_checks(vec![
                    DoctorCheck {
                        name: "model.server".into(),
                        status: CheckStatus::Ok,
                        detail: "ready".into(),
                    },
                    DoctorCheck {
                        name: "disk.f".into(),
                        status: CheckStatus::Warn,
                        detail: "28 GB free (warn below 30 GB)".into(),
                    },
                ]),
            )
            .unwrap(),
        ),
        (
            "res_error",
            Response::err(
                RequestId::Str("abc".into()),
                RpcError::new(error_code::MODEL_UNAVAILABLE, "model server is restarting"),
            ),
        ),
        (
            "res_tool_list",
            Response::ok(
                id(10),
                &ToolList {
                    tools: vec![ToolInfo {
                        server: "filesystem".into(),
                        name: "read_file".into(),
                        description: Some("Read a file from the worktree".into()),
                        input_schema: json!({
                            "type": "object",
                            "properties": { "path": { "type": "string" } },
                            "required": ["path"],
                        }),
                    }],
                },
            )
            .unwrap(),
        ),
        (
            "res_tool_call",
            Response::ok(
                id(11),
                &ToolCallOutcome {
                    content: json!([{ "type": "text", "text": "fn main() {}" }]),
                    is_error: false,
                },
            )
            .unwrap(),
        ),
        (
            "res_task_started",
            Response::ok(
                id(12),
                &TaskStarted {
                    task_id: task_id(),
                    trace_id: trace(),
                },
            )
            .unwrap(),
        ),
        (
            "res_task_stop_all",
            Response::ok(id(14), &TaskStopAllResponse { stopped: 2 }).unwrap(),
        ),
        (
            "res_game_mode_status",
            Response::ok(
                id(16),
                &GameModeStatusResponse {
                    active: true,
                    manual_override: Some(true),
                    reason: Some("manual override".into()),
                    process_name: None,
                },
            )
            .unwrap(),
        ),
        (
            "res_notify_status",
            Response::ok(
                id(17),
                &NotifyStatusResponse {
                    enabled: true,
                    server_url: "http://100.64.1.2:8080".into(),
                    topic: "nebula-alerts".into(),
                    tailscale_only: true,
                },
            )
            .unwrap(),
        ),
        (
            "res_notify_test",
            Response::ok(
                id(18),
                &NotifyTestResponse {
                    success: true,
                    message: "Notification sent to topic nebula-alerts".into(),
                },
            )
            .unwrap(),
        ),
        (
            "res_approval_respond",
            Response::ok(
                id(19),
                &ApprovalRespondResponse {
                    success: true,
                    message: "Approval granted for command 'git push origin main'".into(),
                },
            )
            .unwrap(),
        ),
        (
            "res_language_list",
            Response::ok(
                id(20),
                &LanguageListResponse {
                    languages: vec![
                        LanguageMetadata {
                            id: "cpp".into(),
                            display_name: "C++ (MSVC / Clang / GCC)".into(),
                            description:
                                "Native C++ development with compiler toolchains and CMake".into(),
                            supported_versions: vec!["c++17".into(), "c++20".into()],
                            default_version: Some("c++20".into()),
                        },
                        LanguageMetadata {
                            id: "python".into(),
                            display_name: "Python (uv-managed)".into(),
                            description:
                                "Python development with uv virtual environments and tools".into(),
                            supported_versions: vec!["3.10".into(), "3.11".into(), "3.12".into()],
                            default_version: Some("3.12".into()),
                        },
                    ],
                },
            )
            .unwrap(),
        ),
        (
            "res_language_info",
            Response::ok(
                id(21),
                &LanguageInfoResponse {
                    metadata: LanguageMetadata {
                        id: "python".into(),
                        display_name: "Python (uv-managed)".into(),
                        description: "Python development with uv virtual environments and tools"
                            .into(),
                        supported_versions: vec!["3.10".into(), "3.11".into(), "3.12".into()],
                        default_version: Some("3.12".into()),
                    },
                    tools: vec![ToolInfo {
                        server: "language_python".into(),
                        name: "python_run".into(),
                        description: Some(
                            "Run a python script in the task virtual environment".into(),
                        ),
                        input_schema: json!({
                            "type": "object",
                            "properties": { "script": { "type": "string" } },
                            "required": ["script"],
                        }),
                    }],
                },
            )
            .unwrap(),
        ),
    ];
    for (name, res) in cases {
        check(name, &Message::Response(res));
    }
}

#[test]
fn events() {
    let cases: Vec<(&str, Event)> = vec![
        (
            "evt_chat_token",
            Event::ChatToken(ChatToken { chat_id: chat_id(), text: "39".into(), reasoning: String::new(), streams: None }),
        ),
        (
            "evt_chat_token_reasoning",
            Event::ChatToken(ChatToken {
                chat_id: chat_id(),
                text: String::new(),
                reasoning: "17 * 23 = 391".into(),
                streams: None,
            }),
        ),
        (
            "evt_chat_done",
            Event::ChatDone(ChatDone {
                chat_id: chat_id(),
                stop_reason: StopReason::Stop,
                usage: Usage { prompt_n: 28, prompt_ms: 41.5, predicted_n: 3, predicted_ms: 40.0 },
            }),
        ),
        (
            "evt_chat_error",
            Event::ChatError(ChatError {
                chat_id: chat_id(),
                code: error_code::CANCELLED,
                message: "cancelled by client".into(),
            }),
        ),
        (
            "evt_model_state_changed",
            Event::ModelStateChanged(ModelStateChanged {
                profile: "standard".into(),
                from: ModelState::Ready,
                to: ModelState::Restarting,
                reason: Some("health check timed out".into()),
            }),
        ),
        ("evt_resources_snapshot", Event::ResourcesSnapshot(snapshot())),
        (
            "evt_log_event",
            Event::LogEvent(LogEvent {
                ts: datetime!(2026-10-04 07:00:03.250 UTC),
                level: LogLevel::Info,
                target: "nebula_model::backend".into(),
                event: "model.call".into(),
                trace_id: Some(trace()),
                span_id: Some(span()),
                parent_span_id: None,
                task_id: None,
                step_id: None,
                fields: json!({
                    "profile": "standard",
                    "prompt_blob": "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
                    "predicted_n": 3
                })
                .as_object()
                .unwrap()
                .clone(),
            }),
        ),
        (
            "evt_task_step",
            Event::TaskStep(TaskStep {
                task_id: task_id(),
                step_id: step_id(),
                step_index: 0,
                tool_calls: vec!["test.run".into()],
                streams: None,
            }),
        ),
        (
            "evt_task_done",
            Event::TaskDone(TaskDone {
                task_id: task_id(),
                status: TaskStatus::Completed,
                summary: "Fixed test_calc by correcting off-by-one error".into(),
            }),
        ),
        (
            "evt_game_mode_changed",
            Event::GameModeChanged(GameModeChanged {
                active: true,
                reason: Some("fullscreen window (game.exe)".into()),
            }),
        ),
        (
            "evt_approval_needed",
            Event::ApprovalNeeded(ApprovalNeeded {
                command: "git".into(),
                tier: "System".into(),
                rule_id: Some("git.system".into()),
                trace_id: trace().to_string(),
                summary: "Command 'git' requires approval under rule 'git.system'".into(),
            }),
        ),
    ];
    for (name, event) in cases {
        check(name, &Message::Notification(Notification::new(event)));
    }
}

#[test]
fn missing_params_means_empty() {
    let msg =
        Message::decode(r#"{"jsonrpc":"2.0","id":1,"proto_version":3,"method":"daemon.status"}"#)
            .unwrap();
    assert_eq!(msg, request(1, Method::DaemonStatus(Empty {})));
}

#[test]
fn decode_errors_map_to_rpc_codes() {
    let code = |line: &str| Message::decode(line).unwrap_err().to_rpc_error().code;
    assert_eq!(code("{not json"), error_code::PARSE_ERROR);
    assert_eq!(code("[1,2]"), error_code::INVALID_REQUEST);
    assert_eq!(
        code(r#"{"jsonrpc":"1.0","id":1,"proto_version":3,"method":"daemon.status"}"#),
        error_code::INVALID_REQUEST
    );
    assert_eq!(
        code(r#"{"jsonrpc":"2.0","id":1,"proto_version":2,"method":"daemon.status","params":{}}"#),
        error_code::VERSION_MISMATCH
    );
    assert_eq!(
        code(r#"{"jsonrpc":"2.0","id":1,"proto_version":3,"method":"fs.delete","params":{}}"#),
        error_code::METHOD_NOT_FOUND
    );
    assert_eq!(
        code(
            r#"{"jsonrpc":"2.0","id":1,"proto_version":3,"method":"model.set_profile","params":{}}"#
        ),
        error_code::INVALID_PARAMS
    );
    assert_eq!(
        code(
            r#"{"jsonrpc":"2.0","id":1,"proto_version":3,"method":"model.set_profile","params":{"profile":"x","typo":1}}"#
        ),
        error_code::INVALID_PARAMS,
        "unknown fields are rejected"
    );
    assert_eq!(
        code(
            r#"{"jsonrpc":"2.0","id":1,"proto_version":3,"result":{},"error":{"code":1,"message":"x"}}"#
        ),
        error_code::INVALID_REQUEST,
        "result and error together"
    );
}

#[test]
fn reasoning_high_is_rejected() {
    assert!(serde_json::from_str::<ReasoningEffort>("\"high\"").is_err());
}

#[test]
fn typed_results() {
    let ok = Response::ok(RequestId::Num(1), &model_status()).unwrap();
    assert_eq!(ok.into_result::<ModelStatus>().unwrap(), model_status());
    let err = Response::err(RequestId::Num(1), RpcError::new(error_code::BUSY, "busy"));
    match err.into_result::<ModelStatus>() {
        Err(ProtoError::Rpc(e)) => assert_eq!(e.code, error_code::BUSY),
        other => panic!("expected an RPC error, got {other:?}"),
    }
}

#[test]
fn ids_sort_by_time_and_parse() {
    let a = TraceId::new();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let b = TraceId::new();
    assert!(a < b);
    assert_eq!(a.to_string().parse::<TraceId>().unwrap(), a);
    assert_eq!(a.to_string().len(), 26);
}

#[test]
fn doctor_overall_is_worst() {
    let c = |s| DoctorCheck {
        name: "x".into(),
        status: s,
        detail: String::new(),
    };
    assert_eq!(DoctorReport::from_checks(vec![]).overall, CheckStatus::Ok);
    assert_eq!(
        DoctorReport::from_checks(vec![c(CheckStatus::Warn), c(CheckStatus::Ok)]).overall,
        CheckStatus::Warn
    );
    assert_eq!(
        DoctorReport::from_checks(vec![c(CheckStatus::Warn), c(CheckStatus::Fail)]).overall,
        CheckStatus::Fail
    );
}
