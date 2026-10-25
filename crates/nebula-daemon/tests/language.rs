#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use nebula_config::NebulaConfig;
use nebula_daemon::client::Client;
use nebula_daemon::language::{
    FinalizeReport, LanguageError, LanguageInfoParams, LanguageInfoResponse, LanguageListResponse,
    LanguageMetadata, LanguageModule, ProvisionContext,
};
use nebula_daemon::{Daemon, Deps};
use nebula_model::SupervisorConfig;
use nebula_model::testing::{FakeLauncher, FakeState, content, finish};
use nebula_proto::{
    CheckStatus, DoctorCheck, DoctorReport, Empty, Event, Method, ModelStatus, TaskCreateParams,
    TaskDone, TaskId, TaskStarted, TaskStatus, TaskStep, error_code,
};
use nebula_resources::Sources;
use nebula_resources::sources::{GpuReading, GpuSource, SystemReading, SystemSource};
use nebula_telemetry::{Telemetry, TelemetryConfig};
use nebula_tools::builtins::{BuiltinTool, ToolContext, ToolOutput};
use nebula_tools::permit::Tier;
use nebula_tools::testing::{FakeTool, host_config};
use nebula_tools::{ToolError, ToolHost};
use serde_json::{Value, json};
use tempfile::TempDir;

const TIMEOUT: Duration = Duration::from_secs(10);

fn telemetry() -> Telemetry {
    static T: OnceLock<Telemetry> = OnceLock::new();
    T.get_or_init(|| nebula_telemetry::init(&TelemetryConfig::default()).unwrap())
        .clone()
}

fn supervisor_policy() -> SupervisorConfig {
    SupervisorConfig {
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

struct FakeGpu;
impl GpuSource for FakeGpu {
    fn read(&mut self) -> Result<GpuReading, nebula_resources::ResourceError> {
        Ok(GpuReading {
            name: "Fake GPU".into(),
            vram_used_mib: 9_000,
            vram_total_mib: 12_282,
            processes: vec![(4242, Some(8_000))],
            ..GpuReading::default()
        })
    }
}

struct FakeSystem;
impl SystemSource for FakeSystem {
    fn read(
        &mut self,
        _mounts: &[String],
    ) -> Result<SystemReading, nebula_resources::ResourceError> {
        Ok(SystemReading {
            ram_total_mib: 32_768,
            ..SystemReading::default()
        })
    }
    fn process_names(&mut self, pids: &[u32]) -> HashMap<u32, String> {
        pids.iter()
            .map(|p| (*p, "llama-server.exe".to_owned()))
            .collect()
    }
}

struct Harness {
    daemon: Daemon,
    _launcher: Arc<FakeLauncher>,
    state: Arc<FakeState>,
    pipe: String,
    _worktrees_dir: TempDir,
}

fn retired_drive_off(path: &Path) -> String {
    let on = path
        .components()
        .next()
        .and_then(|c| match c {
            std::path::Component::Prefix(p) => p.as_os_str().to_str(),
            _ => None,
        })
        .map(str::to_ascii_uppercase)
        .unwrap_or_default();
    if on.starts_with('Q') {
        "Z:".to_owned()
    } else {
        "Q:".to_owned()
    }
}

async fn start_harness() -> Harness {
    let unique = TaskId::new().to_string();
    let pipe_name = format!("nebula-test-{unique}");
    let instance = format!(r"Local\NebulaLangTest-{unique}");
    let state = FakeState::new();
    let launcher = FakeLauncher::new(Arc::clone(&state));

    let worktrees_dir = tempfile::tempdir().unwrap();
    let retired_drive = retired_drive_off(worktrees_dir.path());

    let mut config = NebulaConfig::from_toml(None).unwrap();
    config.model = nebula_model::testing::config();
    config.daemon.pipe_name = pipe_name.clone();
    config.daemon.load_on_start = "a".into();
    config.daemon.embedding_profile = String::new();
    config.resources.sample_interval_ms = 100;
    config.worktree.worktrees_dir = worktrees_dir.path().to_path_buf();
    config.resources.retired_drive = retired_drive;

    let external_launcher = nebula_tools::testing::launcher(vec![FakeTool::echo()]);
    let tool_host =
        ToolHost::start_with(&host_config("fs", 1_000), None, external_launcher.as_ref())
            .await
            .unwrap();

    let deps = Deps {
        telemetry: telemetry(),
        launcher: Arc::clone(&launcher) as _,
        tool_host,
        sources: Some(Sources {
            gpu: Some(Box::new(FakeGpu)),
            gpu_processes: None,
            system: Box::new(FakeSystem),
        }),
        preflight: None,
        supervisor: supervisor_policy(),
        instance_name: instance,
        local_checks: Arc::new(|_| {
            DoctorReport::from_checks(vec![DoctorCheck {
                name: "local.fake".into(),
                status: CheckStatus::Ok,
                detail: "fine".into(),
            }])
        }),
        artifact_checks: false,
        foreground_detector: None,
    };

    let daemon = nebula_daemon::start(config.clone(), deps).unwrap();

    Harness {
        daemon,
        _launcher: launcher,
        state,
        pipe: config.daemon.pipe_path(),
        _worktrees_dir: worktrees_dir,
    }
}

impl Harness {
    async fn ready_client(&self) -> Client {
        let mut c = Client::connect(&self.pipe, TIMEOUT).await.unwrap();
        tokio::time::timeout(TIMEOUT, async {
            loop {
                let s: ModelStatus = c.call(Method::ModelStatus(Empty {})).await.unwrap();
                if s.state == nebula_proto::ModelState::Ready {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .unwrap();
        c
    }
}

fn git(dir: &Path, args: &[&str]) {
    let status = std::process::Command::new("git")
        .args(args)
        .current_dir(dir)
        .status()
        .unwrap();
    assert!(status.success(), "git {args:?} failed in {dir:?}");
}

fn seed_repo() -> TempDir {
    let d = TempDir::new().unwrap();
    git(d.path(), &["init"]);
    git(d.path(), &["config", "user.email", "test@example.com"]);
    git(d.path(), &["config", "user.name", "test"]);
    git(d.path(), &["config", "core.autocrlf", "false"]);
    std::fs::write(d.path().join("seed.txt"), "seed\n").unwrap();
    git(d.path(), &["add", "seed.txt"]);
    git(d.path(), &["commit", "-m", "seed"]);
    d
}

fn tool_call_turn(id: &str, name: &str, arguments: &str) -> Vec<Value> {
    vec![
        json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        }
                    }]
                },
                "finish_reason": null,
            }]
        }),
        finish("tool_calls"),
    ]
}

fn completion_turn(text: &str) -> Vec<Value> {
    vec![content(text), finish("stop")]
}

async fn collect_task_events(c: &mut Client, task_id: TaskId) -> (Vec<TaskStep>, TaskDone) {
    let mut steps = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let ev = tokio::time::timeout_at(deadline, c.next_event())
            .await
            .expect("timeout waiting for task event")
            .expect("event error");
        match ev {
            Event::TaskStep(step) if step.task_id == task_id => {
                steps.push(step);
            }
            Event::TaskDone(done) if done.task_id == task_id => {
                return (steps, done);
            }
            _ => {}
        }
    }
}

#[tokio::test]
async fn test_language_list_rpc() {
    let h = start_harness().await;
    let mut c = h.ready_client().await;

    let res: LanguageListResponse = c.call(Method::LanguageList(Empty {})).await.unwrap();
    assert!(!res.languages.is_empty());
    let polyglot = res.languages.iter().find(|l| l.id == "polyglot");
    assert!(polyglot.is_some());
    assert_eq!(polyglot.unwrap().display_name, "Generic / Polyglot");
}

#[tokio::test]
async fn test_language_info_rpc() {
    let h = start_harness().await;
    let mut c = h.ready_client().await;

    let res: LanguageInfoResponse = c
        .call(Method::LanguageInfo(LanguageInfoParams {
            language: "polyglot".into(),
        }))
        .await
        .unwrap();
    assert_eq!(res.metadata.id, "polyglot");
    assert!(res.tools.is_empty());

    let err = c
        .call::<LanguageInfoResponse>(Method::LanguageInfo(LanguageInfoParams {
            language: "nonexistent".into(),
        }))
        .await
        .unwrap_err();

    assert_eq!(err.rpc_code(), Some(error_code::NOT_FOUND));
}

#[tokio::test]
async fn test_executor_with_explicit_polyglot() {
    let h = start_harness().await;
    h.state
        .set_chunks(completion_turn("Task completed under polyglot"));

    let repo = seed_repo();
    let mut c = h.ready_client().await;

    let started: TaskStarted = c
        .call(Method::TaskCreate(TaskCreateParams {
            goal: "Do a polyglot task".into(),
            repo: repo.path().to_path_buf(),
            start_point: None,
            profile: None,
            max_tool_calls: None,
            max_tokens: None,
            wall_time_secs: None,
            create_pr: None,
            language: Some("polyglot".into()),
            language_config: None,
            verification_target: None,
            streams: None,
        }))
        .await
        .unwrap();

    let (_steps, done) = collect_task_events(&mut c, started.task_id).await;
    assert_eq!(done.status, TaskStatus::Completed);
    assert_eq!(done.summary, "Task completed under polyglot");
}

#[tokio::test]
async fn test_executor_unknown_language_fails_fast() {
    let h = start_harness().await;
    let repo = seed_repo();
    let mut c = h.ready_client().await;

    let started: TaskStarted = c
        .call(Method::TaskCreate(TaskCreateParams {
            goal: "Task with unknown language".into(),
            repo: repo.path().to_path_buf(),
            start_point: None,
            profile: None,
            max_tool_calls: None,
            max_tokens: None,
            wall_time_secs: None,
            create_pr: None,
            language: Some("unknown_lang_42".into()),
            language_config: None,
            verification_target: None,
            streams: None,
        }))
        .await
        .unwrap();

    let (_steps, done) = collect_task_events(&mut c, started.task_id).await;
    assert_eq!(done.status, TaskStatus::Failed);
    assert!(done.summary.contains("unknown language module"));
}

struct CustomEchoTool {
    called: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl BuiltinTool for CustomEchoTool {
    fn name(&self) -> &str {
        "custom.echo"
    }

    fn description(&self) -> Option<&str> {
        Some("Echoes back a test message")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "msg": { "type": "string" } },
            "required": ["msg"]
        })
    }

    fn tier(&self) -> Tier {
        Tier::Read
    }

    async fn call(&self, arguments: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        self.called.store(true, Ordering::SeqCst);
        let msg = arguments["msg"].as_str().unwrap_or_default();
        Ok(ToolOutput::text(format!("custom: {msg}")))
    }
}

struct MockLanguageModule {
    tool_called: Arc<AtomicBool>,
    provisioned: Arc<AtomicBool>,
    finalized: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl LanguageModule for MockLanguageModule {
    fn id(&self) -> &str {
        "custom_lang"
    }

    fn metadata(&self) -> LanguageMetadata {
        LanguageMetadata {
            id: "custom_lang".into(),
            display_name: "Custom Language Module".into(),
            description: "Module for integration testing".into(),
            supported_versions: vec!["1.0".into()],
            default_version: Some("1.0".into()),
        }
    }

    async fn detect(&self, repo_root: &Path) -> Option<f32> {
        if repo_root.join("custom.lang").exists() {
            Some(0.95)
        } else {
            None
        }
    }

    async fn provision(&self, _ctx: &ProvisionContext<'_>) -> Result<(), LanguageError> {
        self.provisioned.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn tools(&self) -> Vec<Arc<dyn BuiltinTool>> {
        vec![Arc::new(CustomEchoTool {
            called: Arc::clone(&self.tool_called),
        })]
    }

    fn prompt_guidelines(&self, _ctx: &ProvisionContext<'_>) -> Vec<String> {
        vec!["Always use custom.echo first".into()]
    }

    async fn finalize(&self, _ctx: &ProvisionContext<'_>) -> Result<FinalizeReport, LanguageError> {
        self.finalized.store(true, Ordering::SeqCst);
        Ok(FinalizeReport {
            success: true,
            summary: "Custom finalize ok".into(),
            artifacts: vec![],
        })
    }
}

#[tokio::test]
async fn test_executor_with_custom_module_and_tool() {
    let h = start_harness().await;

    let tool_called = Arc::new(AtomicBool::new(false));
    let provisioned = Arc::new(AtomicBool::new(false));
    let finalized = Arc::new(AtomicBool::new(false));

    h.daemon.languages().register(Arc::new(MockLanguageModule {
        tool_called: Arc::clone(&tool_called),
        provisioned: Arc::clone(&provisioned),
        finalized: Arc::clone(&finalized),
    }));

    // Turn 1: call custom.echo
    // Turn 2: complete
    h.state.push_turn(tool_call_turn(
        "call_custom",
        "custom.echo",
        r#"{"msg":"hello from test"}"#,
    ));
    h.state
        .push_turn(completion_turn("Task done with custom module"));

    let repo = seed_repo();
    let mut c = h.ready_client().await;

    let started: TaskStarted = c
        .call(Method::TaskCreate(TaskCreateParams {
            goal: "Use custom language".into(),
            repo: repo.path().to_path_buf(),
            start_point: None,
            profile: None,
            max_tool_calls: Some(5),
            max_tokens: None,
            wall_time_secs: None,
            create_pr: None,
            language: Some("custom_lang".into()),
            language_config: None,
            verification_target: None,
            streams: None,
        }))
        .await
        .unwrap();

    let (steps, done) = collect_task_events(&mut c, started.task_id).await;
    assert_eq!(done.status, TaskStatus::Completed);
    assert_eq!(done.summary, "Task done with custom module");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].tool_calls, vec!["custom.echo"]);

    assert!(
        provisioned.load(Ordering::SeqCst),
        "module must be provisioned"
    );
    assert!(
        tool_called.load(Ordering::SeqCst),
        "custom tool must have been called"
    );
    assert!(
        finalized.load(Ordering::SeqCst),
        "module must have finalized"
    );
}
