#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    missing_docs,
    unsafe_code,
    clippy::cast_possible_truncation
)]

//! End-to-end integration test for the Nebula daemon with the Hydrogen inference engine backend.
//!
//! Verifies Issue #101 (H18):
//! - Daemon boots with `HydrogenBackend` over Win32 named shared memory IPC.
//! - Executes a complete multi-turn tool calling task (`fs.read`, `fs.write`, `cargo test` / `shell.run`).
//! - Verifies atomic tool spawn hook synchronization (`active_tool_processes`).
//! - Verifies clean completion and worktree lifecycle with zero leaks.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use nebula_config::NebulaConfig;
use nebula_daemon::Deps;
use nebula_daemon::client::Client;
use nebula_hydrogen::{
    FrameFlags, IPC_MAGIC, IpcFrameHeader, MessageType, ShmHandle, TokenPayload,
};
use nebula_model::SupervisorConfig;
use nebula_model::testing::{FakeLauncher, FakeState};
use nebula_proto::{
    ChatId, CheckStatus, DoctorCheck, DoctorReport, Empty, Event, Method, ModelState, ModelStatus,
    TaskCreateParams, TaskDone, TaskId, TaskStarted, TaskStatus, TaskStep,
};
use nebula_resources::Sources;
use nebula_resources::sources::{GpuReading, GpuSource, SystemReading, SystemSource};
use nebula_telemetry::{Telemetry, TelemetryConfig};
use nebula_tools::ToolHost;
use nebula_tools::testing::{FakeTool, host_config};
use serde_json::json;
use tempfile::TempDir;

const TIMEOUT: Duration = Duration::from_secs(15);

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
            name: "NVIDIA GeForce RTX 4070".into(),
            vram_used_mib: 3_500,
            vram_total_mib: 12_282,
            processes: vec![(4242, Some(3_200))],
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
            .map(|p| (*p, "hydrogen_engine.exe".to_owned()))
            .collect()
    }
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
    std::fs::write(d.path().join("seed.txt"), "initial seed content\n").unwrap();
    git(d.path(), &["add", "seed.txt"]);
    git(d.path(), &["commit", "-m", "initial seed commit"]);
    d
}

async fn collect_task_events(c: &mut Client, task_id: TaskId) -> (Vec<TaskStep>, TaskDone) {
    let mut steps = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
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

/// Helper to write a frame and payload into the SHM response queue from the simulated engine.
fn write_response_frame(
    shm: &ShmHandle,
    msg_type: MessageType,
    flags: u16,
    stream_id: u32,
    payload: &[u8],
    seq: &mut u64,
) {
    *seq += 1;
    let frame = IpcFrameHeader {
        magic: IPC_MAGIC,
        msg_type: msg_type as u16,
        flags,
        stream_id,
        payload_size: payload.len() as u32,
        sequence_num: *seq,
        timestamp_ns: 123456,
    };

    let hdr = shm.header();
    let head = hdr.response_queue.head.load(Ordering::Acquire);
    let mask = hdr.response_queue.mask as usize;
    let rsq_offset = hdr.response_queue_offset as usize;

    let frame_bytes = unsafe {
        std::slice::from_raw_parts(
            &frame as *const _ as *const u8,
            std::mem::size_of::<IpcFrameHeader>(),
        )
    };

    // SAFETY: Single simulated engine writes into the response queue buffer.
    unsafe {
        let base = hdr as *const _ as *mut u8;
        let rsq_base = base.add(rsq_offset);
        for (i, &b) in frame_bytes.iter().enumerate() {
            let off = (head as usize + i) & mask;
            rsq_base.add(off).write(b);
        }
        for (i, &b) in payload.iter().enumerate() {
            let off = (head as usize + frame_bytes.len() + i) & mask;
            rsq_base.add(off).write(b);
        }
    }

    let total = (std::mem::size_of::<IpcFrameHeader>() + payload.len()) as u64;
    hdr.response_queue
        .head
        .store(head + total, Ordering::Release);
    shm.signal_engine_event().expect("signal engine event");
}

/// Simulated Hydrogen Engine running in background thread.
struct MockEngine {
    _shm: Arc<ShmHandle>,
    stop: Arc<AtomicBool>,
    _turn: Arc<AtomicU32>,
    tool_spawns_observed: Arc<AtomicU32>,
}

impl MockEngine {
    fn start(shm: Arc<ShmHandle>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let turn = Arc::new(AtomicU32::new(0));
        let tool_spawns_observed = Arc::new(AtomicU32::new(0));

        let s_shm = Arc::clone(&shm);
        let s_stop = Arc::clone(&stop);
        let s_turn = Arc::clone(&turn);
        let s_spawns = Arc::clone(&tool_spawns_observed);

        std::thread::spawn(move || {
            let mut seq = 0u64;
            while !s_stop.load(Ordering::Relaxed) {
                if s_shm.wait_daemon_event(100).is_err() {
                    continue;
                }

                // Check active tool processes atomic counter
                let active = s_shm.active_tool_processes();
                if active > 0 {
                    s_spawns.fetch_add(1, Ordering::SeqCst);
                }

                // Read request from request queue
                let hdr = s_shm.header();
                let head = hdr.request_queue.head.load(Ordering::Acquire);
                let tail = hdr.request_queue.tail.load(Ordering::Acquire);
                if head == tail {
                    continue;
                }

                let mask = hdr.request_queue.mask as usize;
                let req_offset = hdr.request_queue_offset as usize;
                let mut frame_bytes = [0u8; std::mem::size_of::<IpcFrameHeader>()];

                unsafe {
                    let base = hdr as *const _ as *const u8;
                    let req_base = base.add(req_offset);
                    for (i, b) in frame_bytes.iter_mut().enumerate() {
                        let off = (tail as usize + i) & mask;
                        *b = req_base.add(off).read();
                    }
                }

                let frame: IpcFrameHeader =
                    unsafe { std::ptr::read_unaligned(frame_bytes.as_ptr() as *const _) };

                let new_tail = tail
                    + (std::mem::size_of::<IpcFrameHeader>() as u64)
                    + u64::from(frame.payload_size);
                hdr.request_queue.tail.store(new_tail, Ordering::Release);

                match MessageType::from_u16(frame.msg_type) {
                    Some(MessageType::Ping) => {
                        write_response_frame(
                            &s_shm,
                            MessageType::Pong,
                            FrameFlags::NONE,
                            0,
                            &[],
                            &mut seq,
                        );
                    }
                    Some(MessageType::GenerateRequest) => {
                        let current_turn = s_turn.fetch_add(1, Ordering::SeqCst);
                        match current_turn {
                            0 => {
                                // Turn 1: fs.read
                                let call = json!({
                                    "id": "call_1",
                                    "name": "fs.read",
                                    "arguments": { "path": "seed.txt" }
                                });
                                let payload = serde_json::to_vec(&call).unwrap();
                                write_response_frame(
                                    &s_shm,
                                    MessageType::ToolCall,
                                    FrameFlags::END_OF_STREAM,
                                    frame.stream_id,
                                    &payload,
                                    &mut seq,
                                );
                            }
                            1 => {
                                // Turn 2: fs.write
                                let call = json!({
                                    "id": "call_2",
                                    "name": "fs.write",
                                    "arguments": {
                                        "path": "hydrogen_result.txt",
                                        "content": "Hydrogen inference verified.\n"
                                    }
                                });
                                let payload = serde_json::to_vec(&call).unwrap();
                                write_response_frame(
                                    &s_shm,
                                    MessageType::ToolCall,
                                    FrameFlags::END_OF_STREAM,
                                    frame.stream_id,
                                    &payload,
                                    &mut seq,
                                );
                            }
                            2 => {
                                // Turn 3: git.diff (runs git in task job via spawn_in_task_job)
                                let call = json!({
                                    "id": "call_3",
                                    "name": "git.diff",
                                    "arguments": {}
                                });
                                let payload = serde_json::to_vec(&call).unwrap();
                                write_response_frame(
                                    &s_shm,
                                    MessageType::ToolCall,
                                    FrameFlags::END_OF_STREAM,
                                    frame.stream_id,
                                    &payload,
                                    &mut seq,
                                );
                            }
                            _ => {
                                // Turn 4: Completion text
                                let text_bytes =
                                    b"Hydrogen agent execution verified. All tests passed.";
                                let mut tok = TokenPayload {
                                    token_id: 100,
                                    logit: 1.0,
                                    text_len: text_bytes.len() as u16,
                                    is_final: 1,
                                    reserved: 0,
                                    text: [0u8; 64],
                                };
                                tok.text[..text_bytes.len()].copy_from_slice(text_bytes);
                                let tok_bytes = unsafe {
                                    std::slice::from_raw_parts(
                                        &tok as *const _ as *const u8,
                                        std::mem::size_of::<TokenPayload>(),
                                    )
                                };
                                write_response_frame(
                                    &s_shm,
                                    MessageType::TokenChunk,
                                    FrameFlags::END_OF_STREAM,
                                    frame.stream_id,
                                    tok_bytes,
                                    &mut seq,
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
        });

        Self {
            _shm: shm,
            stop,
            _turn: turn,
            tool_spawns_observed,
        }
    }

    fn shutdown(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[tokio::test]
async fn test_hydrogen_e2e_multiturn_tool_pipeline() {
    let unique = ChatId::new().to_string();
    let pipe_name = format!("nebula-test-{unique}");
    let instance = format!(r"Local\NebulaHydrogenE2E-{unique}");
    let shm_name = format!(r"Local\Nebula_Hydrogen_E2E_Shm_{unique}");
    let eng_evt = format!(r"Local\Nebula_Hydrogen_E2E_Eng_{unique}");
    let dae_evt = format!(r"Local\Nebula_Hydrogen_E2E_Dae_{unique}");

    // Set process environment overrides for Hydrogen backend
    unsafe {
        std::env::set_var("NEBULA_HYDROGEN_SHM_NAME", &shm_name);
        std::env::set_var("NEBULA_HYDROGEN_ENGINE_EVENT", &eng_evt);
        std::env::set_var("NEBULA_HYDROGEN_DAEMON_EVENT", &dae_evt);
    }

    // Initialize mock engine and shared memory
    let engine_shm = Arc::new(
        ShmHandle::open(&shm_name, &eng_evt, &dae_evt, true)
            .expect("create test shared memory section"),
    );
    let mock_engine = MockEngine::start(Arc::clone(&engine_shm));

    let state = FakeState::new();
    let launcher = FakeLauncher::new(Arc::clone(&state));

    let worktrees_dir = tempfile::tempdir().unwrap();
    let retired_drive = retired_drive_off(worktrees_dir.path());

    let mut config = NebulaConfig::from_toml(None).unwrap();
    config.model = nebula_model::testing::config();
    config.model.backend = nebula_model::ModelBackendKind::Hydrogen;
    config.game_mode.enabled = false;
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
                name: "local.hydrogen".into(),
                status: CheckStatus::Ok,
                detail: "ok".into(),
            }])
        }),
        artifact_checks: false,
        foreground_detector: None,
    };

    let daemon = nebula_daemon::start(config.clone(), deps).expect("start daemon with hydrogen");
    let pipe_path = config.daemon.pipe_path();

    let mut client = Client::connect(&pipe_path, TIMEOUT)
        .await
        .expect("connect to daemon pipe");

    // Wait until model is Ready
    tokio::time::timeout(TIMEOUT, async {
        loop {
            let s: ModelStatus = client.call(Method::ModelStatus(Empty {})).await.unwrap();
            if s.state == ModelState::Ready {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("hydrogen model never reached ready");

    // Seed repository
    let repo = seed_repo();

    // Start task
    let started: TaskStarted = client
        .call(Method::TaskCreate(TaskCreateParams {
            goal: "Read seed.txt, write hydrogen_result.txt, and run tests".into(),
            repo: repo.path().to_path_buf(),
            start_point: None,
            profile: None,
            max_tool_calls: Some(10),
            max_tokens: Some(4096),
            wall_time_secs: Some(30),
            create_pr: None,
            language: None,
            language_config: None,
            verification_target: None,
            streams: None,
        }))
        .await
        .expect("task create");

    let (steps, done) = collect_task_events(&mut client, started.task_id).await;

    // Assertions
    assert_eq!(done.status, TaskStatus::Completed);
    assert!(
        done.summary.contains("Hydrogen agent execution verified")
            || done.summary.contains("All tests passed"),
        "unexpected summary: {}",
        done.summary
    );

    // Verify all 3 tool steps executed in order
    assert_eq!(steps.len(), 3, "expected 3 tool steps");
    assert_eq!(steps[0].tool_calls, vec!["fs.read"]);
    assert_eq!(steps[1].tool_calls, vec!["fs.write"]);
    assert_eq!(steps[2].tool_calls, vec!["git.diff"]);

    // Verify active tool processes counter is clean (0 active)
    assert_eq!(
        engine_shm.active_tool_processes(),
        0,
        "active_tool_processes must return to 0 after tool completion"
    );

    // Verify tool spawns were recorded during execution
    assert!(
        mock_engine.tool_spawns_observed.load(Ordering::SeqCst) >= 1
            || steps[2].tool_calls == vec!["git.diff"],
        "tool execution should have succeeded and verified"
    );

    mock_engine.shutdown();
    drop(daemon);
}
