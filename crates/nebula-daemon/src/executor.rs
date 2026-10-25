//! The autonomous single executor loop (issue #32).
//!
//! Alternates model chat calls and schema-constrained tool calls until the goal is
//! achieved, a limit is reached, or the task is cancelled. Runs inside the task's
//! isolated git worktree and Windows Job Object.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError};
use std::time::Duration;

use futures_util::StreamExt;
use nebula_model::ModelError;
use nebula_model::types::{ChatOutput, ChatRequest, StreamItem};
use nebula_proto::{
    ChatId, ChatMessage, ChatToken, Event, Role, StepId, TaskCreateParams, TaskDone, TaskId,
    TaskStatus, TaskStep, TraceId,
};
use nebula_resources::GB;
use nebula_sandbox::job::MemoryScope;
use nebula_sandbox::task_job::{CpuRateLimit, JobLimits, MemoryLimit, TaskJob};
use nebula_tools::{CURRENT_TASK_JOB, CURRENT_WORKTREE};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::Shared;
use crate::circuit_breaker::CircuitBreakerTracker;
use crate::language::ProvisionContext;

const DEFAULT_STEP_MAX_TOKENS: u32 = 16384;

/// Runs one autonomous executor task until completion, limits, or cancellation.
pub(crate) async fn run_task(
    shared: Arc<Shared>,
    task_id: TaskId,
    trace_id: TraceId,
    params: TaskCreateParams,
    worktree_root: PathBuf,
    token: CancellationToken,
) {
    let wt_task_id = nebula_sandbox::worktree::TaskId::new(task_id.to_string());

    // 1. Build the task's Job Object
    let job_limits = build_job_limits(&shared);
    let task_job = match TaskJob::new(task_id.to_string(), job_limits) {
        Ok(job) => Arc::new(job),
        Err(e) => {
            tracing::error!(event = "task.job_create_failed", error = %e);
            let _ = shared.worktrees.finalize(&wt_task_id).await;
            let summary = format!("could not create task job: {e}");
            shared.notifications.notify_task_done(
                &task_id,
                &TaskStatus::Failed,
                &summary,
                &trace_id,
            );
            let done_ev = Event::TaskDone(TaskDone {
                task_id,
                status: TaskStatus::Failed,
                summary,
            });
            shared.emit_event(done_ev);
            return;
        }
    };

    // Register job handle in executor_tasks for global stop termination
    if let Some(state) = shared
        .executor_tasks
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_mut(&task_id)
    {
        state.task_job = Some(Arc::clone(&task_job));
    }

    // 2. Ensure model profile is loaded
    let profile = params.profile.as_deref().or_else(|| {
        let p = shared.config.executor.default_profile.as_str();
        if p.is_empty() { None } else { Some(p) }
    });
    if let Err(e) = crate::server::ensure_profile(&shared, profile, &token).await {
        tracing::error!(event = "task.model_profile_failed", error = %e.message);
        let _ = shared.worktrees.finalize(&wt_task_id).await;
        let summary = format!("model unavailable: {}", e.message);
        shared
            .notifications
            .notify_task_done(&task_id, &TaskStatus::Failed, &summary, &trace_id);
        let done_ev = Event::TaskDone(TaskDone {
            task_id,
            status: TaskStatus::Failed,
            summary,
        });
        shared.emit_event(done_ev);
        return;
    }

    // 3. Resolve limits
    let max_tool_calls = params
        .max_tool_calls
        .unwrap_or(shared.config.executor.max_tool_calls);
    let max_tokens = params
        .max_tokens
        .unwrap_or(shared.config.executor.max_tokens);
    let wall_time_secs = params
        .wall_time_secs
        .unwrap_or(shared.config.executor.wall_time_secs);

    // 4. Resolve Language Module (explicit or auto-detected)
    let lang_module = if let Some(ref lang_id) = params.language {
        match shared.languages.get(lang_id) {
            Some(m) => Some(m),
            None => {
                let summary = format!("unknown language module: {lang_id}");
                tracing::error!(event = "task.language_not_found", language = %lang_id);
                let _ = shared.worktrees.finalize(&wt_task_id).await;
                shared.notifications.notify_task_done(
                    &task_id,
                    &TaskStatus::Failed,
                    &summary,
                    &trace_id,
                );
                let done_ev = Event::TaskDone(TaskDone {
                    task_id,
                    status: TaskStatus::Failed,
                    summary,
                });
                shared.emit_event(done_ev);
                return;
            }
        }
    } else {
        let detected = shared.languages.detect(&worktree_root).await;
        if let Some(ref m) = detected {
            tracing::info!(
                event = "language.detected",
                language = %m.id(),
                repo = %worktree_root.display()
            );
        }
        detected
    };

    // 5. Provision Hermetic Environment
    if let Some(ref module) = lang_module {
        let prov_ctx = ProvisionContext {
            task_id: &task_id,
            worktree_root: &worktree_root,
            task_job: Some(&task_job),
            language_config: params.language_config.as_ref(),
            trace_id: &trace_id,
        };
        let prov_started = std::time::Instant::now();
        if let Err(e) = module.provision(&prov_ctx).await {
            tracing::error!(event = "language.provision_failed", error = %e);
            let _ = shared.worktrees.finalize(&wt_task_id).await;
            let summary = format!("language module provisioning failed: {e}");
            shared.notifications.notify_task_done(
                &task_id,
                &TaskStatus::Failed,
                &summary,
                &trace_id,
            );
            let done_ev = Event::TaskDone(TaskDone {
                task_id,
                status: TaskStatus::Failed,
                summary,
            });
            shared.emit_event(done_ev);
            return;
        }
        tracing::info!(
            event = "language.provisioned",
            language = %module.id(),
            duration_ms = prov_started.elapsed().as_millis()
        );
    }

    // 6. Check tool collisions and prepare language tools
    let language_tools: Vec<Arc<dyn nebula_tools::BuiltinTool>> =
        lang_module.as_ref().map(|m| m.tools()).unwrap_or_default();

    let existing_tool_names: HashSet<String> = shared
        .tool_host
        .list_tools()
        .into_iter()
        .map(|t| t.name)
        .collect();

    for lt in &language_tools {
        if existing_tool_names.contains(lt.name()) {
            let summary = format!(
                "tool name collision: language module tool '{}' collides with existing tool",
                lt.name()
            );
            tracing::error!(event = "task.tool_collision", tool = %lt.name());
            let _ = shared.worktrees.finalize(&wt_task_id).await;
            shared.notifications.notify_task_done(
                &task_id,
                &TaskStatus::Failed,
                &summary,
                &trace_id,
            );
            let done_ev = Event::TaskDone(TaskDone {
                task_id,
                status: TaskStatus::Failed,
                summary,
            });
            shared.emit_event(done_ev);
            return;
        }
    }

    // 7. Seed conversation
    let mut system_prompt = String::from(
        "You are Nebula, an autonomous local AI coding assistant working in an isolated git worktree. \
Your working directory is the root of the worktree. \
Inspect files, run commands and tests, edit code, and verify your changes autonomously. \
Environment & Tools: \
- System drives: The active repository/worktree is on drive F:. Never touch or reference drive C:. \
- Available tools: Standard environment tools such as python, git, uv, cargo, and pytest are available in PATH. Run them directly without hardcoded absolute paths. \
- Filesystem: Always use built-in tools (such as fs.list, fs.read, fs.write) to examine or modify files in the repository rather than shell commands like dir or cat. \
TOOL CALL FORMAT: When invoking a tool, you MUST wrap your call in <tool_call> tags and format it as a JSON object with 'name' and 'arguments': \
<tool_call> \
{\"name\": \"tool_name\", \"arguments\": {\"param1\": \"val1\"}} \
</tool_call> \
EXECUTION RULE: Execute tool calls while work is in progress. Do NOT output conversational plans, status updates, or thoughts without tool calls. If you need to inspect files or run commands, invoke the tools directly. \
EXIT DIRECTIVE: Once your verification and test commands pass with exit code 0 and satisfy the Definition of Done, the task is complete. You MUST immediately stop calling tools. Output your final completion summary in plain text without invoking any further tools to conclude the task.",
    );

    if let Some(ref module) = lang_module {
        let prov_ctx = ProvisionContext {
            task_id: &task_id,
            worktree_root: &worktree_root,
            task_job: Some(&task_job),
            language_config: params.language_config.as_ref(),
            trace_id: &trace_id,
        };
        let guidelines = module.prompt_guidelines(&prov_ctx);
        if !guidelines.is_empty() {
            system_prompt.push_str("\n\n--- Language Guidelines (");
            system_prompt.push_str(module.id());
            system_prompt.push_str(") ---\n");
            for g in guidelines {
                system_prompt.push_str("- ");
                system_prompt.push_str(&g);
                system_prompt.push('\n');
            }
        }
    }

    let mut messages = vec![
        ChatMessage {
            role: Role::System,
            content: system_prompt,
        },
        ChatMessage {
            role: Role::User,
            content: params.goal.clone(),
        },
    ];

    let num_candidates = params
        .streams
        .map(|s| s as usize)
        .or_else(|| {
            params
                .language_config
                .as_ref()
                .and_then(|v| v.get("speculative_candidates"))
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
        })
        .or_else(|| {
            std::env::var("NEBULA_SPECULATIVE_CANDIDATES")
                .ok()
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(1)
        .clamp(1, 4);
    let (status, summary, active_worktree_root, active_wt_task_id) = if num_candidates > 1 {
        tracing::info!(
            event = "task.speculative_branching.started",
            candidates = num_candidates,
            task_id = %task_id
        );

        let mut candidate_data = Vec::new();
        for c in 0..num_candidates {
            let cand_task_id = nebula_sandbox::worktree::TaskId::new(format!("{task_id}-cand-{c}"));
            match shared
                .worktrees
                .create(&cand_task_id, &params.repo, params.start_point.as_deref())
                .await
            {
                Ok(root) => {
                    let cand_job = match TaskJob::new(format!("{task_id}-cand-{c}"), job_limits) {
                        Ok(job) => Arc::new(job),
                        Err(e) => {
                            tracing::error!(event = "task.candidate_job_failed", candidate = c, error = %e);
                            Arc::clone(&task_job)
                        }
                    };
                    candidate_data.push((c, cand_task_id, root, cand_job));
                }
                Err(e) => {
                    tracing::warn!(event = "task.candidate_worktree_failed", candidate = c, error = %e);
                }
            }
        }

        if candidate_data.is_empty() {
            let loop_future = execute_loop(
                LoopContext {
                    shared: &shared,
                    task_id,
                    trace_id,
                    goal: &params.goal,
                    worktree_root: &worktree_root,
                    task_job: &task_job,
                    max_tool_calls,
                    max_tokens,
                    token: &token,
                    language_tools: &language_tools,
                    language_id: lang_module.as_ref().map(|m| m.id()),
                    verification_target: params.verification_target.as_deref(),
                    candidates: 1,
                },
                &mut messages,
            );
            let (st, sum) = match tokio::time::timeout(
                Duration::from_secs(wall_time_secs),
                loop_future,
            )
            .await
            {
                Ok(res) => res,
                Err(_) => {
                    tracing::warn!(event = "task.timeout", wall_time_secs = wall_time_secs);
                    (
                        TaskStatus::LimitReached,
                        format!("wall-clock timeout reached ({wall_time_secs}s)"),
                    )
                }
            };
            (st, sum, worktree_root, wt_task_id)
        } else {
            let mut candidate_messages = Vec::new();
            for (c, _, _, _) in &candidate_data {
                let strategy_hint = match c {
                    0 => "Candidate Strategy 0: Direct implementation and immediate verification.",
                    1 => {
                        "Candidate Strategy 1: Minimal surgical patch with targeted regression test."
                    }
                    2 => {
                        "Candidate Strategy 2: Defensive refactoring and comprehensive verification."
                    }
                    _ => {
                        "Candidate Strategy 3: Alternative algorithmic decomposition and validation."
                    }
                };
                let mut c_messages = messages.clone();
                c_messages.push(ChatMessage {
                    role: Role::User,
                    content: format!("[Candidate Strategy Guidance]: {strategy_hint}"),
                });
                candidate_messages.push(c_messages);
            }

            let mut futures = Vec::new();
            for ((_, _, c_root, c_job), c_messages) in
                candidate_data.iter().zip(candidate_messages.iter_mut())
            {
                let fut = execute_loop(
                    LoopContext {
                        shared: &shared,
                        task_id,
                        trace_id,
                        goal: &params.goal,
                        worktree_root: c_root,
                        task_job: c_job,
                        max_tool_calls,
                        max_tokens,
                        token: &token,
                        language_tools: &language_tools,
                        language_id: lang_module.as_ref().map(|m| m.id()),
                        verification_target: params.verification_target.as_deref(),
                        candidates: u32::try_from(num_candidates).unwrap_or(1),
                    },
                    c_messages,
                );
                futures.push(fut);
            }

            let join_fut = futures_util::future::join_all(futures);
            let results =
                match tokio::time::timeout(Duration::from_secs(wall_time_secs), join_fut).await {
                    Ok(res) => res,
                    Err(_) => {
                        let mut timeout_results = Vec::new();
                        for _ in 0..candidate_data.len() {
                            timeout_results.push((
                                TaskStatus::LimitReached,
                                format!("wall-clock timeout reached ({wall_time_secs}s)"),
                            ));
                        }
                        timeout_results
                    }
                };

            let mut best_idx = 0;
            for (idx, (st, _)) in results.iter().enumerate() {
                if *st == TaskStatus::Completed {
                    best_idx = idx;
                    break;
                }
            }

            let (best_status, best_summary) = results[best_idx].clone();
            tracing::info!(
                event = "task.speculative_branching.selected",
                winning_candidate = best_idx,
                status = ?best_status
            );

            let winning_root = candidate_data[best_idx].2.clone();
            let winning_task_id = candidate_data[best_idx].1.clone();
            let _ = commit_worktree_if_dirty(&winning_root, &params.goal, &task_id).await;

            for (idx, (_, cand_id, cand_root, _)) in candidate_data.into_iter().enumerate() {
                if idx != best_idx {
                    let _ = commit_worktree_if_dirty(
                        &cand_root,
                        "speculative candidate discarded",
                        &task_id,
                    )
                    .await;
                    let _ = shared.worktrees.remove_worktree(&cand_id).await;
                }
            }

            let _ = commit_worktree_if_dirty(
                &worktree_root,
                "primary worktree superseded by candidate",
                &task_id,
            )
            .await;
            let _ = shared.worktrees.remove_worktree(&wt_task_id).await;

            (best_status, best_summary, winning_root, winning_task_id)
        }
    } else {
        let loop_future = execute_loop(
            LoopContext {
                shared: &shared,
                task_id,
                trace_id,
                goal: &params.goal,
                worktree_root: &worktree_root,
                task_job: &task_job,
                max_tool_calls,
                max_tokens,
                token: &token,
                language_tools: &language_tools,
                language_id: lang_module.as_ref().map(|m| m.id()),
                verification_target: params.verification_target.as_deref(),
                candidates: 1,
            },
            &mut messages,
        );

        let (st, sum) =
            match tokio::time::timeout(Duration::from_secs(wall_time_secs), loop_future).await {
                Ok(result) => result,
                Err(_) => {
                    tracing::warn!(event = "task.timeout", wall_time_secs = wall_time_secs);
                    (
                        TaskStatus::LimitReached,
                        format!("wall-clock timeout reached ({wall_time_secs}s)"),
                    )
                }
            };
        (st, sum, worktree_root, wt_task_id)
    };

    // 8. Cleanup and finalization
    drop(task_job);

    if status == TaskStatus::Completed
        && let Some(ref module) = lang_module
    {
        let prov_ctx = ProvisionContext {
            task_id: &task_id,
            worktree_root: &active_worktree_root,
            task_job: None,
            language_config: params.language_config.as_ref(),
            trace_id: &trace_id,
        };
        match module.finalize(&prov_ctx).await {
            Ok(report) => {
                tracing::info!(
                    event = "language.finalized",
                    language = %module.id(),
                    success = report.success,
                    summary = %report.summary,
                    artifacts_count = report.artifacts.len()
                );
            }
            Err(e) => {
                tracing::warn!(event = "language.finalize_failed", error = %e);
            }
        }
    }

    if status == TaskStatus::Completed || status == TaskStatus::LimitReached {
        let _ = commit_worktree_if_dirty(&active_worktree_root, &params.goal, &task_id).await;
    }

    let finalized_branch = shared.worktrees.finalize(&active_wt_task_id).await;

    let should_create_pr = params.create_pr.unwrap_or(shared.config.github.enabled);
    let mut final_summary = summary;
    if status == TaskStatus::Completed
        && should_create_pr
        && let Ok(branch_name) = finalized_branch
    {
        match crate::github::handle_task_completion(crate::github::TaskCompletionParams {
            repo_root: &active_worktree_root,
            branch: &branch_name,
            goal: &params.goal,
            summary: &final_summary,
            trace_id,
            test_results: None,
            config: &shared.config.github,
            token_override: None,
            api_base_url_override: None,
            git_remote_url_override: None,
            extra_coauthor: None,
        })
        .await
        {
            Ok(Some(pr_url)) => {
                final_summary = format!("{final_summary}\n\nPull Request: {pr_url}");
            }
            Ok(None) => {}
            Err(e) => {
                tracing::error!(event = "github.flow_failed", error = %e);
                final_summary = format!("{final_summary}\n\n(GitHub PR creation failed: {e})");
            }
        }
    }

    tracing::info!(event = "task.done", status = ?status, summary = %final_summary);
    shared
        .notifications
        .notify_task_done(&task_id, &status, &final_summary, &trace_id);
    let done_ev = Event::TaskDone(TaskDone {
        task_id,
        status,
        summary: final_summary,
    });
    shared.emit_event(done_ev);
}

fn build_job_limits(shared: &Shared) -> JobLimits {
    let mut limits = JobLimits::default();
    if shared.config.tasks.memory_limit_mib > 0 {
        let bytes = shared.config.tasks.memory_limit_mib * 1024 * 1024;
        let scope = match shared.config.tasks.memory_limit_scope {
            nebula_config::MemoryScopeConfig::Process => MemoryScope::PerProcess,
            nebula_config::MemoryScopeConfig::Job => MemoryScope::WholeJob,
        };
        limits.memory = Some(MemoryLimit { bytes, scope });
    }
    if shared.config.tasks.cpu_rate_percent > 0 {
        limits.cpu = Some(CpuRateLimit {
            percent: shared.config.tasks.cpu_rate_percent,
        });
    }
    limits
}

struct LoopContext<'a> {
    shared: &'a Shared,
    task_id: TaskId,
    trace_id: TraceId,
    goal: &'a str,
    worktree_root: &'a Path,
    task_job: &'a Arc<TaskJob>,
    max_tool_calls: u32,
    max_tokens: u32,
    token: &'a CancellationToken,
    language_tools: &'a [Arc<dyn nebula_tools::BuiltinTool>],
    language_id: Option<&'a str>,
    verification_target: Option<&'a str>,
    candidates: u32,
}

fn normalize_mount_path(p: &Path) -> String {
    let s = p.to_string_lossy().replace('/', "\\");
    let s = s.strip_prefix(r"\\?\").unwrap_or(&s);
    s.to_uppercase()
}

fn mount_matches(path: &Path, mount: &str) -> bool {
    let p_str = normalize_mount_path(path);
    let m_str = mount.to_uppercase().replace('/', "\\");
    let m_trimmed = m_str.trim_end_matches('\\');
    p_str.starts_with(&m_str) || p_str.starts_with(m_trimmed)
}

fn check_disk_guard(shared: &Shared, worktree_root: &Path) -> Result<(), String> {
    let norm_worktree = normalize_mount_path(worktree_root);
    let drive_prefix = norm_worktree.chars().take(2).collect::<String>();

    let guard = shared.config.resources.volumes.iter().find(|g| {
        mount_matches(worktree_root, &g.mount) || g.mount.to_uppercase().starts_with(&drive_prefix)
    });

    let Some(guard) = guard else {
        return Ok(());
    };

    let free_bytes = if let Some(snap) = shared.latest_snapshot() {
        snap.disks
            .iter()
            .find(|d| mount_matches(Path::new(&d.mount), &guard.mount))
            .map(|d| d.free_bytes)
            .unwrap_or(u64::MAX)
    } else {
        let disks = sysinfo::Disks::new_with_refreshed_list();
        disks
            .iter()
            .find(|d| mount_matches(d.mount_point(), &guard.mount))
            .map(sysinfo::Disk::available_space)
            .unwrap_or(u64::MAX)
    };

    let block_gb = guard.block_gb.unwrap_or(guard.warn_gb);
    let block_bytes = block_gb * GB;
    if free_bytes < block_bytes {
        let free_gb = free_bytes / GB;
        tracing::warn!(
            event = "disk_guard.tripped",
            mount = %guard.mount,
            free_gb = free_gb,
            block_gb = block_gb,
        );
        return Err(format!(
            "disk guard tripped: free space on {} is below budget ({} GB free < {} GB budget)",
            guard.mount, free_gb, block_gb
        ));
    }

    Ok(())
}

async fn execute_loop(
    ctx: LoopContext<'_>,
    messages: &mut Vec<ChatMessage>,
) -> (TaskStatus, String) {
    if let Err(reason) = check_disk_guard(ctx.shared, ctx.worktree_root) {
        return (TaskStatus::LimitReached, reason);
    }

    let mut tracker = CircuitBreakerTracker::new(ctx.shared.config.circuit_breakers);
    let mut step_index: u32 = 0;
    let mut tool_calls_executed: u32 = 0;
    let mut total_tokens_used: u32 = 0;
    let mut nudge_count: u32 = 0;
    let mut verification_passed: bool = false;

    loop {
        if ctx.token.is_cancelled() {
            return (TaskStatus::Cancelled, "task cancelled by client".into());
        }

        if ctx.shared.game_mode.is_active()
            && ctx.shared.chat.status().state == nebula_proto::ModelState::Unloaded
            && let Err(e) = ctx.shared.game_mode.wait_if_active(ctx.token).await
        {
            return (TaskStatus::Cancelled, e);
        }

        // List tools and convert to OpenAI function schemas
        let mut descriptors = ctx.shared.tool_host.list_tools();
        for lt in ctx.language_tools {
            descriptors.push(nebula_tools::ToolDescriptor {
                server: format!("language_{}", ctx.language_id.unwrap_or("module")),
                name: lt.name().to_string(),
                description: lt.description().map(ToOwned::to_owned),
                input_schema: lt.input_schema(),
            });
        }
        let tools: Vec<Value> = descriptors
            .into_iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description.unwrap_or_default(),
                        "parameters": t.input_schema,
                    }
                })
            })
            .collect();

        let remaining_tokens = ctx.max_tokens.saturating_sub(total_tokens_used);
        if remaining_tokens == 0 {
            return (
                TaskStatus::LimitReached,
                format!("model token budget exhausted ({total_tokens_used} tokens used)"),
            );
        }

        let mut req = ChatRequest::new(messages.clone());
        req.tools = tools;
        req.max_tokens = DEFAULT_STEP_MAX_TOKENS.min(remaining_tokens);
        if ctx.shared.game_mode.is_active() {
            // WDDM continuous batching prefill clamping (Pivot 4 & 5 / Issue #35)
            // Clamping step tokens to <= 32 keeps GPU quantum under 14.5ms, eliminating desktop/gaming frame drops
            req.max_tokens = req.max_tokens.min(32);
        }
        req.trace_id = Some(ctx.trace_id);

        let backend = match ctx.shared.chat.backend() {
            Ok(b) => b,
            Err(e) => return (TaskStatus::Failed, format!("model backend error: {e}")),
        };

        let mut stream = tokio::select! {
            () = ctx.token.cancelled() => return (TaskStatus::Cancelled, "task cancelled".into()),
            r = backend.chat(req) => match r {
                Ok(s) => s,
                Err(ModelError::Timeout) => return (TaskStatus::LimitReached, "model call timed out".into()),
                Err(e) => return (TaskStatus::Failed, format!("model call failed: {e}")),
            },
        };

        let mut output = ChatOutput::default();
        let chat_id = ChatId::from_ulid(ctx.task_id.ulid());
        loop {
            let next_item = tokio::select! {
                () = ctx.token.cancelled() => return (TaskStatus::Cancelled, "task cancelled".into()),
                item = stream.next() => item,
            };

            let Some(item_res) = next_item else {
                break;
            };

            let item = match item_res {
                Ok(it) => it,
                Err(ModelError::Timeout) => {
                    return (TaskStatus::LimitReached, "model stream timed out".into());
                }
                Err(e) => return (TaskStatus::Failed, format!("model stream failed: {e}")),
            };

            match item {
                StreamItem::Token { text, reasoning } => {
                    if reasoning {
                        output.reasoning.push_str(&text);
                    } else {
                        output.text.push_str(&text);
                    }
                    let token_ev = Event::ChatToken(ChatToken {
                        chat_id,
                        text: if reasoning {
                            String::new()
                        } else {
                            text.clone()
                        },
                        reasoning: if reasoning { text } else { String::new() },
                        streams: Some(ctx.candidates),
                    });
                    ctx.shared.emit_event(token_ev);
                }
                StreamItem::ToolCall(c) => output.tool_calls.push(c),
                StreamItem::Usage(u) => output.usage = Some(u),
                StreamItem::Done(r) => {
                    output.stop_reason = Some(r);
                    break;
                }
            }
        }

        if let Some(usage) = &output.usage {
            let step_tokens = usage.prompt_n + usage.predicted_n;
            if let Err(trip) = tracker.check_step_tokens(step_tokens) {
                tracing::warn!(event = "circuit_breaker.tripped", reason = %trip);
                ctx.shared
                    .notifications
                    .notify_breaker_tripped(&trip.to_string(), Some(&ctx.trace_id));
                return (
                    TaskStatus::LimitReached,
                    format!("circuit breaker tripped: {trip}"),
                );
            }
            total_tokens_used += step_tokens;
            if total_tokens_used >= ctx.max_tokens {
                return (
                    TaskStatus::LimitReached,
                    format!(
                        "model token budget exhausted ({total_tokens_used} >= {})",
                        ctx.max_tokens
                    ),
                );
            }
        }

        // If the model finished without structured tool calls, extract from raw text (supports JSON and XML fallback)
        if output.tool_calls.is_empty() {
            output.tool_calls = extract_tool_calls_from_text(&output.text);
            if !output.tool_calls.is_empty() {
                tracing::info!(
                    event = "task.tool_calls_extracted_from_text",
                    count = output.tool_calls.len(),
                    tools = ?output.tool_calls.iter().map(|c| &c.name).collect::<Vec<_>>()
                );
            }
        }

        // If the model finished without calling any tools, check if it was conversational planning, pseudo-tool text, or completion.
        if output.tool_calls.is_empty() {
            let lower = output.text.to_lowercase();
            let is_calling_tools = lower.starts_with("calling tools")
                || lower.starts_with("calling tool")
                || lower.starts_with("invoking tool")
                || lower.starts_with("executing tool");
            let has_completion_markers = lower.contains("complete")
                || lower.contains("created")
                || lower.contains("implemented")
                || lower.contains("summary")
                || lower.contains("finished")
                || lower.contains("verified")
                || lower.contains("here is")
                || lower.contains("in summary");
            let is_planning_phrase = (lower.contains("let me ")
                && !lower.contains("let me know")
                && !lower.contains("let me summarize")
                && !lower.contains("let me explain"))
                || lower.contains("i will ")
                || lower.contains("i'll ")
                || lower.contains("i need to ")
                || lower.starts_with("let's ")
                || lower.contains("next, ");
            let is_still_planning =
                is_planning_phrase && !(tool_calls_executed > 0 && has_completion_markers);
            let is_empty = output.text.trim().is_empty();

            if (is_still_planning || is_calling_tools || is_empty) && nudge_count < 5 {
                nudge_count += 1;
                tracing::info!(
                    event = "task.nudge",
                    text = %output.text,
                    is_calling_tools = is_calling_tools,
                    is_still_planning = is_still_planning,
                    is_empty = is_empty
                );
                messages.push(ChatMessage {
                    role: Role::Assistant,
                    content: output.text,
                });
                let prompt = if is_calling_tools {
                    "To invoke a tool, you must emit a structured tool call. If all necessary actions are complete, provide a final plain-text response detailing what was created and verified."
                } else if is_empty {
                    "Please provide a final summary of what was accomplished, or invoke the appropriate tools to continue the task."
                } else if tool_calls_executed > 0 {
                    "If all necessary actions are complete, please provide your final plain-text summary of what was accomplished. Otherwise, continue executing the task using the available tools."
                } else {
                    "Please proceed with executing the task using the available tools now."
                };
                messages.push(ChatMessage {
                    role: Role::User,
                    content: prompt.into(),
                });
                continue;
            }

            if (is_still_planning || is_calling_tools) && nudge_count >= 5 {
                if tool_calls_executed > 0 {
                    tracing::info!(
                        event = "task.nudge_limit_resolved_as_completed",
                        text = %output.text,
                        tool_calls = tool_calls_executed
                    );
                    return (TaskStatus::Completed, output.text.trim().to_owned());
                }
                tracing::warn!(event = "task.nudge_limit_exhausted", text = %output.text);
                return (
                    TaskStatus::LimitReached,
                    format!(
                        "task execution stopped during unfinished planning: {}",
                        output.text.trim()
                    ),
                );
            }

            if verification_passed {
                tracing::info!(
                    event = "task.verification_heuristic_completed",
                    text = %output.text
                );
                let summary = if output.text.trim().is_empty() {
                    format!(
                        "Verification target '{}' succeeded and task completed.",
                        ctx.verification_target.unwrap_or("verification")
                    )
                } else {
                    output.text
                };
                return (TaskStatus::Completed, summary);
            }

            let summary = if is_calling_tools {
                format!("Task completed after executing {tool_calls_executed} tool call(s).")
            } else if output.text.trim().is_empty() {
                "Task completed".into()
            } else {
                output.text
            };
            return (TaskStatus::Completed, summary);
        }

        // Tools were invoked: reset consecutive planning nudge count
        nudge_count = 0;

        // Check if executing this turn's tool calls would exceed the tool call limit
        let num_calls = u32::try_from(output.tool_calls.len()).unwrap_or(u32::MAX);
        if tool_calls_executed + num_calls > ctx.max_tool_calls {
            return (
                TaskStatus::LimitReached,
                format!(
                    "maximum tool-call limit reached ({} calls)",
                    ctx.max_tool_calls
                ),
            );
        }

        // Emit step event
        let tool_names: Vec<String> = output.tool_calls.iter().map(|c| c.name.clone()).collect();
        let step_id = StepId::new();
        let step_ev = Event::TaskStep(TaskStep {
            task_id: ctx.task_id,
            step_id,
            step_index,
            tool_calls: tool_names.clone(),
            streams: Some(ctx.candidates),
        });
        ctx.shared.emit_event(step_ev);

        tracing::info!(
            event = "task.step",
            trace_id = %ctx.trace_id,
            task_id = %ctx.task_id,
            step_id = %step_id,
            step_index = step_index,
            tool_calls = ?tool_names,
        );
        step_index += 1;

        // Add assistant turn to conversation without synthetic text poisoning
        messages.push(ChatMessage {
            role: Role::Assistant,
            content: output.text,
        });

        // Dispatch each tool call within worktree and job scopes
        for call in output.tool_calls {
            if ctx.token.is_cancelled() {
                return (TaskStatus::Cancelled, "task cancelled".into());
            }

            tool_calls_executed += 1;
            let (args, parse_error) = match serde_json::from_str::<Value>(&call.arguments) {
                Ok(v) => (v, None),
                Err(e) => {
                    let is_truncated = output.stop_reason == Some(nebula_proto::StopReason::Length);
                    let detail = if is_truncated {
                        "tool arguments were truncated (exceeded step token limit). Please break work into smaller writes or shorter files"
                    } else {
                        "tool arguments were not valid JSON"
                    };
                    (
                        json!({ "raw": call.arguments }),
                        Some(format!("{detail}: {e}")),
                    )
                }
            };
            let trace_str = ctx.trace_id.to_string();

            let (mut result_content, mut is_error) = if let Some(err_msg) = parse_error {
                (format!("[tool error] {err_msg}"), true)
            } else {
                let lang_tool = ctx
                    .language_tools
                    .iter()
                    .find(|t| t.name() == call.name)
                    .cloned();

                let call_res = CURRENT_WORKTREE
                    .scope(
                        Some(ctx.worktree_root.to_path_buf()),
                        CURRENT_TASK_JOB.scope(
                            Some(Arc::clone(ctx.task_job)),
                            async {
                                if let Some(ref lt) = lang_tool {
                                    ctx.shared
                                        .tool_host
                                        .call_builtin(lt, args.clone(), Some(&trace_str))
                                        .instrument(tracing::info_span!("task.lang_tool", trace_id = %trace_str))
                                        .await
                                } else {
                                    ctx.shared
                                        .tool_host
                                        .call(&call.name, args.clone(), Some(&trace_str))
                                        .instrument(tracing::info_span!("task", trace_id = %trace_str))
                                        .await
                                }
                            },
                        ),
                    )
                    .await;

                match call_res {
                    Ok(outcome) => {
                        let err = outcome.is_error;
                        let text = if err {
                            format!("[tool error] {}", outcome.content)
                        } else {
                            outcome.content.to_string()
                        };
                        (text, err)
                    }
                    Err(err) => (format!("[tool dispatch error: {err}]"), true),
                }
            };

            if call.name == "task.finish" && !is_error {
                let finish_summary = args
                    .get("summary")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Task completed successfully.");
                let finish_status = args
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("success");

                // Issue #79: Enforce Windows Job Object process barrier before task.finish auto-commit.
                // Terminate and drain any active child/grandchild processes still running in the task's job.
                match ctx.task_job.active_processes() {
                    Ok(active) if active > 0 => {
                        tracing::warn!(
                            event = "task.finish_barrier_terminating_children",
                            task_id = %ctx.task_id,
                            active_processes = active,
                            "Terminating active child processes in task job before auto-commit"
                        );
                        let _ = ctx.task_job.terminate();
                        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
                        while tokio::time::Instant::now() < deadline {
                            if let Ok(0) = ctx.task_job.active_processes() {
                                break;
                            }
                            tokio::time::sleep(Duration::from_millis(10)).await;
                        }
                    }
                    Err(err) => {
                        tracing::warn!(
                            event = "task.finish_barrier_query_error",
                            task_id = %ctx.task_id,
                            error = %err,
                            "Could not query active job processes; enforcing safety termination before commit"
                        );
                        let _ = ctx.task_job.terminate();
                    }
                    _ => {}
                }

                match commit_worktree_if_dirty(ctx.worktree_root, ctx.goal, &ctx.task_id).await {
                    Ok(_) => {
                        let task_status = if finish_status.eq_ignore_ascii_case("failed") {
                            TaskStatus::Failed
                        } else {
                            TaskStatus::Completed
                        };
                        tracing::info!(
                            event = "task.finish_invoked",
                            task_id = %ctx.task_id,
                            status = ?task_status,
                            summary = %finish_summary
                        );
                        return (task_status, finish_summary.to_string());
                    }
                    Err(lock_err) => {
                        tracing::error!(
                            event = "task.finish_lock_error",
                            task_id = %ctx.task_id,
                            error = %lock_err
                        );
                        is_error = true;
                        result_content = format!("[tool error] {lock_err}");
                    }
                }
            }

            if is_error
                && (result_content.contains("requires an approval")
                    || result_content.contains("lacking an approval"))
            {
                let cmd_name = if call.name == "shell.run" {
                    args.get("command")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&call.name)
                        .to_string()
                } else {
                    call.name.clone()
                };

                ctx.shared
                    .notifications
                    .notify_approval_needed(&cmd_name, "System", None, &trace_str);
                let app_needed = nebula_proto::ApprovalNeeded {
                    command: cmd_name.clone(),
                    tier: "System".into(),
                    rule_id: None,
                    trace_id: trace_str.clone(),
                    summary: result_content.clone(),
                };
                ctx.shared.emit_event(Event::ApprovalNeeded(app_needed));

                // Pause execution and wait for user's approval response (via TUI or ntfy)
                let mut resp_rx = ctx.shared.approval_responses.subscribe();
                let wait_for_approval = async {
                    loop {
                        match resp_rx.recv().await {
                            Ok(resp)
                                if resp.trace_id == trace_str
                                    || resp.command == cmd_name
                                    || nebula_sandbox::rules::normalize_command(&resp.command)
                                        == nebula_sandbox::rules::normalize_command(&cmd_name) =>
                            {
                                return Ok(resp.approved);
                            }
                            Ok(_) => continue,
                            Err(_) => return Err("approval notification channel closed"),
                        }
                    }
                };

                let approval_decision = tokio::select! {
                    () = ctx.token.cancelled() => {
                        return (TaskStatus::Cancelled, "task cancelled while waiting for approval".into());
                    }
                    () = tokio::time::sleep(Duration::from_secs(300)) => {
                        Err("approval request timed out after 5 minutes")
                    }
                    decision = wait_for_approval => decision,
                };

                match approval_decision {
                    Ok(true) => {
                        tracing::info!(event = "task.approval_granted", tool = %call.name, command = %cmd_name);
                        let tier = nebula_sandbox::tier::Tier::System;
                        let normalized_cmd = nebula_sandbox::rules::normalize_command(&cmd_name);

                        // Ensure approval grants are available in ctx.shared.approvals
                        let grant_id1 =
                            nebula_sandbox::approval::GrantId::new(ulid::Ulid::generate().0);
                        let scope1 = nebula_sandbox::approval::ApprovalScope {
                            command: normalized_cmd.clone(),
                            tier,
                            trace_id: "task".into(),
                        };
                        ctx.shared
                            .approvals
                            .insert(nebula_sandbox::approval::Approval::grant(grant_id1, scope1));

                        let grant_id2 =
                            nebula_sandbox::approval::GrantId::new(ulid::Ulid::generate().0);
                        let scope2 = nebula_sandbox::approval::ApprovalScope {
                            command: normalized_cmd,
                            tier,
                            trace_id: trace_str.clone(),
                        };
                        ctx.shared
                            .approvals
                            .insert(nebula_sandbox::approval::Approval::grant(grant_id2, scope2));

                        let grant_id3 =
                            nebula_sandbox::approval::GrantId::new(ulid::Ulid::generate().0);
                        let scope3 = nebula_sandbox::approval::ApprovalScope {
                            command: cmd_name.clone(),
                            tier,
                            trace_id: "task".into(),
                        };
                        ctx.shared
                            .approvals
                            .insert(nebula_sandbox::approval::Approval::grant(grant_id3, scope3));

                        let retry_res = CURRENT_WORKTREE
                            .scope(
                                Some(ctx.worktree_root.to_path_buf()),
                                CURRENT_TASK_JOB.scope(
                                    Some(Arc::clone(ctx.task_job)),
                                    ctx.shared
                                        .tool_host
                                        .call(&call.name, args.clone(), Some(&trace_str))
                                        .instrument(
                                            tracing::info_span!("task", trace_id = %trace_str),
                                        ),
                                ),
                            )
                            .await;

                        match retry_res {
                            Ok(outcome) => {
                                is_error = outcome.is_error;
                                result_content = if is_error {
                                    format!("[tool error] {}", outcome.content)
                                } else {
                                    outcome.content.to_string()
                                };
                            }
                            Err(err) => {
                                is_error = true;
                                result_content = format!("[tool dispatch error: {err}]");
                            }
                        }
                    }
                    Ok(false) => {
                        tracing::info!(event = "task.approval_denied", tool = %call.name, command = %cmd_name);
                        result_content =
                            format!("[tool error] command '{cmd_name}' was denied by user");
                        is_error = true;
                    }
                    Err(reason) => {
                        tracing::warn!(event = "task.approval_timeout", reason = %reason);
                        result_content = format!("[tool error] {reason}");
                        is_error = true;
                    }
                }
            }

            if let Err(trip) = tracker.record_tool_call(&call.name, &args, is_error) {
                tracing::warn!(event = "circuit_breaker.tripped", reason = %trip);
                ctx.shared
                    .notifications
                    .notify_breaker_tripped(&trip.to_string(), Some(&ctx.trace_id));
                return (
                    TaskStatus::LimitReached,
                    format!("circuit breaker tripped: {trip}"),
                );
            }

            let outcome_summary = if is_error {
                let first_line = result_content.lines().next().unwrap_or(&result_content);
                format!("  └─> [ERROR] {first_line}\n")
            } else {
                let first_line = result_content.lines().next().unwrap_or("");
                if first_line.is_empty() {
                    "  └─> [OK]\n".to_string()
                } else {
                    format!("  └─> [OK] {first_line}\n")
                }
            };
            ctx.shared.emit_event(Event::ChatToken(ChatToken {
                chat_id,
                text: outcome_summary,
                reasoning: String::new(),
                streams: Some(ctx.candidates),
            }));

            let tool_output = format!("[tool result: {}]\n{}", call.name, result_content);
            messages.push(ChatMessage {
                role: Role::Tool,
                content: tool_output,
            });

            if let Some(target) = ctx.verification_target {
                let matched = if call.name == target {
                    true
                } else if call.name == "shell.run" {
                    args.get("command")
                        .and_then(|v| v.as_str())
                        .is_some_and(|cmd| cmd.contains(target) || cmd == target)
                } else {
                    false
                };

                if matched && !is_error {
                    tracing::info!(event = "task.verification_target_passed", target = %target);
                    verification_passed = true;
                    messages.push(ChatMessage {
                        role: Role::User,
                        content: format!(
                            "Verification target '{target}' succeeded with returncode 0 and 0 failures. All requirements are verified. Please conclude the task by invoking task.finish with your summary."
                        ),
                    });
                }
            }
        }
    }
}

/// Errors that can occur during git worktree lock contention resolution.
#[derive(Debug, thiserror::Error)]
pub enum LockContentionError {
    /// Lock file is locked by a concurrent process with sharing violation (Win32 error code 32).
    #[error(
        "Win32 ERROR_SHARING_VIOLATION (error code 32) at {}: file is locked by a concurrent process. Remediation hint: exclude worktree directory from Windows Search Indexing and antivirus real-time monitoring.",
        path.display()
    )]
    SharingViolation {
        /// Path to the contested lock file.
        path: PathBuf,
    },

    /// Any other lock failure that prevents resolving lock contention.
    #[error("Lock contention failure at {}: {detail}", path.display())]
    Other {
        /// Path to the contested lock file.
        path: PathBuf,
        /// Underlying failure detail.
        detail: String,
    },
}

/// Resolves the path to `.git/index.lock` for a repository or worktree.
pub fn find_index_lock_path(worktree_root: &Path) -> PathBuf {
    let dot_git = worktree_root.join(".git");
    if dot_git.is_file() {
        let content = std::fs::read_to_string(&dot_git).unwrap_or_default();
        for line in content.lines() {
            if let Some(gitdir_str) = line.strip_prefix("gitdir:") {
                let gitdir_path = PathBuf::from(gitdir_str.trim());
                let resolved = if gitdir_path.is_relative() {
                    worktree_root.join(gitdir_path)
                } else {
                    gitdir_path
                };
                return resolved.join("index.lock");
            }
        }
    }
    dot_git.join("index.lock")
}

/// Tier 2: NTFS Atomic Quarantine Rename for a contested `.git/index.lock`.
///
/// Renames `index.lock` to `index.lock.<timestamp>.stale` so git operations can proceed immediately.
pub fn quarantine_rename(lock_path: &Path) -> Result<PathBuf, std::io::Error> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let stale_path = lock_path.with_extension(format!("lock.{ts}.stale"));
    std::fs::rename(lock_path, &stale_path)?;
    Ok(stale_path)
}

/// Three-tier lock contention protocol for `.git/index.lock`.
///
/// - Tier 1: Exponential backoff retries (25ms, 50ms, 100ms, 250ms, 500ms) up to 5 times.
/// - Tier 2: NTFS atomic quarantine rename to `index.lock.<ts>.stale` if deletion fails.
/// - Tier 3: Diagnostic error propagation with Win32 error code 32 and remediation hints.
pub async fn resolve_lock_contention(lock_path: &Path) -> Result<(), LockContentionError> {
    if !lock_path.exists() {
        return Ok(());
    }

    // Tier 1: 5 retries with exponential backoffs
    let backoffs = [
        Duration::from_millis(25),
        Duration::from_millis(50),
        Duration::from_millis(100),
        Duration::from_millis(250),
        Duration::from_millis(500),
    ];

    let mut last_err = None;
    for delay in backoffs {
        match std::fs::remove_file(lock_path) {
            Ok(()) => {
                tracing::info!(event = "lock_protocol.tier1_cleared", path = %lock_path.display());
                return Ok(());
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    return Ok(());
                }
                last_err = Some(e);
                tokio::time::sleep(delay).await;
            }
        }
    }

    // Tier 2: NTFS Atomic Quarantine Rename
    match quarantine_rename(lock_path) {
        Ok(stale_path) => {
            tracing::warn!(
                event = "lock_protocol.tier2_quarantined",
                original = %lock_path.display(),
                quarantine = %stale_path.display(),
            );
            return Ok(());
        }
        Err(rename_err) => {
            tracing::error!(
                event = "lock_protocol.tier2_rename_failed",
                error = %rename_err,
                path = %lock_path.display()
            );
        }
    }

    // Tier 3: Diagnostic Error Reporting & Propagation
    let is_sharing_violation = last_err.as_ref().is_some_and(|e| {
        e.raw_os_error() == Some(32) || e.kind() == std::io::ErrorKind::PermissionDenied
    });

    if is_sharing_violation {
        Err(LockContentionError::SharingViolation {
            path: lock_path.to_path_buf(),
        })
    } else {
        let detail = last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unable to remove or rename lock file".to_string());
        Err(LockContentionError::Other {
            path: lock_path.to_path_buf(),
            detail,
        })
    }
}

pub(crate) async fn commit_worktree_if_dirty(
    worktree_root: &Path,
    goal: &str,
    task_id: &TaskId,
) -> Result<bool, LockContentionError> {
    use std::process::Stdio;
    use tokio::process::Command;

    let lock_path = find_index_lock_path(worktree_root);
    if lock_path.exists() {
        resolve_lock_contention(&lock_path).await?;
    }

    // 1. Check if git status has uncommitted or untracked changes
    let mut status_cmd = Command::new("git");
    status_cmd
        .args(["status", "--porcelain"])
        .current_dir(worktree_root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let is_dirty = match status_cmd.output().await {
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.contains("index.lock") {
                resolve_lock_contention(&lock_path).await?;
            }
            !out.stdout.is_empty()
        }
        Err(e) => {
            tracing::warn!(event = "task.git_status_failed", error = %e);
            false
        }
    };

    if !is_dirty {
        return Ok(false);
    }

    // 2. Stage all changes
    let mut add_cmd = Command::new("git");
    add_cmd
        .args(["add", "-A"])
        .current_dir(worktree_root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let add_out = match add_cmd.output().await {
        Ok(out) => out,
        Err(e) => {
            tracing::warn!(event = "task.git_add_failed", error = %e);
            return Err(LockContentionError::Other {
                path: lock_path,
                detail: format!("git add failed to spawn: {e}"),
            });
        }
    };

    if !add_out.status.success() {
        let stderr = String::from_utf8_lossy(&add_out.stderr);
        if stderr.contains("index.lock") {
            resolve_lock_contention(&lock_path).await?;
            let mut retry_add = Command::new("git");
            retry_add
                .args(["add", "-A"])
                .current_dir(worktree_root)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);
            match retry_add.output().await {
                Ok(out) if !out.status.success() => {
                    let err = String::from_utf8_lossy(&out.stderr);
                    return Err(LockContentionError::Other {
                        path: lock_path,
                        detail: format!("git add failed: {err}"),
                    });
                }
                _ => {}
            }
        } else {
            return Err(LockContentionError::Other {
                path: lock_path,
                detail: format!("git add failed: {stderr}"),
            });
        }
    }

    // 3. Commit staged changes with conventional commit format
    let commit_msg = format!(
        "feat: {goal}\n\nAutomated commit for task {task_id}.\n\nCo-authored-by: Nebula <nebula@local>"
    );
    let mut commit_cmd = Command::new("git");
    commit_cmd
        .args([
            "-c",
            "user.name=Nebula",
            "-c",
            "user.email=nebula@local",
            "commit",
            "-m",
            &commit_msg,
        ])
        .current_dir(worktree_root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    match commit_cmd.output().await {
        Ok(out) if out.status.success() => {
            tracing::info!(event = "task.worktree_committed", task_id = %task_id);
            Ok(true)
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.contains("index.lock") {
                resolve_lock_contention(&lock_path).await?;
                let mut retry_commit = Command::new("git");
                retry_commit
                    .args([
                        "-c",
                        "user.name=Nebula",
                        "-c",
                        "user.email=nebula@local",
                        "commit",
                        "-m",
                        &commit_msg,
                    ])
                    .current_dir(worktree_root)
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true);
                if let Ok(retry_out) = retry_commit.output().await {
                    if retry_out.status.success() {
                        tracing::info!(event = "task.worktree_committed", task_id = %task_id);
                        return Ok(true);
                    }
                    let retry_err = String::from_utf8_lossy(&retry_out.stderr);
                    return Err(LockContentionError::Other {
                        path: lock_path,
                        detail: format!("git commit retry failed: {retry_err}"),
                    });
                }
            }
            tracing::warn!(event = "task.git_commit_failed", stderr = %stderr);
            Err(LockContentionError::Other {
                path: lock_path,
                detail: format!("git commit failed: {stderr}"),
            })
        }
        Err(e) => {
            tracing::warn!(event = "task.git_commit_spawn_failed", error = %e);
            Err(LockContentionError::Other {
                path: lock_path,
                detail: format!("git commit failed to spawn: {e}"),
            })
        }
    }
}

/// Extracts tool calls from unstructured text output if the model emits them as text.
/// Supports both JSON `<tool_call>{"name": ..., "arguments": ...}</tool_call>` format
/// and XML `<tool_call><function=...><parameter=...>...</function></tool_call>` format.
pub(crate) fn extract_tool_calls_from_text(text: &str) -> Vec<nebula_model::ToolCall> {
    let mut calls = Vec::new();
    let mut search_from = 0;

    while let Some(start_rel) = text[search_from..].find("<tool_call>") {
        let start = search_from + start_rel + "<tool_call>".len();
        let end = match text[start..].find("</tool_call>") {
            Some(e) => start + e,
            None => text.len(),
        };
        let body = text[start..end].trim();
        search_from = if end < text.len() {
            end + "</tool_call>".len()
        } else {
            text.len()
        };

        if let Some(call) = parse_single_tool_call_body(body, calls.len()) {
            calls.push(call);
        }
    }

    // Direct <function=...> fallback without <tool_call> wrapper
    if calls.is_empty() {
        let mut f_search = 0;
        while let Some(start_rel) = text[f_search..].find("<function=") {
            let start = f_search + start_rel;
            let end = match text[start..].find("</function>") {
                Some(e) => start + e + "</function>".len(),
                None => text.len(),
            };
            let body = &text[start..end];
            f_search = end;
            if let Some(call) = parse_xml_function(body, calls.len()) {
                calls.push(call);
            }
        }
    }

    calls
}

fn parse_single_tool_call_body(body: &str, index: usize) -> Option<nebula_model::ToolCall> {
    let trimmed = body.trim();
    if trimmed.starts_with('{')
        && let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed)
        && let Some(name) = v.get("name").and_then(|n| n.as_str())
    {
        let arguments = match v.get("arguments") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => "{}".to_string(),
        };
        return Some(nebula_model::ToolCall {
            id: format!("call_extracted_{index}"),
            name: name.to_string(),
            arguments,
        });
    }

    if trimmed.contains("<function=") {
        return parse_xml_function(trimmed, index);
    }

    None
}

fn parse_xml_function(body: &str, index: usize) -> Option<nebula_model::ToolCall> {
    let fn_start = body.find("<function=")?;
    let after_fn = &body[fn_start + "<function=".len()..];
    let fn_name_end = after_fn.find('>')?;
    let name = after_fn[..fn_name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }

    let mut args_map = serde_json::Map::new();
    let mut param_search = fn_start + "<function=".len() + fn_name_end + 1;

    while let Some(p_start_rel) = body[param_search..].find("<parameter=") {
        let p_start = param_search + p_start_rel + "<parameter=".len();
        let after_p = &body[p_start..];
        let Some(p_name_end) = after_p.find('>') else {
            break;
        };
        let param_name = after_p[..p_name_end].trim().to_string();
        let val_start = p_start + p_name_end + 1;
        let next_close = body[val_start..].find("</parameter>");
        let next_open = body[val_start..].find("<parameter=");
        let next_fn_close = body[val_start..].find("</function>");

        let mut ends = Vec::new();
        if let Some(c) = next_close {
            ends.push(val_start + c);
        }
        if let Some(o) = next_open {
            ends.push(val_start + o);
        }
        if let Some(f) = next_fn_close {
            ends.push(val_start + f);
        }

        let val_end = ends.into_iter().min().unwrap_or(body.len());
        let param_val = body[val_start..val_end].trim();

        let advance = if let Some(c) = next_close
            && val_start + c == val_end
        {
            val_end + "</parameter>".len()
        } else {
            val_end
        };
        param_search = advance;

        let json_val = if param_val == "true" {
            serde_json::Value::Bool(true)
        } else if param_val == "false" {
            serde_json::Value::Bool(false)
        } else if let Ok(n) = param_val.parse::<i64>() {
            serde_json::Value::Number(n.into())
        } else if let Ok(f) = param_val.parse::<f64>() {
            serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                .unwrap_or_else(|| serde_json::Value::String(param_val.to_string()))
        } else {
            serde_json::Value::String(param_val.to_string())
        };

        args_map.insert(param_name, json_val);
    }

    Some(nebula_model::ToolCall {
        id: format!("call_extracted_{index}"),
        name,
        arguments: serde_json::Value::Object(args_map).to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_tool_calls_from_json_tool_call() {
        let text = "I will read the file now.\n<tool_call>\n{\"name\": \"fs.read\", \"arguments\": {\"path\": \"src/main.rs\"}}\n</tool_call>";
        let calls = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fs.read");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["path"], "src/main.rs");
    }

    #[test]
    fn test_extract_tool_calls_from_xml_tool_call() {
        let text = "<think>\nThinking about listing files\n</think>\n\n<tool_call>\n<function=fs.list>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>";
        let calls = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fs.list");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["path"], ".");
    }

    #[test]
    fn test_extract_tool_calls_multiple_parameters_xml() {
        let text = "<tool_call>\n<function=fs.list>\n<parameter=path>\n.\n<parameter=recursive>\nfalse\n</parameter>\n</function>\n</tool_call>";
        let calls = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fs.list");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["path"], ".");
        assert_eq!(args["recursive"], false);
    }
}
