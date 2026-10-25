//! `nebula task`: run or cancel autonomous executor tasks.

use std::io::Write;
use std::path::PathBuf;

use anyhow::Context as _;
use nebula_proto::{
    Empty, Event, Method, TaskCancelParams, TaskCreateParams, TaskId, TaskStarted, TaskStatus,
};

use crate::{Ctx, connect, emit_json, format};

/// Arguments for `nebula task run`.
#[derive(Debug, Clone, clap::Args)]
pub struct TaskRunArgs {
    /// Goal / prompt describing the objective for Nebula.
    pub goal: String,
    /// Path to the target git repository (defaults to current directory).
    #[arg(long)]
    pub repo: Option<PathBuf>,
    /// Git ref/commit to branch from (defaults to HEAD).
    #[arg(long)]
    pub start_point: Option<String>,
    /// Model profile override.
    #[arg(long)]
    pub profile: Option<String>,
    /// Maximum tool calls before stopping.
    #[arg(long)]
    pub max_tool_calls: Option<u32>,
    /// Maximum total tokens before stopping.
    #[arg(long)]
    pub max_tokens: Option<u32>,
    /// Wall-clock timeout in seconds.
    #[arg(long)]
    pub wall_time_secs: Option<u64>,
    /// Automatically push branch and open a GitHub PR upon task completion.
    #[arg(long, conflicts_with = "no_pr")]
    pub pr: bool,
    /// Do not open a GitHub PR upon task completion.
    #[arg(long, conflicts_with = "pr")]
    pub no_pr: bool,
    /// Target language module identifier (e.g. "python", "cpp", "sql").
    #[arg(long = "lang")]
    pub language: Option<String>,
    /// Optional Python runtime version constraint (e.g. "3.11", "3.12").
    #[arg(long = "python-version")]
    pub python_version: Option<String>,
    /// Optional observable verification command target (e.g. "pytest" or "test.run").
    #[arg(long)]
    pub verification_target: Option<String>,
    /// Number of concurrent streams assigned to this task (defaults to 1).
    #[arg(long, default_value = "1")]
    pub streams: u32,
}

/// Heuristically detects the workspace language from repository manifest files.
fn detect_workspace_language(repo: &std::path::Path) -> Option<String> {
    if repo.join("Cargo.toml").is_file() {
        Some("rust".into())
    } else if repo.join("pyproject.toml").is_file() || repo.join("requirements.txt").is_file() {
        Some("python".into())
    } else if repo.join("CMakeLists.txt").is_file() || repo.join("compile_commands.json").is_file()
    {
        Some("cpp".into())
    } else if repo.join("schema.sql").is_file() {
        Some("sql".into())
    } else {
        None
    }
}

/// Run an autonomous task to completion, streaming steps.
pub(crate) async fn run(ctx: &Ctx, args: &TaskRunArgs, out: &mut dyn Write) -> anyhow::Result<u8> {
    let repo = match &args.repo {
        Some(p) => p.clone(),
        None => std::env::current_dir().context("getting current directory for repo")?,
    };

    let create_pr = if args.pr {
        Some(true)
    } else if args.no_pr {
        Some(false)
    } else {
        None
    };

    let language = args
        .language
        .clone()
        .or_else(|| detect_workspace_language(&repo));

    let language_config = args.python_version.as_ref().map(|pv| {
        serde_json::json!({
            "python_version": pv
        })
    });

    let mut c = connect(ctx).await?;
    let params = TaskCreateParams {
        goal: args.goal.clone(),
        repo,
        start_point: args.start_point.clone(),
        profile: args.profile.clone(),
        max_tool_calls: args.max_tool_calls,
        max_tokens: args.max_tokens,
        wall_time_secs: args.wall_time_secs,
        create_pr,
        language,
        language_config,
        verification_target: args.verification_target.clone(),
        streams: Some(args.streams),
    };

    let started: TaskStarted = c.call(Method::TaskCreate(params)).await?;
    let task_id = started.task_id;

    if ctx.json {
        emit_json(out, &started)?;
    } else {
        writeln!(
            out,
            "task {} started {}",
            task_id,
            ctx.style.dim(&format!("(trace {})", started.trace_id))
        )?;
        out.flush()?;
    }

    let mut cancel_sent = false;
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        let ev = tokio::select! {
            ev = c.next_event() => ev?,
            _ = &mut ctrl_c, if !cancel_sent => {
                cancel_sent = true;
                if !ctx.json {
                    writeln!(out, "{}", ctx.style.warn("cancelling task..."))?;
                    out.flush()?;
                }
                let _ = c.call::<Empty>(Method::TaskCancel(TaskCancelParams { task_id })).await;
                continue;
            }
        };

        match ev {
            Event::TaskStep(step) if step.task_id == task_id => {
                if ctx.json {
                    emit_json(out, &step)?;
                } else {
                    writeln!(out, "{}", format::task_step(&step, ctx.style))?;
                    out.flush()?;
                }
            }
            Event::ChatToken(t) if t.text.starts_with("  └─>") => {
                if !ctx.json {
                    write!(out, "{}", t.text)?;
                    out.flush()?;
                }
            }
            Event::ApprovalNeeded(req) => {
                if !ctx.json {
                    writeln!(
                        out,
                        "{}",
                        ctx.style.warn(&format!(
                            "approval needed for '{}' ({}) - approve in TUI or via ntfy",
                            req.command, req.tier
                        ))
                    )?;
                    out.flush()?;
                }
            }
            Event::TaskDone(done) if done.task_id == task_id => {
                if ctx.json {
                    emit_json(out, &done)?;
                } else {
                    writeln!(out, "{}", format::task_done(&done, ctx.style))?;
                    out.flush()?;
                }
                return match done.status {
                    TaskStatus::Completed => Ok(0),
                    TaskStatus::Cancelled => Ok(130),
                    _ => Ok(1),
                };
            }
            _ => {}
        }
    }
}

/// Cancel a running task.
pub(crate) async fn cancel(ctx: &Ctx, task_id: TaskId, out: &mut dyn Write) -> anyhow::Result<u8> {
    let mut c = connect(ctx).await?;
    let _: Empty = c
        .call(Method::TaskCancel(TaskCancelParams { task_id }))
        .await?;

    if ctx.json {
        emit_json(
            out,
            &serde_json::json!({ "task_id": task_id, "cancelled": true }),
        )?;
    } else {
        writeln!(out, "task {task_id} cancelled")?;
    }
    Ok(0)
}

/// Stop all running tasks immediately.
pub(crate) async fn stop_all(
    ctx: &Ctx,
    reason: Option<String>,
    out: &mut dyn Write,
) -> anyhow::Result<u8> {
    let mut c = connect(ctx).await?;
    let res: nebula_proto::TaskStopAllResponse = c
        .call(Method::TaskStopAll(nebula_proto::TaskStopAllParams {
            reason,
        }))
        .await?;

    if ctx.json {
        emit_json(out, &serde_json::json!({ "stopped": res.stopped }))?;
    } else if res.stopped == 0 {
        writeln!(out, "no running tasks to stop")?;
    } else {
        writeln!(out, "stopped {} running task(s)", res.stopped)?;
    }
    Ok(0)
}
