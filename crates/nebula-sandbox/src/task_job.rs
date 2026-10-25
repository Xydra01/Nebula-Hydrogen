//! The per-task Job Object wrapper (issue #30): one [`TaskJob`] per task, built on the audited
//! [`job`](crate::job) FFI surface.
//!
//! A [`TaskJob`] owns one kill-on-close Windows Job Object for a single task, applies the task's
//! optional memory and CPU limits before any member can run, and — when a memory limit is set —
//! associates an IO completion port and runs a background pump that turns the OS's
//! `JOB_OBJECT_MSG_*` notifications into a [`TerminationReason`] the executor can read. Dropping the
//! `TaskJob` closes the last handle, so the OS terminates the whole process tree (Requirement 5, 6).
//!
//! The `CURRENT_TASK_JOB` task-local and the `spawn_in_task_job` seam that `shell.run` / `git.*`
//! call live in `nebula-tools` (beside the `WorktreeRootProvider` they sit next to), not here, for
//! the same dependency-cycle reason the worktree provider does — see
//! [`crate::worktree::provider`]. This module provides the `TaskJob` the executor creates and the
//! seam spawns into.

use std::sync::Arc;

use tokio::sync::Mutex;
use tokio::sync::mpsc;

use crate::job::{CompletionPort, Job, MemoryScope};

/// `JOB_OBJECT_MSG_*` notification codes this pump cares about.
///
/// Defined as plain constants (their documented numeric values) rather than pulled from the
/// `windows` crate so the mapping is unit-testable off-Windows and in one obvious place.
mod msg {
    /// A member exceeded the per-process committed-memory limit.
    pub const PROCESS_MEMORY_LIMIT: u32 = 9;
    /// The job exceeded its whole-job committed-memory limit.
    pub const JOB_MEMORY_LIMIT: u32 = 10;
    /// A member exited abnormally (crash / nonzero via unhandled exception).
    pub const ABNORMAL_EXIT_PROCESS: u32 = 6;
}

/// Why a task's processes were terminated, surfaced to the executor (issue #34 consumes this as a
/// task-status reason). Produced by the completion-port pump; `None` from
/// [`TaskJob::next_termination_reason`] means no reason has been observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminationReason {
    /// A configured committed-memory limit was exceeded. Carries the limit and the scope that was
    /// applied, so the executor can explain the kill as a limit breach rather than an ordinary exit.
    MemoryLimit {
        /// The committed-memory cap in bytes that was exceeded.
        limit_bytes: u64,
        /// Whether the cap was per-process or whole-job.
        scope: MemoryScope,
    },
    /// A member process exited abnormally (for example a crash), distinct from a limit breach.
    ProcessCrashed,
}

/// Map a job notification code to a [`TerminationReason`], given the configured limit.
///
/// Pure and `const`-friendly so it is exercised by a property test off-Windows: the memory-limit
/// messages carry the configured limit and scope, an abnormal exit is a crash, and every other code
/// (including the benign "last member exited") yields `None`.
#[must_use]
pub fn reason_for(
    msg_code: u32,
    limit_bytes: u64,
    scope: MemoryScope,
) -> Option<TerminationReason> {
    match msg_code {
        msg::PROCESS_MEMORY_LIMIT | msg::JOB_MEMORY_LIMIT => {
            Some(TerminationReason::MemoryLimit { limit_bytes, scope })
        }
        msg::ABNORMAL_EXIT_PROCESS => Some(TerminationReason::ProcessCrashed),
        _ => None,
    }
}

/// A committed-memory cap for a task's job (Requirement 7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryLimit {
    /// The cap in bytes. Validated at [`TaskJob::new`] to be `1..=installed_physical_memory`.
    pub bytes: u64,
    /// Whether the cap is enforced per-process or across the whole job.
    pub scope: MemoryScope,
}

/// A hard CPU-rate cap for a task's job (Requirement 8), expressed as a percentage `1..=100`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuRateLimit {
    /// Percentage of total CPU the job may use, `1..=100`.
    pub percent: u8,
}

/// Optional per-task limits applied to a [`TaskJob`]. Both default to off (Phase 1): with no
/// limits a task's job is kill-on-close only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JobLimits {
    /// Optional committed-memory cap.
    pub memory: Option<MemoryLimit>,
    /// Optional hard CPU-rate cap.
    pub cpu: Option<CpuRateLimit>,
}

/// Errors from the per-task Job Object API (Requirement 10.5). The single Tool-level error type the
/// executor and the spawn seam surface; the executor never sees a raw Win32 error.
#[derive(Debug, thiserror::Error)]
pub enum JobError {
    /// The OS job could not be created or configured. No job is left behind.
    #[error("could not create task job: {0}")]
    Create(String),
    /// A configured memory limit is outside `1..=installed_physical_memory`; no job was created
    /// (Requirement 7.2).
    #[error(
        "memory limit {bytes} bytes is out of range (installed physical memory {physical} bytes)"
    )]
    LimitOutOfRange {
        /// The rejected cap.
        bytes: u64,
        /// Installed physical memory at the time of the check.
        physical: u64,
    },
    /// A spawn through the seam was attempted with no task in scope; nothing was spawned
    /// (Requirement 10.4).
    #[error("no task in scope: a built-in tried to spawn a child outside any task's job")]
    NoTaskInScope,
    /// The child process could not be spawned.
    #[error("could not spawn child process: {0}")]
    Spawn(String),
    /// The spawned child could not be assigned to its task's job; the still-suspended child was
    /// terminated before it could run (Requirement 4.5, 4.6).
    #[error("could not assign child to task job for {task_id}: {detail}")]
    Assign {
        /// The task whose job the child could not join.
        task_id: String,
        /// The underlying failure detail.
        detail: String,
    },
    /// The suspended child exited before its assignment could be confirmed (Requirement 2.7).
    #[error("child exited before it could be assigned to its task job")]
    ExitedBeforeAssign,
}

/// One Windows Job Object per task (Requirement 1).
///
/// Owns the kill-on-close [`Job`], its configured [`JobLimits`], and — when a memory limit is set —
/// the completion-port pump that surfaces a [`TerminationReason`]. Dropping it closes the last
/// handle so kill-on-close terminates the whole process tree (Requirement 5, 6). Held by the
/// executor as `Arc<TaskJob>` and read by the spawn seam to assign children.
pub struct TaskJob {
    task_id: String,
    job: Job,
    limits: JobLimits,
    /// Present only when a memory limit is configured: the receiver end of the pump's reason
    /// channel. The pump task and the `CompletionPort` are owned by `_pump` so they live as long as
    /// the `TaskJob`.
    reasons: Option<Mutex<mpsc::UnboundedReceiver<TerminationReason>>>,
    /// Keeps the pump task and its completion port alive for the `TaskJob`'s lifetime; dropped with
    /// the `TaskJob`, which closes the port and ends the pump.
    _pump: Option<Pump>,
}

/// Owns the completion port and the background pump task; dropping it closes the port (which
/// unblocks and ends the pump).
struct Pump {
    _port: Arc<CompletionPort>,
    handle: tokio::task::JoinHandle<()>,
}

impl Drop for Pump {
    fn drop(&mut self) {
        // Dropping `_port` closes the completion-port handle, which makes the pump's blocking
        // `next_message` return `Ok(None)` so the task exits; abort as a backstop in case it is
        // parked elsewhere.
        self.handle.abort();
    }
}

impl TaskJob {
    /// Create the task's job, applying `limits` before any member can be assigned.
    ///
    /// A configured memory limit is first range-checked against installed physical memory
    /// (`1..=physical`); an out-of-range value is rejected with [`JobError::LimitOutOfRange`] and no
    /// job is created (Requirement 7.2). Otherwise the kill-on-close job is built with the limits in
    /// force (Requirement 1.1, 6.4, 7.1, 8.1), and when a memory limit is set a completion port is
    /// associated and a background pump started so [`next_termination_reason`](Self::next_termination_reason)
    /// can report a limit breach (Requirement 7.5).
    ///
    /// # Errors
    /// [`JobError::LimitOutOfRange`] for an out-of-range memory cap; [`JobError::Create`] if the OS
    /// job cannot be created or configured.
    pub fn new(task_id: impl Into<String>, limits: JobLimits) -> Result<Self, JobError> {
        let task_id = task_id.into();

        if let Some(mem) = limits.memory {
            let physical = installed_physical_memory_bytes();
            if mem.bytes < 1 || mem.bytes > physical {
                return Err(JobError::LimitOutOfRange {
                    bytes: mem.bytes,
                    physical,
                });
            }
        }

        let memory = limits.memory.map(|m| (m.bytes, m.scope));
        let cpu = limits.cpu.map(|c| u32::from(c.percent) * 100); // percent → hundredths
        let job = Job::new_with_limits(memory, cpu).map_err(|e| JobError::Create(e.to_string()))?;

        // Only run a pump when a memory limit is configured — that is the only reason we need to
        // learn *why* a member died. A limit-free job still kills on close; it just has no reason to
        // report.
        let (reasons, pump) = if let Some(mem) = limits.memory {
            let port = Arc::new(
                CompletionPort::associate(&job).map_err(|e| JobError::Create(e.to_string()))?,
            );
            let (tx, rx) = mpsc::unbounded_channel();
            let pump_port = Arc::clone(&port);
            let limit_bytes = mem.bytes;
            let scope = mem.scope;
            // The pump blocks on `next_message`, so it runs on a blocking thread.
            let handle = tokio::task::spawn_blocking(move || {
                while let Ok(Some((code, _pid))) = pump_port.next_message() {
                    if let Some(reason) = reason_for(code, limit_bytes, scope) {
                        // The receiver may already be gone if the TaskJob was dropped; ignore.
                        let _ = tx.send(reason);
                    }
                }
            });
            (
                Some(Mutex::new(rx)),
                Some(Pump {
                    _port: port,
                    handle,
                }),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            task_id,
            job,
            limits,
            reasons,
            _pump: pump,
        })
    }

    /// The task this job belongs to.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// The limits applied to this job.
    #[must_use]
    pub fn limits(&self) -> JobLimits {
        self.limits
    }

    /// Assign an already-spawned (suspended) child to this task's job.
    ///
    /// The spawn seam calls this between [`crate::job::spawn_suspended`] and
    /// [`crate::job::ResumeHandle::resume`], so the child joins the job before it can run
    /// (Requirement 2.1, 4.1).
    ///
    /// # Errors
    /// [`JobError::Assign`] if the OS assignment fails.
    pub fn assign(&self, process: std::os::windows::io::RawHandle) -> Result<(), JobError> {
        self.job.assign(process).map_err(|e| JobError::Assign {
            task_id: self.task_id.clone(),
            detail: e.to_string(),
        })
    }

    /// Await the next termination reason the pump has observed (for example a memory-limit breach),
    /// or `None` when no memory limit is configured (so no pump runs) or the pump has ended
    /// (Requirement 7.5).
    pub async fn next_termination_reason(&self) -> Option<TerminationReason> {
        match &self.reasons {
            Some(rx) => rx.lock().await.recv().await,
            None => None,
        }
    }

    /// Immediately terminate all member processes belonging to this task's job.
    ///
    /// Used by global stop (`nebula stop` / `task.stop_all`) to end active task execution.
    ///
    /// # Errors
    /// [`JobError`] if OS termination fails.
    pub fn terminate(&self) -> Result<(), JobError> {
        self.job.terminate(1).map_err(|e| JobError::Assign {
            task_id: self.task_id.clone(),
            detail: format!("terminate failed: {e}"),
        })
    }

    /// Query the number of currently active processes belonging to this task's job.
    ///
    /// # Errors
    /// [`JobError`] if the OS query fails.
    pub fn active_processes(&self) -> Result<u32, JobError> {
        self.job.active_processes().map_err(|e| JobError::Assign {
            task_id: self.task_id.clone(),
            detail: format!("query active processes failed: {e}"),
        })
    }
}

/// Installed physical memory in bytes, used to range-check a configured memory cap.
///
/// Falls back to `u64::MAX` if the OS query fails, so a query failure never spuriously rejects a
/// legitimate limit (the OS still enforces the cap it was given).
#[cfg(windows)]
fn installed_physical_memory_bytes() -> u64 {
    use windows::Win32::System::SystemInformation::GetPhysicallyInstalledSystemMemory;
    let mut kib: u64 = 0;
    // SAFETY: `kib` is a valid out-pointer that outlives the call; the function writes the
    // installed memory in kibibytes on success.
    #[allow(unsafe_code)]
    let ok = unsafe { GetPhysicallyInstalledSystemMemory(&mut kib) }.is_ok();
    if ok {
        kib.saturating_mul(1024)
    } else {
        u64::MAX
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::{JobLimits, MemoryLimit, MemoryScope, TaskJob, TerminationReason, msg, reason_for};

    // ---- reason_for mapping (Property 8, decidable half; Requirement 7.3, 7.4, 7.5) ----

    #[test]
    fn per_process_and_job_memory_messages_map_to_memory_limit() {
        let got = reason_for(msg::PROCESS_MEMORY_LIMIT, 4096, MemoryScope::PerProcess);
        assert_eq!(
            got,
            Some(TerminationReason::MemoryLimit {
                limit_bytes: 4096,
                scope: MemoryScope::PerProcess,
            }),
        );
        let got = reason_for(msg::JOB_MEMORY_LIMIT, 8192, MemoryScope::WholeJob);
        assert_eq!(
            got,
            Some(TerminationReason::MemoryLimit {
                limit_bytes: 8192,
                scope: MemoryScope::WholeJob,
            }),
        );
    }

    #[test]
    fn abnormal_exit_maps_to_process_crashed() {
        assert_eq!(
            reason_for(msg::ABNORMAL_EXIT_PROCESS, 1, MemoryScope::WholeJob),
            Some(TerminationReason::ProcessCrashed),
        );
    }

    #[test]
    fn benign_messages_map_to_none() {
        // "last member exited" (7) and any unknown code carry no termination reason.
        for code in [7u32, 0, 4, 100] {
            assert_eq!(
                reason_for(code, 4096, MemoryScope::WholeJob),
                None,
                "code {code} must not surface a termination reason",
            );
        }
    }

    // Feature: job-objects-for-tasks, Property 8: a memory-limit termination surfaces the correct reason
    //
    // Validates: Requirements 7.3, 7.4, 7.5 (the decidable mapping half; the real-OS half is a
    // Windows integration test). For any (message code, limit, scope): the two memory-limit codes
    // yield exactly MemoryLimit carrying the applied limit and scope, abnormal-exit yields
    // ProcessCrashed, and every other code yields None.
    mod property_reason_mapping {
        use super::super::{MemoryScope, TerminationReason, msg, reason_for};
        use proptest::prelude::*;

        fn any_scope() -> impl Strategy<Value = MemoryScope> {
            prop_oneof![Just(MemoryScope::PerProcess), Just(MemoryScope::WholeJob)]
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn reason_mapping_is_correct(code in any::<u32>(), limit in any::<u64>(), scope in any_scope()) {
                let got = reason_for(code, limit, scope);
                let expected = if code == msg::PROCESS_MEMORY_LIMIT || code == msg::JOB_MEMORY_LIMIT {
                    Some(TerminationReason::MemoryLimit { limit_bytes: limit, scope })
                } else if code == msg::ABNORMAL_EXIT_PROCESS {
                    Some(TerminationReason::ProcessCrashed)
                } else {
                    None
                };
                prop_assert_eq!(got, expected);
            }
        }
    }

    // ---- memory-limit range check at TaskJob::new (Property 9; Requirement 7.2) ----

    #[test]
    fn zero_byte_memory_limit_is_rejected_before_any_job_is_created() {
        let limits = JobLimits {
            memory: Some(MemoryLimit {
                bytes: 0,
                scope: MemoryScope::WholeJob,
            }),
            cpu: None,
        };
        match TaskJob::new("task-zero", limits) {
            Err(super::JobError::LimitOutOfRange { bytes: 0, .. }) => {}
            Err(other) => panic!("expected LimitOutOfRange, got {other:?}"),
            Ok(_) => panic!("a zero-byte memory limit is out of range and must be rejected"),
        }
    }

    #[test]
    fn over_physical_memory_limit_is_rejected() {
        // u64::MAX bytes cannot be <= installed physical memory, so this is always out of range,
        // and no job is created.
        let limits = JobLimits {
            memory: Some(MemoryLimit {
                bytes: u64::MAX,
                scope: MemoryScope::PerProcess,
            }),
            cpu: None,
        };
        match TaskJob::new("task-huge", limits) {
            Err(super::JobError::LimitOutOfRange { bytes, .. }) if bytes == u64::MAX => {}
            Err(other) => panic!("expected LimitOutOfRange, got {other:?}"),
            Ok(_) => panic!("a limit larger than installed physical memory must be rejected"),
        }
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn no_limits_builds_a_job_and_reports_no_reason() {
        // A limit-free task job is kill-on-close only: it builds, runs no pump, and surfaces no
        // termination reason.
        let job = TaskJob::new("task-plain", JobLimits::default())
            .expect("a limit-free task job must build");
        assert_eq!(job.task_id(), "task-plain");
        assert_eq!(job.limits(), JobLimits::default());
        assert_eq!(
            job.next_termination_reason().await,
            None,
            "with no memory limit there is no pump, so no reason is ever surfaced",
        );
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn an_in_range_memory_limit_builds_a_job() {
        // 64 MiB is comfortably within installed physical memory on any machine that runs the
        // suite, so the job builds with the limit applied.
        let limits = JobLimits {
            memory: Some(MemoryLimit {
                bytes: 64 * 1024 * 1024,
                scope: MemoryScope::WholeJob,
            }),
            cpu: None,
        };
        let job =
            TaskJob::new("task-mem", limits).expect("an in-range memory limit must build a job");
        assert_eq!(job.limits().memory.unwrap().bytes, 64 * 1024 * 1024);
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn active_processes_tracks_and_terminates_children() {
        use std::os::windows::io::AsRawHandle;
        use std::process::Command;

        let job = TaskJob::new("task-active-proc", JobLimits::default()).unwrap();
        assert_eq!(job.active_processes().unwrap(), 0);

        // Spawn a child process and assign to job
        let mut child = Command::new("cmd")
            .args(["/C", "ping", "-n", "30", "127.0.0.1"])
            .spawn()
            .expect("spawn ping child");

        job.assign(child.as_raw_handle() as _)
            .expect("assign child to job");
        let active = job.active_processes().unwrap();
        assert!(active >= 1, "active processes should be >= 1, got {active}");

        job.terminate().expect("terminate job");
        // Active processes should drop to 0
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut final_count = u32::MAX;
        while std::time::Instant::now() < deadline {
            if let Ok(c) = job.active_processes() {
                final_count = c;
                if c == 0 {
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(final_count, 0, "processes must drain to 0 after terminate");
        let _ = child.kill();
    }
}
