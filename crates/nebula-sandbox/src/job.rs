//! The one audited `unsafe` Win32 Job Object surface in the workspace (issue #30).
//!
//! This module owns every job-object FFI call Nebula makes: [`CreateJobObjectW`], [`SetInformationJobObject`],
//! [`AssignProcessToJobObject`], the job-handle [`CloseHandle`], the IO-completion-port calls
//! ([`CreateIoCompletionPort`]/[`GetQueuedCompletionStatus`]) used to learn *why* a member died,
//! and the `CREATE_SUSPENDED` + [`ResumeThread`] plumbing that assigns a child to its job before it
//! can run. No other module in the workspace declares these calls — `nebula-model` (its
//! per-process, daemon-lifetime model-server job) and `nebula-tools` (`shell.run` / `git.*` task
//! children, and the daemon-lifetime tool-server job) both build on this one surface rather than
//! keeping duplicate copies (Requirement 9).
//!
//! Per AGENTS.md, `unsafe` is denied everywhere except isolated modules like this one, and every
//! `unsafe` block carries a `// SAFETY:` comment stating the invariant it relies on.
//!
//! # Why a job per task
//!
//! A Windows Job Object groups processes: a child spawned by a member is itself a member (unless it
//! is explicitly allowed to break away, which this module never permits), and closing the last
//! handle to a job configured with [`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`] terminates every member —
//! children and grandchildren alike — even if the owning process crashed. Assigning every process a
//! task starts to one per-task job therefore makes "cancel the task" and "the daemon died" both
//! reduce to "close the job handle", with no survivors (issue #30, Phase 1 exit criterion 3).
//!
//! # Breakaway is never permitted
//!
//! This module never sets `JOB_OBJECT_LIMIT_BREAKAWAY_OK` or `JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK`,
//! so a descendant that passes `CREATE_BREAKAWAY_FROM_JOB` is denied and stays a member. Job
//! membership is thus inherited by the whole process tree at any depth (Requirement 3).

// This is the isolated, audited Win32 Job Object FFI surface; `unsafe` is allowed here (and only
// here, plus the model/tools launchers that will migrate to it) per AGENTS.md, with a `// SAFETY:`
// comment on every `unsafe` block.
#![allow(unsafe_code)]

use std::io;
use std::os::windows::io::RawHandle;

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_CPU_RATE_CONTROL_ENABLE,
    JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP, JOB_OBJECT_LIMIT_JOB_MEMORY,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_CPU_RATE_CONTROL_INFORMATION,
    JOBOBJECT_CPU_RATE_CONTROL_INFORMATION_0, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JobObjectBasicAccountingInformation, JobObjectCpuRateControlInformation,
    JobObjectExtendedLimitInformation, QueryInformationJobObject, SetInformationJobObject,
    TerminateJobObject,
};

use windows::core::PCWSTR;

/// Which committed-memory cap a [`Job`] enforces (Requirement 7).
///
/// A per-process cap terminates only the offending member when it is exceeded; a whole-job cap
/// terminates the entire job. The scope is chosen by configuration and reported on the resulting
/// termination reason so the executor can explain why a task stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryScope {
    /// `JOB_OBJECT_LIMIT_PROCESS_MEMORY`: the cap applies to each member process individually.
    PerProcess,
    /// `JOB_OBJECT_LIMIT_JOB_MEMORY`: the cap applies to the job's total committed memory.
    WholeJob,
}

/// Map a `windows` error to a plain [`io::Error`], so callers never see the `windows` crate.
fn win_err(e: &windows::core::Error) -> io::Error {
    io::Error::from_raw_os_error(e.code().0)
}

/// A Windows Job Object configured with `KILL_ON_JOB_CLOSE` and, optionally, a committed-memory
/// cap and a CPU-rate cap.
///
/// Dropping the handle — including on a crash of the owning process — closes it; when it is the
/// last handle to the job, Windows terminates every member at any depth (Requirement 5, 6). The
/// handle is created without inheritance and is never duplicated into a child, so a spawned child
/// cannot keep the job alive after the owner dies (Requirement 6.5).
///
/// This type is the raw FFI primitive. The per-task lifecycle wrapper that associates a completion
/// port, enforces the limit-range check, and runs the notification pump is
/// [`TaskJob`](crate::task_job::TaskJob); `nebula-model` uses [`Job::new_kill_on_close`] directly
/// for its per-process model-server job.
pub struct Job(HANDLE);

// SAFETY: a job handle is a kernel HANDLE, which Windows documents as usable from any thread; the
// `Job` owns it exclusively and closes it exactly once on drop.
unsafe impl Send for Job {}
// SAFETY: as above — the handle is only read (assign/set-information/close) and those calls are
// internally synchronized by the kernel, so sharing `&Job` across threads is sound.
unsafe impl Sync for Job {}

impl Job {
    /// Create a kill-on-close job with no memory or CPU cap.
    ///
    /// Used by `nebula-model` for each llama-server's own per-process, daemon-lifetime job and by
    /// `nebula-tools` for the daemon-lifetime tool-server job: both want kill-on-close grouping
    /// without per-task limits.
    ///
    /// # Errors
    /// The underlying Win32 error if the job cannot be created or configured.
    pub fn new_kill_on_close() -> io::Result<Self> {
        Self::new_with_limits(None, None)
    }

    /// Create a kill-on-close job and apply the given optional memory and CPU limits.
    ///
    /// Every limit is set **before** this returns, so no member can be assigned (and therefore no
    /// member can run) before the limits are in force (Requirement 6.4, 7.1, 8.1). `memory` is a
    /// `(cap_in_bytes, scope)` pair; `cpu_rate_hundredths_percent` is a hard CPU cap in hundredths
    /// of a percent (for example `2500` = 25%).
    ///
    /// # Errors
    /// The underlying Win32 error if the job cannot be created or any limit cannot be set.
    pub fn new_with_limits(
        memory: Option<(u64, MemoryScope)>,
        cpu_rate_hundredths_percent: Option<u32>,
    ) -> io::Result<Self> {
        // SAFETY: no security attributes and no name; the returned handle is owned by `Job` and
        // closed exactly once on drop. A failure returns an `Err` and leaves nothing to close.
        let handle = unsafe { CreateJobObjectW(None, PCWSTR::null()) }.map_err(|e| win_err(&e))?;
        let job = Self(handle);

        // Always kill-on-close; add a memory cap when configured. Breakaway flags are never set,
        // so membership is always inherited by descendants (Requirement 3.4).
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if let Some((bytes, scope)) = memory {
            let cap = usize::try_from(bytes).map_err(io::Error::other)?;
            match scope {
                MemoryScope::PerProcess => {
                    info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;
                    info.ProcessMemoryLimit = cap;
                }
                MemoryScope::WholeJob => {
                    info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY;
                    info.JobMemoryLimit = cap;
                }
            }
        }
        let size = u32::try_from(size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>())
            .map_err(io::Error::other)?;
        // SAFETY: `info` is a valid, fully-initialized JOBOBJECT_EXTENDED_LIMIT_INFORMATION of
        // `size` bytes that outlives the call; `job.0` is a live job handle created just above.
        unsafe {
            SetInformationJobObject(
                job.0,
                JobObjectExtendedLimitInformation,
                (&raw const info).cast(),
                size,
            )
        }
        .map_err(|e| win_err(&e))?;

        if let Some(rate) = cpu_rate_hundredths_percent {
            let mut cpu = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION {
                ControlFlags: JOB_OBJECT_CPU_RATE_CONTROL_ENABLE
                    | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
                Anonymous: JOBOBJECT_CPU_RATE_CONTROL_INFORMATION_0 { CpuRate: rate },
            };
            let cpu_size = u32::try_from(size_of::<JOBOBJECT_CPU_RATE_CONTROL_INFORMATION>())
                .map_err(io::Error::other)?;
            // SAFETY: `cpu` is a valid, fully-initialized JOBOBJECT_CPU_RATE_CONTROL_INFORMATION of
            // `cpu_size` bytes that outlives the call; `job.0` is a live job handle.
            unsafe {
                SetInformationJobObject(
                    job.0,
                    JobObjectCpuRateControlInformation,
                    (&raw mut cpu).cast(),
                    cpu_size,
                )
            }
            .map_err(|e| win_err(&e))?;
        }

        Ok(job)
    }

    /// Assign a process (by its raw OS handle) to this job.
    ///
    /// Placing a process in a job does not grant the process a handle *to* the job, so this never
    /// duplicates an inheritable job handle into the child (Requirement 6.5). Must be called while
    /// the process is still suspended, so the child cannot spawn an unassigned descendant before it
    /// joins the job (Requirement 4.1).
    ///
    /// # Errors
    /// The underlying Win32 error if the assignment fails.
    pub fn assign(&self, process: RawHandle) -> io::Result<()> {
        // SAFETY: both handles are valid for the duration of the call; `self.0` is this `Job`'s
        // live handle and `process` is borrowed from a live `Child`.
        unsafe { AssignProcessToJobObject(self.0, HANDLE(process)) }.map_err(|e| win_err(&e))
    }

    /// Immediately terminate all processes currently associated with this job.
    ///
    /// Used by global stop (`nebula stop` / `task.stop_all`) to kill all running task processes
    /// synchronously without waiting for handle drop (Requirement 3.2, 3.3).
    ///
    /// # Errors
    /// The underlying Win32 error if termination fails.
    pub fn terminate(&self, exit_code: u32) -> io::Result<()> {
        // SAFETY: `self.0` is a valid, live Job Object handle.
        unsafe { TerminateJobObject(self.0, exit_code) }.map_err(|e| win_err(&e))
    }

    /// Query the number of active processes currently associated with this job.
    ///
    /// # Errors
    /// The underlying Win32 error if the query fails.
    pub fn active_processes(&self) -> io::Result<u32> {
        let mut info = std::mem::MaybeUninit::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>::zeroed();
        let size = u32::try_from(std::mem::size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>())
            .map_err(io::Error::other)?;
        let mut return_length = 0;
        // SAFETY: `self.0` is a live job handle, `info` is a valid buffer of `size` bytes.
        unsafe {
            QueryInformationJobObject(
                Some(self.0),
                JobObjectBasicAccountingInformation,
                info.as_mut_ptr().cast(),
                size,
                Some(&mut return_length),
            )
        }
        .map_err(|e| win_err(&e))?;

        // SAFETY: QueryInformationJobObject succeeded and initialized the struct.
        let info = unsafe { info.assume_init() };
        Ok(info.ActiveProcesses)
    }

    /// The raw job handle, for associating a completion port (see [`crate::job::CompletionPort`]).
    pub(crate) fn raw(&self) -> HANDLE {
        self.0
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `new_with_limits` and is closed exactly once here.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

// ---------------------------------------------------------------------------------------------
// IO completion port: learning *why* a member died (Requirement 7).
// ---------------------------------------------------------------------------------------------

use windows::Win32::System::IO::{CreateIoCompletionPort, GetQueuedCompletionStatus};
use windows::Win32::System::JobObjects::{
    JOBOBJECT_ASSOCIATE_COMPLETION_PORT, JobObjectAssociateCompletionPortInformation,
};

/// An IO completion port associated with a [`Job`], used to receive `JOB_OBJECT_MSG_*`
/// notifications so the daemon can tell a memory-limit kill from an ordinary exit (Requirement 7).
///
/// A job with no completion port still enforces its limits — the OS terminates the offending
/// member or job regardless. The port exists only to *report the reason*, so it is created only
/// when a memory limit is configured.
pub struct CompletionPort(HANDLE);

// SAFETY: a completion-port handle is a kernel HANDLE usable from any thread; `CompletionPort`
// owns it exclusively and closes it once on drop. The pump calls `next_message` from a single
// dedicated blocking task.
unsafe impl Send for CompletionPort {}
// SAFETY: as above — the handle is only polled and closed, both kernel-synchronized.
unsafe impl Sync for CompletionPort {}

impl CompletionPort {
    /// Create a completion port and associate it with `job`.
    ///
    /// The association uses the job handle as the port's completion key, so a later
    /// [`next_message`](Self::next_message) returns the job's `JOB_OBJECT_MSG_*` codes.
    ///
    /// # Errors
    /// The underlying Win32 error if the port cannot be created or associated.
    pub fn associate(job: &Job) -> io::Result<Self> {
        // SAFETY: creating a brand-new port — no existing handle to extend, a null key and zero
        // threads are the documented "new port" arguments. The returned handle is owned here.
        let port = unsafe {
            CreateIoCompletionPort(windows::Win32::Foundation::INVALID_HANDLE_VALUE, None, 0, 0)
        }
        .map_err(|e| win_err(&e))?;
        let port = Self(port);

        let assoc = JOBOBJECT_ASSOCIATE_COMPLETION_PORT {
            // Use the job handle as the completion key so messages are attributable to it.
            CompletionKey: job.raw().0,
            CompletionPort: port.0,
        };
        let size = u32::try_from(size_of::<JOBOBJECT_ASSOCIATE_COMPLETION_PORT>())
            .map_err(io::Error::other)?;
        // SAFETY: `assoc` is a valid, fully-initialized association struct of `size` bytes that
        // outlives the call; `job.raw()` and `port.0` are both live handles.
        unsafe {
            SetInformationJobObject(
                job.raw(),
                JobObjectAssociateCompletionPortInformation,
                (&raw const assoc).cast(),
                size,
            )
        }
        .map_err(|e| win_err(&e))?;

        Ok(port)
    }

    /// Block for the next job notification, returning `(message_code, member_pid)`.
    ///
    /// Returns `Ok(None)` when the port is closed (the pump should then exit). Call this from a
    /// dedicated blocking task — it blocks the calling thread until a message arrives.
    ///
    /// # Errors
    /// The underlying Win32 error for a genuine failure (not a normal close).
    pub fn next_message(&self) -> io::Result<Option<(u32, u32)>> {
        let mut code: u32 = 0;
        let mut key: usize = 0;
        let mut overlapped = std::ptr::null_mut();
        // SAFETY: all out-params are valid local pointers that outlive the call; `self.0` is a
        // live completion-port handle. `GetQueuedCompletionStatus` writes the message code into
        // `code`, the completion key into `key`, and the per-message value (the member PID, for
        // job messages) into `overlapped`.
        let result = unsafe {
            GetQueuedCompletionStatus(
                self.0,
                &mut code,
                &mut key,
                &mut overlapped,
                windows::Win32::System::Threading::INFINITE,
            )
        };
        match result {
            Ok(()) => {
                // For job messages the "overlapped" out-param carries the member PID as a value,
                // not a real pointer. Narrow it to a u32 pid (PIDs are 32-bit on Windows, so the
                // low 32 bits are the whole value — the truncation is intentional).
                #[allow(clippy::cast_possible_truncation)]
                let pid = (overlapped as usize as u64 & u64::from(u32::MAX)) as u32;
                Ok(Some((code, pid)))
            }
            Err(e) => {
                // A closed port surfaces as an abort/invalid-handle error; treat that as "no more
                // messages" so the pump exits cleanly rather than logging a spurious failure.
                let raw = e.code().0;
                const ERROR_ABANDONED_WAIT_0: i32 = 0x8007_02DF_u32 as i32;
                const ERROR_INVALID_HANDLE: i32 = 0x8007_0006_u32 as i32;
                if raw == ERROR_ABANDONED_WAIT_0 || raw == ERROR_INVALID_HANDLE {
                    Ok(None)
                } else {
                    Err(win_err(&e))
                }
            }
        }
    }
}

impl Drop for CompletionPort {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `associate` and is closed exactly once here. Closing
        // it also unblocks a pump parked in `next_message` (it returns the abandoned-wait error,
        // which `next_message` maps to `Ok(None)`).
        let _ = unsafe { CloseHandle(self.0) };
    }
}

// ---------------------------------------------------------------------------------------------
// Suspended spawn: assign the child to its job before it can run (Requirement 4).
// ---------------------------------------------------------------------------------------------

use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows::Win32::System::Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME};

/// `CREATE_SUSPENDED | CREATE_NO_WINDOW` — spawn the child suspended and without a console window.
const CREATE_SUSPENDED_NO_WINDOW: u32 = 0x0000_0004 | 0x0800_0000;

/// Owns the suspended child's main-thread handle; [`resume`](Self::resume) releases the child to
/// run.
///
/// Dropping a `ResumeHandle` without calling `resume` closes the thread handle without resuming —
/// the child stays suspended and is terminated when its job closes (so an assign failure leaves no
/// running process, Requirement 4.5).
pub struct ResumeHandle(HANDLE);

// SAFETY: a thread handle is a kernel HANDLE usable from any thread; owned exclusively here.
unsafe impl Send for ResumeHandle {}

impl ResumeHandle {
    /// Resume the suspended main thread, releasing the child to execute (Requirement 4.3).
    ///
    /// Consumes the handle so a child is resumed at most once; the handle is closed afterwards.
    ///
    /// # Errors
    /// The underlying Win32 error if the thread cannot be resumed.
    pub fn resume(self) -> io::Result<()> {
        // SAFETY: `self.0` is a live thread handle opened with THREAD_SUSPEND_RESUME. `ResumeThread`
        // returns the previous suspend count or `u32::MAX` on failure.
        let prev = unsafe { ResumeThread(self.0) };
        // `self` (and thus the handle) is dropped at end of scope, closing it exactly once.
        if prev == u32::MAX {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

impl Drop for ResumeHandle {
    fn drop(&mut self) {
        // SAFETY: the thread handle was opened by `spawn_suspended` and is closed exactly once.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

/// Spawn `cmd` suspended and return the child together with a handle to resume its main thread.
///
/// The caller must assign the returned child to a [`Job`] and only then call
/// [`ResumeHandle::resume`], so the child cannot run — and therefore cannot spawn any descendant —
/// before it is a job member (Requirement 4.1–4.3). `cmd` is otherwise configured by the caller
/// (program, args, env, stdio, `current_dir`); this function adds `CREATE_SUSPENDED | CREATE_NO_WINDOW`
/// and `kill_on_drop(true)` and spawns.
///
/// The child's main-thread handle is found via a ToolHelp thread snapshot: a freshly
/// `CREATE_SUSPENDED` process has exactly one thread, so the single `THREADENTRY32` whose
/// `th32OwnerProcessID` equals the child PID is unambiguous. `tokio::process` does not expose the
/// main-thread handle, so this is the robust documented way to obtain it without re-implementing
/// stdio piping.
///
/// # Errors
/// The spawn error if the process cannot start, or the Win32 error if its main thread cannot be
/// located or opened. On a post-spawn failure the just-spawned (still-suspended) child is killed
/// before returning, so no suspended orphan is left behind.
pub fn spawn_suspended(
    cmd: &mut tokio::process::Command,
) -> io::Result<(tokio::process::Child, ResumeHandle)> {
    cmd.creation_flags(CREATE_SUSPENDED_NO_WINDOW)
        .kill_on_drop(true);
    let mut child = cmd.spawn()?;
    let Some(pid) = child.id() else {
        // The child already exited (before we could even snapshot its thread). Reap it and
        // report, so the caller never treats this as a running, assignable child.
        let _ = child.start_kill();
        return Err(io::Error::other(
            "suspended child exited before it could be assigned",
        ));
    };

    match main_thread_handle(pid) {
        Ok(resume) => Ok((child, resume)),
        Err(e) => {
            // Could not get a resume handle: kill the suspended child so it never runs, then
            // surface the error. `kill_on_drop` also backs this up when `child` drops.
            let _ = child.start_kill();
            Err(e)
        }
    }
}

/// Open a `THREAD_SUSPEND_RESUME` handle to the (single) main thread of suspended process `pid`.
fn main_thread_handle(pid: u32) -> io::Result<ResumeHandle> {
    // SAFETY: snapshot of all threads; the returned handle is closed before this function returns.
    let snapshot =
        unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) }.map_err(|e| win_err(&e))?;
    // Ensure the snapshot handle is closed on every path.
    struct SnapGuard(HANDLE);
    impl Drop for SnapGuard {
        fn drop(&mut self) {
            // SAFETY: the snapshot handle was created just above and is closed exactly once.
            let _ = unsafe { CloseHandle(self.0) };
        }
    }
    let _guard = SnapGuard(snapshot);

    let mut entry = THREADENTRY32 {
        dwSize: u32::try_from(size_of::<THREADENTRY32>()).map_err(io::Error::other)?,
        ..Default::default()
    };
    // SAFETY: `snapshot` is live and `entry` is a valid THREADENTRY32 with `dwSize` set.
    let mut ok = unsafe { Thread32First(snapshot, &mut entry) }.is_ok();
    while ok {
        if entry.th32OwnerProcessID == pid {
            // SAFETY: opening a thread by id for suspend/resume; the returned handle is owned by
            // the `ResumeHandle` and closed exactly once (on resume or drop).
            let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME, false, entry.th32ThreadID) }
                .map_err(|e| win_err(&e))?;
            return Ok(ResumeHandle(thread));
        }
        // SAFETY: as `Thread32First`, iterating the same live snapshot.
        ok = unsafe { Thread32Next(snapshot, &mut entry) }.is_ok();
    }
    Err(io::Error::other(format!(
        "no main thread found for suspended process {pid}"
    )))
}
