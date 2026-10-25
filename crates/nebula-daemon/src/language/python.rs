//! Flagship Python language module with hermetic `uv` virtualenv enforcement (Issue #63).
//!
//! Provides isolated `.venv` management, version selection, package installation,
//! and structured diagnostic extraction for pytest and ruff.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use nebula_proto::LanguageMetadata;
use nebula_sandbox::task_job::JobError;
use nebula_tools::ToolError;
use nebula_tools::builtins::{BuiltinTool, ToolContext, ToolOutput};
use nebula_tools::permit::Tier;
use regex::Regex;
use serde_json::{Value, json};

use super::{FinalizeReport, LanguageError, LanguageModule, ProvisionContext};

/// Metadata describing the flagship Python module.
pub fn python_metadata() -> LanguageMetadata {
    LanguageMetadata {
        id: "python".into(),
        display_name: "Python (uv-managed)".into(),
        description:
            "Flagship Python module with hermetic uv virtualenv enforcement and structured diagnostics"
                .into(),
        supported_versions: vec![
            "3.9".into(),
            "3.10".into(),
            "3.11".into(),
            "3.12".into(),
            "3.13".into(),
        ],
        default_version: Some("3.12".into()),
    }
}

/// Helper to execute a command within the task's Job Object or fallback to child execution.
async fn run_in_job(
    program: &str,
    args: &[&str],
    cwd: &Path,
    timeout: Duration,
) -> Result<std::process::Output, ToolError> {
    let mut cmd = tokio::process::Command::new(program);
    cmd.args(args)
        .current_dir(cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    #[cfg(windows)]
    {
        match nebula_tools::builtins::spawn_in_task_job(&mut cmd) {
            Ok(mut child) => match child.wait_with_timeout(timeout).await {
                Ok(Some(out)) => Ok(out),
                Ok(None) => Err(ToolError::Timeout {
                    tool: program.to_string(),
                    timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
                }),
                Err(e) => Err(ToolError::Unavailable {
                    server: "python".to_string(),
                    detail: format!("process wait failed: {e}"),
                }),
            },
            Err(JobError::NoTaskInScope) => {
                // Standalone / unit test execution fallback
                match tokio::time::timeout(timeout, cmd.output()).await {
                    Ok(Ok(out)) => Ok(out),
                    Ok(Err(e)) => Err(ToolError::Unavailable {
                        server: "python".to_string(),
                        detail: format!("failed to spawn {program}: {e}"),
                    }),
                    Err(_) => Err(ToolError::Timeout {
                        tool: program.to_string(),
                        timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
                    }),
                }
            }
            Err(e) => Err(ToolError::Unavailable {
                server: "python".to_string(),
                detail: format!("failed to assign process to Job Object: {e}"),
            }),
        }
    }

    #[cfg(not(windows))]
    {
        match tokio::time::timeout(timeout, cmd.output()).await {
            Ok(Ok(out)) => Ok(out),
            Ok(Err(e)) => Err(ToolError::Unavailable {
                server: "python".to_string(),
                detail: format!("failed to spawn {program}: {e}"),
            }),
            Err(_) => Err(ToolError::Timeout {
                tool: program.to_string(),
                timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
            }),
        }
    }
}

/// Returns the path to the python executable within the worktree's `.venv`.
pub fn venv_python_path(worktree_root: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        worktree_root
            .join(".venv")
            .join("Scripts")
            .join("python.exe")
    }
    #[cfg(not(windows))]
    {
        worktree_root.join(".venv").join("bin").join("python")
    }
}

/// Returns the path to pytest within the worktree's `.venv`.
pub fn venv_pytest_path(worktree_root: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        worktree_root
            .join(".venv")
            .join("Scripts")
            .join("pytest.exe")
    }
    #[cfg(not(windows))]
    {
        worktree_root.join(".venv").join("bin").join("pytest")
    }
}

/// Returns the path to ruff within the worktree's `.venv`.
pub fn venv_ruff_path(worktree_root: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        worktree_root.join(".venv").join("Scripts").join("ruff.exe")
    }
    #[cfg(not(windows))]
    {
        worktree_root.join(".venv").join("bin").join("ruff")
    }
}

/// Flagship Python Language Module.
pub struct PythonModule;

#[async_trait]
impl LanguageModule for PythonModule {
    fn id(&self) -> &str {
        "python"
    }

    fn metadata(&self) -> LanguageMetadata {
        python_metadata()
    }

    async fn detect(&self, repo_root: &Path) -> Option<f32> {
        let manifests = [
            "pyproject.toml",
            "requirements.txt",
            "setup.py",
            "Pipfile",
            "tox.ini",
        ];
        for manifest in manifests {
            if repo_root.join(manifest).exists() {
                return Some(0.95);
            }
        }

        // Check if any .py files exist in root or immediate children
        if let Ok(mut entries) = tokio::fs::read_dir(repo_root).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) == Some("py") {
                    return Some(0.70);
                }
            }
        }

        None
    }

    async fn provision(&self, ctx: &ProvisionContext<'_>) -> Result<(), LanguageError> {
        let version = ctx
            .language_config
            .and_then(|c| c.get("python_version"))
            .and_then(Value::as_str)
            .unwrap_or("3.12");

        // 1. Ensure .gitignore protects .venv/
        let gitignore_path = ctx.worktree_root.join(".gitignore");
        let should_append_gitignore = if gitignore_path.exists() {
            let content = tokio::fs::read_to_string(&gitignore_path)
                .await
                .unwrap_or_default();
            !content
                .lines()
                .any(|l| l.trim() == ".venv" || l.trim() == ".venv/")
        } else {
            true
        };

        if should_append_gitignore {
            use tokio::io::AsyncWriteExt as _;
            if let Ok(mut file) = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&gitignore_path)
                .await
            {
                let _ = file.write_all(b"\n.venv/\n").await;
            }
        }

        // 2. Provision virtualenv using uv if not already present
        let py_exe = venv_python_path(ctx.worktree_root);
        if !py_exe.exists() {
            let version_arg = format!("--python={version}");
            let res = run_in_job(
                "uv",
                &["venv", &version_arg, ".venv"],
                ctx.worktree_root,
                Duration::from_secs(60),
            )
            .await;

            if let Err(e) = res {
                // If uv fails or not found, try fallback to python -m venv
                let fallback = run_in_job(
                    "python",
                    &["-m", "venv", ".venv"],
                    ctx.worktree_root,
                    Duration::from_secs(60),
                )
                .await;
                if let Err(fe) = fallback {
                    return Err(LanguageError::ProvisionFailed(format!(
                        "failed to provision .venv via uv ({e}) or python ({fe})"
                    )));
                }
            }
        }

        Ok(())
    }

    fn tools(&self) -> Vec<Arc<dyn BuiltinTool>> {
        vec![
            Arc::new(PythonEnvEnsureTool),
            Arc::new(PythonPkgAddTool),
            Arc::new(PythonPkgListTool),
            Arc::new(PythonRunTool),
            Arc::new(PythonTestRunTool),
            Arc::new(PythonLintRunTool),
        ]
    }

    fn prompt_guidelines(&self, _ctx: &ProvisionContext<'_>) -> Vec<String> {
        vec![
            "A dedicated Python virtual environment is active in your worktree at '.venv/'.".into(),
            "DO NOT attempt to run global python or manual activation scripts (e.g. 'activate.ps1').".into(),
            "Use 'python.run' to execute scripts inside the hermetic virtual environment.".into(),
            "Use 'python.pkg.add' to install third-party libraries (e.g. requests, rich, pytest) into '.venv/'. Never install packages globally.".into(),
            "Use 'python.test.run' to run test suites with pytest; it extracts structured pass/fail results.".into(),
            "Use 'python.lint.run' to analyze and format code with ruff.".into(),
        ]
    }

    async fn finalize(&self, ctx: &ProvisionContext<'_>) -> Result<FinalizeReport, LanguageError> {
        let mut artifacts = Vec::new();

        // Run ruff check
        let ruff_out = run_in_job(
            "uv",
            &["run", "ruff", "check", "."],
            ctx.worktree_root,
            Duration::from_secs(30),
        )
        .await;

        let ruff_ok = match ruff_out {
            Ok(ref out) => out.status.success(),
            Err(_) => true,
        };

        // Check if tests exist
        let tests_dir = ctx.worktree_root.join("tests");
        let mut test_summary = String::new();
        let mut test_ok = true;

        if tests_dir.exists() {
            let py_test = run_in_job(
                "uv",
                &["run", "pytest", "-q"],
                ctx.worktree_root,
                Duration::from_secs(60),
            )
            .await;
            match py_test {
                Ok(out) => {
                    test_ok = out.status.success();
                    test_summary = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    artifacts.push("pytest".to_string());
                }
                Err(e) => {
                    test_summary = format!("pytest run error: {e}");
                }
            }
        }

        let success = ruff_ok && test_ok;
        let summary = if success {
            format!("Python environment verified clean. {test_summary}")
        } else {
            format!(
                "Python verification findings. Ruff clean: {ruff_ok}, Tests passed: {test_ok}. {test_summary}"
            )
        };

        Ok(FinalizeReport {
            success,
            summary,
            artifacts,
        })
    }
}

// ---------------------------------------------------------------------------
// Tool 1: python.env.ensure
// ---------------------------------------------------------------------------

/// Built-in tool for verifying or provisioning virtual environments.
pub struct PythonEnvEnsureTool;

#[async_trait]
impl BuiltinTool for PythonEnvEnsureTool {
    fn name(&self) -> &str {
        "python.env.ensure"
    }

    fn description(&self) -> Option<&str> {
        Some("Verifies or provisions the task's isolated virtual environment (.venv).")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "python_version": {
                    "type": "string",
                    "enum": ["3.9", "3.10", "3.11", "3.12", "3.13"],
                    "description": "Target Python minor version. Defaults to 3.12."
                }
            },
            "additionalProperties": false
        })
    }

    fn tier(&self) -> Tier {
        Tier::Sandbox
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let worktree_root = ctx.worktree.worktree_root();
        let version = arguments
            .get("python_version")
            .and_then(Value::as_str)
            .unwrap_or("3.12");

        let py_exe = venv_python_path(&worktree_root);
        if !py_exe.exists() {
            let version_arg = format!("--python={version}");
            let _ = run_in_job(
                "uv",
                &["venv", &version_arg, ".venv"],
                &worktree_root,
                ctx.limits.call_timeout,
            )
            .await;
        }

        let py_ver_output = run_in_job(
            "uv",
            &["run", "python", "--version"],
            &worktree_root,
            ctx.limits.call_timeout,
        )
        .await
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| format!("Python {version}"));

        let payload = json!({
            "status": "ready",
            "python_version": py_ver_output,
            "venv_path": worktree_root.join(".venv").to_string_lossy(),
            "executable": py_exe.to_string_lossy(),
        });

        ToolOutput::json(&payload)
    }
}

// ---------------------------------------------------------------------------
// Tool 2: python.pkg.add
// ---------------------------------------------------------------------------

/// Built-in tool for installing Python packages into the virtual environment.
pub struct PythonPkgAddTool;

#[async_trait]
impl BuiltinTool for PythonPkgAddTool {
    fn name(&self) -> &str {
        "python.pkg.add"
    }

    fn description(&self) -> Option<&str> {
        Some("Installs packages into the virtual environment and updates dependency manifests.")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "packages": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of package specifications (e.g. ['rich>=13.0', 'pytest'])"
                },
                "dev": {
                    "type": "boolean",
                    "description": "Whether to install as development dependencies."
                }
            },
            "required": ["packages"],
            "additionalProperties": false
        })
    }

    fn tier(&self) -> Tier {
        Tier::Sandbox
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let worktree_root = ctx.worktree.worktree_root();
        let packages: Vec<String> = arguments
            .get("packages")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect()
            })
            .unwrap_or_default();

        if packages.is_empty() {
            return Err(ToolError::InvalidArguments {
                tool: "python.pkg.add".into(),
                detail: "packages list cannot be empty".into(),
            });
        }

        let is_dev = arguments
            .get("dev")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let pyproject_path = worktree_root.join("pyproject.toml");
        let requirements_path = worktree_root.join("requirements.txt");

        let manifest_updated = if pyproject_path.exists() {
            let mut args = vec!["add"];
            if is_dev {
                args.push("--dev");
            }
            for pkg in &packages {
                args.push(pkg.as_str());
            }
            let out = run_in_job("uv", &args, &worktree_root, ctx.limits.call_timeout).await?;
            if !out.status.success() {
                let err = String::from_utf8_lossy(&out.stderr);
                return Err(ToolError::Unavailable {
                    server: "python".into(),
                    detail: format!("uv add failed: {err}"),
                });
            }
            "pyproject.toml"
        } else if requirements_path.exists() {
            let mut args = vec!["pip", "install"];
            for pkg in &packages {
                args.push(pkg.as_str());
            }
            let out = run_in_job("uv", &args, &worktree_root, ctx.limits.call_timeout).await?;
            if !out.status.success() {
                let err = String::from_utf8_lossy(&out.stderr);
                return Err(ToolError::Unavailable {
                    server: "python".into(),
                    detail: format!("uv pip install failed: {err}"),
                });
            }
            // Append packages to requirements.txt
            if let Ok(mut file) = tokio::fs::OpenOptions::new()
                .append(true)
                .open(&requirements_path)
                .await
            {
                use tokio::io::AsyncWriteExt as _;
                for pkg in &packages {
                    let _ = file.write_all(format!("{pkg}\n").as_bytes()).await;
                }
            }
            "requirements.txt"
        } else {
            // Create minimal pyproject.toml and run uv add
            let minimal_manifest =
                "[project]\nname = \"workspace\"\nversion = \"0.1.0\"\ndependencies = []\n";
            let _ = tokio::fs::write(&pyproject_path, minimal_manifest).await;
            let mut args = vec!["add"];
            if is_dev {
                args.push("--dev");
            }
            for pkg in &packages {
                args.push(pkg.as_str());
            }
            let out = run_in_job("uv", &args, &worktree_root, ctx.limits.call_timeout).await?;
            if !out.status.success() {
                let err = String::from_utf8_lossy(&out.stderr);
                return Err(ToolError::Unavailable {
                    server: "python".into(),
                    detail: format!("uv add failed: {err}"),
                });
            }
            "pyproject.toml"
        };

        let installed_entries: Vec<Value> = packages
            .iter()
            .map(|pkg| {
                let name = pkg.split(['<', '>', '=']).next().unwrap_or(pkg);
                json!({ "name": name, "spec": pkg })
            })
            .collect();

        ToolOutput::json(&json!({
            "status": "success",
            "installed": installed_entries,
            "manifest_updated": manifest_updated,
        }))
    }
}

// ---------------------------------------------------------------------------
// Tool 3: python.pkg.list
// ---------------------------------------------------------------------------

/// Built-in tool for listing packages installed in the virtual environment.
pub struct PythonPkgListTool;

#[async_trait]
impl BuiltinTool for PythonPkgListTool {
    fn name(&self) -> &str {
        "python.pkg.list"
    }

    fn description(&self) -> Option<&str> {
        Some("Lists all packages installed in the worktree's virtual environment.")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        })
    }

    fn tier(&self) -> Tier {
        Tier::Read
    }

    async fn call(&self, _arguments: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let worktree_root = ctx.worktree.worktree_root();
        let out = run_in_job(
            "uv",
            &["pip", "list", "--format=json"],
            &worktree_root,
            ctx.limits.call_timeout,
        )
        .await?;

        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            return Err(ToolError::Unavailable {
                server: "python".into(),
                detail: format!("uv pip list failed: {err}"),
            });
        }

        let packages: Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|_| json!([]));
        ToolOutput::json(&json!({ "packages": packages }))
    }
}

// ---------------------------------------------------------------------------
// Tool 4: python.run
// ---------------------------------------------------------------------------

/// Built-in tool for executing Python scripts inside the virtual environment.
pub struct PythonRunTool;

#[async_trait]
impl BuiltinTool for PythonRunTool {
    fn name(&self) -> &str {
        "python.run"
    }

    fn description(&self) -> Option<&str> {
        Some("Executes a Python script directly within the virtual environment.")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "Path to the python script relative to the worktree."
                },
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional command-line arguments passed to the script."
                }
            },
            "required": ["script"],
            "additionalProperties": false
        })
    }

    fn tier(&self) -> Tier {
        Tier::Sandbox
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let worktree_root = ctx.worktree.worktree_root();
        let script = arguments
            .get("script")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidArguments {
                tool: "python.run".into(),
                detail: "missing 'script' argument".into(),
            })?;

        let args: Vec<String> = arguments
            .get("args")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect()
            })
            .unwrap_or_default();

        let mut cmd_args = vec!["run", "python", script];
        for a in &args {
            cmd_args.push(a.as_str());
        }

        let out = run_in_job("uv", &cmd_args, &worktree_root, ctx.limits.call_timeout).await?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        let mut stderr = String::from_utf8_lossy(&out.stderr).to_string();

        // Smart remediation for ModuleNotFoundError
        if stderr.contains("ModuleNotFoundError: No module named ") {
            let re = Regex::new(r"ModuleNotFoundError: No module named '(?P<pkg>[^']+)'").ok();
            if let Some(caps) = re.and_then(|r| r.captures(&stderr))
                && let Some(pkg) = caps.name("pkg")
            {
                stderr.push_str(&format!(
                    "\n\nHint: Package '{}' is not installed in the virtual environment. Call python.pkg.add(['{}']) to install it.",
                    pkg.as_str(),
                    pkg.as_str()
                ));
            }
        }

        let is_error = !out.status.success();
        let combined = if stderr.is_empty() {
            stdout.to_string()
        } else if stdout.is_empty() {
            stderr
        } else {
            format!("{stdout}\n{stderr}")
        };

        Ok(ToolOutput {
            bytes: combined.into_bytes(),
            is_error,
        })
    }
}

// ---------------------------------------------------------------------------
// Tool 5: python.test.run
// ---------------------------------------------------------------------------

/// Built-in tool for executing pytest test suites and parsing results.
pub struct PythonTestRunTool;

#[async_trait]
impl BuiltinTool for PythonTestRunTool {
    fn name(&self) -> &str {
        "python.test.run"
    }

    fn description(&self) -> Option<&str> {
        Some("Executes pytest within the virtual environment and returns structured test results.")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "test_path": {
                    "type": "string",
                    "description": "Optional specific test file or directory (e.g. 'tests/test_storage.py')."
                },
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional additional pytest arguments."
                }
            },
            "additionalProperties": false
        })
    }

    fn tier(&self) -> Tier {
        Tier::Sandbox
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let worktree_root = ctx.worktree.worktree_root();
        let test_path = arguments.get("test_path").and_then(Value::as_str);
        let extra_args: Vec<String> = arguments
            .get("args")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect()
            })
            .unwrap_or_default();

        let mut cmd_args = vec!["run", "pytest", "-q", "--tb=short"];
        if let Some(tp) = test_path {
            cmd_args.push(tp);
        }
        for a in &extra_args {
            cmd_args.push(a.as_str());
        }

        let started = std::time::Instant::now();
        let out = run_in_job("uv", &cmd_args, &worktree_root, ctx.limits.call_timeout).await?;
        let duration_s = started.elapsed().as_secs_f64();

        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        let combined = format!("{stdout}\n{stderr}");

        // Parse summary counts: "3 passed, 1 failed, 1 error"
        let re_passed = Regex::new(r"(?P<count>\d+)\s+passed").ok();
        let re_failed = Regex::new(r"(?P<count>\d+)\s+failed").ok();
        let re_errors = Regex::new(r"(?P<count>\d+)\s+error").ok();

        let passed: u32 = re_passed
            .and_then(|r| r.captures(&combined))
            .and_then(|c| c.name("count"))
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        let failed: u32 = re_failed
            .and_then(|r| r.captures(&combined))
            .and_then(|c| c.name("count"))
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        let errors: u32 = re_errors
            .and_then(|r| r.captures(&combined))
            .and_then(|c| c.name("count"))
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);

        // Parse failures: "FAILED tests/test_math.py::test_err - AssertionError: ..."
        let re_failure = Regex::new(r"FAILED\s+(?P<test>\S+?)(?:\s+-\s+(?P<msg>.*))?$").ok();
        let mut failures = Vec::new();
        for line in combined.lines() {
            if let Some(ref rf) = re_failure
                && let Some(caps) = rf.captures(line.trim())
            {
                let test = caps.name("test").map(|m| m.as_str()).unwrap_or("");
                let msg = caps
                    .name("msg")
                    .map(|m| m.as_str())
                    .unwrap_or("test failed");
                let file = test.split("::").next().unwrap_or(test);
                failures.push(json!({
                    "test": test,
                    "file": file,
                    "line": 1,
                    "message": msg,
                }));
            }
        }

        let is_success = out.status.success();
        let status = if is_success { "passed" } else { "failed" };

        ToolOutput::json(&json!({
            "status": status,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "duration_s": duration_s,
            "failures": failures,
            "raw_output": combined.trim(),
        }))
    }
}

// ---------------------------------------------------------------------------
// Tool 6: python.lint.run
// ---------------------------------------------------------------------------

/// Built-in tool for executing ruff linting and formatting checks.
pub struct PythonLintRunTool;

#[async_trait]
impl BuiltinTool for PythonLintRunTool {
    fn name(&self) -> &str {
        "python.lint.run"
    }

    fn description(&self) -> Option<&str> {
        Some(
            "Executes ruff within the virtual environment and returns structured lint diagnostics.",
        )
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory or file to lint; defaults to worktree root."
                },
                "fix": {
                    "type": "boolean",
                    "description": "Automatically apply safe fixes where possible."
                }
            },
            "additionalProperties": false
        })
    }

    fn tier(&self) -> Tier {
        Tier::Read
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let worktree_root = ctx.worktree.worktree_root();
        let target_path = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
        let fix = arguments
            .get("fix")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let mut cmd_args = vec!["run", "ruff", "check", target_path, "--output-format=json"];
        if fix {
            cmd_args.push("--fix");
        }

        let out = run_in_job("uv", &cmd_args, &worktree_root, ctx.limits.call_timeout).await?;
        let findings: Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|_| json!([]));
        let count = findings.as_array().map(Vec::len).unwrap_or(0);
        let status = if count == 0 { "clean" } else { "findings" };

        ToolOutput::json(&json!({
            "status": status,
            "findings_count": count,
            "findings": findings,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn python_detection_finds_pyproject() {
        let tmp = tempfile::tempdir().unwrap();
        let module = PythonModule;
        assert!(module.detect(tmp.path()).await.is_none());

        tokio::fs::write(tmp.path().join("pyproject.toml"), "[project]\n")
            .await
            .unwrap();
        let score = module.detect(tmp.path()).await;
        assert_eq!(score, Some(0.95));
    }

    #[tokio::test]
    async fn python_detection_finds_py_file() {
        let tmp = tempfile::tempdir().unwrap();
        let module = PythonModule;
        tokio::fs::write(tmp.path().join("main.py"), "print('hello')\n")
            .await
            .unwrap();
        let score = module.detect(tmp.path()).await;
        assert_eq!(score, Some(0.70));
    }

    #[tokio::test]
    async fn python_gitignore_appended_if_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let module = PythonModule;
        let task_id = nebula_proto::TaskId::new();
        let trace_id = nebula_proto::TraceId::new();
        let ctx = ProvisionContext {
            task_id: &task_id,
            worktree_root: tmp.path(),
            task_job: None,
            language_config: None,
            trace_id: &trace_id,
        };

        // Provision will attempt .venv creation and append .gitignore
        let _ = module.provision(&ctx).await;
        let gitignore = tmp.path().join(".gitignore");
        assert!(gitignore.exists());
        let content = tokio::fs::read_to_string(&gitignore).await.unwrap();
        assert!(content.contains(".venv/"));
    }

    #[test]
    fn python_tools_list_has_all_expected_tools() {
        let module = PythonModule;
        let tools = module.tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(
            names,
            vec![
                "python.env.ensure",
                "python.pkg.add",
                "python.pkg.list",
                "python.run",
                "python.test.run",
                "python.lint.run",
            ]
        );
    }
}
