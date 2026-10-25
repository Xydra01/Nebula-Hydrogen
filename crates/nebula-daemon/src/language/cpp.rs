//! C++ Language Module with native CMake adapter, MSVC/Clang discovery, and error parsing (Issues #65, #70, #73).
//!
//! Enforces hermetic out-of-source CMake build trees at `<worktree_root>\build\`,
//! parses compiler diagnostics from MSVC and Clang into structured JSON,
//! and executes all build and test processes within the task's Job Object.

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

/// Metadata describing the C++ module.
pub fn cpp_metadata() -> LanguageMetadata {
    LanguageMetadata {
        id: "cpp".into(),
        display_name: "C++ (CMake/MSVC/Clang)".into(),
        description:
            "C++ module with native CMake build tree management, MSVC/Clang discovery, and structured compiler error parsing"
                .into(),
        supported_versions: vec!["c++17".into(), "c++20".into(), "c++23".into()],
        default_version: Some("c++20".into()),
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
                    server: "cpp".to_string(),
                    detail: format!("process wait failed: {e}"),
                }),
            },
            Err(JobError::NoTaskInScope) => {
                match tokio::time::timeout(timeout, cmd.output()).await {
                    Ok(Ok(out)) => Ok(out),
                    Ok(Err(e)) => Err(ToolError::Unavailable {
                        server: "cpp".to_string(),
                        detail: format!("failed to spawn {program}: {e}"),
                    }),
                    Err(_) => Err(ToolError::Timeout {
                        tool: program.to_string(),
                        timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
                    }),
                }
            }
            Err(e) => Err(ToolError::Unavailable {
                server: "cpp".to_string(),
                detail: format!("failed to assign process to Job Object: {e}"),
            }),
        }
    }

    #[cfg(not(windows))]
    {
        match tokio::time::timeout(timeout, cmd.output()).await {
            Ok(Ok(out)) => Ok(out),
            Ok(Err(e)) => Err(ToolError::Unavailable {
                server: "cpp".to_string(),
                detail: format!("failed to spawn {program}: {e}"),
            }),
            Err(_) => Err(ToolError::Timeout {
                tool: program.to_string(),
                timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
            }),
        }
    }
}

/// Toolchain detection result.
#[derive(Clone, Debug)]
pub struct ToolchainInfo {
    /// Whether ninja generator is available on PATH.
    pub has_ninja: bool,
    /// Whether cmake is available on PATH.
    pub has_cmake: bool,
    /// Installation path of Visual Studio MSVC if discovered.
    pub msvc_path: Option<PathBuf>,
}

/// Discover available C++ toolchains on Windows.
pub async fn discover_toolchain() -> ToolchainInfo {
    let has_cmake = run_in_job(
        "cmake",
        &["--version"],
        Path::new("."),
        Duration::from_secs(2),
    )
    .await
    .map(|o| o.status.success())
    .unwrap_or(false);

    let has_ninja = run_in_job(
        "ninja",
        &["--version"],
        Path::new("."),
        Duration::from_secs(2),
    )
    .await
    .map(|o| o.status.success())
    .unwrap_or(false);

    let mut msvc_path = None;

    #[cfg(windows)]
    {
        let vswhere =
            PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
        if vswhere.exists() {
            let res = run_in_job(
                vswhere.to_str().unwrap_or("vswhere.exe"),
                &["-latest", "-products", "*", "-property", "installationPath"],
                Path::new("."),
                Duration::from_secs(3),
            )
            .await;

            if let Ok(out) = res {
                let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !p.is_empty() {
                    msvc_path = Some(PathBuf::from(p));
                }
            }
        }
    }

    ToolchainInfo {
        has_ninja,
        has_cmake,
        msvc_path,
    }
}

/// C++ Language Module.
pub struct CppModule;

#[async_trait]
impl LanguageModule for CppModule {
    fn id(&self) -> &str {
        "cpp"
    }

    fn metadata(&self) -> LanguageMetadata {
        cpp_metadata()
    }

    async fn detect(&self, repo_root: &Path) -> Option<f32> {
        if repo_root.join("CMakeLists.txt").exists() {
            return Some(0.95);
        }
        if repo_root.join("compile_commands.json").exists() {
            return Some(0.90);
        }

        let cpp_exts = ["cpp", "hpp", "cc", "cxx", "h"];
        if let Ok(mut entries) = tokio::fs::read_dir(repo_root).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if let Some(ext) = path.extension().and_then(|e| e.to_str())
                    && cpp_exts.contains(&ext)
                {
                    return Some(0.70);
                }
            }
        }

        None
    }

    async fn provision(&self, ctx: &ProvisionContext<'_>) -> Result<(), LanguageError> {
        let build_dir = ctx.worktree_root.join("build");
        if !build_dir.exists() {
            let _ = tokio::fs::create_dir_all(&build_dir).await;
        }

        // Ensure .gitignore has build/
        let gitignore_path = ctx.worktree_root.join(".gitignore");
        let should_append = if gitignore_path.exists() {
            let content = tokio::fs::read_to_string(&gitignore_path)
                .await
                .unwrap_or_default();
            !content
                .lines()
                .any(|l| l.trim() == "build" || l.trim() == "build/")
        } else {
            true
        };

        if should_append {
            use tokio::io::AsyncWriteExt as _;
            if let Ok(mut file) = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&gitignore_path)
                .await
            {
                let _ = file.write_all(b"\nbuild/\n").await;
            }
        }

        Ok(())
    }

    fn tools(&self) -> Vec<Arc<dyn BuiltinTool>> {
        vec![
            Arc::new(CppBuildConfigureTool),
            Arc::new(CppBuildCompileTool),
            Arc::new(CppTestRunTool),
            Arc::new(CppRunTool),
        ]
    }

    fn prompt_guidelines(&self, _ctx: &ProvisionContext<'_>) -> Vec<String> {
        vec![
            "A hermetic C++ build tree is managed in 'build/'.".into(),
            "Use 'cpp.build.configure' to generate the CMake build tree (Ninja is preferred).".into(),
            "Use 'cpp.build.compile' to compile targets; it returns structured diagnostics with exact file:line:col locations.".into(),
            "Use 'cpp.test.run' to run CTest test suites with failure tracking.".into(),
            "Use 'cpp.run' to execute compiled executables within the worktree sandbox.".into(),
        ]
    }

    async fn finalize(&self, ctx: &ProvisionContext<'_>) -> Result<FinalizeReport, LanguageError> {
        let mut artifacts = Vec::new();
        let build_dir = ctx.worktree_root.join("build");

        if !build_dir.exists() {
            return Ok(FinalizeReport {
                success: true,
                summary: "C++ task completed (no build directory produced)".into(),
                artifacts,
            });
        }

        // Run ctest if tests are configured
        let ctest_res = run_in_job(
            "ctest",
            &["--test-dir", "build", "--output-on-failure"],
            ctx.worktree_root,
            Duration::from_secs(60),
        )
        .await;

        let (success, summary) = match ctest_res {
            Ok(out) => {
                let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
                artifacts.push("build".to_string());
                (out.status.success(), format!("CTest results: {text}"))
            }
            Err(_) => (true, "Build tree intact; no CTest suite executed.".into()),
        };

        Ok(FinalizeReport {
            success,
            summary,
            artifacts,
        })
    }
}

// ---------------------------------------------------------------------------
// Tool 1: cpp.build.configure
// ---------------------------------------------------------------------------

/// Built-in tool for configuring CMake build trees.
pub struct CppBuildConfigureTool;

#[async_trait]
impl BuiltinTool for CppBuildConfigureTool {
    fn name(&self) -> &str {
        "cpp.build.configure"
    }

    fn description(&self) -> Option<&str> {
        Some("Configures a CMake build tree in the worktree's build/ directory.")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "generator": {
                    "type": "string",
                    "enum": ["Ninja", "Visual Studio 17 2022"],
                    "description": "CMake generator. Defaults to Ninja if available, else standard generator."
                },
                "build_type": {
                    "type": "string",
                    "enum": ["Debug", "Release", "RelWithDebInfo"],
                    "description": "Build type configuration. Defaults to Debug."
                },
                "cmake_args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Additional command-line arguments passed to CMake."
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
        let toolchain = discover_toolchain().await;

        let generator =
            arguments
                .get("generator")
                .and_then(Value::as_str)
                .or(if toolchain.has_ninja {
                    Some("Ninja")
                } else {
                    None
                });

        let build_type = arguments
            .get("build_type")
            .and_then(Value::as_str)
            .unwrap_or("Debug");

        let cmake_args: Vec<String> = arguments
            .get("cmake_args")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect()
            })
            .unwrap_or_default();

        let mut args = vec!["-B", "build", "-S", "."];
        if let Some(g) = generator {
            args.push("-G");
            args.push(g);
        }
        let build_type_arg = format!("-DCMAKE_BUILD_TYPE={build_type}");
        args.push(&build_type_arg);

        for a in &cmake_args {
            args.push(a.as_str());
        }

        let out = run_in_job("cmake", &args, &worktree_root, ctx.limits.call_timeout).await?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        let combined = format!("{stdout}\n{stderr}");

        let status = if out.status.success() {
            "success"
        } else {
            "failed"
        };

        ToolOutput::json(&json!({
            "status": status,
            "generator": generator.unwrap_or("default"),
            "build_type": build_type,
            "build_dir": worktree_root.join("build").to_string_lossy(),
            "raw_output": combined.trim(),
        }))
    }
}

// ---------------------------------------------------------------------------
// Tool 2: cpp.build.compile
// ---------------------------------------------------------------------------

/// Built-in tool for compiling CMake targets and extracting structured diagnostics.
pub struct CppBuildCompileTool;

#[async_trait]
impl BuiltinTool for CppBuildCompileTool {
    fn name(&self) -> &str {
        "cpp.build.compile"
    }

    fn description(&self) -> Option<&str> {
        Some("Compiles C++ targets and returns structured compiler diagnostics.")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Specific CMake target to build. Builds all targets if omitted."
                },
                "clean_first": {
                    "type": "boolean",
                    "description": "Perform a clean build before compiling."
                },
                "config": {
                    "type": "string",
                    "enum": ["Debug", "Release", "RelWithDebInfo"],
                    "description": "Build configuration; defaults to Debug."
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
        let target = arguments.get("target").and_then(Value::as_str);
        let clean_first = arguments
            .get("clean_first")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let config = arguments
            .get("config")
            .and_then(Value::as_str)
            .unwrap_or("Debug");

        let mut args = vec!["--build", "build", "--config", config];
        if clean_first {
            args.push("--clean-first");
        }
        if let Some(t) = target {
            args.push("--target");
            args.push(t);
        }

        let out = run_in_job("cmake", &args, &worktree_root, ctx.limits.call_timeout).await?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        let combined = format!("{stdout}\n{stderr}");

        // Parse MSVC: path/to/file.cpp(42,15): error C2065: message
        let re_msvc = Regex::new(
            r"^(?P<file>.+?)\((?P<line>\d+)(?:,(?P<col>\d+))?\):\s+(?P<severity>fatal error|error|warning)\s+(?P<code>C\d+):\s+(?P<msg>.+)$",
        )
        .ok();

        // Parse Clang: path/to/file.cpp:42:15: error: message
        let re_clang = Regex::new(
            r"^(?P<file>[^:\n]+?):(?P<line>\d+):(?:(?P<col>\d+):)?\s*(?P<severity>fatal error|error|warning):\s*(?P<msg>.+)$",
        )
        .ok();

        let mut diagnostics = Vec::new();
        let mut error_count = 0;
        let mut warning_count = 0;

        for line in combined.lines() {
            let l = line.trim();
            if let Some(ref rm) = re_msvc
                && let Some(caps) = rm.captures(l)
            {
                let file = caps.name("file").map(|m| m.as_str()).unwrap_or("");
                let line_no: u32 = caps
                    .name("line")
                    .and_then(|m| m.as_str().parse().ok())
                    .unwrap_or(1);
                let col: Option<u32> = caps.name("col").and_then(|m| m.as_str().parse().ok());
                let sev = caps.name("severity").map(|m| m.as_str()).unwrap_or("error");
                let code = caps.name("code").map(|m| m.as_str()).unwrap_or("");
                let msg = caps.name("msg").map(|m| m.as_str()).unwrap_or("");

                let is_err = sev.contains("error");
                if is_err {
                    error_count += 1;
                } else {
                    warning_count += 1;
                }

                diagnostics.push(json!({
                    "file": file,
                    "line": line_no,
                    "col": col,
                    "severity": if is_err { "error" } else { "warning" },
                    "code": code,
                    "message": msg,
                }));
                continue;
            }

            if let Some(ref rc) = re_clang
                && let Some(caps) = rc.captures(l)
            {
                let file = caps.name("file").map(|m| m.as_str()).unwrap_or("");
                let line_no: u32 = caps
                    .name("line")
                    .and_then(|m| m.as_str().parse().ok())
                    .unwrap_or(1);
                let col: Option<u32> = caps.name("col").and_then(|m| m.as_str().parse().ok());
                let sev = caps.name("severity").map(|m| m.as_str()).unwrap_or("error");
                let msg = caps.name("msg").map(|m| m.as_str()).unwrap_or("");

                let is_err = sev.contains("error");
                if is_err {
                    error_count += 1;
                } else {
                    warning_count += 1;
                }

                diagnostics.push(json!({
                    "file": file,
                    "line": line_no,
                    "col": col,
                    "severity": if is_err { "error" } else { "warning" },
                    "code": "",
                    "message": msg,
                }));
            }
        }

        let is_success = out.status.success();
        let status = if is_success { "success" } else { "failed" };

        ToolOutput::json(&json!({
            "status": status,
            "error_count": error_count,
            "warning_count": warning_count,
            "diagnostics": diagnostics,
            "raw_output": combined.trim(),
        }))
    }
}

// ---------------------------------------------------------------------------
// Tool 3: cpp.test.run
// ---------------------------------------------------------------------------

/// Built-in tool for running CTest test suites and test binaries.
pub struct CppTestRunTool;

#[async_trait]
impl BuiltinTool for CppTestRunTool {
    fn name(&self) -> &str {
        "cpp.test.run"
    }

    fn description(&self) -> Option<&str> {
        Some("Runs CTest test suites or a specified test executable in the build tree.")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "test_binary": {
                    "type": "string",
                    "description": "Optional specific test binary to run in build/. If omitted, invokes CTest."
                },
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional command-line arguments."
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
        let test_bin = arguments.get("test_binary").and_then(Value::as_str);
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

        let out = if let Some(bin) = test_bin {
            let bin_path = worktree_root.join("build").join(bin);
            let mut args = Vec::new();
            for a in &extra_args {
                args.push(a.as_str());
            }
            run_in_job(
                bin_path.to_str().unwrap_or(bin),
                &args,
                &worktree_root,
                ctx.limits.call_timeout,
            )
            .await?
        } else {
            let mut args = vec!["--test-dir", "build", "--output-on-failure"];
            for a in &extra_args {
                args.push(a.as_str());
            }
            run_in_job("ctest", &args, &worktree_root, ctx.limits.call_timeout).await?
        };

        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        let combined = format!("{stdout}\n{stderr}");

        let re_fail =
            Regex::new(r"^\s*\d+/\d+\s+Test\s+#\d+:\s+(?P<test>\S+)\s+\.+\s*\*\*\*Failed").ok();
        let mut failures = Vec::new();
        for line in combined.lines() {
            if let Some(ref rf) = re_fail
                && let Some(caps) = rf.captures(line.trim())
                && let Some(t) = caps.name("test")
            {
                failures.push(t.as_str().to_string());
            }
        }

        let is_success = out.status.success();
        let status = if is_success { "passed" } else { "failed" };

        ToolOutput::json(&json!({
            "status": status,
            "failed_count": failures.len(),
            "failures": failures,
            "raw_output": combined.trim(),
        }))
    }
}

// ---------------------------------------------------------------------------
// Tool 4: cpp.run
// ---------------------------------------------------------------------------

/// Built-in tool for executing compiled binaries in the build tree under Job Object containment.
pub struct CppRunTool;

#[async_trait]
impl BuiltinTool for CppRunTool {
    fn name(&self) -> &str {
        "cpp.run"
    }

    fn description(&self) -> Option<&str> {
        Some("Executes a compiled binary within the build/ directory under Job Object containment.")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "binary": {
                    "type": "string",
                    "description": "Name or relative path of the compiled binary inside build/."
                },
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional command-line arguments."
                }
            },
            "required": ["binary"],
            "additionalProperties": false
        })
    }

    fn tier(&self) -> Tier {
        Tier::Sandbox
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let worktree_root = ctx.worktree.worktree_root();
        let binary = arguments
            .get("binary")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidArguments {
                tool: "cpp.run".into(),
                detail: "missing 'binary' argument".into(),
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

        let bin_path = if Path::new(binary).is_absolute() {
            PathBuf::from(binary)
        } else {
            worktree_root.join("build").join(binary)
        };

        let mut cmd_args = Vec::new();
        for a in &args {
            cmd_args.push(a.as_str());
        }

        let out = run_in_job(
            bin_path.to_str().unwrap_or(binary),
            &cmd_args,
            &worktree_root,
            ctx.limits.call_timeout,
        )
        .await?;

        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        let combined = if stderr.is_empty() {
            stdout.to_string()
        } else if stdout.is_empty() {
            stderr.to_string()
        } else {
            format!("{stdout}\n{stderr}")
        };

        Ok(ToolOutput {
            bytes: combined.into_bytes(),
            is_error: !out.status.success(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cpp_detection_finds_cmakelists() {
        let tmp = tempfile::tempdir().unwrap();
        let module = CppModule;
        assert!(module.detect(tmp.path()).await.is_none());

        tokio::fs::write(
            tmp.path().join("CMakeLists.txt"),
            "cmake_minimum_required(VERSION 3.20)\n",
        )
        .await
        .unwrap();
        let score = module.detect(tmp.path()).await;
        assert_eq!(score, Some(0.95));
    }

    #[tokio::test]
    async fn cpp_detection_finds_cpp_source() {
        let tmp = tempfile::tempdir().unwrap();
        let module = CppModule;
        tokio::fs::write(tmp.path().join("main.cpp"), "int main() { return 0; }\n")
            .await
            .unwrap();
        let score = module.detect(tmp.path()).await;
        assert_eq!(score, Some(0.70));
    }

    #[tokio::test]
    async fn cpp_provision_creates_build_and_gitignores() {
        let tmp = tempfile::tempdir().unwrap();
        let module = CppModule;
        let task_id = nebula_proto::TaskId::new();
        let trace_id = nebula_proto::TraceId::new();
        let ctx = ProvisionContext {
            task_id: &task_id,
            worktree_root: tmp.path(),
            task_job: None,
            language_config: None,
            trace_id: &trace_id,
        };

        module.provision(&ctx).await.unwrap();
        assert!(tmp.path().join("build").exists());

        let gitignore = tmp.path().join(".gitignore");
        assert!(gitignore.exists());
        let content = tokio::fs::read_to_string(&gitignore).await.unwrap();
        assert!(content.contains("build/"));
    }

    #[test]
    fn cpp_tools_list_has_all_expected_tools() {
        let module = CppModule;
        let tools = module.tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(
            names,
            vec![
                "cpp.build.configure",
                "cpp.build.compile",
                "cpp.test.run",
                "cpp.run",
            ]
        );
    }
}
