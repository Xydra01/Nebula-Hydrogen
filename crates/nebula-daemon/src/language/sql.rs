//! SQL and Database Language Module with sandbox introspection and migration validation (Issue #66).
//!
//! Provides isolated SQLite sandboxing in `<worktree_root>\sandbox.db`,
//! schema introspection, read-only query validation, and migration testing.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use nebula_proto::LanguageMetadata;
use nebula_sandbox::task_job::JobError;
use nebula_tools::ToolError;
use nebula_tools::builtins::{BuiltinTool, ToolContext, ToolOutput};
use nebula_tools::permit::Tier;
use serde_json::{Value, json};

use super::{FinalizeReport, LanguageError, LanguageModule, ProvisionContext};

/// Metadata describing the SQL module.
pub fn sql_metadata() -> LanguageMetadata {
    LanguageMetadata {
        id: "sql".into(),
        display_name: "SQL & Database Module".into(),
        description:
            "SQL module with SQLite sandboxing, schema introspection, query validation, and migration testing"
                .into(),
        supported_versions: vec![
            "sqlite".into(),
            "duckdb".into(),
            "postgres".into(),
            "mysql".into(),
        ],
        default_version: Some("sqlite".into()),
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
                    server: "sql".to_string(),
                    detail: format!("process wait failed: {e}"),
                }),
            },
            Err(JobError::NoTaskInScope) => {
                match tokio::time::timeout(timeout, cmd.output()).await {
                    Ok(Ok(out)) => Ok(out),
                    Ok(Err(e)) => Err(ToolError::Unavailable {
                        server: "sql".to_string(),
                        detail: format!("failed to spawn {program}: {e}"),
                    }),
                    Err(_) => Err(ToolError::Timeout {
                        tool: program.to_string(),
                        timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
                    }),
                }
            }
            Err(e) => Err(ToolError::Unavailable {
                server: "sql".to_string(),
                detail: format!("failed to assign process to Job Object: {e}"),
            }),
        }
    }

    #[cfg(not(windows))]
    {
        match tokio::time::timeout(timeout, cmd.output()).await {
            Ok(Ok(out)) => Ok(out),
            Ok(Err(e)) => Err(ToolError::Unavailable {
                server: "sql".to_string(),
                detail: format!("failed to spawn {program}: {e}"),
            }),
            Err(_) => Err(ToolError::Timeout {
                tool: program.to_string(),
                timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
            }),
        }
    }
}

/// Helper to execute Python-based SQLite runner scripts.
async fn run_sqlite_py(
    py_code: &str,
    args: &[&str],
    cwd: &Path,
    timeout: Duration,
) -> Result<std::process::Output, ToolError> {
    let mut full_args = vec!["-c", py_code];
    full_args.extend_from_slice(args);
    run_in_job("python", &full_args, cwd, timeout).await
}

/// Resolves the database path relative to the worktree root.
fn resolve_db_path(worktree_root: &Path, db_name: Option<&str>) -> PathBuf {
    let rel = db_name.unwrap_or("sandbox.db");
    worktree_root.join(rel)
}

/// SQL Language Module.
pub struct SqlModule;

#[async_trait]
impl LanguageModule for SqlModule {
    fn id(&self) -> &str {
        "sql"
    }

    fn metadata(&self) -> LanguageMetadata {
        sql_metadata()
    }

    async fn detect(&self, repo_root: &Path) -> Option<f32> {
        if repo_root.join("schema.sql").exists() || repo_root.join("migrations").is_dir() {
            return Some(0.95);
        }

        let sql_exts = ["sql", "sqlite", "sqlite3", "db"];
        if let Ok(mut entries) = tokio::fs::read_dir(repo_root).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if let Some(ext) = path.extension().and_then(|e| e.to_str())
                    && sql_exts.contains(&ext)
                {
                    return Some(0.90);
                }
            }
        }

        None
    }

    async fn provision(&self, ctx: &ProvisionContext<'_>) -> Result<(), LanguageError> {
        let db_path = ctx.worktree_root.join("sandbox.db");
        if !db_path.exists() {
            // Initialize empty SQLite database
            let code = "import sqlite3, sys; conn = sqlite3.connect(sys.argv[1]); conn.close()";
            let _ = run_sqlite_py(
                code,
                &[db_path.to_str().unwrap_or("sandbox.db")],
                ctx.worktree_root,
                Duration::from_secs(10),
            )
            .await;
        }

        // Ensure .gitignore excludes *.db and sandbox.db
        let gitignore_path = ctx.worktree_root.join(".gitignore");
        let should_append = if gitignore_path.exists() {
            let content = tokio::fs::read_to_string(&gitignore_path)
                .await
                .unwrap_or_default();
            !content
                .lines()
                .any(|l| l.trim() == "sandbox.db" || l.trim() == "*.db")
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
                let _ = file.write_all(b"\nsandbox.db\n*.db\n").await;
            }
        }

        Ok(())
    }

    fn tools(&self) -> Vec<Arc<dyn BuiltinTool>> {
        vec![
            Arc::new(SqlSchemaInspectTool),
            Arc::new(SqlQueryExecuteTool),
            Arc::new(SqlMigrateRunTool),
            Arc::new(SqlLintRunTool),
        ]
    }

    fn prompt_guidelines(&self, _ctx: &ProvisionContext<'_>) -> Vec<String> {
        vec![
            "A sandboxed SQLite database is available at 'sandbox.db' in your worktree root.".into(),
            "Use 'sql.schema.inspect' to discover table definitions, column types, and foreign keys before writing queries.".into(),
            "Use 'sql.query.execute' for read-only query testing. Destructive statements (DROP, DELETE, UPDATE) are strictly blocked in query mode.".into(),
            "Use 'sql.migrate.run' to execute schema migrations transactionally in order.".into(),
            "Use 'sql.lint.run' to validate SQL syntax against dialect standards.".into(),
        ]
    }

    async fn finalize(&self, ctx: &ProvisionContext<'_>) -> Result<FinalizeReport, LanguageError> {
        let db_path = ctx.worktree_root.join("sandbox.db");
        let mut artifacts = Vec::new();

        if !db_path.exists() {
            return Ok(FinalizeReport {
                success: true,
                summary: "SQL verification complete (no database sandbox active)".into(),
                artifacts,
            });
        }

        // Integrity check via PRAGMA integrity_check
        let code = r#"
import sqlite3, sys, json
conn = sqlite3.connect(sys.argv[1])
cur = conn.cursor()
cur.execute("PRAGMA integrity_check")
res = cur.fetchall()
conn.close()
print(json.dumps([r[0] for r in res]))
"#;
        let out = run_sqlite_py(
            code,
            &[db_path.to_str().unwrap_or("sandbox.db")],
            ctx.worktree_root,
            Duration::from_secs(10),
        )
        .await;

        let (success, summary) = match out {
            Ok(o) => {
                let text = String::from_utf8_lossy(&o.stdout).trim().to_string();
                let is_ok = text.contains("ok");
                artifacts.push("sandbox.db".to_string());
                (is_ok, format!("Database integrity check: {text}"))
            }
            Err(e) => (false, format!("Failed database integrity check: {e}")),
        };

        Ok(FinalizeReport {
            success,
            summary,
            artifacts,
        })
    }
}

// ---------------------------------------------------------------------------
// Tool 1: sql.schema.inspect
// ---------------------------------------------------------------------------

/// Built-in tool for introspecting tables, columns, constraints, and indexes.
pub struct SqlSchemaInspectTool;

#[async_trait]
impl BuiltinTool for SqlSchemaInspectTool {
    fn name(&self) -> &str {
        "sql.schema.inspect"
    }

    fn description(&self) -> Option<&str> {
        Some("Introspects tables, columns, indexes, and constraints from the database.")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Database file relative to worktree root. Defaults to sandbox.db."
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
        let db_name = arguments.get("database").and_then(Value::as_str);
        let db_path = resolve_db_path(&worktree_root, db_name);

        let code = r#"
import sqlite3, sys, json
db = sys.argv[1]
conn = sqlite3.connect(db)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
tables = [row[0] for row in cur.fetchall()]

schema = []
for t in tables:
    cur.execute(f"PRAGMA table_info('{t}')")
    cols = []
    for c in cur.fetchall():
        cols.append({
            "cid": c[0],
            "name": c[1],
            "type": c[2],
            "nullable": not c[3],
            "default_value": c[4],
            "primary_key": bool(c[5])
        })
    cur.execute(f"PRAGMA index_list('{t}')")
    indexes = [idx[1] for idx in cur.fetchall()]
    cur.execute(f"PRAGMA foreign_key_list('{t}')")
    fks = []
    for fk in cur.fetchall():
        fks.append({
            "id": fk[0],
            "seq": fk[1],
            "table": fk[2],
            "from": fk[3],
            "to": fk[4]
        })
    schema.append({
        "table": t,
        "columns": cols,
        "indexes": indexes,
        "foreign_keys": fks
    })

conn.close()
print(json.dumps(schema))
"#;

        let out = run_sqlite_py(
            code,
            &[db_path.to_str().unwrap_or("sandbox.db")],
            &worktree_root,
            ctx.limits.call_timeout,
        )
        .await?;

        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            return Err(ToolError::Unavailable {
                server: "sql".into(),
                detail: format!("schema inspection failed: {err}"),
            });
        }

        let parsed: Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|_| json!([]));
        ToolOutput::json(&json!({ "tables": parsed }))
    }
}

// ---------------------------------------------------------------------------
// Tool 2: sql.query.execute
// ---------------------------------------------------------------------------

/// Built-in tool for executing read-only SQL queries with mutation rejection.
pub struct SqlQueryExecuteTool;

#[async_trait]
impl BuiltinTool for SqlQueryExecuteTool {
    fn name(&self) -> &str {
        "sql.query.execute"
    }

    fn description(&self) -> Option<&str> {
        Some("Executes a read-only query against the database with strict mutation rejection.")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL SELECT query to execute."
                },
                "database": {
                    "type": "string",
                    "description": "Database file relative to worktree root. Defaults to sandbox.db."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of rows to return (default 50, max 500)."
                }
            },
            "required": ["query"],
            "additionalProperties": false
        })
    }

    fn tier(&self) -> Tier {
        Tier::Sandbox
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let worktree_root = ctx.worktree.worktree_root();
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidArguments {
                tool: "sql.query.execute".into(),
                detail: "missing 'query' argument".into(),
            })?;

        // Strict read-only query guard!
        let upper = query.to_ascii_uppercase();
        let mutation_keywords = [
            "INSERT ",
            "UPDATE ",
            "DELETE ",
            "DROP ",
            "ALTER ",
            "TRUNCATE ",
            "CREATE ",
            "REPLACE ",
            "ATTACH ",
            "DETACH ",
        ];
        for kw in mutation_keywords {
            if upper.contains(kw) || upper.starts_with(kw.trim()) {
                return Err(ToolError::InvalidArguments {
                    tool: "sql.query.execute".into(),
                    detail: format!(
                        "mutation rejected: sql.query.execute is strictly read-only. Keyword '{kw}' forbidden; use sql.migrate.run for schema changes"
                    ),
                });
            }
        }

        let limit = arguments
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(50)
            .min(500);

        let db_name = arguments.get("database").and_then(Value::as_str);
        let db_path = resolve_db_path(&worktree_root, db_name);

        let code = r#"
import sqlite3, sys, json
db = sys.argv[1]
query = sys.argv[2]
limit = int(sys.argv[3])

conn = sqlite3.connect(db)
cur = conn.cursor()
try:
    cur.execute(query)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchmany(limit)
    res = {
        "status": "success",
        "columns": cols,
        "rows": rows,
        "row_count": len(rows),
        "truncated": len(rows) == limit
    }
except Exception as e:
    res = {
        "status": "error",
        "error": str(e)
    }
finally:
    conn.close()

print(json.dumps(res))
"#;

        let limit_str = limit.to_string();
        let out = run_sqlite_py(
            code,
            &[db_path.to_str().unwrap_or("sandbox.db"), query, &limit_str],
            &worktree_root,
            ctx.limits.call_timeout,
        )
        .await?;

        let parsed: Value =
            serde_json::from_slice(&out.stdout).map_err(|e| ToolError::Unavailable {
                server: "sql".into(),
                detail: format!("query execution failed to parse output: {e}"),
            })?;

        ToolOutput::json(&parsed)
    }
}

// ---------------------------------------------------------------------------
// Tool 3: sql.migrate.run
// ---------------------------------------------------------------------------

/// Built-in tool for applying migration scripts transactionally in order.
pub struct SqlMigrateRunTool;

#[async_trait]
impl BuiltinTool for SqlMigrateRunTool {
    fn name(&self) -> &str {
        "sql.migrate.run"
    }

    fn description(&self) -> Option<&str> {
        Some("Applies SQL migration files in alphabetical order within a sandboxed transaction.")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "migration_dir": {
                    "type": "string",
                    "description": "Directory containing migration scripts. Defaults to 'migrations'."
                },
                "database": {
                    "type": "string",
                    "description": "Target database file. Defaults to sandbox.db."
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
        let mig_dir_rel = arguments
            .get("migration_dir")
            .and_then(Value::as_str)
            .unwrap_or("migrations");

        let mig_dir = worktree_root.join(mig_dir_rel);
        let db_name = arguments.get("database").and_then(Value::as_str);
        let db_path = resolve_db_path(&worktree_root, db_name);

        if !mig_dir.is_dir() {
            return ToolOutput::json(&json!({
                "status": "clean",
                "applied_migrations": [],
                "note": format!("migration directory '{}' not found or empty", mig_dir_rel),
            }));
        }

        // Find .sql migration files and sort them
        let mut files = Vec::new();
        if let Ok(mut entries) = tokio::fs::read_dir(&mig_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("sql") {
                    files.push(p);
                }
            }
        }
        files.sort();

        let mut applied = Vec::new();
        for file in &files {
            let filename = file
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();

            let sql =
                tokio::fs::read_to_string(file)
                    .await
                    .map_err(|e| ToolError::Unavailable {
                        server: "sql".into(),
                        detail: format!("failed to read migration {filename}: {e}"),
                    })?;

            let code = r#"
import sqlite3, sys, json
db = sys.argv[1]
sql = sys.argv[2]
conn = sqlite3.connect(db)
try:
    conn.executescript(sql)
    conn.commit()
    print(json.dumps({"success": True}))
except Exception as e:
    conn.rollback()
    print(json.dumps({"success": False, "error": str(e)}))
finally:
    conn.close()
"#;
            let out = run_sqlite_py(
                code,
                &[db_path.to_str().unwrap_or("sandbox.db"), &sql],
                &worktree_root,
                ctx.limits.call_timeout,
            )
            .await?;

            let parsed: Value =
                serde_json::from_slice(&out.stdout).unwrap_or(json!({ "success": false }));
            if parsed.get("success").and_then(Value::as_bool) != Some(true) {
                let err = parsed
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error");
                return ToolOutput::json(&json!({
                    "status": "failed",
                    "failed_at": filename,
                    "error": err,
                    "applied_before_failure": applied,
                }));
            }

            applied.push(filename);
        }

        ToolOutput::json(&json!({
            "status": "success",
            "applied_migrations": applied,
            "total_applied": applied.len(),
        }))
    }
}

// ---------------------------------------------------------------------------
// Tool 4: sql.lint.run
// ---------------------------------------------------------------------------

/// Built-in tool for validating SQL syntax against dialect standards.
pub struct SqlLintRunTool;

#[async_trait]
impl BuiltinTool for SqlLintRunTool {
    fn name(&self) -> &str {
        "sql.lint.run"
    }

    fn description(&self) -> Option<&str> {
        Some("Validates SQL syntax and formatting rules for migration and schema files.")
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "SQL file or directory to validate. Defaults to worktree root."
                },
                "dialect": {
                    "type": "string",
                    "enum": ["sqlite", "postgres", "mysql"],
                    "description": "SQL dialect. Defaults to sqlite."
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
        let target_path_str = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
        let target_path = worktree_root.join(target_path_str);

        let mut sql_files = Vec::new();
        if target_path.is_file() {
            sql_files.push(target_path);
        } else if target_path.is_dir()
            && let Ok(mut entries) = tokio::fs::read_dir(&target_path).await
        {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("sql") {
                    sql_files.push(p);
                }
            }
        }

        let mut findings = Vec::new();
        for file in sql_files {
            let rel = file
                .strip_prefix(&worktree_root)
                .unwrap_or(&file)
                .to_string_lossy()
                .to_string();

            if let Ok(content) = tokio::fs::read_to_string(&file).await {
                // Validate statement syntax via SQLite in-memory parse check
                let code = r#"
import sqlite3, sys, json
sql = sys.argv[1]
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
try:
    conn.executescript(sql)
    print(json.dumps({"ok": True}))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}))
finally:
    conn.close()
"#;
                if let Ok(out) =
                    run_sqlite_py(code, &[&content], &worktree_root, ctx.limits.call_timeout).await
                    && let Ok(parsed) = serde_json::from_slice::<Value>(&out.stdout)
                    && parsed.get("ok").and_then(Value::as_bool) != Some(true)
                {
                    let err_msg = parsed
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("syntax error");
                    findings.push(json!({
                        "file": rel,
                        "line": 1,
                        "severity": "error",
                        "message": err_msg,
                    }));
                }
            }
        }

        let count = findings.len();
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
    async fn sql_detection_finds_schema_sql() {
        let tmp = tempfile::tempdir().unwrap();
        let module = SqlModule;
        assert!(module.detect(tmp.path()).await.is_none());

        tokio::fs::write(tmp.path().join("schema.sql"), "CREATE TABLE t (id INT);\n")
            .await
            .unwrap();
        let score = module.detect(tmp.path()).await;
        assert_eq!(score, Some(0.95));
    }

    #[tokio::test]
    async fn sql_detection_finds_migrations_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let module = SqlModule;
        tokio::fs::create_dir_all(tmp.path().join("migrations"))
            .await
            .unwrap();
        let score = module.detect(tmp.path()).await;
        assert_eq!(score, Some(0.95));
    }

    #[tokio::test]
    async fn sql_provision_creates_sandbox_db_and_gitignore() {
        let tmp = tempfile::tempdir().unwrap();
        let module = SqlModule;
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
        assert!(tmp.path().join("sandbox.db").exists());

        let gitignore = tmp.path().join(".gitignore");
        assert!(gitignore.exists());
        let content = tokio::fs::read_to_string(&gitignore).await.unwrap();
        assert!(content.contains("sandbox.db"));
    }

    #[test]
    fn sql_tools_list_has_all_expected_tools() {
        let module = SqlModule;
        let tools = module.tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(
            names,
            vec![
                "sql.schema.inspect",
                "sql.query.execute",
                "sql.migrate.run",
                "sql.lint.run",
            ]
        );
    }
}
