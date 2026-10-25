# Architecture and Code Map

A guide to what exists in the repo **today**: where things are and how they fit together. The full target design (agent loop, tools, sandboxing, self-improvement) is in [NEBULA_DESIGN.md](NEBULA_DESIGN.md). This page only describes what's built, which is Phase 0.

## The big picture

```mermaid
flowchart LR
  CLI["nebula (nebula-cli)"] -->|"JSON-RPC over \\.\pipe\nebula (ADR-007)"| D["nebula-daemon"]
  D --> MM["ModelManager (nebula-model)"]
  MM -->|"Win32 Shared Memory (Local\Nebula_Hydrogen_IPC)"| HY["hydrogen_engine (GPU C++20/CUDA)"]
  MM -->|"Job Object, HTTP + SSE (fallback)"| LS["llama-server: chat profile (GPU)"]
  MM -->|"Job Object, HTTP"| ES["llama-server: embedding (CPU)"]
  D --> RS["Resource sampler + doctor (nebula-resources)"]
  D --> T["Telemetry (nebula-telemetry)"]
  T --> LOGS["F:\Nebula\logs\*.jsonl + blobs"]
  CLI --> BK["nebula backup (nebula-backup)"]
  BK -->|rclone crypt| GD["Google Drive"]
```

- **One daemon per machine**, started by `nebula daemon start` (detached). It owns the model servers, samples resources every 2 s, writes structured logs, and serves the CLI over a named pipe that only the current user can open.
- **The model** runs as either the native `hydrogen_engine.exe` (communicating over a 64 MiB zero-copy Win32 Named Shared Memory section with sub-50 µs event signaling) or `llama-server.exe` child processes inside a Windows Job Object. If the daemon dies, they die too. The supervisor restarts a crashed server with backoff and switches profiles on request.
- **The CLI** is a thin client. `doctor` and `resources` fall back to local checks when the daemon is down. `backup` runs in the CLI process, without the daemon.
- **Phase 1 & Strategic Realignment** adds the agent itself: tools over MCP, permission tiers, worktrees, an executor loop and a TUI, realigned across 5 strategic pivots (speculative multi-agent branching via Radix KV deduplication, default zero-copy shared memory IPC, GPU-fused FSM schema constraints, tool-spawn CUDA yield seam, and WDDM quantum-safe continuous scheduling). See [HYDROGEN_INFERENCE_ENGINE_DESIGN.md](HYDROGEN_INFERENCE_ENGINE_DESIGN.md#53-nebula-production-alignment-5-strategic-realignment-pivots) and the GitHub issues.

## Crates (`crates/`)

Dependency order, from the bottom up. Each crate's `lib.rs` starts with a module doc that explains it in more detail.

| Crate | What it does | Key files | Depends on |
| --- | --- | --- | --- |
| `nebula-proto` | IPC wire types: JSON-RPC envelopes, `Method` (requests), `Event` (notifications), shared types (`ModelStatus`, `ResourceSnapshot`, `DoctorReport`, `LogEvent`), ULID trace IDs. NDJSON framing in `Message::encode`/`decode`. | `envelope.rs`, `methods.rs`, `events.rs`, `types.rs`; golden snapshots in `tests/snapshots/` | — |
| `nebula-telemetry` | `tracing` subscriber: every event becomes a `LogEvent` and is redacted, then written to daily JSONL, the console and a broadcast channel (for `logs.subscribe`). zstd blob store for large payloads. Log budget, plus the archive to `D:` with its 8 GB auto-prune. | `layer.rs`, `writer.rs`, `blobs.rs`, `budget.rs`, `redact.rs` | proto |
| `nebula-hydrogen` | Safe Rust FFI mirror and bindings to the Hydrogen inference engine's Win32 shared memory IPC (64 MiB ring buffers, sub-50 µs token delivery, atomic active tool processes barriers). | `lib.rs`, `ipc.rs` | — |
| `nebula-model` | `LlamaServerBackend` (HTTP/SSE) and `HydrogenBackend` (Win32 shared memory). `ModelManager` supervisor (launch in a Job Object, health loop, restart and backoff, `Failed` state, profile switches that drain requests). `testing` module: a fake llama-server for tests. | `backend.rs`, `hydrogen.rs`, `supervisor.rs`, `launcher.rs`, `sse.rs`, `config.rs`, `testing.rs` | proto, telemetry, hydrogen |
| `nebula-config` | Typed config. `config/default.toml` is **embedded in the binaries**, and `F:\Nebula\config\nebula.toml` (or `$NEBULA_CONFIG`) is merged over it. `deny_unknown_fields` everywhere. | `lib.rs` | model, telemetry |
| `nebula-resources` | GPU (NVML, plus PDH for per-process VRAM on WDDM), memory and commit charge, disks. The commit preflight before loading a model. Doctor checks that don't need the daemon (disks, SMART trend, `C:` retired, SSH firewall, logs, backup age and sign-in). | `sources.rs`, `sampler.rs`, `doctor.rs` | config, model, proto, telemetry |
| `nebula-sandbox` | **(Phase 1, protected set.)** The permission-tier engine and command classifier: the `Tier` vocabulary (Read/Sandbox/Workspace/System) + `effective_tier`, the single-use scope-matched `Approval` grant contract, the data-driven rules table (embedded `rules.toml`, validated loader), and the `RulesClassifier` — it segments a command, forces encoded/indirection sinks to Tier 3, matches rules highest-tier-wins, resolves path conditions through a `PathResolver` seam, takes the max over segments, gates above-threshold commands on an authorizing single-use approval, and logs one `permit.decision` event. A ≥50-entry red-team fixture pins expected tiers. Performs **no** filesystem I/O (delegates to the resolver). Also hosts the **per-task git worktree lifecycle** (`worktree` module, issue #29): the `WorktreeManager` (create/finalize/remove/recover), the `[worktree]` config, the single `WorktreeError`, the git-shelling helper, and the `Delete_Limit` cleanup gate that **reuses** the same `Approval` contract. The per-task confinement provider (`TaskWorktreeProvider`) lives in `nebula-tools`, not here — see the per-task worktree note. | `tier.rs`, `approval.rs`, `classifier.rs`, `rules.rs`, `engine.rs`, `redteam.rs`, `data/rules.toml`, `data/red_team.toml`; `worktree/` (`manager.rs`, `config.rs`, `error.rs`, `git.rs`, `measure.rs`, `recovery.rs`) | — (a leaf; `nebula-tools` and `nebula-daemon` depend on it) |
| `nebula-daemon` | `start()`/`run()`: a single-instance mutex, the model manager, the sampler, the pipe server. `client.rs` is the pipe client the CLI uses. `checks.rs` holds the runtime and model-hash checks against the lock files, which are **compiled in**. `win.rs` holds the Win32 code (mutex, pipe ACL, Credential Manager). Built-in tool wiring in `builtin_providers.rs` constructs the real `RulesClassifier::embedded` (startup fails on a bad table). `executor.rs` implements the bounded single executor loop (issue #32). `github/` implements automated bot branch push and PR creation with strict safety guardrails and zero token leakage (issue #36). `notifications.rs` implements push notifications over Tailscale via ntfy with strict IP/DNS confinement, secret masking, and debounced health alerts (issue #37). | `lib.rs`, `server.rs`, `client.rs`, `checks.rs`, `win.rs`, `builtin_providers.rs`, `executor.rs`, `github/`, `notifications.rs` | config, model, proto, resources, sandbox, tools, telemetry |
| `nebula-backup` | Backups: `tar.zst` with a SHA-256 manifest, retention (recent, daily, weekly, monthly), an rclone wrapper, restore with verification. | `lib.rs`, `archive.rs`, `retention.rs`, `rclone.rs` | config |
| `nebula-cli` | The `nebula` binary (clap). Commands: `daemon start\|stop\|status`, `chat`, `model status\|profile <name>`, `logs tail`, `resources`, `doctor`, `backup now\|list\|restore\|reauth`, `task run\|cancel`, `game-mode on\|off\|status`, `resume`, `notify status\|test`, `tui`. `--json` on everything. Hosts the ratatui TUI client (`tui/` module). | `lib.rs` (commands), `chat.rs`, `task.rs`, `daemon_ctl.rs`, `backup.rs`, `format.rs`, `win.rs`, `tui/` (`app.rs`, `ui.rs`, `event.rs`, `terminal.rs`) | all of the above |

Binaries: `nebula.exe` (from `nebula-cli`) and `nebula-daemon.exe` (from `nebula-daemon`), installed to `F:\Nebula\bin\` by `scripts/install-nebula.ps1`.

### Where to look for…

| Question | Place |
| --- | --- |
| Add or change an IPC method | `nebula-proto/src/methods.rs` (+ `NAMES`), handler in `nebula-daemon/src/server.rs`, client call in `nebula-cli/src/lib.rs`; accept the snapshot diff |
| Add a doctor check | Local checks: `nebula-resources/src/doctor.rs`. Checks that need the daemon or the lock files: `nebula-daemon/src/checks.rs` |
| Add a config key | `config/default.toml` and the struct in `nebula-config/src/lib.rs` (unknown keys fail) |
| Change a command's permission tier | `nebula-sandbox/src/data/rules.toml` (the embedded rules table; editing it needs a rebuild/reinstall), the engine in `nebula-sandbox/src/engine.rs`, and the red-team fixture `data/red_team.toml`. Protected set — human-reviewed |
| Change the per-task worktree lifecycle | `nebula-sandbox/src/worktree/` (the `WorktreeManager` and its `create`/`finalize`/`remove_worktree`/`recover_stale` in `manager.rs`, the `Delete_Limit` gate in `measure.rs`); the per-task confinement provider `TaskWorktreeProvider` + `CURRENT_WORKTREE` task-local in `nebula-tools`; the daemon wiring (recovery at start, provider install) in `nebula-daemon/src/builtin_providers.rs`. Protected set — human-reviewed |
| Model launch flags and profiles | `[model.profiles.*]` in `config/default.toml`, turned into `llama-server` or `hydrogen_engine` arguments by `ModelProfile::args` in `nebula-model/src/config.rs` |
| Native Hydrogen engine & shared memory IPC | Model backend in `crates/nebula-model/src/hydrogen.rs`, safe FFI in `crates/nebula-hydrogen/src/`, C++20/CUDA engine in `hydrogen/src/` |
| Restart and backoff behaviour | `nebula-model/src/supervisor.rs` |
| What gets logged and redacted | `nebula-telemetry/src/layer.rs`, `redact.rs`; the secret targets are `daemon.secrets` in config |
| CLI output format | `nebula-cli/src/format.rs` (snapshot-tested in `tests/snapshots/`) |
| Executor loop & autonomous tasks | `nebula-daemon/src/executor.rs`, dispatch in `nebula-daemon/src/server.rs`, CLI in `nebula-cli/src/task.rs` |
| Circuit breakers, global stop & disk guard | `nebula-daemon/src/circuit_breaker.rs`, disk guard in `nebula-daemon/src/executor.rs`, job termination in `nebula-sandbox/src/job.rs`, and CLI `nebula stop` in `nebula-cli/src/task.rs` |
| Automated GitHub flow & bot PRs | `crates/nebula-daemon/src/github/` (`safety.rs`, `push.rs`, `api.rs`, `format.rs`); config in `[github]` |
| Game mode detection & GPU yielding | `crates/nebula-daemon/src/game_mode.rs`, Win32 foreground detector in `crates/nebula-daemon/src/win.rs`, executor pause in `crates/nebula-daemon/src/executor.rs`, CLI in `crates/nebula-cli/src/lib.rs` (`nebula game-mode`, `nebula resume`); config in `[game_mode]` |
| Push notifications over Tailscale (ntfy) | `crates/nebula-daemon/src/notifications.rs`, CLI in `crates/nebula-cli/src/lib.rs` (`nebula notify`), config in `[notifications]` |
| Terminal user interface (TUI) | `crates/nebula-cli/src/tui/` (`app.rs`, `ui.rs`, `event.rs`, `terminal.rs`), headless snapshot tests in `crates/nebula-cli/tests/tui.rs` |
| Dedicated Language Modules & dynamic tools | `crates/nebula-daemon/src/language/` (`LanguageModule` trait, `LanguageRegistry`, `PolyglotModule`), task-scoped dynamic tools in `crates/nebula-tools/src/host.rs` (`call_builtin`), CLI in `crates/nebula-cli/src/lib.rs` (`nebula lang list`, `nebula lang info`) |

### Per-task worktree

**(Phase 1, protected set.)** Issue #29 gives every agent task its own git worktree and branch, so concurrent tasks are isolated and the owner's main checkout is never modified. It replaces the single static confinement root (`ConfigWorktreeRoot`, issue #27) with a per-task one, split across two crates:

- **`nebula-sandbox::worktree`** owns the lifecycle. The `WorktreeManager` runs `git worktree add -b nebula/<task-id> <worktrees_dir>/<task-id> <start-point>` on **create**; **finalize** keeps the `Task_Branch` for issue #36's PR flow (no `git push`, no PR, no merge — AGENTS.md hard rule 1); **remove** deletes only that task's directory (`git worktree remove`, which refuses a dirty tree) and prunes; **recover** runs at daemon start to reconcile worktrees left under `worktrees_dir` by a crash, retaining any with uncommitted work. The `Delete_Limit` cleanup gate (1 GiB / 500 files) **reuses** the issue #28 `Approval` contract rather than adding a second approval mechanism.
- **`nebula-tools`** owns the confinement seam: `TaskWorktreeProvider` reads the executing task's root from the `CURRENT_WORKTREE` task-local so the `Path_Resolver` confines every `fs.*` / `shell.run` call to that root, and resolves to a retired-drive sentinel the resolver rejects when no task is in scope. It lives with the `WorktreeRootProvider` trait it implements, because `nebula-sandbox` must not depend on `nebula-tools` (cycle).
- **`nebula-daemon`** wires both together in `builtin_providers.rs`: it runs recovery at start and installs the `TaskWorktreeProvider` in the tool context.

The executor loop that drives the lifecycle (issue #32) and push/PR/merge of the task branch (issue #36) are **not** part of this feature; it creates the worktree and branch and keeps the branch.

## Runtime layout on disk

Hard-coded defaults; all of them can be changed in config. Tests redirect every path to a temp dir.

| Path | Contents |
| --- | --- |
| `F:\Nebula\bin\` | Installed `nebula.exe` and `nebula-daemon.exe` |
| `F:\Nebula\runtime\llama-prism\…`, `llama-stock\…` | Pinned llama.cpp builds (`config/runtime.lock.toml`) |
| `F:\Nebula\models\` | GGUF models (`config/models.lock.toml`) |
| `F:\Nebula\logs\` | Daily `nebula-YYYY-MM-DD.jsonl`, `blobs\<2 hex>\*.zst` |
| `F:\Nebula\state\` | `smart\` (SMART JSON), the model-hash cache, `doctor.json`, `backup-last.json`, `backup-auth.json` |
| `F:\Nebula\worktrees\` | Per-task git worktrees, one `<task-id>\` directory each (`[worktree] worktrees_dir`, issue #29) |
| `F:\Nebula\config\` | `nebula.toml` (local override), `rclone.conf` (encrypted) |
| `F:\Nebula\build\target` | `CARGO_TARGET_DIR` (with sccache) |
| `D:\NebulaCold\` | Cold data: `logs-archive\`, `backups-local\`, `models-archive\`, `restore\`, `downloads\` |
| `C:` | **Retired, failing drive.** Nothing may live there; doctor fails if a configured path resolves to it |

Secrets live in Windows Credential Manager (`nebula/github_bot_token`, `nebula/rclone_config_pass`, …) and are masked in all logs.

## Model profiles

From [ADR-004](adr/ADR-004-model-profiles.md), [ADR-005](adr/ADR-005-fallback-model.md) and [ADR-006](adr/ADR-006-mtp-speculative-decoding.md); defined in `config/default.toml`.

| Profile | Model | Context | Runtime |
| --- | --- | --- | --- |
| `standard` (default) | Bonsai 2 27B PQ2_0 with an MTP head (speculative decoding) | 32K | llama-prism |
| `long` | Bonsai 2 27B PTQ1_0 | 128K | llama-prism |
| `lean` | Bonsai 2 27B PTQ1_0 | 16K | llama-prism |
| `embedding` | Qwen3-Embedding 0.6B, CPU | 1K | llama-stock |
| fallback | Gemma 4 26B-A4B MoE (ADR-005); bench profile only, not yet a daemon profile | 32K | llama-stock |

## Other top-level folders

| Folder | What's there |
| --- | --- |
| `bench/` | Python (uv) benchmark and smoke-test harness: `nebula_bench/` (server launcher, perf, quality, needle, tool calls, MTP, embedding), `profiles/*.toml` (llama-server profiles), `results/<date>/` (reports and raw data behind the ADRs). Run with `uv run nebula-smoke --profile pq2mtp`. |
| `tools/nebula-devtools/` | The **first external Python MCP tool server** (issue #33): build/test/lint adapters exposed as the MCP tools `test.run`, `build.run`, `lint.run` for Python (uv/pytest/ruff) and Rust (cargo build/nextest/clippy). A uv package; the daemon launches it over stdio and routes `tools/call`. Returns a short structured result (overall outcome + `file:line` diagnostics) with the raw log as a second content block the host spills to a blob when over the cap. `pyright` is deferred (needs Node). Intended Sandbox tier is wired by the executor (#32); the classifier does not tier MCP tool names. Gate: `uv run ruff check . && uv run ruff format --check . && uv run pytest -q`. |
| `config/` | `default.toml` (embedded defaults), `runtime.lock.toml` and `models.lock.toml` (pins with SHA-256; embedded in the daemon for doctor) |
| `scripts/` | PowerShell ops scripts: `install-nebula.ps1`, `bot-push-pr.ps1`, `register-backup-tasks.ps1`, `store-*.ps1` (Credential Manager), `install-hooks.ps1`, and the one-off Phase 0 machine setup (`phase0-*.ps1`, `set-pagefile.ps1`, `smart-snapshot.ps1`) |
| `docs/` | Design, plan, ADRs, ops runbooks; see [the docs index](#docs-index) |
| `.githooks/` | gitleaks pre-commit and pre-push hooks |
| `.github/` | CI (`workflows/ci.yml`: gitleaks, bench, Rust fmt/clippy/nextest, aggregated as **CI ok**), issue templates, `CODEOWNERS` |

## Docs index

| Doc | Read it when |
| --- | --- |
| [NEBULA_DESIGN.md](NEBULA_DESIGN.md) | You need the target architecture or the reasoning. It's long; use the table of contents. Section 4 is architecture, 5 the agent, 6 tools, 7 command-line access and security, 9 cloud escalation and Cursor, 10 logging, 11 self-improvement, 12 the roadmap |
| [LANGUAGE_MODULES_DESIGN.md](LANGUAGE_MODULES_DESIGN.md) | Dedicated Language Modules (Language Packs) architecture: virtualenv enforcement via uv, task-scoped tool activation, structured diagnostics, and language blueprints (Python, C++, SQL) |
| [RECURSIVE_SELF_IMPROVEMENT_HARNESS.md](RECURSIVE_SELF_IMPROVEMENT_HARNESS.md) | Post-Phase 1 recursive agent harnessing architecture: Google Research Harness² analysis, 8-component surface, two-level recursion, zero-label contrastive selection, and single-GPU local adaptation |
| [PROMPT_TRANSLATION_ROSETTA_STONE.md](PROMPT_TRANSLATION_ROSETTA_STONE.md) | Prompt translation layer and execution invariants: exit directives, completion barriers, and dependency gate contracts |
| [HYDROGEN_INFERENCE_ENGINE_DESIGN.md](HYDROGEN_INFERENCE_ENGINE_DESIGN.md) | Project Hydrogen inference engine specification: agent-native Radix-Paged KV cache, Cascade Attention, Split-Fuse batching, in-engine tool dispatch, and host-pinned DMA swapping for single-GPU RTX 4070 |
| [HYDROGEN_ELI5_EXPLAINER.md](HYDROGEN_ELI5_EXPLAINER.md) | Plain-English (ELI5) architectural guide to Project Hydrogen: intuitive analogies for Radix-Paged KV cache, Cascade Attention, Split-Fuse batching, host swapping, and Windows WDDM hardening |
| [WORK_ORDER.md](WORK_ORDER.md) | Master Execution Work Order & Issue Dependency Roadmap: Tier 0 blockers (SSM/RoPE, binary IPC, batched GEMM), Tier 1 safety, Tier 2 realignment pivots, Tier 3 language modules, Tier 4 evals/ops |
| [PHASE0_PLAN.md](PHASE0_PLAN.md) | You want what Phase 0 built and why. Look for the "As built" notes under each workstream; Section 9 has the exit criteria |
| [adr/](adr/README.md) | You're about to change a decision: model profiles, fallback, MTP, IPC framing, Rust/Python split, MCP boundary |
| [ops/setup.md](ops/setup.md) | You're rebuilding the machine, or need to know how something was installed |
| [ops/phase0-notes.md](ops/phase0-notes.md) | You want dated results and findings from the real machine (the hardware quirks are here) |
| [ops/backup.md](ops/backup.md) | Backups, restore, the weekly Google sign-in renewal |
| [ops/runtime-upgrade.md](ops/runtime-upgrade.md), [ops/replace-nvme.md](ops/replace-nvme.md), [ops/retire-c-drive.md](ops/retire-c-drive.md) | Runbooks |
| [bench/results/](../bench/results/) | Benchmark reports behind ADR-004 to ADR-006 |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Branch, commit, PR and Rust conventions |
| [AGENTS.md](../AGENTS.md) | You're an AI agent working in this repo: start here |
