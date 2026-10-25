# Protected Set & Permission Rules Audit Protocol

This document defines the formal security review protocol and audit checklist required for any modification to the **Protected Set** (permission rules in `rules.toml`, `config/default.toml`, permission tiers in `crates/nebula-sandbox`, and tool confinement).

Adheres to **AGENTS.md Hard Rule 9** and resolves **Issue #74** (Security review & worktree confinement validation for Issue #71).

---

## 1. Safety Invariants & Threat Model

Nebula operates in autonomous, unattended execution modes (`/goal`, benchmark evaluations). A model could attempt to escape worktree sandbox confinement through:
1. **Directory Traversal**: Using `..` sequences to climb outside `<worktree_root>` (e.g. `cargo test --manifest-path ../../Cargo.toml`).
2. **Absolute Path / Retired Drive Escape**: Accessing system directories or the retired drive (e.g. `cmake -B C:\evil`, `cat C:\Windows\System32\drivers\etc\hosts`).
3. **Flag-Attached Path Injection**: Embedding paths inside flags using `=` or `:` syntax to bypass simple non-flag positional filters (e.g. `--manifest-path=../../Cargo.toml`, `-BC:\evil`).
4. **Shell Chaining & Indirection**: Hiding commands behind encoded payloads or pipes (`cmd /c ...`, `powershell -enc ...`).

---

## 2. Mandatory Audit Checklist for Reviewers

Every Pull Request modifying `rules.toml`, `crates/nebula-sandbox/src/data/rules.toml`, or `config/default.toml` MUST satisfy:

- [ ] **1. Strict Confinement & Floor Guarantee**:
  - Newly whitelisted patterns must be tagged with a non-destructive or destructive `[rule.match.path]` condition.
  - Path condition must resolve targets against the active task worktree root via `PathResolver`.
  - In-worktree execution classifies at `Tier::Sandbox` (Tier 1) or `Tier::Read` (Tier 0).
  - Out-of-worktree execution (`WorktreeEscape`) MUST escalate to at least `Tier::Workspace` (Tier 2).
  - Retired drive targets (`RetiredDrive` -> `C:\`) MUST escalate to `Tier::System` (Tier 3) with zero approval.

- [ ] **2. Flag-Injection & Traversal Resistance**:
  - The pattern must be validated against flag-attached syntax (`--flag=path`, `-B<path>`, `--flag:path`).
  - Positional non-flag arguments and flag values must both be submitted to `target_paths` resolution.

- [ ] **3. Red-Team Proptest Coverage**:
  - Property-based tests in `crates/nebula-sandbox/src/redteam.rs` (`property_toolchain_confinement` and `property_red_team`) must pass across 100+ randomized cases.
  - New commands must be represented in `data/red_team.toml` with both adversarial escape cases and benign in-worktree controls.

- [ ] **4. Dependency DAG Gate**:
  - Prerequisite validation tests (#74) must be merged into `main` before whitelist expansion rules (#71) can take effect.
  - `scripts/check-pr-dependencies.ps1` must pass without errors.

- [ ] **5. Zero Unconstrained Execution**:
  - No whitelist rule may permit arbitrary binary execution outside task confinement bounds.
  - Command environment must remain scrubbed (`CONTROLLED_PATH`) within Windows Job Object bounds.

---

## 3. Verification Commands

Run the full security test suite:
```powershell
cargo nextest run -p nebula-sandbox --locked
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check-pr-dependencies.ps1
```
