## Summary
<!-- Brief explanation of what changes this PR introduces and why. -->

## Linked Issues
<!-- e.g. Closes #123, Refs #456 -->
- Resolves: 

## Protected Set & Security Dependency Checklist
<!-- If this PR touches crates in the protected set, rules.toml, or config/default.toml: -->
- [ ] **Dependency Ordering Gate**: If this PR touches `rules.toml` or `config/default.toml` (e.g. Issue #71), I have confirmed that required security property tests (e.g. Issue #74) are already merged into `main`.
- [ ] **Path Confinement**: All added command patterns are strictly anchored within `<worktree_root>` and forbid directory traversal (`..`) or paths targeting retired drives (e.g. `C:\`).
- [ ] **Fuzz-Testing & Flag Injection**: All added/whitelisted patterns were fuzz-tested against path traversal (`..`), flag injection (`--flag=value`, `-B<path>`), and retired drive (`C:\`) access via `crates/nebula-sandbox/src/redteam.rs`.
- [ ] **Audit Checklist**: Verified compliance with `docs/PROTECTED_SET_AUDIT_CHECKLIST.md`.
- [ ] **No Secrets**: No secrets, tokens, or personal identifiers are committed or printed.

## Test Plan & Proof
<!-- Concrete commands run and verifiable test output. -->
```powershell
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo nextest run --workspace --locked
```
Output:
<!-- Paste proof here -->
