# Architecture Decision Records

| ADR | Title | Status |
| --- | --- | --- |
| [ADR-001](ADR-001-rust-core-python-tools.md) | Rust core with Python tools | Accepted (2026-09-30) |
| [ADR-002](ADR-002-mcp-tool-boundary.md) | MCP as the universal tool boundary | Accepted (2026-09-30) |
| [ADR-003](ADR-003-llama-cpp-backend.md) | llama.cpp (PrismML fork) behind a backend trait | Accepted (2026-09-30) |
| [ADR-004](ADR-004-model-profiles.md) | Model profiles for Bonsai 2 27B | Accepted (2026-10-02) |
| [ADR-005](ADR-005-fallback-model.md) | Fallback model | Accepted (2026-10-02) |
| [ADR-006](ADR-006-mtp-speculative-decoding.md) | MTP speculative decoding for the `standard` profile | Accepted (2026-10-04) |
| [ADR-007](ADR-007-ipc-framing.md) | IPC framing: NDJSON JSON-RPC over a named pipe | Accepted (2026-10-04) |
| [ADR-008](ADR-008-cpp-language-module-and-devtools-boundary.md) | C++ language module and devtools boundary | Accepted (2026-10-24) |

These first three were extracted from [NEBULA_DESIGN.md](../NEBULA_DESIGN.md) Section 3. From now on, the ADR files are the source of truth for decisions, and the design doc links to them.

New ADRs: copy the structure below into `ADR-NNN-<slug>.md`, open a PR, and add a row here.

```markdown
# ADR-NNN: Title

**Status:** Proposed | Accepted (date) | Superseded by ADR-MMM

## Context
## Decision
## Options considered
## Consequences
```
