# ADR-008: C++ Language Module and Devtools Adapter Architecture Boundary

**Status:** Accepted (2026-10-24)

## Context
Nebula provides polyglot task execution across Rust, Python, C++, and SQL. In C++ development on Windows (Issues #65, #70, #73), there are two execution surfaces:
1. **The In-Daemon Language Module (`nebula-daemon::language::CppModule`)**: An in-process `LanguageModule` providing fine-grained task-scoped tools (`cpp.build.configure`, `cpp.build.compile`, `cpp.test.run`, `cpp.run`) and prompt guidelines directly into the agent's inner loop.
2. **The External Devtools MCP Adapter (`nebula-devtools::CppAdapter`)**: An MCP server providing macro verb tools (`build.run`, `test.run`, `lint.run`) for language-agnostic build/test/lint interfaces.

Without an explicit architectural contract, risks include duplicate toolchain discovery, conflicting build directories, uncoordinated process lifecycles, and memory overhead from pre-warming C++ on non-C++ tasks.

## Decision
1. **Toolchain Discovery Single Source of Truth**:
   - Toolchain discovery probes for Visual Studio MSVC (`vswhere.exe`), LLVM/Clang (`clang-cl.exe`), CMake (`cmake.exe`), and Ninja (`ninja.exe`).
   - Discovery results are lazily resolved and cached per-daemon lifecycle. Initial discovery budget is `<2.0s`; subsequent tool invocations execute in `<50ms`.
2. **Hermetic Build Directory Standard**:
   - All CMake build trees reside strictly at `<worktree_root>\build\`.
   - The build directory is excluded from git branches via `.gitignore` auto-configuration.
   - Out-of-source builds ensure worktree teardown completely reclaims compiler artifacts without polluting parent repositories.
3. **Diagnostic Schema Parity**:
   - Compiler errors from MSVC (`cl.exe`) and Clang (`clang-cl.exe`) are normalized into structured `{ file, line, col, severity, code, message }` JSON objects.
   - Diagnostic positions are relativized to the worktree root.
4. **Lazy Post-Detection Pre-warming (Zero Non-C++ Overhead)**:
   - C++ toolchains and adapters are never initialized globally or for tasks where `detect()` yields no C++ signature.
   - Pre-warming triggers asynchronously during initial prompt ingestion only when `language.detected` identifies C++.
5. **Confinement & Job Object Invariant**:
   - All compilation, configure, and execution commands execute within the task's `TaskJob` Job Object and are confined to `CURRENT_WORKTREE`.

## Options Considered
- *Single Monolithic In-Process Module*: Rejected because external MCP tools require language-agnostic verbs (`build.run`, `test.run`).
- *In-Source CMake Build Trees*: Rejected because intermediate build objects pollute git worktrees and break candidate branch diffing.
- *Unconditional Global Toolchain Pre-warming*: Rejected because running `vswhere.exe` and loading MSVC environment variables on non-C++ tasks wastes RAM and delays startup.

## Consequences
- Clean separation between fine-grained model tools (`cpp.build.compile`) and MCP adapter verbs (`build.run`).
- Zero memory and CPU footprint for non-C++ tasks.
- Unified compiler error diagnostic parsing across both interfaces.
