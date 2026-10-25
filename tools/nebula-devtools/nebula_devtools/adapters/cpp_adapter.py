"""The C++ adapter: CMake configure/build, CTest, and compiler diagnostic linting.

Maintains hermetic out-of-source CMake build trees at ``<worktree_root>/build``.
Parses MSVC (cl.exe) and Clang errors into structured diagnostics.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..model import Outcome, StructuredResult
from ..parsers.cpp_parser import parse_cpp_compiler_output, parse_ctest_output
from ..run import RunOutput, run_command
from . import errored_result, outcome_from_exit

LANGUAGE = "cpp"


class CppAdapter:
    """Runs C++ CMake and CTest runners in the worktree and returns structured results."""

    language = LANGUAGE

    def _ensure_configured(self, cwd: str) -> RunOutput | None:
        """Configure the build tree if build/CMakeCache.txt does not exist."""
        build_dir = Path(cwd) / "build"
        cache_file = build_dir / "CMakeCache.txt"
        if cache_file.is_file():
            return None

        # Build command: prefer Ninja if found on PATH
        has_ninja = shutil.which("ninja") is not None
        cmd = ["cmake", "-B", "build", "-S", "."]
        if has_ninja and os.name == "nt":
            cmd.extend(["-G", "Ninja"])

        return run_command(cmd, cwd=cwd)

    def build(self, cwd: str) -> tuple[StructuredResult, RunOutput]:
        """Configure (if needed) and run ``cmake --build build --config Debug``."""
        # Auto-configure if build tree doesn't exist yet
        conf_run = self._ensure_configured(cwd)
        if conf_run is not None and conf_run.exit_code != 0:
            if conf_run.exit_code is None:
                return errored_result(
                    "cmake", LANGUAGE, conf_run.spawn_error or "cmake unavailable"
                ), conf_run
            diags = parse_cpp_compiler_output(
                conf_run.combined_log, worktree_root=cwd, source="cmake"
            )
            errors = [d for d in diags if d.severity.value == "error"]
            outcome = outcome_from_exit(conf_run, len(errors))
            note = "cmake configure failed"
            return StructuredResult.build(outcome, "cmake", LANGUAGE, diags, note=note), conf_run

        run = run_command(
            ["cmake", "--build", "build", "--config", "Debug"],
            cwd=cwd,
        )
        if run.exit_code is None:
            return errored_result("cmake", LANGUAGE, run.spawn_error or "cmake unavailable"), run

        diags = parse_cpp_compiler_output(run.combined_log, worktree_root=cwd, source="cmake")
        errors = [d for d in diags if d.severity.value == "error"]
        outcome = outcome_from_exit(run, len(errors))
        note = None
        if outcome is Outcome.ERRORED and not errors:
            note = "cmake --build exited non-zero but no compiler errors were parsed"
        return StructuredResult.build(outcome, "cmake", LANGUAGE, diags, note=note), run

    def test(self, cwd: str) -> tuple[StructuredResult, RunOutput]:
        """Run ``ctest --test-dir build --output-on-failure``."""
        run = run_command(
            ["ctest", "--test-dir", "build", "--output-on-failure"],
            cwd=cwd,
        )
        if run.exit_code is None:
            return errored_result("ctest", LANGUAGE, run.spawn_error or "ctest unavailable"), run

        failures = parse_ctest_output(run.combined_log, worktree_root=cwd, source="ctest")
        outcome = outcome_from_exit(run, len(failures))
        note = None
        if outcome is Outcome.ERRORED and not failures:
            note = "ctest exited non-zero but no failing test records were parsed"
        result = StructuredResult.build(
            outcome,
            "ctest",
            LANGUAGE,
            failures,
            tests_failed=len(failures) if outcome is not Outcome.PASSED else 0,
            note=note,
        )
        return result, run

    def lint(self, cwd: str) -> tuple[StructuredResult, RunOutput]:
        """Inspect build tree diagnostics or syntax check via compiler."""
        run = run_command(
            ["cmake", "--build", "build", "--config", "Debug"],
            cwd=cwd,
        )
        if run.exit_code is None:
            return errored_result("cmake", LANGUAGE, run.spawn_error or "cmake unavailable"), run

        diags = parse_cpp_compiler_output(run.combined_log, worktree_root=cwd, source="cmake")
        errors = [d for d in diags if d.severity.value == "error"]
        outcome = outcome_from_exit(run, len(errors))
        note = None
        if outcome is Outcome.ERRORED and not errors:
            note = "compiler lint check exited non-zero but no diagnostics were parsed"
        return StructuredResult.build(outcome, "cmake", LANGUAGE, diags, note=note), run
