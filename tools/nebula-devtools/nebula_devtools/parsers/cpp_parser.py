"""Parse C++ compiler output (MSVC cl.exe, Clang) and CTest results into structured diagnostics.

MSVC diagnostics take the form:
    path/to/file.cpp(42,15): error C2065: 'x': undeclared identifier
Clang/GCC diagnostics take the form:
    path/to/file.cpp:42:15: error: use of undeclared identifier 'x'
CTest failures take the form:
    1/2 Test #1: TestName ...***Failed
"""

from __future__ import annotations

import re
from pathlib import PurePath

from ..model import Diagnostic, Severity

# MSVC: path\to\file.cpp(42,15): error C2065: message
_MSVC_DIAG = re.compile(
    r"^(?P<file>.+?)\((?P<line>\d+)(?:,(?P<col>\d+))?\):\s+"
    r"(?P<severity>fatal error|error|warning)\s+(?P<code>C\d+):\s+(?P<msg>.+)$",
    re.IGNORECASE,
)

# Clang / GCC: path/to/file.cpp:42:15: error: message
_CLANG_DIAG = re.compile(
    r"^(?P<file>[^:\n]+?):(?P<line>\d+):(?:(?P<col>\d+):)?\s*"
    r"(?P<severity>fatal error|error|warning):\s*(?P<msg>.+)$",
    re.IGNORECASE,
)

# CTest: 1/2 Test #1: test_math .....................***Failed  0.01 sec
_CTEST_FAILED = re.compile(
    r"^\s*\d+/\d+\s+Test\s+#\d+:\s+(?P<test>\S+)\s+\.+\s*\*\*\*Failed",
    re.IGNORECASE,
)

_CTEST_SUMMARY_ITEM = re.compile(
    r"^\s*\d+\s+-\s+(?P<test>\S+)\s+\((?:Failed|Subprocess aborted)\)",
    re.IGNORECASE,
)


def _relative(file: str, worktree_root: str) -> str:
    try:
        return str(PurePath(file).relative_to(PurePath(worktree_root)))
    except ValueError:
        return file


def _map_severity(s: str) -> Severity:
    s_lower = s.lower()
    if "error" in s_lower:
        return Severity.ERROR
    if "warning" in s_lower:
        return Severity.WARNING
    return Severity.NOTE


def parse_cpp_compiler_output(
    output: str, worktree_root: str, source: str = "cmake"
) -> list[Diagnostic]:
    """Parse MSVC or Clang compiler error and warning lines into structured diagnostics."""
    diagnostics: list[Diagnostic] = []
    seen = set()

    for line in output.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue

        # Check MSVC first
        m = _MSVC_DIAG.match(line_clean)
        if m:
            file = m.group("file").strip()
            line_no = int(m.group("line"))
            col = int(m.group("col")) if m.group("col") else None
            sev = _map_severity(m.group("severity"))
            code = m.group("code").upper()
            msg = f"{m.group('msg').strip()} [{code}]"
            rel_file = _relative(file, worktree_root)
            key = (rel_file, line_no, col, msg)
            if key not in seen:
                seen.add(key)
                diagnostics.append(
                    Diagnostic(
                        file=rel_file,
                        line=line_no,
                        column=col,
                        message=msg,
                        source=source,
                        severity=sev,
                    )
                )
            continue

        # Check Clang / GCC
        m = _CLANG_DIAG.match(line_clean)
        if m:
            file = m.group("file").strip()
            line_no = int(m.group("line"))
            col = int(m.group("col")) if m.group("col") else None
            sev = _map_severity(m.group("severity"))
            msg = m.group("msg").strip()
            rel_file = _relative(file, worktree_root)
            key = (rel_file, line_no, col, msg)
            if key not in seen:
                seen.add(key)
                diagnostics.append(
                    Diagnostic(
                        file=rel_file,
                        line=line_no,
                        column=col,
                        message=msg,
                        source=source,
                        severity=sev,
                    )
                )

    return diagnostics


def parse_ctest_output(output: str, worktree_root: str, source: str = "ctest") -> list[Diagnostic]:
    """Parse CTest stdout/stderr to recover failing test names as diagnostics."""
    failures: list[Diagnostic] = []
    seen_tests = set()

    in_failed_summary = False
    for line in output.splitlines():
        line_s = line.strip()
        if "The following tests FAILED:" in line_s:
            in_failed_summary = True
            continue

        if in_failed_summary:
            m = _CTEST_SUMMARY_ITEM.match(line_s)
            if m:
                test_name = m.group("test")
                if test_name not in seen_tests:
                    seen_tests.add(test_name)
                    failures.append(
                        Diagnostic(
                            file="build",
                            line=1,
                            message=f"CTest {test_name} failed",
                            source=source,
                            severity=Severity.ERROR,
                        )
                    )
                continue
            if line_s.startswith("Errors were encountered"):
                in_failed_summary = False

        m = _CTEST_FAILED.match(line_s)
        if m:
            test_name = m.group("test")
            if test_name not in seen_tests:
                seen_tests.add(test_name)
                failures.append(
                    Diagnostic(
                        file="build",
                        line=1,
                        message=f"CTest {test_name} failed",
                        source=source,
                        severity=Severity.ERROR,
                    )
                )

    return failures
