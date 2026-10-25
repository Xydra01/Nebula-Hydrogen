"""Unit and integration tests for CppAdapter."""

from __future__ import annotations

from unittest.mock import patch

from nebula_devtools.adapters.cpp_adapter import CppAdapter
from nebula_devtools.model import Outcome, Severity
from nebula_devtools.run import RunOutput


def test_cpp_adapter_build_success(tmp_path):
    adapter = CppAdapter()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text("# cache\n", encoding="utf-8")

    mock_run = RunOutput(
        exit_code=0,
        stdout="[1/1] Linking CXX executable test.exe\n",
        stderr="",
    )
    with patch("nebula_devtools.adapters.cpp_adapter.run_command", return_value=mock_run):
        result, run = adapter.build(str(tmp_path))
        assert result.outcome is Outcome.PASSED
        assert result.source == "cmake"
        assert result.language == "cpp"
        assert len(result.diagnostics) == 0


def test_cpp_adapter_build_compiler_error(tmp_path):
    adapter = CppAdapter()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text("# cache\n", encoding="utf-8")

    error_output = "src/main.cpp(10,5): error C2065: 'x': undeclared identifier\n"
    mock_run = RunOutput(
        exit_code=1,
        stdout=error_output,
        stderr="",
    )
    with patch("nebula_devtools.adapters.cpp_adapter.run_command", return_value=mock_run):
        result, run = adapter.build(str(tmp_path))
        assert result.outcome is Outcome.FAILED
        assert len(result.diagnostics) == 1
        d = result.diagnostics[0]
        assert d.severity is Severity.ERROR
        assert d.line == 10
        assert d.column == 5
        assert "C2065" in d.message


def test_cpp_adapter_test_failure(tmp_path):
    adapter = CppAdapter()
    ctest_output = """
    Start 1: MathTest
1/1 Test #1: MathTest .........................***Failed    0.01 sec
"""
    mock_run = RunOutput(
        exit_code=8,
        stdout=ctest_output,
        stderr="",
    )
    with patch("nebula_devtools.adapters.cpp_adapter.run_command", return_value=mock_run):
        result, run = adapter.test(str(tmp_path))
        assert result.outcome is Outcome.FAILED
        assert result.tests_failed == 1
        assert len(result.diagnostics) == 1
        assert "MathTest" in result.diagnostics[0].message
