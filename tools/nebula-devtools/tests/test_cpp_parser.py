"""Unit tests for the C++ compiler and CTest parser."""

from __future__ import annotations

from nebula_devtools.model import Severity
from nebula_devtools.parsers.cpp_parser import parse_cpp_compiler_output, parse_ctest_output


def test_parse_msvc_compiler_error_and_warning():
    output = """
src\\main.cpp(42,15): error C2065: 'undefined_var': undeclared identifier
src\\math.cpp(10): warning C4101: 'unused': unreferenced local variable
"""
    diags = parse_cpp_compiler_output(output, worktree_root=".")
    assert len(diags) == 2
    err = diags[0]
    assert err.file.replace("\\", "/") == "src/main.cpp"
    assert err.line == 42
    assert err.column == 15
    assert err.severity is Severity.ERROR
    assert "undefined_var" in err.message
    assert "C2065" in err.message

    warn = diags[1]
    assert warn.file.replace("\\", "/") == "src/math.cpp"
    assert warn.line == 10
    assert warn.column is None
    assert warn.severity is Severity.WARNING
    assert "C4101" in warn.message


def test_parse_clang_compiler_error():
    output = """
src/calc.cpp:25:9: error: no matching function for call to 'compute'
    compute(1, 2);
    ^~~~~~~
src/calc.cpp:12:5: note: candidate function not viable
"""
    diags = parse_cpp_compiler_output(output, worktree_root=".")
    assert len(diags) == 1
    err = diags[0]
    assert err.file.replace("\\", "/") == "src/calc.cpp"
    assert err.line == 25
    assert err.column == 9
    assert err.severity is Severity.ERROR
    assert "no matching function" in err.message


def test_parse_ctest_failures():
    output = """
Test project F:/Nebula/worktrees/test/build
    Start 1: TestMath
1/2 Test #1: TestMath .........................***Failed    0.02 sec
    Start 2: TestString
2/2 Test #2: TestString .......................   Passed    0.01 sec

50% tests passed, 1 tests failed out of 2

Total Test time (real) =   0.05 sec

The following tests FAILED:
\t  1 - TestMath (Failed)
Errors were encountered while processing:
\t1 - TestMath
"""
    failures = parse_ctest_output(output, worktree_root=".")
    assert len(failures) == 1
    f = failures[0]
    assert "TestMath" in f.message
    assert f.severity is Severity.ERROR
