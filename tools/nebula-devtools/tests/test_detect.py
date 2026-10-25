"""Tests for language detection and worktree confinement (issue #33, Requirements 3.5, 6.2)."""

from __future__ import annotations

import pytest

from nebula_devtools.detect import (
    ConfinementError,
    LanguageError,
    detect_language,
    ensure_within_worktree,
    resolve_language,
)


def test_detects_rust_from_cargo_toml(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    assert detect_language(str(tmp_path)) == "rust"


def test_detects_python_from_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    assert detect_language(str(tmp_path)) == "python"


def test_rust_wins_when_both_present(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    assert detect_language(str(tmp_path)) == "rust"


def test_detects_cpp_from_cmakelists(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8"
    )
    assert detect_language(str(tmp_path)) == "cpp"


def test_detects_cpp_from_compile_commands(tmp_path):
    (tmp_path / "compile_commands.json").write_text("[]\n", encoding="utf-8")
    assert detect_language(str(tmp_path)) == "cpp"


def test_no_manifest_detects_none(tmp_path):
    assert detect_language(str(tmp_path)) is None


def test_explicit_arg_overrides_detection(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    assert resolve_language("python", str(tmp_path)) == "python"


def test_unsupported_explicit_arg_errors(tmp_path):
    with pytest.raises(LanguageError):
        resolve_language("haskell", str(tmp_path))


def test_undetectable_without_arg_errors(tmp_path):
    with pytest.raises(LanguageError):
        resolve_language(None, str(tmp_path))


def test_path_inside_worktree_is_allowed(tmp_path):
    sub = tmp_path / "crate"
    sub.mkdir()
    resolved = ensure_within_worktree("crate", str(tmp_path), retired_drive="Q:")
    assert resolved == sub.resolve()


def test_path_outside_worktree_is_rejected(tmp_path):
    other = tmp_path.parent / "elsewhere"
    with pytest.raises(ConfinementError):
        ensure_within_worktree(str(other), str(tmp_path), retired_drive="Q:")


def test_dotdot_escape_is_rejected(tmp_path):
    with pytest.raises(ConfinementError):
        ensure_within_worktree("..", str(tmp_path), retired_drive="Q:")


def test_retired_drive_is_rejected(tmp_path):
    # An absolute path on the retired drive is rejected regardless of the worktree root.
    with pytest.raises(ConfinementError):
        ensure_within_worktree(r"C:\Windows\System32", str(tmp_path), retired_drive="C:")
