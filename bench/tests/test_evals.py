"""Tests for evals tasks and runner (Issue #39)."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so evals can be imported
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evals.runner import list_all_tasks, run_dry_validation  # noqa: E402


def test_evals_task_suite_has_ten_tasks():
    tasks = list_all_tasks()
    assert len(tasks) == 10, f"Expected 10 tasks, found {len(tasks)}"

    languages = {t[0].language for t in tasks}
    assert "python" in languages
    assert "rust" in languages
    assert "cpp" in languages

    families = {t[0].family for t in tasks}
    assert "F-bugfix" in families
    assert "F-feature" in families
    assert "F-refactor" in families


def test_evals_dry_validation_all_pass():
    tasks = list_all_tasks()
    for task_cfg, task_dir in tasks:
        res = run_dry_validation(task_cfg, task_dir)
        assert res.status == "passed"
        assert res.task_id == task_cfg.id
