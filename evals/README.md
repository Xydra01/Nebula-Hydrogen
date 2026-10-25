# Nebula Regression Benchmark (`evals/`)

This directory houses the authoritative regression benchmark suite for Nebula (Phase 1, Section 11.4 of `NEBULA_DESIGN.md`).

## Overview

The regression benchmark verifies that autonomous agent executor capabilities remain stable and improve over time without regressing. Each task belongs to one of three canonical families:

| Family | Description | Pass Condition |
| :--- | :--- | :--- |
| **`F-bugfix`** | Diagnose a failing test or bug report, trace root cause across files, and apply a targeted fix. | Failing test passes, no regressions, minimal diff. |
| **`F-feature`** | Implement a non-trivial feature across files with new and existing tests. | New tests pass, hidden verification tests pass. |
| **`F-refactor`** | Modularize monolithic code or optimize implementations while preserving exact behavior. | Test suite stays green across every step. |

## Tasks

| ID | Language | Family | Title |
| :--- | :--- | :--- | :--- |
| `task_01_py_interval_merge` | Python | `F-bugfix` | Fix touching interval merge and empty list handling |
| `task_02_py_lru_cache` | Python | `F-feature` | Implement $O(1)$ LRU Cache with capacity eviction |
| `task_03_py_fast_tokenizer` | Python | `F-refactor` | Modularize monolithic regex BPE tokenizer |
| `task_04_py_smart_archive_chunker` | Python | `F-feature` | Semantic markdown document chunker with header tracking |
| `task_05_rs_roman_numeral` | Rust | `F-bugfix` | Canonical Roman numeral validation and subtractive parsing |
| `task_06_rs_word_frequency` | Rust | `F-feature` | Frequency-ranked alphanumeric word counter |
| `task_07_rs_bounded_ring` | Rust | `F-refactor` | Safe zero-allocation circular ring buffer |
| `task_08_cpp_bounds_check` | C++ | `F-bugfix` | Buffer view off-by-one boundary fix |
| `task_09_cpp_ring_buffer` | C++ | `F-feature` | Lock-free atomic single-producer ring buffer |
| `task_10_cpp_matrix_multiply` | C++ | `F-refactor` | Cache-blocked tiled GEMM optimization |

## Running the Benchmark

```powershell
# Run all benchmark tasks
python evals/runner.py --all

# Run a specific task
python evals/runner.py --task task_01_py_interval_merge

# Run dry-run validation (checks task integrity without starting full agent sessions)
python evals/runner.py --dry-run
```

Results are stored in `evals/results/baseline_report.json` and mirrored to `bench/results/evals_baseline.json`.
