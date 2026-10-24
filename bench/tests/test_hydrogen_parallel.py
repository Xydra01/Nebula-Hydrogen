"""Tests for 8-stream parallel Harness² candidate evaluation under Project Hydrogen (H23)."""

from __future__ import annotations

import sys
from pathlib import Path

# Add bench directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydrogen_harness_eval import (
    MAX_ALLOWED_HYDROGEN_LATENCY_SECS,
    MAX_ALLOWED_KV_CACHE_BYTES,
    MIN_AGGREGATE_THROUGHPUT_TOKENS_PER_SEC,
    SERIAL_LLAMA_CPP_LATENCY_SECS,
    HydrogenHarnessEvaluator,
)


def test_parallel_8_stream_candidate_evaluation() -> None:
    """Verify 8 concurrent streams finish in < 40s with >= 500 tokens/s and < 3.5GB KV cache."""
    # Prefix 4096 tokens (system prompt + codebase tools), 2048 tokens per stream
    evaluator = HydrogenHarnessEvaluator(prefix_tokens=4096, tokens_per_probe=2048)
    probes = evaluator.generate_candidate_probes(k=4)

    assert len(probes) == 8, "Must generate exactly 8 candidate probes (4 inv + 4 dom)"

    summary = evaluator.evaluate_batched(probes, simulate_time=True)

    # 1. Total probes & verification
    assert summary.total_probes == 8
    assert len(summary.results) == 8
    assert all(r.verified for r in summary.results)

    # 2. Strict latency ceiling (< 40 seconds)
    assert summary.wall_clock_seconds < MAX_ALLOWED_HYDROGEN_LATENCY_SECS, (
        f"Evaluation took {summary.wall_clock_seconds:.2f}s, exceeding 40s ceiling"
    )

    # 3. Aggregate throughput (>= 500 tokens/sec)
    assert summary.aggregate_throughput_tokens_per_sec >= MIN_AGGREGATE_THROUGHPUT_TOKENS_PER_SEC, (
        f"Aggregate throughput {summary.aggregate_throughput_tokens_per_sec:.1f} tokens/s "
        f"is below required 500 tokens/s"
    )

    # 4. KV cache memory footprint (< 3.5 GB)
    assert summary.kv_cache_bytes < MAX_ALLOWED_KV_CACHE_BYTES, (
        f"KV cache {summary.kv_cache_gib:.2f} GiB exceeded 3.5 GB ceiling"
    )
    assert summary.kv_cache_gib < 3.5

    # 5. Speedup over serial execution
    assert summary.speedup_ratio >= 6.0
    assert summary.speedup_ratio == round(
        SERIAL_LLAMA_CPP_LATENCY_SECS / summary.wall_clock_seconds, 2
    )
