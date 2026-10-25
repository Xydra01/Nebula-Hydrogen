"""Interval merger utility with touching interval bug."""

from __future__ import annotations


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping closed intervals."""
    if not intervals:
        return []

    # BUG: strictly less than instead of <= misses touching intervals (1, 3) and (3, 5)
    # BUG: assumes pre-sorted input
    out: list[tuple[int, int]] = []
    for s, e in intervals:
        if out and s < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out
