from __future__ import annotations

from interval_merger import merge_intervals


def test_empty_list():
    assert merge_intervals([]) == []


def test_standard_overlap():
    assert merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]) == [
        (1, 6),
        (8, 10),
        (15, 18),
    ]


def test_touching_intervals():
    assert merge_intervals([(1, 4), (4, 5)]) == [(1, 5)]


def test_unsorted_input():
    assert merge_intervals([(5, 7), (1, 2)]) == [(1, 2), (5, 7)]


def test_contained_interval():
    assert merge_intervals([(1, 10), (2, 3), (4, 5)]) == [(1, 10)]


def test_single_point():
    assert merge_intervals([(3, 3)]) == [(3, 3)]
