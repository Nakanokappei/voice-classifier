"""Tests for the pipeline's interactive column-selection parser."""

from __future__ import annotations

import pytest

from src import pipeline


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1", [1]),
        ("3", [3]),
        ("1,3,5", [1, 3, 5]),
        ("1;3;5", [1, 3, 5]),
        ("1-3", [1, 2, 3]),
        ("1-3,5", [1, 2, 3, 5]),
        ("5,1-3", [5, 1, 2, 3]),          # order preserved
        ("1;3-4,2", [1, 3, 4, 2]),        # mixed separators
        ("1,1,2", [1, 2]),                # duplicates collapsed
        ("  1 , 3 ", [1, 3]),             # whitespace tolerated
    ],
)
def test_parse_column_selection_happy_paths(raw: str, expected: list[int]) -> None:
    assert pipeline._parse_column_selection(raw, max_index=10) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",                     # empty
        "   ",                  # whitespace only
        "abc",                  # not a number
        "0",                    # below range
        "11",                   # above range (max=10)
        "1-11",                 # range overflows
        "3-1",                  # descending not allowed
        "1--3",                 # malformed
        "1,",                   # trailing sep is fine (empty tokens skipped)
    ],
)
def test_parse_column_selection_rejects_invalid(raw: str) -> None:
    if raw in {"1,"}:
        # trailing separator yields the same single-index result; not an error
        assert pipeline._parse_column_selection(raw, max_index=10) == [1]
        return
    with pytest.raises(ValueError):
        pipeline._parse_column_selection(raw, max_index=10)
