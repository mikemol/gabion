from __future__ import annotations

import os

import pytest

from gabion.exceptions import NeverThrown
from gabion.order_contract import ordered_or_sorted


def test_ordered_or_sorted_sorts_by_default() -> None:
    values = ["b", "a", "c"]
    assert ordered_or_sorted(values, source="test") == ["a", "b", "c"]


def test_ordered_or_sorted_accepts_sorted_when_required() -> None:
    values = ["a", "b", "c"]
    assert ordered_or_sorted(values, source="test", require_sorted=True) == values


def test_ordered_or_sorted_rejects_unsorted_when_required() -> None:
    with pytest.raises(NeverThrown):
        ordered_or_sorted(["b", "a"], source="test", require_sorted=True)


def test_ordered_or_sorted_respects_env_toggle() -> None:
    previous = os.environ.get("GABION_CALLER_SORTED")
    try:
        os.environ["GABION_CALLER_SORTED"] = "1"
        with pytest.raises(NeverThrown):
            ordered_or_sorted(["b", "a"], source="test")
    finally:
        if previous is None:
            os.environ.pop("GABION_CALLER_SORTED", None)
        else:
            os.environ["GABION_CALLER_SORTED"] = previous
