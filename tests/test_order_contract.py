from __future__ import annotations

import os

import pytest

from gabion.exceptions import NeverThrown
from gabion.order_contract import OrderPolicy, order_policy, ordered_or_sorted


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


def test_ordered_or_sorted_check_policy_sorts_only_on_regression() -> None:
    values = ["b", "a", "c"]
    observed: list[dict[str, object]] = []
    with order_policy(OrderPolicy.CHECK):
        ordered = ordered_or_sorted(
            values,
            source="test",
            on_unsorted=lambda payload: observed.append(payload),
        )
    assert ordered == ["a", "b", "c"]
    assert len(observed) == 1
    assert observed[0]["violation_kind"] == "out_of_order"


def test_ordered_or_sorted_trust_policy_keeps_caller_order() -> None:
    values = ["b", "a", "c"]
    with order_policy(OrderPolicy.TRUST):
        assert ordered_or_sorted(values, source="test") == values


def test_ordered_or_sorted_policy_argument_overrides_context() -> None:
    values = ["b", "a", "c"]
    with order_policy(OrderPolicy.TRUST):
        assert ordered_or_sorted(
            values,
            source="test",
            policy=OrderPolicy.SORT,
        ) == ["a", "b", "c"]


def test_ordered_or_sorted_enforce_policy_raises_on_regression() -> None:
    with order_policy(OrderPolicy.ENFORCE):
        with pytest.raises(NeverThrown):
            ordered_or_sorted(["b", "a"], source="test")
