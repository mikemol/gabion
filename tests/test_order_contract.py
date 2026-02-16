from __future__ import annotations

import os

import pytest

from gabion.exceptions import NeverThrown
from gabion.order_contract import (
    OrderPolicy,
    canonical_sort_allowlist,
    get_order_policy,
    get_order_telemetry_events,
    order_policy,
    order_telemetry,
    ordered_or_sorted,
)


def test_ordered_or_sorted_sorts_by_default() -> None:
    values = ["b", "a", "c"]
    assert ordered_or_sorted(values, source="test") == ["a", "b", "c"]


def test_ordered_or_sorted_accepts_sorted_when_required() -> None:
    values = ["a", "b", "c"]
    assert ordered_or_sorted(values, source="test", require_sorted=True) == values


def test_ordered_or_sorted_rejects_unsorted_when_required() -> None:
    with pytest.raises(NeverThrown):
        ordered_or_sorted(["b", "a"], source="test", require_sorted=True)


def test_ordered_or_sorted_respects_env_toggle(env_scope, restore_env) -> None:
    previous = env_scope({"GABION_CALLER_SORTED": "1"})
    try:
        with pytest.raises(NeverThrown):
            ordered_or_sorted(["b", "a"], source="test")
    finally:
        restore_env(previous)


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


def test_ordered_or_sorted_check_records_context_telemetry() -> None:
    with order_policy(OrderPolicy.CHECK):
        with order_telemetry() as events:
            ordered_or_sorted(["b", "a"], source="test")
    assert len(events) == 1
    assert events[0]["action"] == "fallback_sort"
    assert events[0]["source"] == "test"


def test_ordered_or_sorted_check_records_global_telemetry_env(
    env_scope, restore_env
) -> None:
    previous = env_scope({"GABION_ORDER_TELEMETRY": "1"})
    get_order_telemetry_events(clear=True)
    try:
        with order_policy(OrderPolicy.CHECK):
            ordered_or_sorted(["b", "a"], source="test-global")
        events = get_order_telemetry_events(clear=True)
        assert len(events) == 1
        assert events[0]["source"] == "test-global"
    finally:
        get_order_telemetry_events(clear=True)
        restore_env(previous)


def test_get_order_telemetry_events_can_clear_global_buffer(
    env_scope, restore_env
) -> None:
    previous = env_scope({"GABION_ORDER_TELEMETRY": "1"})
    try:
        with order_policy(OrderPolicy.CHECK):
            ordered_or_sorted(["b", "a"], source="clear-buffer")
        assert get_order_telemetry_events(clear=False)
        assert get_order_telemetry_events(clear=True)
        assert get_order_telemetry_events(clear=False) == []
    finally:
        get_order_telemetry_events(clear=True)
        restore_env(previous)


def test_ordered_or_sorted_enforces_canonical_sort_allowlist(
    env_scope, restore_env
) -> None:
    previous = env_scope({"GABION_ENFORCE_CANONICAL_SORT_ALLOWLIST": "1"})
    try:
        with pytest.raises(NeverThrown):
            ordered_or_sorted(
                ["b", "a"],
                source="non_canonical_source",
                policy=OrderPolicy.SORT,
            )
    finally:
        restore_env(previous)


def test_ordered_or_sorted_canonical_sort_allowlist_accepts_known_source(
    env_scope, restore_env
) -> None:
    previous = env_scope({"GABION_ENFORCE_CANONICAL_SORT_ALLOWLIST": "1"})
    try:
        ordered = ordered_or_sorted(
            ["b", "a"],
            source="canonical_type_key.union_parts",
            policy=OrderPolicy.SORT,
        )
        assert ordered == ["a", "b"]
    finally:
        restore_env(previous)


def test_ordered_or_sorted_canonical_sort_allowlist_accepts_context_scope(
    env_scope, restore_env
) -> None:
    previous = env_scope({"GABION_ENFORCE_CANONICAL_SORT_ALLOWLIST": "1"})
    try:
        with canonical_sort_allowlist("scoped_sort_prefix"):
            ordered = ordered_or_sorted(
                ["b", "a"],
                source="scoped_sort_prefix.dynamic",
                policy=OrderPolicy.SORT,
            )
        assert ordered == ["a", "b"]
    finally:
        restore_env(previous)


def test_get_order_policy_reads_env_alias_values(env_scope, restore_env) -> None:
    previous = env_scope({"GABION_ORDER_POLICY": "off"})
    try:
        assert get_order_policy() is OrderPolicy.SORT
        os.environ["GABION_ORDER_POLICY"] = "   "
        assert get_order_policy() is OrderPolicy.SORT
        os.environ["GABION_ORDER_POLICY"] = "on"
        assert get_order_policy() is OrderPolicy.ENFORCE
    finally:
        restore_env(previous)


def test_get_order_policy_rejects_unknown_env_policy(env_scope, restore_env) -> None:
    previous = env_scope({"GABION_ORDER_POLICY": "nonsense"})
    try:
        with pytest.raises(NeverThrown):
            get_order_policy()
    finally:
        restore_env(previous)


def test_ordered_or_sorted_check_handles_incomparable_values() -> None:
    with order_policy(OrderPolicy.CHECK):
        with pytest.raises(TypeError):
            ordered_or_sorted([{"a": 1}, {"b": 2}], source="incomparable")


def test_ordered_or_sorted_enforce_rejects_incomparable_values() -> None:
    with order_policy(OrderPolicy.ENFORCE):
        with pytest.raises(NeverThrown):
            ordered_or_sorted([{"a": 1}, {"b": 2}], source="incomparable")


def test_ordered_or_sorted_canonical_allowlist_skips_without_explicit_sort(
    env_scope, restore_env
) -> None:
    previous = env_scope({"GABION_ENFORCE_CANONICAL_SORT_ALLOWLIST": "1"})
    try:
        assert ordered_or_sorted(["z", "a"], source="not_allowlisted") == ["a", "z"]
    finally:
        restore_env(previous)


def test_ordered_or_sorted_explicit_sort_from_require_sorted_false_obeys_allowlist(
    env_scope, restore_env
) -> None:
    previous = env_scope({"GABION_ENFORCE_CANONICAL_SORT_ALLOWLIST": "1"})
    try:
        with pytest.raises(NeverThrown):
            ordered_or_sorted(
                ["z", "a"],
                source="not_allowlisted.explicit",
                require_sorted=False,
            )
    finally:
        restore_env(previous)


def test_ordered_or_sorted_accepts_string_policy() -> None:
    with order_policy(OrderPolicy.TRUST):
        ordered = ordered_or_sorted(["a", "b"], source="string-policy", policy="enforce")
    assert ordered == ["a", "b"]
