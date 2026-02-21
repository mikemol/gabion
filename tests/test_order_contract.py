from __future__ import annotations

import pytest

from gabion.analysis.timeout_context import (
    Deadline,
    GasMeter,
    deadline_clock_scope,
    deadline_scope,
)
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


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_sorts_by_default::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_sorts_by_default() -> None:
    values = ["b", "a", "c"]
    assert ordered_or_sorted(values, source="test") == ["a", "b", "c"]


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_accepts_sorted_when_required::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_accepts_sorted_when_required() -> None:
    values = ["a", "b", "c"]
    assert ordered_or_sorted(values, source="test", require_sorted=True) == values


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_rejects_unsorted_when_required::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_rejects_unsorted_when_required() -> None:
    with pytest.raises(NeverThrown):
        ordered_or_sorted(["b", "a"], source="test", require_sorted=True)


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_respects_env_toggle::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_respects_env_toggle(env_scope, restore_env) -> None:
    previous = env_scope({"GABION_CALLER_SORTED": "1"})
    try:
        with pytest.raises(NeverThrown):
            ordered_or_sorted(["b", "a"], source="test")
    finally:
        restore_env(previous)


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_check_policy_sorts_only_on_regression::order_contract.py::gabion.order_contract.order_policy::order_contract.py::gabion.order_contract.ordered_or_sorted
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


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_trust_policy_keeps_caller_order::order_contract.py::gabion.order_contract.order_policy::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_trust_policy_keeps_caller_order() -> None:
    values = ["b", "a", "c"]
    with order_policy(OrderPolicy.TRUST):
        assert ordered_or_sorted(values, source="test") == values


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_policy_argument_overrides_context::order_contract.py::gabion.order_contract.order_policy::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_policy_argument_overrides_context() -> None:
    values = ["b", "a", "c"]
    with order_policy(OrderPolicy.TRUST):
        assert ordered_or_sorted(
            values,
            source="test",
            policy=OrderPolicy.SORT,
        ) == ["a", "b", "c"]


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_enforce_policy_raises_on_regression::order_contract.py::gabion.order_contract.order_policy::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_enforce_policy_raises_on_regression() -> None:
    with order_policy(OrderPolicy.ENFORCE):
        with pytest.raises(NeverThrown):
            ordered_or_sorted(["b", "a"], source="test")


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_check_records_context_telemetry::order_contract.py::gabion.order_contract.order_policy::order_contract.py::gabion.order_contract.order_telemetry::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_check_records_context_telemetry() -> None:
    with order_policy(OrderPolicy.CHECK):
        with order_telemetry() as events:
            ordered_or_sorted(["b", "a"], source="test")
    assert len(events) == 1
    assert events[0]["action"] == "fallback_sort"
    assert events[0]["source"] == "test"


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_check_records_global_telemetry_env::order_contract.py::gabion.order_contract.get_order_telemetry_events::order_contract.py::gabion.order_contract.order_policy::order_contract.py::gabion.order_contract.ordered_or_sorted
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


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_get_order_telemetry_events_can_clear_global_buffer::order_contract.py::gabion.order_contract.get_order_telemetry_events::order_contract.py::gabion.order_contract.order_policy::order_contract.py::gabion.order_contract.ordered_or_sorted
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


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_enforces_canonical_sort_allowlist::order_contract.py::gabion.order_contract.ordered_or_sorted
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


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_canonical_sort_allowlist_accepts_known_source::order_contract.py::gabion.order_contract.ordered_or_sorted
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


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_canonical_sort_allowlist_accepts_context_scope::order_contract.py::gabion.order_contract.canonical_sort_allowlist::order_contract.py::gabion.order_contract.ordered_or_sorted
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


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_get_order_policy_reads_env_alias_values::order_contract.py::gabion.order_contract.get_order_policy
def test_get_order_policy_reads_env_alias_values(env_scope, restore_env) -> None:
    previous = env_scope({"GABION_ORDER_POLICY": "off"})
    try:
        assert get_order_policy() is OrderPolicy.SORT
        blank_previous = env_scope({"GABION_ORDER_POLICY": "   "})
        try:
            assert get_order_policy() is OrderPolicy.SORT
        finally:
            restore_env(blank_previous)
        on_previous = env_scope({"GABION_ORDER_POLICY": "on"})
        try:
            assert get_order_policy() is OrderPolicy.ENFORCE
        finally:
            restore_env(on_previous)
    finally:
        restore_env(previous)


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_get_order_policy_rejects_unknown_env_policy::order_contract.py::gabion.order_contract.get_order_policy
def test_get_order_policy_rejects_unknown_env_policy(env_scope, restore_env) -> None:
    previous = env_scope({"GABION_ORDER_POLICY": "nonsense"})
    try:
        with pytest.raises(NeverThrown):
            get_order_policy()
    finally:
        restore_env(previous)


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_check_handles_incomparable_values::order_contract.py::gabion.order_contract.order_policy::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_check_handles_incomparable_values() -> None:
    with order_policy(OrderPolicy.CHECK):
        with pytest.raises(TypeError):
            ordered_or_sorted([{"a": 1}, {"b": 2}], source="incomparable")


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_enforce_rejects_incomparable_values::order_contract.py::gabion.order_contract.order_policy::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_enforce_rejects_incomparable_values() -> None:
    with order_policy(OrderPolicy.ENFORCE):
        with pytest.raises(NeverThrown):
            ordered_or_sorted([{"a": 1}, {"b": 2}], source="incomparable")


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_canonical_allowlist_skips_without_explicit_sort::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_canonical_allowlist_skips_without_explicit_sort(
    env_scope, restore_env
) -> None:
    previous = env_scope({"GABION_ENFORCE_CANONICAL_SORT_ALLOWLIST": "1"})
    try:
        assert ordered_or_sorted(["z", "a"], source="not_allowlisted") == ["a", "z"]
    finally:
        restore_env(previous)


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_explicit_sort_from_require_sorted_false_obeys_allowlist::order_contract.py::gabion.order_contract.ordered_or_sorted
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


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_accepts_string_policy::order_contract.py::gabion.order_contract.order_policy::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_accepts_string_policy() -> None:
    with order_policy(OrderPolicy.TRUST):
        ordered = ordered_or_sorted(["a", "b"], source="string-policy", policy="enforce")
    assert ordered == ["a", "b"]


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_deadline_probe_paths::order_contract.py::gabion.order_contract.ordered_or_sorted::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.deadline_clock_scope::timeout_context.py::gabion.analysis.timeout_context.deadline_scope
def test_ordered_or_sorted_deadline_probe_paths(env_scope, restore_env) -> None:
    previous = env_scope(
        {
            "GABION_ORDER_DEADLINE_PROBE": "1",
            "GABION_ENFORCE_CANONICAL_SORT_ALLOWLIST": "1",
        }
    )
    try:
        with deadline_scope(Deadline.from_timeout_ms(1_000)):
            with deadline_clock_scope(GasMeter(limit=10_000)):
                ordered_or_sorted(["b", "a"], source="deadline-probe.default")
                ordered_or_sorted(
                    ["b", "a"],
                    source="canonical_type_key.deadline_probe",
                    policy="sort",
                )
                ordered_or_sorted(
                    ["b", "a"],
                    source="deadline-probe.check",
                    policy=OrderPolicy.CHECK,
                )
    finally:
        restore_env(previous)


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_order_deadline_tick_budget_allows_check_non_meter_clock::order_contract.py::gabion.order_contract._deadline_tick_budget_allows_check
def test_order_deadline_tick_budget_allows_check_non_meter_clock() -> None:
    class _Clock:
        pass

    from gabion import order_contract

    assert order_contract._deadline_tick_budget_allows_check(_Clock()) is True
