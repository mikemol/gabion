from __future__ import annotations

import pytest

from gabion import order_contract
from gabion.analysis.timeout_context import (
    Deadline,
    GasMeter,
    deadline_clock_scope,
    deadline_scope,
)
from gabion.exceptions import NeverThrown
from gabion.order_contract import (
    OrderPolicy,
    OrderRuntimeConfig,
    canonical_sort_allowlist,
    enforce_ordered,
    get_order_policy,
    get_order_telemetry_events,
    is_sorted_once_carrier,
    order_policy,
    order_telemetry,
    ordered_or_sorted,
    order_runtime_config_scope,
    sort_once,
    sorted_once_source,
)


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_sorts_by_default::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_sorts_by_default() -> None:
    values = ["b", "a", "c"]
    assert ordered_or_sorted(values, source="test") == ["a", "b", "c"]


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_sort_once_sorts_without_markerizing_plain_lists::order_contract.py::gabion.order_contract.sort_once
def test_sort_once_sorts_without_markerizing_plain_lists() -> None:
    carrier = sort_once(["b", "a"], source="test.sort_once.first")
    assert carrier == ["a", "b"]
    assert is_sorted_once_carrier(carrier) is False
    assert sorted_once_source(carrier) is None


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_sort_once_accepts_require_sorted_false_compatibility_path::order_contract.py::gabion.order_contract.sort_once
def test_sort_once_accepts_require_sorted_false_compatibility_path() -> None:
    carrier = sort_once(
        ["b", "a"],
        source="tests.test_order_contract.require_sorted_false",
        require_sorted=False,
    )
    assert carrier == ["a", "b"]


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_sort_once_auto_source_path::order_contract.py::gabion.order_contract.sort_once
def test_sort_once_auto_source_path() -> None:
    carrier = sort_once(["b", "a"])
    assert carrier == ["a", "b"]


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_sort_once_trust_policy_keeps_order::order_contract.py::gabion.order_contract.sort_once
def test_sort_once_trust_policy_keeps_order() -> None:
    values = ["b", "a", "c"]
    assert (
        sort_once(
            values,
            source="tests.test_order_contract.sort_once_trust",
            policy=OrderPolicy.TRUST,
        )
        == values
    )


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_sort_once_check_policy_records_unsorted_payload::order_contract.py::gabion.order_contract.sort_once
def test_sort_once_check_policy_records_unsorted_payload() -> None:
    observed: list[dict[str, object]] = []
    ordered = sort_once(
        ["b", "a", "c"],
        source="tests.test_order_contract.sort_once_check",
        policy=OrderPolicy.CHECK,
        on_unsorted=observed.append,
    )
    assert ordered == ["a", "b", "c"]
    assert len(observed) == 1
    assert observed[0]["violation_kind"] == "out_of_order"


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_sort_once_check_policy_without_callback_still_sorts::order_contract.py::gabion.order_contract.sort_once
def test_sort_once_check_policy_without_callback_still_sorts() -> None:
    ordered = sort_once(
        ["c", "a", "b"],
        source="tests.test_order_contract.sort_once_check_no_callback",
        policy=OrderPolicy.CHECK,
    )
    assert ordered == ["a", "b", "c"]


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_sort_once_enforce_policy_raises_on_unsorted::order_contract.py::gabion.order_contract.sort_once
def test_sort_once_enforce_policy_raises_on_unsorted() -> None:
    with pytest.raises(NeverThrown):
        sort_once(
            ["b", "a"],
            source="tests.test_order_contract.sort_once_enforce",
            policy=OrderPolicy.ENFORCE,
        )


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_enforce_ordered_validates_without_sort_fallback::order_contract.py::gabion.order_contract.enforce_ordered
def test_enforce_ordered_validates_without_sort_fallback() -> None:
    ordered = enforce_ordered(["a", "b"], source="test.enforce")
    assert ordered == ["a", "b"]
    with pytest.raises(NeverThrown):
        enforce_ordered(["b", "a"], source="test.enforce.unsorted")


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_accepts_sorted_when_required::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_accepts_sorted_when_required() -> None:
    values = ["a", "b", "c"]
    assert ordered_or_sorted(values, source="test", require_sorted=True) == values


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_rejects_unsorted_when_required::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_rejects_unsorted_when_required() -> None:
    with pytest.raises(NeverThrown):
        ordered_or_sorted(["b", "a"], source="test", require_sorted=True)


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_respects_env_toggle::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_respects_runtime_toggle() -> None:
    with order_runtime_config_scope(OrderRuntimeConfig(legacy_caller_sorted=True)):
        with pytest.raises(NeverThrown):
            ordered_or_sorted(["b", "a"], source="test")


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
def test_ordered_or_sorted_check_records_global_telemetry_runtime() -> None:
    get_order_telemetry_events(clear=True)
    with order_runtime_config_scope(OrderRuntimeConfig(telemetry_enabled=True)):
        with order_policy(OrderPolicy.CHECK):
            ordered_or_sorted(["b", "a"], source="test-global")
    events = get_order_telemetry_events(clear=True)
    assert len(events) == 1
    assert events[0]["source"] == "test-global"


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_get_order_telemetry_events_can_clear_global_buffer::order_contract.py::gabion.order_contract.get_order_telemetry_events::order_contract.py::gabion.order_contract.order_policy::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_get_order_telemetry_events_can_clear_global_buffer() -> None:
    with order_runtime_config_scope(OrderRuntimeConfig(telemetry_enabled=True)):
        with order_policy(OrderPolicy.CHECK):
            ordered_or_sorted(["b", "a"], source="clear-buffer")
    assert get_order_telemetry_events(clear=False)
    assert get_order_telemetry_events(clear=True)
    assert get_order_telemetry_events(clear=False) == []


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_enforces_canonical_sort_allowlist::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_enforces_canonical_sort_allowlist() -> None:
    with order_runtime_config_scope(OrderRuntimeConfig(enforce_canonical_allowlist=True)):
        with pytest.raises(NeverThrown):
            ordered_or_sorted(
                ["b", "a"],
                source="non_canonical_source",
                policy=OrderPolicy.SORT,
            )


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_canonical_sort_allowlist_accepts_known_source::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_canonical_sort_allowlist_accepts_known_source() -> None:
    with order_runtime_config_scope(OrderRuntimeConfig(enforce_canonical_allowlist=True)):
        ordered = ordered_or_sorted(
            ["b", "a"],
            source="canonical_type_key.union_parts",
            policy=OrderPolicy.SORT,
        )
    assert ordered == ["a", "b"]


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_canonical_sort_allowlist_accepts_context_scope::order_contract.py::gabion.order_contract.canonical_sort_allowlist::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_canonical_sort_allowlist_accepts_context_scope() -> None:
    with order_runtime_config_scope(OrderRuntimeConfig(enforce_canonical_allowlist=True)):
        with canonical_sort_allowlist("scoped_sort_prefix"):
            ordered = ordered_or_sorted(
                ["b", "a"],
                source="scoped_sort_prefix.dynamic",
                policy=OrderPolicy.SORT,
            )
    assert ordered == ["a", "b"]


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_get_order_policy_reads_env_alias_values::order_contract.py::gabion.order_contract.get_order_policy
def test_get_order_policy_reads_runtime_values() -> None:
    with order_runtime_config_scope(OrderRuntimeConfig(default_policy=OrderPolicy.SORT)):
        assert get_order_policy() is OrderPolicy.SORT
    assert get_order_policy() is OrderPolicy.SORT
    with order_runtime_config_scope(OrderRuntimeConfig(default_policy=OrderPolicy.ENFORCE)):
        assert get_order_policy() is OrderPolicy.ENFORCE


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_get_order_policy_rejects_unknown_env_policy::order_contract.py::gabion.order_contract.get_order_policy
def test_get_order_policy_rejects_unknown_runtime_policy() -> None:
    with pytest.raises(ValueError):
        OrderPolicy("nonsense")


def test_normalize_policy_rejects_unknown_policy_string() -> None:
    with pytest.raises(NeverThrown):
        order_contract._normalize_policy("not-a-policy")  # type: ignore[name-defined]


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
def test_ordered_or_sorted_canonical_allowlist_skips_without_explicit_sort() -> None:
    with order_runtime_config_scope(OrderRuntimeConfig(enforce_canonical_allowlist=True)):
        assert ordered_or_sorted(["z", "a"], source="not_allowlisted") == ["a", "z"]


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_explicit_sort_from_require_sorted_false_obeys_allowlist::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_explicit_sort_from_require_sorted_false_obeys_allowlist() -> None:
    with order_runtime_config_scope(OrderRuntimeConfig(enforce_canonical_allowlist=True)):
        with pytest.raises(NeverThrown):
            ordered_or_sorted(
                ["z", "a"],
                source="not_allowlisted.explicit",
                require_sorted=False,
            )


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_accepts_string_policy::order_contract.py::gabion.order_contract.order_policy::order_contract.py::gabion.order_contract.ordered_or_sorted
def test_ordered_or_sorted_accepts_string_policy() -> None:
    with order_policy(OrderPolicy.TRUST):
        ordered = ordered_or_sorted(["a", "b"], source="string-policy", policy="enforce")
    assert ordered == ["a", "b"]


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_ordered_or_sorted_deadline_probe_paths::order_contract.py::gabion.order_contract.ordered_or_sorted::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.deadline_clock_scope::timeout_context.py::gabion.analysis.timeout_context.deadline_scope
def test_ordered_or_sorted_deadline_probe_paths() -> None:
    with order_runtime_config_scope(
        OrderRuntimeConfig(
            deadline_probe_enabled=True,
            enforce_canonical_allowlist=True,
        )
    ):
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


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_order_deadline_tick_budget_allows_check_non_meter_clock::order_contract.py::gabion.order_contract._deadline_tick_budget_allows_check
def test_order_deadline_tick_budget_allows_check_non_meter_clock() -> None:
    class _Clock:
        pass

    from gabion import order_contract

    assert order_contract._deadline_tick_budget_allows_check(_Clock()) is True


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_deadline_clock_if_available_returns_none_without_clock_scope::order_contract.py::gabion.order_contract._deadline_clock_if_available
def test_deadline_clock_if_available_returns_none_without_clock_scope() -> None:
    from gabion import order_contract
    from gabion.analysis import timeout_context

    with deadline_scope(Deadline.from_timeout_ms(1_000)):
        token = timeout_context._deadline_clock_var.set(None)
        try:
            assert order_contract._deadline_clock_if_available() is None
        finally:
            timeout_context._deadline_clock_var.reset(token)


# gabion:evidence E:call_footprint::tests/test_order_contract.py::test_auto_sort_source_handles_missing_frame_paths::order_contract.py::gabion.order_contract._auto_sort_source
def test_auto_sort_source_handles_missing_frame_paths() -> None:
    from types import SimpleNamespace

    from gabion import order_contract

    class _Frame:
        def __init__(self, name: str, filename: str, lineno: int, f_back: object | None) -> None:
            self.f_code = SimpleNamespace(co_name=name, co_filename=filename)
            self.f_lineno = lineno
            self.f_back = f_back

    assert (
        order_contract._auto_sort_source(
            currentframe_fn=lambda: None,
        )
        == "sort_once.auto"
    )
    sentinel = _Frame("_auto_sort_source", "auto.py", 1, None)
    assert (
        order_contract._auto_sort_source(
            currentframe_fn=lambda: sentinel,
        )
        == "sort_once.auto"
    )
