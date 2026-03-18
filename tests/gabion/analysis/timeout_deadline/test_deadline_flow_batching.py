from __future__ import annotations

import pytest

from gabion.analysis.foundation import timeout_context as tc
from gabion.deadline_clock import GasMeter


# gabion:behavior primary=desired
def test_project_deadline_flow_window_uses_model_budget_and_growth_guardrail(
    monkeypatch,
) -> None:
    monkeypatch.setattr(tc, "get_deadline", lambda: tc.Deadline(deadline_ns=1))
    monkeypatch.setattr(tc, "_deadline_ns_remaining", lambda _deadline: 1600)

    projection = tc._project_deadline_flow_window(
        previous_window=4,
        previous_items_per_tick_ewma=0.5,
        has_previous_items_per_tick_ewma=True,
        previous_items_per_ns_ewma=0.05,
        has_previous_items_per_ns_ewma=True,
        previous_best_items_per_tick=0.5,
        has_previous_best_items_per_tick=True,
        previous_best_items_per_ns=0.05,
        has_previous_best_items_per_ns=True,
        observation=tc._DeadlineFlowObservation(
            items_delta=8,
            tick_delta=16,
            ns_delta=160,
        ),
        remaining_tick_budget=tc._RemainingTickBudget(known=True, remaining_ticks=100),
    )
    assert projection.next_window == 25
    assert projection.items_per_tick_ewma == pytest.approx(0.5)
    assert projection.items_per_ns_ewma == pytest.approx(0.05)


# gabion:behavior primary=desired
def test_project_deadline_flow_window_uses_time_budget_when_it_is_tighter(
    monkeypatch,
) -> None:
    monkeypatch.setattr(tc, "get_deadline", lambda: tc.Deadline(deadline_ns=1))
    monkeypatch.setattr(tc, "_deadline_ns_remaining", lambda _deadline: 80)

    projection = tc._project_deadline_flow_window(
        previous_window=32,
        previous_items_per_tick_ewma=0.5,
        has_previous_items_per_tick_ewma=True,
        previous_items_per_ns_ewma=0.05,
        has_previous_items_per_ns_ewma=True,
        previous_best_items_per_tick=0.5,
        has_previous_best_items_per_tick=True,
        previous_best_items_per_ns=0.05,
        has_previous_best_items_per_ns=True,
        observation=tc._DeadlineFlowObservation(
            items_delta=8,
            tick_delta=16,
            ns_delta=160,
        ),
        remaining_tick_budget=tc._RemainingTickBudget(
            known=True,
            remaining_ticks=10_000,
        ),
    )

    assert projection.next_window == 2


# gabion:behavior primary=desired
def test_project_deadline_flow_window_uses_vegas_style_backoff_on_cost_spike(
    monkeypatch,
) -> None:
    monkeypatch.setattr(tc, "get_deadline", lambda: tc.Deadline(deadline_ns=1))
    monkeypatch.setattr(tc, "_deadline_ns_remaining", lambda _deadline: 20_000)

    projection = tc._project_deadline_flow_window(
        previous_window=16,
        previous_items_per_tick_ewma=0.5,
        has_previous_items_per_tick_ewma=True,
        previous_items_per_ns_ewma=0.05,
        has_previous_items_per_ns_ewma=True,
        previous_best_items_per_tick=0.5,
        has_previous_best_items_per_tick=True,
        previous_best_items_per_ns=0.05,
        has_previous_best_items_per_ns=True,
        observation=tc._DeadlineFlowObservation(
            items_delta=8,
            tick_delta=64,
            ns_delta=640,
        ),
        remaining_tick_budget=tc._RemainingTickBudget(
            known=True,
            remaining_ticks=4_000,
        ),
    )

    assert projection.next_window == 12
    assert projection.items_per_tick_ewma == pytest.approx(0.36875)
    assert projection.items_per_ns_ewma == pytest.approx(0.036875)


# gabion:behavior primary=desired
def test_project_deadline_flow_window_prefers_best_recent_throughput_over_ewma(
    monkeypatch,
) -> None:
    monkeypatch.setattr(tc, "get_deadline", lambda: tc.Deadline(deadline_ns=1))
    monkeypatch.setattr(tc, "_deadline_ns_remaining", lambda _deadline: 10_000)

    projection = tc._project_deadline_flow_window(
        previous_window=32,
        previous_items_per_tick_ewma=0.25,
        has_previous_items_per_tick_ewma=True,
        previous_items_per_ns_ewma=0.0025,
        has_previous_items_per_ns_ewma=True,
        previous_best_items_per_tick=1.0,
        has_previous_best_items_per_tick=True,
        previous_best_items_per_ns=0.01,
        has_previous_best_items_per_ns=True,
        observation=tc._DeadlineFlowObservation(
            items_delta=8,
            tick_delta=32,
            ns_delta=3200,
        ),
        remaining_tick_budget=tc._RemainingTickBudget(known=True, remaining_ticks=100),
    )

    assert projection.next_window == 47
    assert projection.best_items_per_tick == pytest.approx(0.95)
    assert projection.best_items_per_ns == pytest.approx(0.0095)


# gabion:behavior primary=desired
def test_deadline_loop_iter_batches_polls_by_observed_tick_rate(monkeypatch) -> None:
    deadline_checks: list[int] = []
    consume_deadline_ticks = tc.consume_deadline_ticks
    monkeypatch.setattr(tc, "_TIMEOUT_PROGRESS_CHECKS_FLOOR", 2)
    monkeypatch.setattr(tc, "_TIMEOUT_FLOW_INITIAL_WINDOW", 2)

    def _batched_check_deadline(*_args, **_kwargs) -> None:
        deadline_checks.append(int(tc.get_deadline_clock().get_mark()))
        consume_deadline_ticks()

    monkeypatch.setattr(tc, "check_deadline", _batched_check_deadline)

    seen: list[int] = []
    with tc.deadline_scope(tc.Deadline.from_timeout_ms(100)):
        with tc.deadline_clock_scope(GasMeter(limit=64)):
            for value in tc.deadline_loop_iter(range(10)):
                seen.append(value)
                consume_deadline_ticks(3)

    assert seen == list(range(10))
    assert 2 <= len(deadline_checks) < len(seen)
