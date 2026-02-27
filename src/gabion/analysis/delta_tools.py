# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once

_AUTO_TRANSITION_DELTA = object()


def coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def format_delta(value: object) -> str:
    normalized = coerce_int(value, 0)
    sign = "+" if normalized > 0 else ""
    return f"{sign}{normalized}"


@dataclass(frozen=True)
class TransitionPair:
    baseline: object
    current: object


def format_transition(
    pair: TransitionPair,
    delta: object = _AUTO_TRANSITION_DELTA,
) -> str:
    base = coerce_int(pair.baseline, 0)
    curr = coerce_int(pair.current, 0)
    delta_value = coerce_int(delta, curr - base)
    return f"{base} -> {curr} ({format_delta(delta_value)})"


def count_delta(
    baseline: Mapping[str, object],
    current: Mapping[str, object],
) -> dict[str, dict[str, int]]:
    check_deadline(allow_frame_fallback=True)
    keys = sort_once(
        {*baseline.keys(), *current.keys()},
        source="delta_tools.count_delta.keys",
    )
    baseline_counts = {key: coerce_int(baseline.get(key), 0) for key in keys}
    current_counts = {key: coerce_int(current.get(key), 0) for key in keys}
    delta_counts = {
        key: current_counts.get(key, 0) - baseline_counts.get(key, 0) for key in keys
    }
    return {
        "baseline": baseline_counts,
        "current": current_counts,
        "delta": delta_counts,
    }
