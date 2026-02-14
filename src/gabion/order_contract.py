from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from enum import Enum
from typing import Any, Callable, Iterable, Iterator, TypeVar

from gabion.invariants import never


T = TypeVar("T")

_CALLER_SORTED_ENV = "GABION_CALLER_SORTED"
_ORDER_POLICY_ENV = "GABION_ORDER_POLICY"
_ORDER_TELEMETRY_ENV = "GABION_ORDER_TELEMETRY"
_ORDER_POLICY_CONTEXT: ContextVar["OrderPolicy | None"] = ContextVar(
    "gabion_order_policy",
    default=None,
)
_ORDER_TELEMETRY_CONTEXT: ContextVar[list[dict[str, object]] | None] = ContextVar(
    "gabion_order_telemetry",
    default=None,
)
_ORDER_TELEMETRY_GLOBAL: list[dict[str, object]] = []


class OrderPolicy(str, Enum):
    SORT = "sort"
    CHECK = "check"
    TRUST = "trust"
    ENFORCE = "enforce"


def ordered_or_sorted(
    values: Iterable[T],
    *,
    source: str,
    key: Callable[[T], Any] | None = None,
    reverse: bool = False,
    policy: OrderPolicy | str | None = None,
    require_sorted: bool | None = None,
    on_unsorted: Callable[[dict[str, object]], None] | None = None,
) -> list[T]:
    """Return deterministic order with configurable caller-order policy.

    - `OrderPolicy.SORT`: always apply sorting.
    - `OrderPolicy.CHECK`: validate caller order, then sort only on regression.
    - `OrderPolicy.TRUST`: trust caller order without validation or sorting.
    - `OrderPolicy.ENFORCE`: require caller-monotonic order, fail via `never()` on regression.

    Policy resolution precedence:
    1. `require_sorted` (legacy override)
    2. explicit `policy`
    3. context policy (`order_policy(...)`)
    4. `GABION_ORDER_POLICY`
    5. `GABION_CALLER_SORTED=1` legacy toggle => `ENFORCE`
    6. default `SORT`
    """

    items = list(values)
    resolved_policy = _resolve_policy(
        policy=policy,
        require_sorted=require_sorted,
    )
    if resolved_policy is OrderPolicy.SORT:
        return sorted(items, key=key, reverse=reverse)
    if resolved_policy is OrderPolicy.TRUST:
        return items
    violation = _first_order_violation(items, key=key, reverse=reverse)
    if violation is None:
        return items
    payload: dict[str, object] = {
        "source": source,
        "previous_index": violation[0],
        "current_index": violation[1],
        "previous_key": repr(violation[2]),
        "current_key": repr(violation[3]),
        "violation_kind": violation[4],
        "reverse": reverse,
        "policy": resolved_policy.value,
    }
    if resolved_policy is OrderPolicy.CHECK:
        _record_order_telemetry(payload)
        if on_unsorted is not None:
            on_unsorted(payload)
        return sorted(items, key=key, reverse=reverse)
    _raise_order_violation(payload, reason=violation[4])
    return items  # pragma: no cover - never() raises


def _resolve_policy(
    *,
    policy: OrderPolicy | str | None,
    require_sorted: bool | None,
) -> OrderPolicy:
    if require_sorted is not None:
        return OrderPolicy.ENFORCE if require_sorted else OrderPolicy.SORT
    if policy is not None:
        return _normalize_policy(policy)
    context_policy = _ORDER_POLICY_CONTEXT.get()
    if context_policy is not None:
        return context_policy
    env_policy = _order_policy_from_env()
    if env_policy is not None:
        return env_policy
    if _caller_sorted_mode_enabled():
        return OrderPolicy.ENFORCE
    return OrderPolicy.SORT


def get_order_policy() -> OrderPolicy:
    return _resolve_policy(policy=None, require_sorted=None)


def get_order_telemetry_events(*, clear: bool = False) -> list[dict[str, object]]:
    events = [dict(entry) for entry in _ORDER_TELEMETRY_GLOBAL]
    if clear:
        _ORDER_TELEMETRY_GLOBAL.clear()
    return events


def set_order_policy(policy: OrderPolicy | str) -> Token[OrderPolicy | None]:
    return _ORDER_POLICY_CONTEXT.set(_normalize_policy(policy))


def reset_order_policy(token: Token[OrderPolicy | None]) -> None:
    _ORDER_POLICY_CONTEXT.reset(token)


@contextmanager
def order_policy(policy: OrderPolicy | str) -> Iterator[None]:
    token = set_order_policy(policy)
    try:
        yield
    finally:
        reset_order_policy(token)


@contextmanager
def order_telemetry() -> Iterator[list[dict[str, object]]]:
    events: list[dict[str, object]] = []
    token = _ORDER_TELEMETRY_CONTEXT.set(events)
    try:
        yield events
    finally:
        _ORDER_TELEMETRY_CONTEXT.reset(token)


def _order_policy_from_env() -> OrderPolicy | None:
    raw = os.environ.get(_ORDER_POLICY_ENV)
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value in {"off", "false", "0"}:
        return OrderPolicy.SORT
    if value in {"on", "true", "1"}:
        return OrderPolicy.ENFORCE
    return _normalize_policy(value)


def _caller_sorted_mode_enabled() -> bool:
    return os.environ.get(_CALLER_SORTED_ENV) == "1"


def _order_telemetry_enabled() -> bool:
    return os.environ.get(_ORDER_TELEMETRY_ENV) == "1"


def _normalize_policy(policy: OrderPolicy | str) -> OrderPolicy:
    if isinstance(policy, OrderPolicy):
        return policy
    normalized = policy.strip().lower()
    for candidate in OrderPolicy:
        if candidate.value == normalized:
            return candidate
    never(
        "unknown order policy",
        policy=policy,
        allowed=[candidate.value for candidate in OrderPolicy],
    )
    return OrderPolicy.SORT  # pragma: no cover - never() raises


def _first_order_violation(
    values: Iterable[T],
    *,
    key: Callable[[T], Any] | None = None,
    reverse: bool = False,
) -> tuple[int, int, Any, Any, str] | None:
    previous_marker: Any | None = None
    previous_index = -1
    has_previous = False
    for index, value in enumerate(values):
        marker = key(value) if key is not None else value
        if has_previous:
            try:
                out_of_order = (
                    bool(previous_marker < marker)
                    if reverse
                    else bool(previous_marker > marker)
                )
            except TypeError:
                return (previous_index, index, previous_marker, marker, "incomparable")
            if out_of_order:
                return (previous_index, index, previous_marker, marker, "out_of_order")
        previous_marker = marker
        previous_index = index
        has_previous = True
    return None


def _raise_order_violation(payload: dict[str, object], *, reason: str) -> None:
    message = (
        "caller-ordered invariant requires comparable keys"
        if reason == "incomparable"
        else "caller-ordered invariant violated"
    )
    never(message, **payload)


def _record_order_telemetry(payload: dict[str, object]) -> None:
    event = dict(payload)
    event["action"] = "fallback_sort"
    sink = _ORDER_TELEMETRY_CONTEXT.get()
    if sink is not None:
        sink.append(event)
    if _order_telemetry_enabled():
        _ORDER_TELEMETRY_GLOBAL.append(event)
