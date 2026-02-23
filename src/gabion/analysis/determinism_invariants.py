# gabion:decision_protocol_module
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from gabion.analysis.timeout_context import check_deadline
from gabion.invariants import never, proof_mode


T = TypeVar("T")


def require_sorted(
    name: str,
    xs: Iterable[T],
    *,
    key: Callable[[T], object] | None = None,
    reverse: bool = False,
    on_violation: Callable[[dict[str, object]], None] | None = None,
    **env: object,
) -> None:
    check_deadline()
    if not proof_mode():
        return
    iterator = iter(xs)
    try:
        previous = next(iterator)
    except StopIteration:
        return
    previous_key = key(previous) if key is not None else previous
    for current in iterator:
        check_deadline()
        current_key = key(current) if key is not None else current
        is_ordered = current_key <= previous_key if reverse else previous_key <= current_key
        if is_ordered:
            previous_key = current_key
            continue
        payload: dict[str, object] = {
            "constraint": "sorted",
            "name": name,
            "previous_key": repr(previous_key),
            "current_key": repr(current_key),
            "reverse": reverse,
        }
        payload.update({str(k): v for k, v in env.items()})
        if on_violation is not None:
            on_violation(payload)
        never("determinism invariant: sorted order violated", **payload)


def require_no_dupes(
    name: str,
    xs: Iterable[T],
    *,
    key: Callable[[T], object] | None = None,
    on_violation: Callable[[dict[str, object]], None] | None = None,
    **env: object,
) -> None:
    check_deadline()
    if not proof_mode():
        return
    seen: set[object] = set()
    for item in xs:
        check_deadline()
        entry_key = key(item) if key is not None else item
        if entry_key not in seen:
            seen.add(entry_key)
            continue
        payload: dict[str, object] = {
            "constraint": "no_dupes",
            "name": name,
            "duplicate_key": repr(entry_key),
        }
        payload.update({str(k): v for k, v in env.items()})
        if on_violation is not None:
            on_violation(payload)
        never("determinism invariant: duplicate entry detected", **payload)


def require_canonical_multiset(
    name: str,
    pairs: Iterable[tuple[str, int]],
    *,
    on_violation: Callable[[dict[str, object]], None] | None = None,
    **env: object,
) -> None:
    check_deadline()
    if not proof_mode():
        return
    seen_keys: set[str] = set()
    previous_key: str | None = None
    for key, count in pairs:
        check_deadline()
        if count <= 0:
            payload: dict[str, object] = {
                "constraint": "canonical_multiset",
                "name": name,
                "invalid_count": int(count),
                "key": key,
            }
            payload.update({str(k): v for k, v in env.items()})
            if on_violation is not None:
                on_violation(payload)
            never("determinism invariant: multiset count must be positive", **payload)
        if key in seen_keys:
            payload = {
                "constraint": "canonical_multiset",
                "name": name,
                "duplicate_key": key,
            }
            payload.update({str(k): v for k, v in env.items()})
            if on_violation is not None:
                on_violation(payload)
            never("determinism invariant: multiset key duplicated", **payload)
        seen_keys.add(key)
        if previous_key is not None and key < previous_key:
            payload = {
                "constraint": "canonical_multiset",
                "name": name,
                "previous_key": previous_key,
                "current_key": key,
            }
            payload.update({str(k): v for k, v in env.items()})
            if on_violation is not None:
                on_violation(payload)
            never("determinism invariant: multiset keys out of order", **payload)
        previous_key = key


def require_no_python_hash(
    name: str,
    *,
    on_violation: Callable[[dict[str, object]], None] | None = None,
    **env: object,
) -> None:
    if not proof_mode():
        return
    payload: dict[str, object] = {
        "constraint": "no_python_hash",
        "name": name,
    }
    payload.update({str(k): v for k, v in env.items()})
    if on_violation is not None:
        on_violation(payload)
    never(
        "determinism invariant: Python hash/randomized ordering is forbidden in this path",
        **payload,
    )
