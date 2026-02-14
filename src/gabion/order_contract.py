from __future__ import annotations

import os
from typing import Any, Callable, Iterable, TypeVar

from gabion.invariants import never


T = TypeVar("T")

_CALLER_SORTED_ENV = "GABION_CALLER_SORTED"


def ordered_or_sorted(
    values: Iterable[T],
    *,
    source: str,
    key: Callable[[T], Any] | None = None,
    require_sorted: bool | None = None,
) -> list[T]:
    """Return deterministic order with optional caller-order enforcement.

    - `require_sorted=False`: apply sorting.
    - `require_sorted=True`: trust caller order and fail via `never()` on regression.
    - `require_sorted=None`: defer to `GABION_CALLER_SORTED` (`1` => enforce).
    """

    items = list(values)
    enforce_caller_order = (
        _caller_sorted_mode_enabled() if require_sorted is None else require_sorted
    )
    if not enforce_caller_order:
        return sorted(items, key=key)
    _assert_monotonic_order(items, source=source, key=key)
    return items


def _caller_sorted_mode_enabled() -> bool:
    return os.environ.get(_CALLER_SORTED_ENV) == "1"


def _assert_monotonic_order(
    values: Iterable[T],
    *,
    source: str,
    key: Callable[[T], Any] | None = None,
) -> None:
    previous_marker: Any | None = None
    previous_index = -1
    has_previous = False
    for index, value in enumerate(values):
        marker = key(value) if key is not None else value
        if has_previous:
            try:
                out_of_order = bool(previous_marker > marker)
            except TypeError:
                never(
                    "caller-ordered invariant requires comparable keys",
                    source=source,
                    previous_index=previous_index,
                    current_index=index,
                    previous_key=repr(previous_marker),
                    current_key=repr(marker),
                )
            if out_of_order:
                never(
                    "caller-ordered invariant violated",
                    source=source,
                    previous_index=previous_index,
                    current_index=index,
                    previous_key=repr(previous_marker),
                    current_key=repr(marker),
                )
        previous_marker = marker
        previous_index = index
        has_previous = True
