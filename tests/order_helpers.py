from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from gabion.order_contract import ordered_or_sorted

T = TypeVar("T")


def contract_sorted(
    values: Iterable[T],
    *,
    key: Callable[[T], Any] | None = None,
    reverse: bool = False,
) -> list[T]:
    return list(
        ordered_or_sorted(
            values,
            key=key,
            reverse=reverse,
            source="tests.order_helpers.contract_sorted",
        )
    )
