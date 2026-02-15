from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from gabion.analysis.aspf import Forest
from gabion.analysis.timeout_context import (
    Deadline,
    deadline_clock_scope,
    deadline_scope,
    forest_scope,
)
from gabion.deadline_clock import GasMeter
from gabion.lsp_client import _env_timeout_ticks, _has_env_timeout

_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000


def timeout_ticks_from_lsp_env(
    *,
    default_ticks: int = _DEFAULT_TIMEOUT_TICKS,
    default_tick_ns: int = _DEFAULT_TIMEOUT_TICK_NS,
) -> tuple[int, int]:
    if _has_env_timeout():
        return _env_timeout_ticks()
    return int(default_ticks), int(default_tick_ns)


@contextmanager
def deadline_scope_from_ticks(
    ticks: int,
    tick_ns: int,
    *,
    gas_limit: int | None = None,
) -> Iterator[None]:
    ticks_value = int(ticks)
    tick_ns_value = int(tick_ns)
    limit = ticks_value if gas_limit is None else int(gas_limit)
    with forest_scope(Forest()):
        with deadline_scope(Deadline.from_timeout_ticks(ticks_value, tick_ns_value)):
            with deadline_clock_scope(GasMeter(limit=limit)):  # pragma: no branch
                yield


@contextmanager
def deadline_scope_from_lsp_env(
    *,
    default_ticks: int = _DEFAULT_TIMEOUT_TICKS,
    default_tick_ns: int = _DEFAULT_TIMEOUT_TICK_NS,
    gas_limit: int | None = None,
) -> Iterator[None]:
    ticks, tick_ns = timeout_ticks_from_lsp_env(
        default_ticks=default_ticks,
        default_tick_ns=default_tick_ns,
    )
    with deadline_scope_from_ticks(
        ticks=ticks,
        tick_ns=tick_ns,
        gas_limit=ticks if gas_limit is None else gas_limit,
    ):
        yield
