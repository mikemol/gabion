from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from gabion.analysis.aspf import Forest
from gabion.analysis.timeout_context import (
    Deadline,
    deadline_clock_scope,
    deadline_scope,
    forest_scope,
)
from gabion.deadline_clock import GasMeter
from gabion.invariants import never
from gabion.lsp_client import _env_timeout_ticks, _has_env_timeout

_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_TIMEOUT_BUDGET: "DeadlineBudget"


@dataclass(frozen=True)
class DeadlineBudget:
    ticks: int
    tick_ns: int

    def __post_init__(self) -> None:
        ticks_value = int(self.ticks)
        tick_ns_value = int(self.tick_ns)
        if ticks_value <= 0:
            never("invalid deadline budget ticks", ticks=self.ticks)
        if tick_ns_value <= 0:
            never("invalid deadline budget tick_ns", tick_ns=self.tick_ns)
        object.__setattr__(self, "ticks", ticks_value)
        object.__setattr__(self, "tick_ns", tick_ns_value)


_DEFAULT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_TIMEOUT_TICK_NS,
)


def timeout_budget_from_lsp_env(
    *,
    default_budget: DeadlineBudget = _DEFAULT_TIMEOUT_BUDGET,
) -> DeadlineBudget:
    if _has_env_timeout():
        ticks, tick_ns = _env_timeout_ticks()
        return DeadlineBudget(ticks=ticks, tick_ns=tick_ns)
    return DeadlineBudget(
        ticks=default_budget.ticks,
        tick_ns=default_budget.tick_ns,
    )


@contextmanager
def deadline_scope_from_ticks(
    budget: DeadlineBudget,
    *,
    gas_limit: int | None = None,
) -> Iterator[None]:
    limit = budget.ticks if gas_limit is None else int(gas_limit)
    if limit <= 0:
        never("invalid deadline gas limit", gas_limit=gas_limit)
    with forest_scope(Forest()):
        with deadline_scope(Deadline.from_timeout_ticks(budget.ticks, budget.tick_ns)):
            with deadline_clock_scope(GasMeter(limit=limit)):  # pragma: no branch
                yield


@contextmanager
def deadline_scope_from_lsp_env(
    *,
    default_budget: DeadlineBudget = _DEFAULT_TIMEOUT_BUDGET,
    gas_limit: int | None = None,
) -> Iterator[None]:
    budget = timeout_budget_from_lsp_env(
        default_budget=default_budget,
    )
    with deadline_scope_from_ticks(
        budget=budget,
        gas_limit=budget.ticks if gas_limit is None else gas_limit,
    ):
        yield
