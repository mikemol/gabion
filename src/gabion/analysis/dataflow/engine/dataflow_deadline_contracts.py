from __future__ import annotations

"""Canonical deadline/callee carrier contracts extracted from monolith."""

from dataclasses import dataclass, field
from pathlib import Path

from gabion.analysis.dataflow.engine.dataflow_contracts import FunctionInfo, OptionalSpan4


@dataclass
class _DeadlineLoopFacts:
    span: OptionalSpan4
    kind: str
    depth: int = 1
    check_params: set[str] = field(default_factory=set)
    ambient_check: bool = False
    call_spans: set[tuple[int, int, int, int]] = field(default_factory=set)


@dataclass(frozen=True)
class _DeadlineLocalInfo:
    origin_vars: set[str]
    origin_spans: dict[str, tuple[int, int, int, int]]
    alias_to_param: dict[str, str]


@dataclass(frozen=True)
class _DeadlineFunctionFacts:
    path: Path
    qual: str
    span: OptionalSpan4
    loop: bool
    check_params: set[str]
    ambient_check: bool
    loop_sites: list[_DeadlineLoopFacts]
    local_info: _DeadlineLocalInfo


@dataclass(frozen=True)
class _CalleeResolutionOutcome:
    status: str
    phase: str
    callee_key: str
    candidates: tuple[FunctionInfo, ...] = ()


__all__ = [
    "_CalleeResolutionOutcome",
    "_DeadlineFunctionFacts",
    "_DeadlineLocalInfo",
    "_DeadlineLoopFacts",
]
