# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Callable, Iterable, Iterator


@dataclass(frozen=True)
class LensSite:
    site_id: str
    path: str
    qualname: str
    line: int
    column: int
    node_kind: str
    surface: str
    fiber_id: str
    taint_class: str
    input_slot: str


@dataclass(frozen=True)
class LensEvent:
    ordinal: int
    site_id: str
    path: str
    qualname: str
    line: int
    column: int
    node_kind: str
    surface: str
    fiber_id: str
    event_kind: str
    event_phase: str
    input_slot: str
    taint_class: str
    action: str


@dataclass(frozen=True)
class ProjectionLensSpec:
    name: str
    project: Callable[[LensSite], Iterable[LensEvent]]


def run_projection_lenses(
    *,
    site: LensSite,
    specs: tuple[ProjectionLensSpec, ...],
) -> Iterator[LensEvent]:
    ordered_events = sorted(
        chain.from_iterable(map(lambda spec: spec.project(site), specs)),
        key=lambda event: (
            int(event.ordinal),
            str(event.action),
            str(event.event_phase),
            str(event.event_kind),
        ),
    )
    for event in ordered_events:
        yield event


__all__ = [
    "LensEvent",
    "LensSite",
    "ProjectionLensSpec",
    "run_projection_lenses",
]
