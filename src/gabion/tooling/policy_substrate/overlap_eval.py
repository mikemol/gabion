# gabion:decision_protocol_module
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import Iterable, Iterator

from gabion.tooling.policy_substrate.projection_lens import LensEvent
from gabion.tooling.policy_substrate.taint_intervals import TaintInterval


@dataclass(frozen=True)
class ConditionOverlap:
    condition_overlap_id: str
    fiber_id: str
    taint_interval_id: str
    condition_event: LensEvent
    start_ordinal: int
    end_ordinal: int


def evaluate_condition_overlaps(
    *,
    intervals: Iterable[TaintInterval],
    condition_events: Iterable[LensEvent],
) -> Iterator[ConditionOverlap]:
    events_by_fiber = _group_condition_events_by_fiber(condition_events)
    overlaps = chain.from_iterable(
        map(
            lambda interval: _overlaps_for_interval(
                interval=interval,
                fiber_events=events_by_fiber.get(interval.fiber_id, ()),
            ),
            intervals,
        )
    )
    ordered_overlaps = sorted(
        overlaps,
        key=lambda overlap: (
            overlap.fiber_id,
            overlap.taint_interval_id,
            int(overlap.start_ordinal),
        ),
    )
    for overlap in ordered_overlaps:
        yield overlap


def _group_condition_events_by_fiber(
    condition_events: Iterable[LensEvent],
) -> dict[str, tuple[LensEvent, ...]]:
    return reduce(_append_condition_event, condition_events, {})


def _append_condition_event(
    grouped: dict[str, tuple[LensEvent, ...]],
    event: LensEvent,
) -> dict[str, tuple[LensEvent, ...]]:
    existing = grouped.get(event.fiber_id, ())
    grouped[event.fiber_id] = (*existing, event)
    return grouped


def _overlaps_for_interval(
    *,
    interval: TaintInterval,
    fiber_events: tuple[LensEvent, ...],
) -> Iterator[ConditionOverlap]:
    for condition_event in filter(
        lambda event: _event_within_interval(
            condition_event=event,
            interval=interval,
        ),
        fiber_events,
    ):
        yield ConditionOverlap(
            condition_overlap_id=_stable_hash(
                interval.interval_id,
                condition_event.event_kind,
                str(condition_event.ordinal),
            ),
            fiber_id=interval.fiber_id,
            taint_interval_id=interval.interval_id,
            condition_event=condition_event,
            start_ordinal=condition_event.ordinal,
            end_ordinal=condition_event.ordinal,
        )


def _event_within_interval(*, condition_event: LensEvent, interval: TaintInterval) -> bool:
    ordinal = int(condition_event.ordinal)
    if interval.is_closed:
        return int(interval.start_ordinal) <= ordinal <= int(interval.end_ordinal)
    return int(interval.start_ordinal) <= ordinal


def _stable_hash(*parts: str) -> str:
    return reduce(_digest_update, parts, hashlib.sha256()).hexdigest()


def _digest_update(digest: object, part: str):
    digest.update(part.encode("utf-8"))
    digest.update(b"\x00")
    return digest


__all__ = [
    "ConditionOverlap",
    "evaluate_condition_overlaps",
]
