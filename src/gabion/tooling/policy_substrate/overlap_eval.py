# gabion:decision_protocol_module
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import reduce
from collections import defaultdict
from itertools import chain
from typing import Iterable, Iterator

from gabion.tooling.policy_substrate.policy_event_kind import (
    PolicyEventKind,
    policy_event_kind_segments,
)
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
                fiber_events=events_by_fiber.get(interval.fiber_id, _empty_lens_events()),
            ),
            intervals,
        )
    )
    ordered_overlaps = sorted(
        overlaps,
        key=lambda overlap: (
            overlap.fiber_id,
            overlap.taint_interval_id,
            overlap.start_ordinal,
        ),
    )
    for overlap in ordered_overlaps:
        yield overlap


def _group_condition_events_by_fiber(
    condition_events: Iterable[LensEvent],
) -> defaultdict[str, list[LensEvent]]:
    return reduce(
        _append_condition_event,
        condition_events,
        defaultdict(list),
    )


def _append_condition_event(
    grouped: defaultdict[str, list[LensEvent]],
    event: LensEvent,
) -> defaultdict[str, list[LensEvent]]:
    grouped[event.fiber_id].append(event)
    return grouped


def _overlaps_for_interval(
    *,
    interval: TaintInterval,
    fiber_events: list[LensEvent],
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
                condition_event.ordinal,
            ),
            fiber_id=interval.fiber_id,
            taint_interval_id=interval.interval_id,
            condition_event=condition_event,
            start_ordinal=condition_event.ordinal,
            end_ordinal=condition_event.ordinal,
        )


def _event_within_interval(*, condition_event: LensEvent, interval: TaintInterval) -> bool:
    ordinal = condition_event.ordinal
    if interval.is_closed:
        return interval.start_ordinal <= ordinal <= interval.end_ordinal
    return interval.start_ordinal <= ordinal


def _stable_hash(*parts: object) -> str:
    return reduce(_digest_update, parts, hashlib.sha256()).hexdigest()


def _digest_update(digest: object, part: object):
    digest.update(_hash_part_bytes(part))
    digest.update(b"\x00")
    return digest


def _empty_lens_events() -> list[LensEvent]:
    return []


def _hash_part_bytes(value: object) -> bytes:
    match value:
        case PolicyEventKind() as event_kind:
            return b"\x1f".join(
                map(lambda segment: segment.encode("utf-8"), policy_event_kind_segments(kind=event_kind))
            )
        case bool() as flag:
            return b"1" if flag else b"0"
        case int() as integer:
            return _int_bytes(integer)
        case str() as text:
            return text.encode("utf-8")
        case bytes() as raw:
            return raw
        case _:
            return b"<unsupported>"


def _int_bytes(value: int) -> bytes:
    magnitude = abs(value)
    width = max(1, (magnitude.bit_length() + 7) // 8)
    sign = b"-" if value < 0 else b"+"
    return sign + magnitude.to_bytes(width, byteorder="big", signed=False)


__all__ = [
    "ConditionOverlap",
    "evaluate_condition_overlaps",
]
