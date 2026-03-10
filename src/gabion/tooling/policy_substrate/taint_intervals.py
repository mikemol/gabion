# gabion:decision_protocol_module
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import Iterable, Iterator

from gabion.tooling.policy_substrate.policy_event_kind import (
    PolicyEventKind,
    policy_event_kind_segments,
)
from gabion.tooling.policy_substrate.projection_lens import LensEvent


@dataclass(frozen=True)
class TaintInterval:
    interval_id: str
    fiber_id: str
    taint_class: str
    intro_event: LensEvent
    erase_event: LensEvent | None
    start_ordinal: int
    end_ordinal: int
    is_closed: bool


@dataclass(frozen=True)
class _GroupIntervalState:
    stack: list[LensEvent]
    intervals: list[TaintInterval]


@dataclass(frozen=True)
class _FiberGroupKey:
    fiber_id: str
    taint_class: str


def build_taint_intervals(*, events: Iterable[LensEvent]) -> Iterator[TaintInterval]:
    grouped = _group_events_by_fiber(events)
    interval_stream = chain.from_iterable(
        map(_intervals_for_group_item, grouped.items())
    )
    ordered_intervals = sorted(
        interval_stream,
        key=lambda interval: (
            interval.fiber_id,
            interval.taint_class,
            interval.start_ordinal,
            interval.end_ordinal,
        ),
    )
    for interval in ordered_intervals:
        yield interval


def _group_events_by_fiber(
    events: Iterable[LensEvent],
) -> defaultdict[_FiberGroupKey, list[LensEvent]]:
    return reduce(_append_event_to_group, events, defaultdict(list))


def _append_event_to_group(
    grouped: defaultdict[_FiberGroupKey, list[LensEvent]],
    event: LensEvent,
) -> defaultdict[_FiberGroupKey, list[LensEvent]]:
    key = _FiberGroupKey(
        fiber_id=event.fiber_id,
        taint_class=event.taint_class,
    )
    grouped[key].append(event)
    return grouped


def _intervals_for_group_item(
    group_item: tuple[_FiberGroupKey, list[LensEvent]],
) -> Iterator[TaintInterval]:
    group_key, grouped_events = group_item
    fiber_id = group_key.fiber_id
    taint_class = group_key.taint_class
    sorted_events = list(sorted(grouped_events, key=lambda item: item.ordinal))
    max_ordinal = max(map(lambda event: event.ordinal, sorted_events), default=0)
    state = reduce(
        lambda current, event: _consume_group_event(
            state=current,
            event=event,
            fiber_id=fiber_id,
            taint_class=taint_class,
        ),
        sorted_events,
        _GroupIntervalState(stack=[], intervals=[]),
    )
    yield from state.intervals
    yield from map(
        lambda intro_event: _new_interval(
            fiber_id=fiber_id,
            taint_class=taint_class,
            intro_event=intro_event,
            erase_event=None,
            start_ordinal=intro_event.ordinal,
            end_ordinal=max_ordinal + 1,
            is_closed=False,
        ),
        state.stack,
    )


def _consume_group_event(
    *,
    state: _GroupIntervalState,
    event: LensEvent,
    fiber_id: str,
    taint_class: str,
) -> _GroupIntervalState:
    match event.action:
        case "taint_intro":
            return _GroupIntervalState(
                stack=[*state.stack, event],
                intervals=state.intervals,
            )
        case "taint_erase":
            return _consume_erase_event(
                state=state,
                event=event,
                fiber_id=fiber_id,
                taint_class=taint_class,
            )
        case _:
            return state


def _consume_erase_event(
    *,
    state: _GroupIntervalState,
    event: LensEvent,
    fiber_id: str,
    taint_class: str,
) -> _GroupIntervalState:
    match state.stack:
        case (*rest, intro_event):
            interval = _new_interval(
                fiber_id=fiber_id,
                taint_class=taint_class,
                intro_event=intro_event,
                erase_event=event,
                start_ordinal=intro_event.ordinal,
                end_ordinal=event.ordinal,
                is_closed=True,
            )
            return _GroupIntervalState(
                stack=list(rest),
                intervals=[*state.intervals, interval],
            )
        case _:
            return state


def _new_interval(
    *,
    fiber_id: str,
    taint_class: str,
    intro_event: LensEvent,
    erase_event: LensEvent | None,
    start_ordinal: int,
    end_ordinal: int,
    is_closed: bool,
) -> TaintInterval:
    interval_id = _stable_hash(
        fiber_id,
        taint_class,
        start_ordinal,
        end_ordinal,
        is_closed,
        intro_event.event_kind,
        erase_event.event_kind if erase_event is not None else "<open>",
    )
    return TaintInterval(
        interval_id=interval_id,
        fiber_id=fiber_id,
        taint_class=taint_class,
        intro_event=intro_event,
        erase_event=erase_event,
        start_ordinal=start_ordinal,
        end_ordinal=end_ordinal,
        is_closed=is_closed,
    )


def _stable_hash(*parts: object) -> str:
    return reduce(_digest_update, parts, hashlib.sha256()).hexdigest()


def _digest_update(digest: object, part: object):
    digest.update(_hash_part_bytes(part))
    digest.update(b"\x00")
    return digest


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
    "TaintInterval",
    "build_taint_intervals",
]
