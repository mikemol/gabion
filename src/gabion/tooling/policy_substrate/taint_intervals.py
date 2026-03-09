# gabion:decision_protocol_module
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import Iterable, Iterator

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
    stack: tuple[LensEvent, ...]
    intervals: tuple[TaintInterval, ...]


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
            int(interval.start_ordinal),
            int(interval.end_ordinal),
        ),
    )
    for interval in ordered_intervals:
        yield interval


def _group_events_by_fiber(
    events: Iterable[LensEvent],
) -> dict[tuple[str, str], tuple[LensEvent, ...]]:
    return reduce(_append_event_to_group, events, {})


def _append_event_to_group(
    grouped: dict[tuple[str, str], tuple[LensEvent, ...]],
    event: LensEvent,
) -> dict[tuple[str, str], tuple[LensEvent, ...]]:
    key = (event.fiber_id, event.taint_class)
    existing = grouped.get(key, ())
    grouped[key] = (*existing, event)
    return grouped


def _intervals_for_group_item(
    group_item: tuple[tuple[str, str], tuple[LensEvent, ...]],
) -> Iterator[TaintInterval]:
    (fiber_id, taint_class), grouped_events = group_item
    sorted_events = tuple(sorted(grouped_events, key=lambda item: item.ordinal))
    max_ordinal = max(map(lambda event: event.ordinal, sorted_events), default=0)
    state = reduce(
        lambda current, event: _consume_group_event(
            state=current,
            event=event,
            fiber_id=fiber_id,
            taint_class=taint_class,
        ),
        sorted_events,
        _GroupIntervalState(stack=(), intervals=()),
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
            return _GroupIntervalState(stack=(*state.stack, event), intervals=state.intervals)
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
                stack=tuple(rest),
                intervals=(*state.intervals, interval),
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
        str(start_ordinal),
        str(end_ordinal),
        str(is_closed),
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


def _stable_hash(*parts: str) -> str:
    return reduce(_digest_update, parts, hashlib.sha256()).hexdigest()


def _digest_update(digest: object, part: str):
    digest.update(part.encode("utf-8"))
    digest.update(b"\x00")
    return digest


__all__ = [
    "TaintInterval",
    "build_taint_intervals",
]
