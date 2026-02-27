# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import inspect
import os
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import TypeVar

from contextvars import ContextVar, Token

from gabion.analysis.aspf import Forest
from gabion.deadline_clock import (
    DeadlineClock,
    DeadlineClockExhausted,
    GasMeter,
    MonotonicClock,
)
from gabion.exceptions import NeverThrown
from gabion.invariants import never
from gabion.json_types import JSONValue
from gabion.order_contract import OrderPolicy, sort_once


_TIMEOUT_PROGRESS_CHECKS_FLOOR = 32
_TIMEOUT_PROGRESS_SITE_FLOOR = 4
_NO_DEADLINE_PROFILE_SNAPSHOT = None
_LoopItem = TypeVar("_LoopItem")


@dataclass(frozen=True)
class PackedCallStack:
    site_table: tuple["_CallSite", ...]
    stack: tuple[int, ...]

    def as_payload(self) -> dict[str, JSONValue]:
        return {
            "site_table": [entry.as_payload() for entry in self.site_table],
            "stack": [value for value in self.stack],
        }


@dataclass(frozen=True)
class TimeoutContext:
    call_stack: PackedCallStack
    forest_spec_id: object = None
    forest_signature: object = None
    deadline_profile: object = None
    progress: object = None

    def as_payload(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {"call_stack": self.call_stack.as_payload()}
        if self.forest_spec_id:
            payload["forest_spec_id"] = self.forest_spec_id
        if self.forest_signature:
            payload["forest_signature"] = self.forest_signature
        if self.deadline_profile:
            payload["deadline_profile"] = self.deadline_profile
        if self.progress:
            payload["progress"] = self.progress
        return payload


@dataclass(frozen=True)
class _FileSite:
    path: str

    def as_payload(self) -> dict[str, JSONValue]:
        # Keep canonical key order ("key" then "kind") so caller-order checks
        # can hold without fallback sorting.
        return {"key": [self.path], "kind": "FileSite"}


@dataclass(frozen=True)
class _CallSite:
    kind: str
    key: tuple[object, ...]

    def as_payload(self) -> dict[str, JSONValue]:
        return {
            "kind": self.kind,
            "key": [_site_part_to_payload(part) for part in self.key],
        }

    def frozen_key(self) -> tuple[object, ...]:
        return _freeze_key(self.key)


@dataclass(frozen=True)
class _InternedCallSite:
    order: int
    site: _CallSite


class TimeoutExceeded(TimeoutError):
    def __init__(self, context: TimeoutContext) -> None:
        super().__init__("Analysis timed out.")
        self.context = context


_SYSTEM_CLOCK = MonotonicClock()


@dataclass(frozen=True)
class Deadline:
    deadline_ns: int

    @classmethod
    # dataflow-bundle: tick_ns, ticks
    def from_timeout_ticks(cls, ticks: int, tick_ns: int) -> "Deadline":
        ticks_value = int(ticks)
        tick_ns_value = int(tick_ns)
        if ticks_value < 0:
            never("invalid timeout ticks", ticks=ticks)
        if tick_ns_value <= 0:
            never("invalid timeout tick_ns", tick_ns=tick_ns)
        total_ns = ticks_value * tick_ns_value
        return cls(deadline_ns=_SYSTEM_CLOCK.get_mark() + total_ns)

    @classmethod
    def from_timeout_ms(cls, milliseconds: int) -> "Deadline":
        return cls.from_timeout_ticks(milliseconds, 1_000_000)

    @classmethod
    def from_timeout(cls, seconds: float) -> "Deadline":
        # Deprecated: prefer from_timeout_ticks/from_timeout_ms (integer-only).
        try:
            value = Decimal(str(seconds))
        except (InvalidOperation, ValueError):
            never("invalid timeout seconds", seconds=seconds)
        if value < 0:
            never("invalid timeout seconds", seconds=seconds)
        millis = int(value * Decimal(1000))
        return cls.from_timeout_ms(millis)

    def expired(self) -> bool:
        return _SYSTEM_CLOCK.get_mark() >= self.deadline_ns

    def check(self, builder: Callable[[], TimeoutContext]) -> None:
        if self.expired():
            raise TimeoutExceeded(builder())


_deadline_var: ContextVar[object] = ContextVar("gabion_deadline", default=None)
_deadline_clock_var: ContextVar[object] = ContextVar(
    "gabion_deadline_clock", default=None
)
_MISSING_FOREST = object()
_forest_var: ContextVar[object] = ContextVar(
    "gabion_forest", default=_MISSING_FOREST
)


@dataclass
class _DeadlineSiteStats:
    checks: int = 0
    elapsed_ns: int = 0
    max_gap_ns: int = 0


@dataclass
class _DeadlineEdgeStats:
    transitions: int = 0
    elapsed_ns: int = 0
    max_gap_ns: int = 0


@dataclass
class _DeadlineIoStats:
    events: int = 0
    elapsed_ns: int = 0
    max_event_ns: int = 0
    bytes_total: int = 0


@dataclass
class _DeadlineProfileState:
    enabled: bool
    started_ns: int
    last_ns: int
    started_wall_ns: int
    last_wall_ns: int
    project_root: object = None
    project_root_key: object = None
    sample_interval: int = 1
    checks_total: int = 0
    sampled_checks_total: int = 0
    sample_pending_checks: int = 0
    sample_pending_elapsed_ns: int = 0
    unattributed_elapsed_ns: int = 0
    last_site_id: object = None
    root_resolution_cache: dict[str, tuple[object, object]] = field(
        default_factory=dict
    )
    site_keys: list[tuple[str, str]] = field(default_factory=list)
    site_ids: dict[tuple[str, str], int] = field(default_factory=dict)
    frame_site_cache: dict[tuple[object, object], int] = field(default_factory=dict)
    site_stats: dict[int, _DeadlineSiteStats] = field(default_factory=dict)
    edge_stats: dict[
        tuple[int, int],
        _DeadlineEdgeStats,
    ] = field(default_factory=dict)
    io_stats: dict[str, _DeadlineIoStats] = field(default_factory=dict)


_deadline_profile_var: ContextVar[object] = ContextVar(
    "gabion_deadline_profile", default=None
)


def _current_deadline_mark() -> int:
    return get_deadline_clock().get_mark()


def set_deadline_profile(
    *,
    project_root = None,
    enabled: bool = True,
    sample_interval: int = 1,
):
    normalized_sample_interval = max(1, int(sample_interval))
    resolved_root = project_root.resolve() if project_root is not None else None
    root_key = str(resolved_root) if resolved_root is not None else None
    now = _current_deadline_mark()
    wall_now = _SYSTEM_CLOCK.get_mark()
    state = _DeadlineProfileState(
        enabled=enabled,
        started_ns=now,
        last_ns=now,
        started_wall_ns=wall_now,
        last_wall_ns=wall_now,
        project_root=resolved_root,
        project_root_key=root_key,
        sample_interval=normalized_sample_interval,
        root_resolution_cache={str(project_root): (resolved_root, root_key)}
        if project_root is not None
        else {},
    )
    return _deadline_profile_var.set(state)


def reset_deadline_profile(token) -> None:
    _deadline_profile_var.reset(token)


def set_deadline(deadline: Deadline):
    if deadline is None:
        never("deadline carrier missing")
    return _deadline_var.set(deadline)


def reset_deadline(token) -> None:
    _deadline_var.reset(token)


def get_deadline() -> Deadline:
    deadline = _deadline_var.get()
    if deadline is None:
        never("deadline carrier missing")
    return deadline


def set_deadline_clock(clock: DeadlineClock):
    if clock is None:
        never("deadline clock missing")
    return _deadline_clock_var.set(clock)


def reset_deadline_clock(token) -> None:
    _deadline_clock_var.reset(token)


def get_deadline_clock() -> DeadlineClock:
    clock = _deadline_clock_var.get()
    if clock is None:
        never("deadline clock missing")
    return clock


def set_forest(forest: Forest):
    return _forest_var.set(forest)


def reset_forest(token) -> None:
    _forest_var.reset(token)


def get_forest() -> Forest:
    forest = _forest_var.get()
    if forest is _MISSING_FOREST:
        never("forest carrier missing")
    match forest:
        case Forest() as valid_forest:
            return valid_forest
        case _:
            never("invalid forest carrier", carrier_type=type(forest).__name__)
            return forest  # pragma: no cover - never() raises


@contextmanager
def deadline_scope(deadline: Deadline):
    if deadline is None:
        never("deadline carrier missing")
    token = set_deadline(deadline)
    try:
        yield
    finally:
        reset_deadline(token)


@contextmanager
def deadline_clock_scope(clock: DeadlineClock):
    token = set_deadline_clock(clock)
    try:
        yield
    finally:
        reset_deadline_clock(token)


@contextmanager
def forest_scope(forest: Forest):
    token = set_forest(forest)
    try:
        yield
    finally:
        reset_forest(token)


@contextmanager
def deadline_profile_scope(
    *,
    project_root = None,
    enabled: bool = True,
    sample_interval: int = 1,
):
    token = set_deadline_profile(
        project_root=project_root,
        enabled=enabled,
        sample_interval=sample_interval,
    )
    try:
        yield
    finally:
        reset_deadline_profile(token)


def _profile_site_key(
    frame: FrameType,
    *,
    project_root,
) -> tuple[str, str]:
    if project_root is None:
        return _frame_site_key(frame, project_root=None)
    try:
        return _frame_site_key(frame, project_root=project_root)
    except NeverThrown:
        _, qual = _frame_site_key(frame, project_root=None)
        return ("<external>", qual)


def _record_deadline_check(
    project_root,
    *,
    frame_getter = inspect.currentframe,
    profile_site_key_fn: Callable[..., tuple[str, str]] = _profile_site_key,
) -> None:
    state = _deadline_profile_var.get()
    if state is not None and state.enabled:
        frame = frame_getter()
        has_caller = (
            frame is not None
            and frame.f_back is not None
            and frame.f_back.f_back is not None
        )
        if has_caller:
            caller_frame = frame.f_back.f_back
            effective_root = state.project_root
            effective_root_key = state.project_root_key
            if project_root is not None:
                cache_key = str(project_root)
                cached_root = state.root_resolution_cache.get(cache_key)
                if cached_root is None:
                    resolved_root = project_root.resolve()
                    resolved_root_key = str(resolved_root)
                    cached_root = (resolved_root, resolved_root_key)
                    state.root_resolution_cache[cache_key] = cached_root
                effective_root, effective_root_key = cached_root
            now = _current_deadline_mark()
            delta = max(0, now - state.last_ns)
            state.last_ns = now
            state.last_wall_ns = _SYSTEM_CLOCK.get_mark()
            state.checks_total += 1
            state.sample_pending_checks += 1
            state.sample_pending_elapsed_ns += delta
            sample_interval = max(1, state.sample_interval)
            if state.sample_pending_checks >= sample_interval:
                sampled_checks = state.sample_pending_checks
                sampled_elapsed_ns = state.sample_pending_elapsed_ns
                state.sample_pending_checks = 0
                state.sample_pending_elapsed_ns = 0
                state.sampled_checks_total += sampled_checks
                frame_cache_key = (caller_frame.f_code, effective_root_key)
                site_id = state.frame_site_cache.get(frame_cache_key)
                if site_id is None:
                    site_key = profile_site_key_fn(
                        caller_frame,
                        project_root=effective_root,
                    )
                    site_id = state.site_ids.get(site_key)
                    if site_id is None:
                        site_id = len(state.site_keys)
                        state.site_ids[site_key] = site_id
                        state.site_keys.append(site_key)
                    state.frame_site_cache[frame_cache_key] = site_id
                stats = state.site_stats.setdefault(site_id, _DeadlineSiteStats())
                stats.checks += sampled_checks
                stats.elapsed_ns += sampled_elapsed_ns
                if sampled_elapsed_ns > stats.max_gap_ns:
                    stats.max_gap_ns = sampled_elapsed_ns
                if state.last_site_id is None:
                    state.unattributed_elapsed_ns += sampled_elapsed_ns
                else:
                    edge_key = (state.last_site_id, site_id)
                    edge_stats = state.edge_stats.setdefault(edge_key, _DeadlineEdgeStats())
                    edge_stats.transitions += 1
                    edge_stats.elapsed_ns += sampled_elapsed_ns
                    if sampled_elapsed_ns > edge_stats.max_gap_ns:
                        edge_stats.max_gap_ns = sampled_elapsed_ns
                state.last_site_id = site_id


def _deadline_profile_snapshot():
    state = _deadline_profile_var.get()
    if state is None or not state.enabled:
        return _NO_DEADLINE_PROFILE_SNAPSHOT
    wall_total_elapsed_ns = max(0, state.last_wall_ns - state.started_wall_ns)
    clock = _deadline_clock_var.get()
    ticks_consumed = None
    ticks_per_ns = None
    match clock:
        case GasMeter() as gas_clock:
            ticks_consumed = int(gas_clock.get_mark())
            if wall_total_elapsed_ns > 0:
                ticks_per_ns = float(ticks_consumed) / float(wall_total_elapsed_ns)
        case _:
            pass
    total_elapsed_ns = max(0, state.last_ns - state.started_ns)
    site_rows: list[dict[str, JSONValue]] = []
    for site_id, stats in sort_once(
        state.site_stats.items(),
        source="_deadline_profile_snapshot.site_rows",
        key=lambda item: (
            -item[1].elapsed_ns,
            state.site_keys[item[0]][0],
            state.site_keys[item[0]][1],
        ),
    ):
        path, qual = state.site_keys[site_id]
        site_rows.append(
            {
                "path": path,
                "qual": qual,
                "check_count": stats.checks,
                "elapsed_between_checks_ns": stats.elapsed_ns,
                "max_gap_ns": stats.max_gap_ns,
            }
        )
    edge_rows: list[dict[str, JSONValue]] = []
    for (source_id, target_id), stats in sort_once(
        state.edge_stats.items(),
        source="_deadline_profile_snapshot.edge_rows",
        key=lambda item: (
            -item[1].elapsed_ns,
            state.site_keys[item[0][0]][0],
            state.site_keys[item[0][0]][1],
            state.site_keys[item[0][1]][0],
            state.site_keys[item[0][1]][1],
        ),
    ):
        source = state.site_keys[source_id]
        target = state.site_keys[target_id]
        edge_rows.append(
            {
                "from_path": source[0],
                "from_qual": source[1],
                "to_path": target[0],
                "to_qual": target[1],
                "transition_count": stats.transitions,
                "elapsed_ns": stats.elapsed_ns,
                "max_gap_ns": stats.max_gap_ns,
            }
        )
    io_rows: list[dict[str, JSONValue]] = []
    for io_name, stats in sort_once(
        state.io_stats.items(),
        source="_deadline_profile_snapshot.io_rows",
        key=lambda item: (-item[1].elapsed_ns, item[0]),
    ):
        io_rows.append(
            {
                "name": io_name,
                "event_count": stats.events,
                "elapsed_ns": stats.elapsed_ns,
                "max_event_ns": stats.max_event_ns,
                "bytes_total": stats.bytes_total,
            }
        )
    return {
        "checks_total": state.checks_total,
        "sample_interval": state.sample_interval,
        "sampled_checks_total": state.sampled_checks_total,
        "sample_pending_checks": state.sample_pending_checks,
        "sample_pending_elapsed_ns": state.sample_pending_elapsed_ns,
        "started_ns": state.started_ns,
        "last_check_ns": state.last_ns,
        "total_elapsed_ns": total_elapsed_ns,
        "wall_total_elapsed_ns": wall_total_elapsed_ns,
        "ticks_consumed": ticks_consumed,
        "ticks_per_ns": ticks_per_ns,
        "unattributed_elapsed_ns": state.unattributed_elapsed_ns,
        "sites": site_rows,
        "edges": edge_rows,
        "io": io_rows,
    }


def _timeout_progress_snapshot(
    *,
    forest: Forest,
    deadline_profile,
) -> dict[str, JSONValue]:
    checks_total = 0
    site_count = 0
    match deadline_profile:
        case Mapping() as profile_map:
            checks_total = int(profile_map.get("checks_total", 0) or 0)
            match profile_map.get("sites"):
                case list() as site_rows:
                    site_count = len(site_rows)
                case _:
                    pass
        case _:
            pass
    forest_nodes = len(forest.nodes)
    forest_alts = len(forest.alts)
    ticks_per_ns = None
    match deadline_profile:
        case Mapping() as profile_map:
            profile_ticks_per_ns = profile_map.get("ticks_per_ns")
            match profile_ticks_per_ns:
                case int() | float():
                    ticks_per_ns = float(profile_ticks_per_ns)
                case _:
                    pass
        case _:
            pass
    progressed = (
        (forest_nodes + forest_alts) > 0
        or checks_total >= _TIMEOUT_PROGRESS_CHECKS_FLOOR
        or site_count >= _TIMEOUT_PROGRESS_SITE_FLOOR
    )
    classification = (
        "timed_out_progress_resume"
        if progressed
        else "timed_out_no_progress"
    )
    clock = get_deadline_clock()
    tick_mark = int(clock.get_mark())
    tick_limit = None
    ticks_remaining = None
    match clock:
        case GasMeter() as gas_clock:
            tick_limit = int(gas_clock.limit)
            ticks_remaining = max(0, tick_limit - tick_mark)
        case _:
            pass
    return {
        "classification": classification,
        "retry_recommended": progressed,
        "resume_supported": False,
        "checks_total": checks_total,
        "site_count": site_count,
        "forest_nodes": forest_nodes,
        "forest_alts": forest_alts,
        "ticks_consumed": tick_mark,
        "tick_limit": tick_limit,
        "ticks_remaining": ticks_remaining,
        "ticks_per_ns": ticks_per_ns,
    }


def render_deadline_profile_markdown(
    profile: Mapping[str, JSONValue],
    *,
    max_rows: int = 25,
) -> str:
    lines: list[str] = ["# Deadline Profile Heat", ""]
    checks_total = int(profile.get("checks_total", 0) or 0)
    total_elapsed_ns = int(profile.get("total_elapsed_ns", 0) or 0)
    wall_total_elapsed_ns = int(profile.get("wall_total_elapsed_ns", 0) or 0)
    ticks_consumed = profile.get("ticks_consumed")
    ticks_per_ns = profile.get("ticks_per_ns")
    unattributed_ns = int(profile.get("unattributed_elapsed_ns", 0) or 0)
    lines.append(f"- checks_total: `{checks_total}`")
    lines.append(f"- total_elapsed_ns: `{total_elapsed_ns}`")
    lines.append(f"- wall_total_elapsed_ns: `{wall_total_elapsed_ns}`")
    if ticks_consumed is not None:
        lines.append(f"- ticks_consumed: `{int(ticks_consumed)}`")
    match ticks_per_ns:
        case int() | float():
            lines.append(f"- ticks_per_ns: `{ticks_per_ns:.9f}`")
        case _:
            pass
    lines.append(f"- unattributed_elapsed_ns: `{unattributed_ns}`")
    lines.append("")
    lines.append("## Site Heat")
    lines.append("")
    lines.append("| path | qual | checks | elapsed_ns | max_gap_ns |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    sites = profile.get("sites", [])
    match sites:
        case list() as site_rows:
            for row in site_rows[:max_rows]:
                match row:
                    case Mapping() as row_map:
                        lines.append(
                            "| {path} | {qual} | {checks} | {elapsed} | {gap} |".format(
                                path=str(row_map.get("path", "") or ""),
                                qual=str(row_map.get("qual", "") or ""),
                                checks=int(row_map.get("check_count", 0) or 0),
                                elapsed=int(
                                    row_map.get("elapsed_between_checks_ns", 0) or 0
                                ),
                                gap=int(row_map.get("max_gap_ns", 0) or 0),
                            )
                        )
                    case _:
                        pass
            if len(site_rows) > max_rows:
                lines.append("| ... | ... | ... | ... | ... |")
        case _:
            pass
    lines.append("")
    lines.append("## Transition Heat")
    lines.append("")
    lines.append("| from | to | transitions | elapsed_ns | max_gap_ns |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    edges = profile.get("edges", [])
    match edges:
        case list() as edge_rows:
            for row in edge_rows[:max_rows]:
                match row:
                    case Mapping() as row_map:
                        lines.append(
                            "| {source} | {target} | {count} | {elapsed} | {gap} |".format(
                                source=f"{str(row_map.get('from_path', '') or '')}:{str(row_map.get('from_qual', '') or '')}",
                                target=f"{str(row_map.get('to_path', '') or '')}:{str(row_map.get('to_qual', '') or '')}",
                                count=int(row_map.get("transition_count", 0) or 0),
                                elapsed=int(row_map.get("elapsed_ns", 0) or 0),
                                gap=int(row_map.get("max_gap_ns", 0) or 0),
                            )
                        )
                    case _:
                        pass
            if len(edge_rows) > max_rows:
                lines.append("| ... | ... | ... | ... | ... |")
        case _:
            pass
    lines.append("")
    lines.append("## I/O Heat")
    lines.append("")
    lines.append("| io | events | elapsed_ns | max_event_ns | bytes_total |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    io_rows = profile.get("io", [])
    match io_rows:
        case list() as deadline_io_rows:
            for row in deadline_io_rows[:max_rows]:
                match row:
                    case Mapping() as row_map:
                        lines.append(
                            "| {name} | {events} | {elapsed} | {max_event} | {bytes_total} |".format(
                                name=str(row_map.get("name", "") or ""),
                                events=int(row_map.get("event_count", 0) or 0),
                                elapsed=int(row_map.get("elapsed_ns", 0) or 0),
                                max_event=int(row_map.get("max_event_ns", 0) or 0),
                                bytes_total=int(row_map.get("bytes_total", 0) or 0),
                            )
                        )
                    case _:
                        pass
            if len(deadline_io_rows) > max_rows:
                lines.append("| ... | ... | ... | ... | ... |")
        case _:
            pass
    lines.append("")
    return "\n".join(lines)


def record_deadline_io(
    *,
    name: str,
    elapsed_ns: int,
    bytes_count = None,
) -> None:
    state = _deadline_profile_var.get()
    if state is not None and state.enabled:
        safe_elapsed_ns = max(0, int(elapsed_ns))
        safe_bytes_count = max(0, int(bytes_count)) if bytes_count is not None else 0
        stats = state.io_stats.setdefault(name, _DeadlineIoStats())
        stats.events += 1
        stats.elapsed_ns += safe_elapsed_ns
        stats.bytes_total += safe_bytes_count
        if safe_elapsed_ns > stats.max_event_ns:
            stats.max_event_ns = safe_elapsed_ns


def check_deadline(
    deadline = None,
    *,
    project_root = None,
    forest_spec_id = None,
    forest_signature = None,
    allow_frame_fallback: bool = True,
) -> None:
    if deadline is None:
        get_deadline()
    clock = _deadline_clock_var.get()
    if clock is None:
        never("deadline clock missing")
    consume_deadline_ticks(
        project_root=project_root,
        forest_spec_id=forest_spec_id,
        forest_signature=forest_signature,
        allow_frame_fallback=allow_frame_fallback,
    )
    _record_deadline_check(project_root)
    return


def deadline_loop_iter(values: Iterable[_LoopItem]) -> Iterator[_LoopItem]:
    for value in values:
        check_deadline()
        yield value


def consume_deadline_ticks(
    ticks: int = 1,
    *,
    project_root = None,
    forest_spec_id = None,
    forest_signature = None,
    allow_frame_fallback: bool = True,
) -> None:
    clock = _deadline_clock_var.get()
    if clock is None:
        never("deadline clock missing")
    try:
        clock.consume(ticks)
    except DeadlineClockExhausted as exc:
        forest = _forest_var.get()
        if forest is _MISSING_FOREST:
            raise
        timeout_context = build_timeout_context_from_stack(
            forest=forest,
            project_root=project_root,
            forest_spec_id=forest_spec_id,
            forest_signature=forest_signature,
            deadline_profile=_deadline_profile_snapshot(),
            allow_frame_fallback=allow_frame_fallback,
        )
        raise TimeoutExceeded(timeout_context) from exc


def _normalize_qualname(qualname: str) -> str:
    return qualname.replace(".<locals>.", ".")


def _frame_site_key(
    frame: FrameType,
    *,
    project_root,
) -> tuple[str, str]:
    module = frame.f_globals.get("__name__") or ""
    qualname = _normalize_qualname(frame.f_code.co_qualname or frame.f_code.co_name)
    if module:
        qual = f"{module}.{qualname}" if not qualname.startswith(f"{module}.") else qualname
    else:
        qual = qualname
    path = Path(frame.f_code.co_filename)
    if project_root is not None:
        resolved_path = str(path.resolve())
        root_text = str(project_root)
        root_prefix = root_text if root_text.endswith(os.sep) else f"{root_text}{os.sep}"
        if resolved_path != root_text and not resolved_path.startswith(root_prefix):
            never(
                "frame outside project_root",
                path=resolved_path,
                project_root=root_text,
            )
    return (path.name, qual)


def _site_key_payload(
    *,
    path: str,
    qual: str,
    span = None,
) -> dict[str, JSONValue]:
    return _function_site(path=path, qual=qual, span=span).as_payload()


def _site_key(
    *,
    path: str,
    qual: str,
    span = None,
) -> tuple[object, ...]:
    key: list[object] = [_FileSite(path), qual]
    if span and len(span) == 4:
        key.extend(span)
    return tuple(key)


def _function_site(
    *,
    path: str,
    qual: str,
    span = None,
) -> _CallSite:
    return _CallSite(kind="FunctionSite", key=_site_key(path=path, qual=qual, span=span))


def _site_part_from_payload(value: object) -> object:
    match value:
        case _FileSite():
            return value
        case Mapping() as mapping_value:
            kind = str(mapping_value.get("kind", "") or "")
            key_payload = mapping_value.get("key")
            match (kind, key_payload):
                case ("FileSite", [str() as path]):
                    return _FileSite(path)
                case _:
                    never("invalid site key mapping payload", payload_kind=kind)
                    return mapping_value  # pragma: no cover - never() raises
        case list() as payload_list:
            return tuple(_site_part_from_payload(part) for part in payload_list)
        case None | str() | int() | float() | bool():
            return value
        case _:
            never("invalid site key payload value", value_type=type(value).__name__)
            return value  # pragma: no cover - never() raises


def _site_part_to_payload(value: object) -> JSONValue:
    match value:
        case _FileSite() as file_site:
            return file_site.as_payload()
        case tuple() as key_tuple:
            return [_site_part_to_payload(part) for part in key_tuple]
        case None | str() | int() | float() | bool():
            return value
        case _:
            never("invalid site key value", value_type=type(value).__name__)
            return _NO_DEADLINE_PROFILE_SNAPSHOT  # pragma: no cover - never() raises


def build_site_index(
    forest: Forest,
) -> dict[tuple[str, str], _CallSite]:
    index: dict[tuple[str, str], _CallSite] = {}
    ordered_nodes = sort_once(
        forest.nodes.items(),
        source="build_site_index.ordered_nodes",
        key=lambda item: item[0].sort_key(),
    )
    for node_id, node in ordered_nodes:
        if node_id.kind != "FunctionSite":
            continue
        path = str(node.meta.get("path", "") or "")
        qual = str(node.meta.get("qual", "") or "")
        if not path or not qual:
            continue
        span = node.meta.get("span")
        span_list = list(span) if span else None
        index.setdefault(
            (path, qual),
            _function_site(path=path, qual=qual, span=span_list),
        )
    return index


def pack_call_stack(
    sites,
) -> PackedCallStack:
    normalized: list[_CallSite] = []
    for site in sites:
        match site:
            case _CallSite() as payload:
                pass
            case _:
                payload = _normalize_site_payload(site)
        normalized.append(payload)
    unique: dict[tuple[str, tuple[object, ...]], _InternedCallSite] = {}
    for entry in normalized:
        key = (entry.kind, entry.frozen_key())
        if key not in unique:
            unique[key] = _InternedCallSite(order=len(unique), site=entry)
    ordered_unique = sort_once(
        unique.items(),
        source="pack_call_stack.site_table",
        policy=OrderPolicy.ENFORCE,
        key=lambda item: item[1].order,
    )
    site_table: list[_CallSite] = []
    index: dict[tuple[str, tuple[object, ...]], int] = {}
    for idx, ((kind, frozen_key), interned) in enumerate(ordered_unique):
        site_table.append(interned.site)
        index[(kind, frozen_key)] = idx
    stack = [
        index[(entry.kind, entry.frozen_key())]
        for entry in normalized
    ]
    return PackedCallStack(
        site_table=tuple(site_table),
        stack=tuple(stack),
    )


def _normalize_site_payload(
    site: Mapping[str, object],
) -> _CallSite:
    kind = str(site.get("kind", "") or "FunctionSite")
    key_payload = site.get("key")
    match key_payload:
        case list() as key_entries if key_entries:
            key = [value for value in key_entries]
            first = key[0]
            second = key[1] if key[1:] else None
            match (first, second):
                case (str() as path_value, str() as qual_value):
                    key = [_FileSite(path_value), qual_value, *key[2:]]
                case _:
                    pass
            key_tuple = tuple(_site_part_from_payload(value) for value in key)
            return _CallSite(
                kind=kind,
                key=key_tuple,
            )
        case _:
            pass
    path = str(site.get("path", "") or "")
    qual = str(site.get("qual", "") or "")
    if not path or not qual:
        never("site payload missing path/qual", site=dict(site))
    span = site.get("span")
    match span:
        case list() as span_values:
            span_list = list(span_values)
        case _:
            span_list = None
    return _function_site(path=path, qual=qual, span=span_list)


def _freeze_value(value: object) -> object:
    match value:
        case _FileSite() as file_site:
            return ("FileSite", file_site.path)
        case tuple() as key_tuple:
            return ("tuple", tuple(_freeze_value(item) for item in key_tuple))
        case None | str() | int() | float() | bool():
            return ("atom", value)
        case _:
            never("invalid site key value for freezing", value_type=type(value).__name__)
            return value  # pragma: no cover - never() raises


def _freeze_key(key: Iterable[object]) -> tuple[object, ...]:
    return tuple(_freeze_value(part) for part in key)


def build_timeout_context_from_stack(
    *,
    forest: Forest,
    project_root,
    forest_spec_id = None,
    forest_signature = None,
    deadline_profile = None,
    allow_frame_fallback: bool = False,
    frames = None,
) -> TimeoutContext:
    site_index = build_site_index(forest)
    frame_list = list(frames) if frames is not None else [frame.frame for frame in inspect.stack()]
    sites: list[_CallSite] = []
    resolved_root = project_root.resolve() if project_root is not None else None
    for frame in frame_list:
        if resolved_root is not None:
            frame_path = Path(frame.f_code.co_filename).resolve()
            try:
                frame_path.relative_to(resolved_root)
            except ValueError:
                continue
        key = _frame_site_key(frame, project_root=resolved_root)
        if key in site_index:
            sites.append(site_index[key])
        elif allow_frame_fallback:
            sites.append(_function_site(path=key[0], qual=key[1]))
    packed = pack_call_stack(reversed(sites))
    profile_payload = deadline_profile or _deadline_profile_snapshot()
    return TimeoutContext(
        call_stack=packed,
        forest_spec_id=forest_spec_id,
        forest_signature=forest_signature,
        deadline_profile=profile_payload,
        progress=_timeout_progress_snapshot(
            forest=forest,
            deadline_profile=profile_payload,
        ),
    )
