from __future__ import annotations

import inspect
import time
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Callable, Iterable, Mapping

from contextvars import ContextVar, Token

from gabion.analysis.aspf import Forest
from gabion.exceptions import NeverThrown
from gabion.invariants import never
from gabion.json_types import JSONValue
from gabion.order_contract import OrderPolicy, ordered_or_sorted


_TIMEOUT_PROGRESS_CHECKS_FLOOR = 32
_TIMEOUT_PROGRESS_SITE_FLOOR = 4


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
    forest_spec_id: str | None = None
    forest_signature: dict[str, JSONValue] | None = None
    deadline_profile: dict[str, JSONValue] | None = None
    progress: dict[str, JSONValue] | None = None

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
        return cls(deadline_ns=time.monotonic_ns() + total_ns)

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
        return time.monotonic_ns() >= self.deadline_ns

    def check(self, builder: Callable[[], TimeoutContext]) -> None:
        if self.expired():
            raise TimeoutExceeded(builder())


_deadline_var: ContextVar[Deadline | None] = ContextVar("gabion_deadline", default=None)
_MISSING_FOREST = object()
_forest_var: ContextVar[Forest | object] = ContextVar(
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
class _DeadlineProfileState:
    enabled: bool
    started_ns: int
    last_ns: int
    project_root: Path | None = None
    checks_total: int = 0
    unattributed_elapsed_ns: int = 0
    last_site: tuple[str, str] | None = None
    site_stats: dict[tuple[str, str], _DeadlineSiteStats] = field(default_factory=dict)
    edge_stats: dict[
        tuple[tuple[str, str], tuple[str, str]],
        _DeadlineEdgeStats,
    ] = field(default_factory=dict)


_deadline_profile_var: ContextVar[_DeadlineProfileState | None] = ContextVar(
    "gabion_deadline_profile", default=None
)


def set_deadline_profile(
    *,
    project_root: Path | None = None,
    enabled: bool = True,
) -> Token[_DeadlineProfileState | None]:
    resolved_root = project_root.resolve() if project_root is not None else None
    now = time.monotonic_ns()
    state = _DeadlineProfileState(
        enabled=enabled,
        started_ns=now,
        last_ns=now,
        project_root=resolved_root,
    )
    return _deadline_profile_var.set(state)


def reset_deadline_profile(token: Token[_DeadlineProfileState | None]) -> None:
    _deadline_profile_var.reset(token)


def set_deadline(deadline: Deadline) -> Token[Deadline | None]:
    if deadline is None:
        never("deadline carrier missing")
    return _deadline_var.set(deadline)


def reset_deadline(token: Token[Deadline | None]) -> None:
    _deadline_var.reset(token)


def get_deadline() -> Deadline:
    deadline = _deadline_var.get()
    if deadline is None:
        never("deadline carrier missing")
    return deadline


def set_forest(forest: Forest) -> Token[Forest | object]:
    return _forest_var.set(forest)


def reset_forest(token: Token[Forest | object]) -> None:
    _forest_var.reset(token)


def get_forest() -> Forest:
    forest = _forest_var.get()
    if forest is _MISSING_FOREST:
        never("forest carrier missing")
    if not isinstance(forest, Forest):
        never("invalid forest carrier", carrier_type=type(forest).__name__)
    return forest


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
def forest_scope(forest: Forest):
    token = set_forest(forest)
    try:
        yield
    finally:
        reset_forest(token)


@contextmanager
def deadline_profile_scope(
    *,
    project_root: Path | None = None,
    enabled: bool = True,
):
    token = set_deadline_profile(project_root=project_root, enabled=enabled)
    try:
        yield
    finally:
        reset_deadline_profile(token)


def _profile_site_key(
    frame: FrameType,
    *,
    project_root: Path | None,
) -> tuple[str, str]:
    if project_root is None:
        return _frame_site_key(frame, project_root=None)
    try:
        return _frame_site_key(frame, project_root=project_root)
    except NeverThrown:
        _, qual = _frame_site_key(frame, project_root=None)
        return ("<external>", qual)


def _record_deadline_check(project_root: Path | None) -> None:
    state = _deadline_profile_var.get()
    if state is None:
        return
    if not state.enabled:
        return
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return
    caller_frame = frame.f_back.f_back
    effective_root = project_root.resolve() if project_root is not None else state.project_root
    now = time.monotonic_ns()
    delta = max(0, now - state.last_ns)
    state.last_ns = now
    state.checks_total += 1
    site = _profile_site_key(caller_frame, project_root=effective_root)
    stats = state.site_stats.setdefault(site, _DeadlineSiteStats())
    stats.checks += 1
    stats.elapsed_ns += delta
    if delta > stats.max_gap_ns:
        stats.max_gap_ns = delta
    if state.last_site is None:
        state.unattributed_elapsed_ns += delta
    else:
        edge_key = (state.last_site, site)
        edge_stats = state.edge_stats.setdefault(edge_key, _DeadlineEdgeStats())
        edge_stats.transitions += 1
        edge_stats.elapsed_ns += delta
        if delta > edge_stats.max_gap_ns:
            edge_stats.max_gap_ns = delta
    state.last_site = site


def _deadline_profile_snapshot() -> dict[str, JSONValue] | None:
    state = _deadline_profile_var.get()
    if state is None:
        return None
    if not state.enabled:
        return None
    total_elapsed_ns = max(0, state.last_ns - state.started_ns)
    site_rows: list[dict[str, JSONValue]] = []
    for (path, qual), stats in ordered_or_sorted(
        state.site_stats.items(),
        source="_deadline_profile_snapshot.site_rows",
        key=lambda item: (-item[1].elapsed_ns, item[0][0], item[0][1]),
    ):
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
    for (source, target), stats in ordered_or_sorted(
        state.edge_stats.items(),
        source="_deadline_profile_snapshot.edge_rows",
        key=lambda item: (
            -item[1].elapsed_ns,
            item[0][0][0],
            item[0][0][1],
            item[0][1][0],
            item[0][1][1],
        ),
    ):
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
    return {
        "checks_total": state.checks_total,
        "started_ns": state.started_ns,
        "last_check_ns": state.last_ns,
        "total_elapsed_ns": total_elapsed_ns,
        "unattributed_elapsed_ns": state.unattributed_elapsed_ns,
        "sites": site_rows,
        "edges": edge_rows,
    }


def _timeout_progress_snapshot(
    *,
    forest: Forest,
    deadline_profile: Mapping[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    checks_total = 0
    site_count = 0
    if isinstance(deadline_profile, Mapping):
        checks_total = int(deadline_profile.get("checks_total", 0) or 0)
        sites = deadline_profile.get("sites")
        if isinstance(sites, list):
            site_count = len(sites)
    forest_nodes = len(forest.nodes)
    forest_alts = len(forest.alts)
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
    return {
        "classification": classification,
        "retry_recommended": progressed,
        "resume_supported": False,
        "checks_total": checks_total,
        "site_count": site_count,
        "forest_nodes": forest_nodes,
        "forest_alts": forest_alts,
    }


def render_deadline_profile_markdown(
    profile: Mapping[str, JSONValue],
    *,
    max_rows: int = 25,
) -> str:
    lines: list[str] = ["# Deadline Profile Heat", ""]
    checks_total = int(profile.get("checks_total", 0) or 0)
    total_elapsed_ns = int(profile.get("total_elapsed_ns", 0) or 0)
    unattributed_ns = int(profile.get("unattributed_elapsed_ns", 0) or 0)
    lines.append(f"- checks_total: `{checks_total}`")
    lines.append(f"- total_elapsed_ns: `{total_elapsed_ns}`")
    lines.append(f"- unattributed_elapsed_ns: `{unattributed_ns}`")
    lines.append("")
    lines.append("## Site Heat")
    lines.append("")
    lines.append("| path | qual | checks | elapsed_ns | max_gap_ns |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    sites = profile.get("sites", [])
    if isinstance(sites, list):
        for row in sites[:max_rows]:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| {path} | {qual} | {checks} | {elapsed} | {gap} |".format(
                    path=str(row.get("path", "") or ""),
                    qual=str(row.get("qual", "") or ""),
                    checks=int(row.get("check_count", 0) or 0),
                    elapsed=int(row.get("elapsed_between_checks_ns", 0) or 0),
                    gap=int(row.get("max_gap_ns", 0) or 0),
                )
            )
        if len(sites) > max_rows:
            lines.append(f"| ... | ... | ... | ... | ... |")
    lines.append("")
    lines.append("## Transition Heat")
    lines.append("")
    lines.append("| from | to | transitions | elapsed_ns | max_gap_ns |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    edges = profile.get("edges", [])
    if isinstance(edges, list):
        for row in edges[:max_rows]:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| {source} | {target} | {count} | {elapsed} | {gap} |".format(
                    source=f"{str(row.get('from_path', '') or '')}:{str(row.get('from_qual', '') or '')}",
                    target=f"{str(row.get('to_path', '') or '')}:{str(row.get('to_qual', '') or '')}",
                    count=int(row.get("transition_count", 0) or 0),
                    elapsed=int(row.get("elapsed_ns", 0) or 0),
                    gap=int(row.get("max_gap_ns", 0) or 0),
                )
            )
        if len(edges) > max_rows:
            lines.append(f"| ... | ... | ... | ... | ... |")
    lines.append("")
    return "\n".join(lines)


def check_deadline(
    deadline: Deadline | None = None,
    *,
    project_root: Path | None = None,
    forest_spec_id: str | None = None,
    forest_signature: dict[str, JSONValue] | None = None,
    allow_frame_fallback: bool = True,
) -> None:
    if deadline is None:
        deadline = get_deadline()
    _record_deadline_check(project_root)
    deadline.check(
        lambda: build_timeout_context_from_stack(
            forest=get_forest(),
            project_root=project_root,
            forest_spec_id=forest_spec_id,
            forest_signature=forest_signature,
            deadline_profile=_deadline_profile_snapshot(),
            allow_frame_fallback=allow_frame_fallback,
        )
    )


def _normalize_qualname(qualname: str) -> str:
    return qualname.replace(".<locals>.", ".")


def _frame_site_key(
    frame: FrameType,
    *,
    project_root: Path | None,
) -> tuple[str, str]:
    module = frame.f_globals.get("__name__") or ""
    qualname = _normalize_qualname(frame.f_code.co_qualname or frame.f_code.co_name)
    if module:
        qual = f"{module}.{qualname}" if not qualname.startswith(f"{module}.") else qualname
    else:
        qual = qualname
    path = Path(frame.f_code.co_filename)
    if project_root is not None:
        try:
            path = path.resolve().relative_to(project_root)
        except ValueError:
            never("frame outside project_root", path=str(path), project_root=str(project_root))
    return (path.name, qual)


def _site_key_payload(
    *,
    path: str,
    qual: str,
    span: list[int] | None = None,
) -> dict[str, JSONValue]:
    return _function_site(path=path, qual=qual, span=span).as_payload()


def _site_key(
    *,
    path: str,
    qual: str,
    span: list[int] | None = None,
) -> tuple[object, ...]:
    key: list[object] = [_FileSite(path), qual]
    if span and len(span) == 4:
        key.extend(span)
    return tuple(key)


def _function_site(
    *,
    path: str,
    qual: str,
    span: list[int] | None = None,
) -> _CallSite:
    return _CallSite(kind="FunctionSite", key=_site_key(path=path, qual=qual, span=span))


def _site_part_from_payload(value: object) -> object:
    if isinstance(value, _FileSite):
        return value
    if isinstance(value, Mapping):
        kind = str(value.get("kind", "") or "")
        key_payload = value.get("key")
        if kind == "FileSite" and isinstance(key_payload, list) and len(key_payload) == 1:
            path = key_payload[0]
            if isinstance(path, str):
                return _FileSite(path)
        never("invalid site key mapping payload", payload_kind=kind)
    if isinstance(value, list):
        return tuple(_site_part_from_payload(part) for part in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    never("invalid site key payload value", value_type=type(value).__name__)
    return value  # pragma: no cover - never() raises


def _site_part_to_payload(value: object) -> JSONValue:
    if isinstance(value, _FileSite):
        return value.as_payload()
    if isinstance(value, tuple):
        return [_site_part_to_payload(part) for part in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    never("invalid site key value", value_type=type(value).__name__)
    return None  # pragma: no cover - never() raises


def build_site_index(
    forest: Forest,
) -> dict[tuple[str, str], _CallSite]:
    index: dict[tuple[str, str], _CallSite] = {}
    ordered_nodes = ordered_or_sorted(
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
        span_list = list(span) if isinstance(span, list) else None
        index.setdefault(
            (path, qual),
            _function_site(path=path, qual=qual, span=span_list),
        )
    return index


def pack_call_stack(
    sites: Iterable[_CallSite | Mapping[str, object]],
) -> PackedCallStack:
    normalized: list[_CallSite] = []
    for site in sites:
        payload = site if isinstance(site, _CallSite) else _normalize_site_payload(site)
        normalized.append(payload)
    unique: dict[tuple[str, tuple[object, ...]], _InternedCallSite] = {}
    for entry in normalized:
        key = (entry.kind, entry.frozen_key())
        if key not in unique:
            unique[key] = _InternedCallSite(order=len(unique), site=entry)
    ordered_unique = ordered_or_sorted(
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
    if isinstance(key_payload, list) and key_payload:
        key = [value for value in key_payload]
        if (
            len(key) >= 2
            and isinstance(key[0], str)
            and isinstance(key[1], str)
        ):
            key = [_FileSite(key[0]), key[1], *key[2:]]
        key_tuple = tuple(_site_part_from_payload(value) for value in key)
        return _CallSite(
            kind=kind,
            key=key_tuple,
        )
    path = str(site.get("path", "") or "")
    qual = str(site.get("qual", "") or "")
    if not path or not qual:
        never("site payload missing path/qual", site=dict(site))
    span = site.get("span")
    span_list = list(span) if isinstance(span, list) else None
    return _function_site(path=path, qual=qual, span=span_list)


def _freeze_value(value: object) -> object:
    if isinstance(value, _FileSite):
        return ("FileSite", value.path)
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze_value(item) for item in value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return ("atom", value)
    never("invalid site key value for freezing", value_type=type(value).__name__)
    return value  # pragma: no cover - never() raises


def _freeze_key(key: Iterable[object]) -> tuple[object, ...]:
    return tuple(_freeze_value(part) for part in key)


def build_timeout_context_from_stack(
    *,
    forest: Forest,
    project_root: Path | None,
    forest_spec_id: str | None = None,
    forest_signature: dict[str, JSONValue] | None = None,
    deadline_profile: dict[str, JSONValue] | None = None,
    allow_frame_fallback: bool = False,
    frames: Iterable[FrameType] | None = None,
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
