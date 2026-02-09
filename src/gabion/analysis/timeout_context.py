from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Callable, Iterable, Mapping

from contextvars import ContextVar, Token

from gabion.analysis.aspf import Forest
from gabion.json_types import JSONValue


@dataclass(frozen=True)
class PackedCallStack:
    site_table: list[dict[str, JSONValue]]
    stack: list[int]

    def as_payload(self) -> dict[str, JSONValue]:
        return {"site_table": self.site_table, "stack": self.stack}


@dataclass(frozen=True)
class TimeoutContext:
    call_stack: PackedCallStack
    forest_spec_id: str | None = None
    forest_signature: dict[str, JSONValue] | None = None

    def as_payload(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {"call_stack": self.call_stack.as_payload()}
        if self.forest_spec_id:
            payload["forest_spec_id"] = self.forest_spec_id
        if self.forest_signature:
            payload["forest_signature"] = self.forest_signature
        return payload


class TimeoutExceeded(TimeoutError):
    def __init__(self, context: TimeoutContext) -> None:
        super().__init__("Analysis timed out.")
        self.context = context


@dataclass(frozen=True)
class Deadline:
    deadline: float

    @classmethod
    def from_timeout(cls, seconds: float) -> "Deadline":
        return cls(deadline=time.monotonic() + max(float(seconds), 0.0))

    def expired(self) -> bool:
        return time.monotonic() >= self.deadline

    def check(self, builder: Callable[[], TimeoutContext]) -> None:
        if self.expired():
            raise TimeoutExceeded(builder())


_deadline_var: ContextVar[Deadline | None] = ContextVar("gabion_deadline", default=None)


def set_deadline(deadline: Deadline | None) -> Token[Deadline | None]:
    return _deadline_var.set(deadline)


def reset_deadline(token: Token[Deadline | None]) -> None:
    _deadline_var.reset(token)


def get_deadline() -> Deadline | None:
    return _deadline_var.get()


@contextmanager
def deadline_scope(deadline: Deadline | None):
    token = set_deadline(deadline)
    try:
        yield
    finally:
        reset_deadline(token)


def check_deadline(
    deadline: Deadline | None = None,
    *,
    forest: Forest | None = None,
    project_root: Path | None = None,
    forest_spec_id: str | None = None,
    forest_signature: dict[str, JSONValue] | None = None,
    allow_frame_fallback: bool = False,
) -> None:
    if deadline is None:
        deadline = get_deadline()
    if deadline is None:
        return
    deadline.check(
        lambda: build_timeout_context_from_stack(
            forest=forest,
            project_root=project_root,
            forest_spec_id=forest_spec_id,
            forest_signature=forest_signature,
            allow_frame_fallback=allow_frame_fallback,
        )
    )


def _normalize_qualname(qualname: str) -> str:
    return qualname.replace(".<locals>.", ".")


def _frame_site_key(
    frame: FrameType,
    *,
    project_root: Path | None,
) -> tuple[str, str] | None:
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
            return None
    return (path.name, qual)


def _site_key_payload(
    *,
    path: str,
    qual: str,
    span: list[int] | None = None,
) -> dict[str, JSONValue]:
    key: list[JSONValue] = [path, qual]
    if span and len(span) == 4:
        key.extend(span)
    return {"kind": "FunctionSite", "key": key}


def build_site_index(
    forest: Forest | None,
) -> dict[tuple[str, str], dict[str, JSONValue]]:
    if forest is None:
        return {}
    index: dict[tuple[str, str], dict[str, JSONValue]] = {}
    ordered_nodes = sorted(forest.nodes.items(), key=lambda item: item[0].sort_key())
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
            _site_key_payload(path=path, qual=qual, span=span_list),
        )
    return index


def pack_call_stack(sites: Iterable[Mapping[str, JSONValue]]) -> PackedCallStack:
    normalized: list[dict[str, JSONValue]] = []
    for site in sites:
        payload = _normalize_site_payload(site)
        if payload is None:
            continue
        normalized.append(payload)
    unique = {(entry["kind"], tuple(entry["key"])) for entry in normalized}
    site_table = [
        {"kind": kind, "key": list(key)}
        for kind, key in sorted(unique, key=_site_sort_key)
    ]
    index = {
        (entry["kind"], tuple(entry["key"])): idx
        for idx, entry in enumerate(site_table)
    }
    stack = [
        index[(entry["kind"], tuple(entry["key"]))]
        for entry in normalized
    ]
    return PackedCallStack(site_table=site_table, stack=stack)


def _normalize_site_payload(
    site: Mapping[str, JSONValue],
) -> dict[str, JSONValue] | None:
    kind = str(site.get("kind", "") or "FunctionSite")
    key_payload = site.get("key")
    if isinstance(key_payload, list) and key_payload:
        key = [value for value in key_payload]
        return {"kind": kind, "key": key}
    path = str(site.get("path", "") or "")
    qual = str(site.get("qual", "") or "")
    if not path or not qual:
        return None
    span = site.get("span")
    span_list = list(span) if isinstance(span, list) else None
    return _site_key_payload(path=path, qual=qual, span=span_list)


def _site_sort_key(entry: tuple[str, tuple[JSONValue, ...]]) -> tuple[str, tuple[str, ...]]:
    kind, key = entry
    return (kind, tuple(str(part) for part in key))


def build_timeout_context_from_stack(
    *,
    forest: Forest | None,
    project_root: Path | None,
    forest_spec_id: str | None = None,
    forest_signature: dict[str, JSONValue] | None = None,
    allow_frame_fallback: bool = False,
    frames: Iterable[FrameType] | None = None,
) -> TimeoutContext:
    site_index = build_site_index(forest)
    frame_list = list(frames) if frames is not None else [frame.frame for frame in inspect.stack()]
    sites: list[dict[str, JSONValue]] = []
    for frame in frame_list:
        key = _frame_site_key(frame, project_root=project_root)
        if key is None:
            continue
        if key in site_index:
            sites.append(site_index[key])
        elif allow_frame_fallback:
            sites.append(_site_key_payload(path=key[0], qual=key[1]))
    packed = pack_call_stack(reversed(sites))
    return TimeoutContext(
        call_stack=packed,
        forest_spec_id=forest_spec_id,
        forest_signature=forest_signature,
    )
