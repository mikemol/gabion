# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyEventKind:
    namespace: str
    family: str
    detail: str
    extra: tuple[str, ...] = ()


def policy_event_kind_from_scalar(*, raw: str) -> PolicyEventKind:
    segments = tuple(raw.split(":")) if raw else ()
    namespace = _segment_or_empty(segments=segments, index=0)
    family = _segment_or_empty(segments=segments, index=1)
    detail = _segment_or_empty(segments=segments, index=2)
    extra = segments[3:] if len(segments) > 3 else ()
    return PolicyEventKind(
        namespace=namespace,
        family=family,
        detail=detail,
        extra=extra,
    )


def coerce_policy_event_kind(*, kind: PolicyEventKind | str) -> PolicyEventKind:
    match kind:
        case PolicyEventKind():
            return kind
        case str() as raw:
            return policy_event_kind_from_scalar(raw=raw)
        case _:
            return policy_event_kind_from_scalar(raw="")


def policy_event_kind_segments(*, kind: PolicyEventKind | str) -> tuple[str, ...]:
    policy_kind = coerce_policy_event_kind(kind=kind)
    segments = (
        policy_kind.namespace,
        policy_kind.family,
        policy_kind.detail,
        *policy_kind.extra,
    )
    return _trim_trailing_empty(segments=segments)


def policy_event_kind_sort_key(*, kind: PolicyEventKind | str) -> tuple[str, ...]:
    policy_kind = coerce_policy_event_kind(kind=kind)
    return (
        policy_kind.namespace,
        policy_kind.family,
        policy_kind.detail,
        *policy_kind.extra,
    )


def policy_event_kind_scalar(*, kind: PolicyEventKind | str) -> str:
    segments = policy_event_kind_segments(kind=kind)
    return ":".join(segments)


def _segment_or_empty(*, segments: tuple[str, ...], index: int) -> str:
    return segments[index] if len(segments) > index else ""


def _trim_trailing_empty(*, segments: tuple[str, ...]) -> tuple[str, ...]:
    trimmed = segments
    while trimmed and trimmed[-1] == "":
        trimmed = trimmed[:-1]
    return trimmed


__all__ = [
    "PolicyEventKind",
    "coerce_policy_event_kind",
    "policy_event_kind_from_scalar",
    "policy_event_kind_segments",
    "policy_event_kind_sort_key",
    "policy_event_kind_scalar",
]
