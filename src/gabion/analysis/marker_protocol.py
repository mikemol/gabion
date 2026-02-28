"""Marker protocol contracts for invariant and analysis surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha1
import json
from typing import Mapping, Sequence


class MarkerKind(StrEnum):
    NEVER = "never"


class MarkerLifecycleState(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class SemanticReference:
    kind: str
    value: str


@dataclass(frozen=True)
class MarkerPayload:
    marker_kind: MarkerKind
    reason: str
    owner: str
    expiry: str
    lifecycle_state: MarkerLifecycleState
    links: tuple[SemanticReference, ...]
    env: dict[str, object]


DEFAULT_MARKER_ALIASES: dict[MarkerKind, tuple[str, ...]] = {
    MarkerKind.NEVER: (
        "never",
        "gabion.never",
        "gabion.invariants.never",
    )
}


def normalize_semantic_links(raw_links: Sequence[Mapping[str, str]]) -> tuple[SemanticReference, ...]:
    normalized = tuple(
        SemanticReference(kind=str(raw.get("kind", "")).strip(), value=str(raw.get("value", "")).strip())
        for raw in raw_links
    )
    return tuple(
        sorted(
            (item for item in normalized if item.kind and item.value),
            key=lambda item: (item.kind, item.value),
        )
    )


def normalize_marker_payload(
    *,
    reason: str,
    env: Mapping[str, object] = {},
    marker_kind: MarkerKind = MarkerKind.NEVER,
    owner: str = "",
    expiry: str = "",
    lifecycle_state: MarkerLifecycleState = MarkerLifecycleState.ACTIVE,
    links: Sequence[Mapping[str, str]] = (),
) -> MarkerPayload:
    normalized_reason = (reason or "never() invariant reached").strip()
    env_payload = {str(key): value for key, value in env.items()}
    return MarkerPayload(
        marker_kind=marker_kind,
        reason=normalized_reason,
        owner=owner.strip(),
        expiry=expiry.strip(),
        lifecycle_state=lifecycle_state,
        links=normalize_semantic_links(links),
        env=env_payload,
    )


def marker_identity(payload: MarkerPayload) -> str:
    identity_payload = {
        "marker_kind": payload.marker_kind.value,
        "reason": payload.reason,
        "owner": payload.owner,
        "expiry": payload.expiry,
        "lifecycle_state": payload.lifecycle_state.value,
        "links": [{"kind": link.kind, "value": link.value} for link in payload.links],
    }
    encoded = json.dumps(identity_payload, separators=(",", ":"), sort_keys=True)
    digest = sha1(encoded.encode("utf-8")).hexdigest()[:12]
    return f"{payload.marker_kind.value}:{digest}"


def never_marker_payload(
    *,
    reason: str = "",
    env: Mapping[str, object] = {},
    owner: str = "",
    expiry: str = "",
    links: Sequence[Mapping[str, str]] = (),
) -> MarkerPayload:
    return normalize_marker_payload(
        reason=reason,
        env=env,
        marker_kind=MarkerKind.NEVER,
        owner=owner,
        expiry=expiry,
        links=links,
    )
