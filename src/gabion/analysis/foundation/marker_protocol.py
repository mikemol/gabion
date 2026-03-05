"""Marker protocol contracts for invariant and analysis surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha1
import json
from types import MappingProxyType
from typing import Mapping, Sequence


class MarkerKind(StrEnum):
    NEVER = "never"
    TODO = "todo"
    DEPRECATED = "deprecated"


class MarkerGovernanceProfile(StrEnum):
    GOVERNANCE = "governance"
    DEBT_LEDGER = "debt_ledger"
    INVARIANT = "invariant"


class MarkerLifecycleState(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    ROLLED_BACK = "rolled_back"


class SemanticLinkKind(StrEnum):
    POLICY_ID = "policy_id"
    DOC_ID = "doc_id"
    INVARIANT_ID = "invariant_id"
    OBJECT_ID = "object_id"


@dataclass(frozen=True)
class SemanticReference:
    kind: SemanticLinkKind
    value: str


@dataclass(frozen=True)
class MarkerReasoning:
    summary: str
    control: str
    blocking_dependencies: tuple[str, ...]


@dataclass(frozen=True)
class MarkerPayload:
    marker_kind: MarkerKind
    reason: str
    owner: str
    expiry: str
    lifecycle_state: MarkerLifecycleState
    links: tuple[SemanticReference, ...]
    reasoning: MarkerReasoning
    env: dict[str, object]

_DEFAULT_REASONING = MarkerReasoning(summary="", control="", blocking_dependencies=())

_EMPTY_ENV: Mapping[str, object] = MappingProxyType({})
_EMPTY_LINKS: tuple[Mapping[str, str], ...] = ()
_LINK_KIND_BY_VALUE: dict[str, SemanticLinkKind] = {kind.value: kind for kind in SemanticLinkKind}


DEFAULT_MARKER_ALIASES: dict[MarkerKind, tuple[str, ...]] = {
    MarkerKind.NEVER: (
        "never",
        "gabion.never",
        "gabion.invariants.never",
    ),
    MarkerKind.TODO: (
        "todo",
        "gabion.todo",
        "gabion.invariants.todo",
    ),
    MarkerKind.DEPRECATED: (
        "deprecated",
        "gabion.deprecated",
        "gabion.invariants.deprecated",
    ),
}


_PROFILE_KIND_MAP: dict[MarkerGovernanceProfile, dict[MarkerKind, MarkerKind]] = {
    MarkerGovernanceProfile.GOVERNANCE: {
        MarkerKind.NEVER: MarkerKind.NEVER,
        MarkerKind.TODO: MarkerKind.TODO,
        MarkerKind.DEPRECATED: MarkerKind.DEPRECATED,
    },
    MarkerGovernanceProfile.DEBT_LEDGER: {
        MarkerKind.NEVER: MarkerKind.NEVER,
        MarkerKind.TODO: MarkerKind.TODO,
        MarkerKind.DEPRECATED: MarkerKind.DEPRECATED,
    },
    MarkerGovernanceProfile.INVARIANT: {
        MarkerKind.NEVER: MarkerKind.NEVER,
        MarkerKind.TODO: MarkerKind.NEVER,
        MarkerKind.DEPRECATED: MarkerKind.NEVER,
    },
}


def normalize_governance_profile(value: object) -> MarkerGovernanceProfile:
    normalized = str(value or "").strip().lower()
    for profile in MarkerGovernanceProfile:
        if profile.value == normalized:
            return profile
    return MarkerGovernanceProfile.GOVERNANCE


def resolve_marker_kind_for_profile(
    marker_kind: MarkerKind,
    *,
    profile: object,
) -> MarkerKind:
    normalized = normalize_governance_profile(profile)
    return _PROFILE_KIND_MAP[normalized].get(marker_kind, MarkerKind.NEVER)


def normalize_semantic_links(raw_links: Sequence[Mapping[str, str]] = _EMPTY_LINKS) -> tuple[SemanticReference, ...]:
    entries = tuple(
        (
            str(raw.get("kind", "")).strip().lower(),
            str(raw.get("value", "")).strip(),
        )
        for raw in raw_links
    )
    links = tuple(
        SemanticReference(kind=_LINK_KIND_BY_VALUE[kind], value=value)
        for kind, value in entries
        if kind in _LINK_KIND_BY_VALUE and value
    )
    return tuple(sorted(links, key=lambda link: (link.kind.value, link.value)))


def normalize_marker_reasoning(
    *,
    summary: object = "",
    control: object = "",
    blocking_dependencies: object = (),
) -> MarkerReasoning:
    summary_text = str(summary or "").strip()
    control_text = str(control or "").strip()
    if type(blocking_dependencies) is list or type(blocking_dependencies) is tuple:
        deps = tuple(
            sorted(
                {
                    str(item).strip()
                    for item in blocking_dependencies
                    if str(item).strip()
                }
            )
        )
    else:
        deps = ()
    return MarkerReasoning(
        summary=summary_text,
        control=control_text,
        blocking_dependencies=deps,
    )


def normalize_marker_payload(
    *,
    reason: str,
    env: Mapping[str, object] = _EMPTY_ENV,
    marker_kind: MarkerKind = MarkerKind.NEVER,
    owner: str = "",
    expiry: str = "",
    lifecycle_state: MarkerLifecycleState = MarkerLifecycleState.ACTIVE,
    links: Sequence[Mapping[str, str]] = _EMPTY_LINKS,
    reasoning: MarkerReasoning = _DEFAULT_REASONING,
) -> MarkerPayload:
    normalized_reason = str(reason or "never() invariant reached").strip()
    env_payload = {str(key): value for key, value in env.items()}
    normalized_reasoning = reasoning
    if not normalized_reasoning.summary and not normalized_reasoning.control and not normalized_reasoning.blocking_dependencies:
        normalized_reasoning = normalize_marker_reasoning(summary=normalized_reason)
    return MarkerPayload(
        marker_kind=marker_kind,
        reason=normalized_reason,
        owner=owner.strip(),
        expiry=expiry.strip(),
        lifecycle_state=lifecycle_state,
        links=normalize_semantic_links(links),
        reasoning=normalized_reasoning,
        env=env_payload,
    )


def marker_identity(payload: MarkerPayload) -> str:
    identity_payload = {
        "marker_kind": payload.marker_kind.value,
        "reason": payload.reason,
        "owner": payload.owner,
        "expiry": payload.expiry,
        "lifecycle_state": payload.lifecycle_state.value,
        "reasoning": {
            "summary": payload.reasoning.summary,
            "control": payload.reasoning.control,
            "blocking_dependencies": list(payload.reasoning.blocking_dependencies),
        },
        "links": [
            {"kind": link.kind.value, "value": link.value}
            for link in payload.links
        ],
    }
    encoded = json.dumps(identity_payload, separators=(",", ":"), sort_keys=True)
    digest = sha1(encoded.encode("utf-8")).hexdigest()[:12]
    return f"{payload.marker_kind.value}:{digest}"


def never_marker_payload(
    *,
    reason: str = "",
    env: Mapping[str, object] = _EMPTY_ENV,
    owner: str = "",
    expiry: str = "",
    links: Sequence[Mapping[str, str]] = _EMPTY_LINKS,
    reasoning: MarkerReasoning = _DEFAULT_REASONING,
) -> MarkerPayload:
    return normalize_marker_payload(
        reason=reason,
        env=env,
        marker_kind=MarkerKind.NEVER,
        owner=owner,
        expiry=expiry,
        links=links,
        reasoning=reasoning,
    )
