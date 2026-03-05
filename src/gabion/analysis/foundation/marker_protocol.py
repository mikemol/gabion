# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
"""Marker protocol contracts for invariant and analysis surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from hashlib import sha1
import json
from contextlib import contextmanager
from contextvars import ContextVar, Token
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence


class MarkerKind(StrEnum):
    NEVER = "never"
    TODO = "todo"
    DEPRECATED = "deprecated"


class MarkerKindProfile(StrEnum):
    NATIVE = "native"
    COLLAPSE_TO_NEVER = "collapse_to_never"


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
    reasoning: MarkerReasoning
    owner: str
    expiry: str
    lifecycle_state: MarkerLifecycleState
    links: tuple[SemanticReference, ...]
    env: dict[str, object]


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


@dataclass(frozen=True)
class MarkerKindMappingConfig:
    profile: MarkerKindProfile
    kind_map: Mapping[MarkerKind, MarkerKind]


DEFAULT_MARKER_KIND_PROFILE_MAPS: Mapping[MarkerKindProfile, Mapping[MarkerKind, MarkerKind]] = MappingProxyType(
    {
        MarkerKindProfile.NATIVE: MappingProxyType({}),
        MarkerKindProfile.COLLAPSE_TO_NEVER: MappingProxyType(
            {
                MarkerKind.TODO: MarkerKind.NEVER,
                MarkerKind.DEPRECATED: MarkerKind.NEVER,
            }
        ),
    }
)

DEFAULT_MARKER_KIND_MAPPING_CONFIG = MarkerKindMappingConfig(
    profile=MarkerKindProfile.NATIVE,
    kind_map=DEFAULT_MARKER_KIND_PROFILE_MAPS[MarkerKindProfile.NATIVE],
)

_MARKER_KIND_MAPPING_CONFIG: ContextVar[MarkerKindMappingConfig] = ContextVar(
    "gabion_marker_kind_mapping_config",
    default=DEFAULT_MARKER_KIND_MAPPING_CONFIG,
)


def marker_kind_mapping_config(profile: MarkerKindProfile) -> MarkerKindMappingConfig:
    return MarkerKindMappingConfig(profile=profile, kind_map=DEFAULT_MARKER_KIND_PROFILE_MAPS[profile])


def runtime_marker_kind_mapping_config() -> MarkerKindMappingConfig:
    return _MARKER_KIND_MAPPING_CONFIG.get()


def set_runtime_marker_kind_mapping_config(
    config: MarkerKindMappingConfig,
) -> Token[MarkerKindMappingConfig]:
    return _MARKER_KIND_MAPPING_CONFIG.set(config)


def reset_runtime_marker_kind_mapping_config(
    token: Token[MarkerKindMappingConfig],
) -> None:
    _MARKER_KIND_MAPPING_CONFIG.reset(token)


@contextmanager
def runtime_marker_kind_mapping_scope(config: MarkerKindMappingConfig) -> Iterator[None]:
    token = set_runtime_marker_kind_mapping_config(config)
    try:
        yield
    finally:
        reset_runtime_marker_kind_mapping_config(token)


def resolve_marker_kind_for_profile(
    marker_kind: MarkerKind,
    *,
    mapping_config: MarkerKindMappingConfig | None = None,
) -> MarkerKind:
    effective_mapping_config = mapping_config or runtime_marker_kind_mapping_config()
    return effective_mapping_config.kind_map.get(marker_kind, marker_kind)


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


def normalize_marker_payload(
    *,
    reason: str,
    env: Mapping[str, object] = _EMPTY_ENV,
    marker_kind: MarkerKind = MarkerKind.NEVER,
    owner: str = "",
    expiry: str = "",
    lifecycle_state: MarkerLifecycleState = MarkerLifecycleState.ACTIVE,
    links: Sequence[Mapping[str, str]] = _EMPTY_LINKS,
    reasoning: object = "",
) -> MarkerPayload:
    normalized_reasoning = normalize_marker_reasoning(reasoning or reason)
    normalized_reason = normalized_reasoning.summary or "never() invariant reached"
    env_payload = {str(key): value for key, value in env.items()}
    return MarkerPayload(
        marker_kind=marker_kind,
        reason=normalized_reason,
        reasoning=normalized_reasoning,
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
        "reasoning": {
            "summary": payload.reasoning.summary,
            "control": payload.reasoning.control,
            "blocking_dependencies": list(payload.reasoning.blocking_dependencies),
        },
        "owner": payload.owner,
        "expiry": payload.expiry,
        "lifecycle_state": payload.lifecycle_state.value,
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
    reasoning: object = "",
) -> MarkerPayload:
    return normalize_marker_payload(
        reason=reason,
        reasoning=reasoning,
        env=env,
        marker_kind=MarkerKind.NEVER,
        owner=owner,
        expiry=expiry,
        links=links,
    )


def _normalize_dependency_values(raw_values: object) -> tuple[str, ...]:
    if raw_values is None:
        return ()
    if type(raw_values) is str:
        candidates = (raw_values,)
    elif isinstance(raw_values, Sequence):
        candidates = tuple(str(value) for value in raw_values)
    else:
        candidates = (str(raw_values),)
    deduped = {value.strip() for value in candidates if value.strip()}
    return tuple(sorted(deduped))


def _normalize_reasoning_mapping(raw_mapping: Mapping[object, object]) -> MarkerReasoning:
    summary = str(raw_mapping.get("summary", "")).strip()
    control = str(raw_mapping.get("control", "")).strip()
    blocking_dependencies = _normalize_dependency_values(raw_mapping.get("blocking_dependencies", ()))
    return MarkerReasoning(
        summary=summary,
        control=control,
        blocking_dependencies=blocking_dependencies,
    )


def normalize_marker_reasoning(raw_reasoning: object = "") -> MarkerReasoning:
    """Boundary normalizer for marker reasoning payloads.

    Supports typed dataclass input, generic mappings, and scalar fallback values.
    """

    if isinstance(raw_reasoning, MarkerReasoning):
        return MarkerReasoning(
            summary=raw_reasoning.summary.strip(),
            control=raw_reasoning.control.strip(),
            blocking_dependencies=_normalize_dependency_values(raw_reasoning.blocking_dependencies),
        )

    if is_dataclass(raw_reasoning):
        raw_dataclass = asdict(raw_reasoning)
        if isinstance(raw_dataclass, dict):
            return _normalize_reasoning_mapping(raw_dataclass)

    if isinstance(raw_reasoning, Mapping):
        return _normalize_reasoning_mapping(raw_reasoning)

    summary = str(raw_reasoning).strip()
    return MarkerReasoning(summary=summary, control="", blocking_dependencies=())
