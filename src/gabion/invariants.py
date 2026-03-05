# gabion:decision_protocol_module
"""Invariant markers for Gabion analysis."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import warnings
from typing import Callable, TypeVar, cast

from gabion.exceptions import NeverThrown

_PROOF_MODE_OVERRIDE: ContextVar[bool | None] = ContextVar(
    "gabion_proof_mode_override",
    default=None,
)


@dataclass(frozen=True)
class ProofModeConfig:
    enabled: bool = False


_PROOF_MODE_CONFIG: ContextVar[ProofModeConfig] = ContextVar(
    "gabion_proof_mode_config",
    default=ProofModeConfig(),
)


@dataclass(frozen=True)
class StructuredReasoning:
    summary: str
    control: str = ""
    blocking_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class InvariantRuntimeBehavior:
    throws: bool
    emits_warning: bool
    warning_limit: int


@dataclass(frozen=True)
class InvariantProfileConfig:
    never: InvariantRuntimeBehavior
    todo: InvariantRuntimeBehavior
    deprecated: InvariantRuntimeBehavior


_DEFAULT_NEVER_BEHAVIOR = InvariantRuntimeBehavior(
    throws=True,
    emits_warning=False,
    warning_limit=0,
)
_DEFAULT_TODO_BEHAVIOR = InvariantRuntimeBehavior(
    throws=False,
    emits_warning=False,
    warning_limit=0,
)
_DEFAULT_DEPRECATED_BEHAVIOR = InvariantRuntimeBehavior(
    throws=False,
    emits_warning=True,
    warning_limit=1024,
)
_DEFAULT_INVARIANT_PROFILE = InvariantProfileConfig(
    never=_DEFAULT_NEVER_BEHAVIOR,
    todo=_DEFAULT_TODO_BEHAVIOR,
    deprecated=_DEFAULT_DEPRECATED_BEHAVIOR,
)


@dataclass(frozen=True)
class MarkerGovernanceConfig:
    profile: str = "governance"
    invariant_profile: InvariantProfileConfig = _DEFAULT_INVARIANT_PROFILE


_MARKER_GOVERNANCE_CONFIG: ContextVar[MarkerGovernanceConfig] = ContextVar(
    "gabion_marker_governance_config",
    default=MarkerGovernanceConfig(),
)


class DeprecatedMarkerWarning(RuntimeWarning):
    """Runtime warning emitted when deprecated() marker is executed."""


_WARNING_CACHE: set[str] = set()

T = TypeVar("T")
FuncT = TypeVar("FuncT", bound=Callable[..., object])


def _normalized_marker_links(raw_links: object) -> tuple[dict[str, str], ...]:
    if type(raw_links) is not list:
        return ()
    links = cast(list[object], raw_links)
    normalized: list[dict[str, str]] = []
    for item in links:
        if type(item) is not dict:
            continue
        payload = cast(dict[object, object], item)
        kind = str(payload.get("kind", "")).strip().lower()
        value = str(payload.get("value", "")).strip()
        if kind and value:
            normalized.append({"kind": kind, "value": value})
    return tuple(normalized)


def _normalize_structured_reasoning(reasoning: object) -> StructuredReasoning:
    if isinstance(reasoning, StructuredReasoning):
        return reasoning
    if type(reasoning) is dict:
        payload = cast(dict[object, object], reasoning)
        raw_dependencies = payload.get("blocking_dependencies", ())
        if type(raw_dependencies) is list or type(raw_dependencies) is tuple:
            dependencies = tuple(
                sorted(
                    {
                        str(item).strip()
                        for item in raw_dependencies
                        if str(item).strip()
                    }
                )
            )
        else:
            dependencies = ()
        return StructuredReasoning(
            summary=str(payload.get("summary", "")).strip(),
            control=str(payload.get("control", "")).strip(),
            blocking_dependencies=dependencies,
        )
    return StructuredReasoning(summary=str(reasoning or "").strip())


def _resolve_reasoning(reasoning: object) -> StructuredReasoning:
    structured = _normalize_structured_reasoning(reasoning)
    if structured.summary:
        return structured
    return StructuredReasoning(
        summary="marker reached",
        control=structured.control,
        blocking_dependencies=structured.blocking_dependencies,
    )


def _runtime_behavior_for(marker_kind: str, config: MarkerGovernanceConfig) -> InvariantRuntimeBehavior:
    if marker_kind == "never":
        return config.invariant_profile.never
    if marker_kind == "todo":
        return config.invariant_profile.todo
    return config.invariant_profile.deprecated


def _warning_key(marker_kind: str, structured: StructuredReasoning) -> str:
    dependencies = ",".join(structured.blocking_dependencies)
    return f"{marker_kind}|{structured.summary}|{structured.control}|{dependencies}"


def _emit_warning_once(*, marker_kind: str, structured: StructuredReasoning, warning_limit: int) -> None:
    if warning_limit <= 0:
        return
    key = _warning_key(marker_kind, structured)
    if key in _WARNING_CACHE:
        return
    if len(_WARNING_CACHE) >= warning_limit:
        return
    _WARNING_CACHE.add(key)
    warnings.warn(
        (
            f"{marker_kind}() marker reached"
            f" summary={structured.summary!r}"
            f" control={structured.control!r}"
            f" blocking_dependencies={list(structured.blocking_dependencies)!r}"
        ),
        category=DeprecatedMarkerWarning,
        stacklevel=2,
    )


def _marker_payload_from_reasoning(marker_kind: str, structured: StructuredReasoning, **env: object):
    from gabion.analysis.foundation.marker_protocol import (
        MarkerKind,
        normalize_governance_profile,
        normalize_marker_payload,
        normalize_marker_reasoning,
        resolve_marker_kind_for_profile,
    )

    resolved_kind = resolve_marker_kind_for_profile(
        MarkerKind(marker_kind),
        profile=normalize_governance_profile(current_marker_governance_config().profile),
    )
    owner = str(env.get("owner", ""))
    expiry = str(env.get("expiry", ""))
    links = _normalized_marker_links(env.get("links", ()))
    extra_env = {
        key: value
        for key, value in env.items()
        if key not in {"owner", "expiry", "links"}
    }
    normalized_reasoning = normalize_marker_reasoning(
        summary=structured.summary,
        control=structured.control,
        blocking_dependencies=structured.blocking_dependencies,
    )
    return normalize_marker_payload(
        reason=structured.summary,
        env=extra_env,
        marker_kind=resolved_kind,
        owner=owner,
        expiry=expiry,
        links=links,
        reasoning=normalized_reasoning,
    )


def invariant_factory(marker_kind: str, reasoning: object = "", **env: object):
    structured = _resolve_reasoning(reasoning)
    payload = _marker_payload_from_reasoning(marker_kind, structured, **env)
    runtime_behavior = _runtime_behavior_for(
        marker_kind,
        current_marker_governance_config(),
    )
    if runtime_behavior.emits_warning:
        _emit_warning_once(
            marker_kind=marker_kind,
            structured=structured,
            warning_limit=runtime_behavior.warning_limit,
        )
    if runtime_behavior.throws:
        raise NeverThrown(payload.reason, marker_payload=payload)
    return payload


def never(reasoning: object = "", **env: object):
    """Mark a code path as intentionally unreachable using structured reasoning."""
    return invariant_factory("never", reasoning, **env)


def todo(reasoning: object = "", **env: object) -> None:
    """Mark a code path as pending implementation without runtime interruption."""
    invariant_factory("todo", reasoning, **env)


def deprecated(reasoning: object = "", **env: object) -> None:
    """Mark a deprecated path and emit profile-configured runtime warning."""
    invariant_factory("deprecated", reasoning, **env)


def proof_mode() -> bool:
    override = _PROOF_MODE_OVERRIDE.get()
    if override is not None:
        return bool(override)
    return bool(_PROOF_MODE_CONFIG.get().enabled)


def set_proof_mode_config(config: ProofModeConfig) -> Token[ProofModeConfig]:
    return _PROOF_MODE_CONFIG.set(config)


def reset_proof_mode_config(token: Token[ProofModeConfig]) -> None:
    _PROOF_MODE_CONFIG.reset(token)


def _normalize_marker_governance_config(config: MarkerGovernanceConfig) -> MarkerGovernanceConfig:
    profile_value = str(config.profile or "").strip() or "governance"
    return MarkerGovernanceConfig(
        profile=profile_value,
        invariant_profile=config.invariant_profile,
    )


def current_marker_governance_config() -> MarkerGovernanceConfig:
    return _MARKER_GOVERNANCE_CONFIG.get()


def set_marker_governance_config(config: MarkerGovernanceConfig) -> Token[MarkerGovernanceConfig]:
    return _MARKER_GOVERNANCE_CONFIG.set(_normalize_marker_governance_config(config))


def reset_marker_governance_config(token: Token[MarkerGovernanceConfig]) -> None:
    _MARKER_GOVERNANCE_CONFIG.reset(token)


@contextmanager
def marker_governance_scope(
    profile: object,
    *,
    invariant_profile: InvariantProfileConfig = _DEFAULT_INVARIANT_PROFILE,
):
    token = set_marker_governance_config(
        MarkerGovernanceConfig(
            profile=str(profile),
            invariant_profile=invariant_profile,
        )
    )
    try:
        yield
    finally:
        reset_marker_governance_config(token)


@contextmanager
def proof_mode_config_scope(config: ProofModeConfig):
    token = set_proof_mode_config(config)
    try:
        yield
    finally:
        reset_proof_mode_config(token)


@contextmanager
def proof_mode_scope(enabled: bool):
    token = _PROOF_MODE_OVERRIDE.set(bool(enabled))
    try:
        yield
    finally:
        _PROOF_MODE_OVERRIDE.reset(token)


def require_not_none(
    value: T | None,
    *,
    reason: str = "",
    strict: bool | None = None,
    **env: object,
) -> T | None:
    if value is None:
        if strict is None:
            strict = proof_mode()
        if strict:
            never({"summary": reason or "required value is None"}, **env)
    return value


def decision_protocol(func: FuncT) -> FuncT:
    """Marker decorator for explicit decision-protocol control surfaces."""
    return func


def boundary_normalization(func: FuncT) -> FuncT:
    """Marker decorator for boundary normalization surfaces."""
    return func
