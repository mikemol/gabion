# gabion:decision_protocol_module
"""Invariant markers for Gabion analysis."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Callable, TypeVar, cast
import warnings

from gabion.exceptions import NeverThrown

if TYPE_CHECKING:
    from gabion.analysis.foundation.marker_protocol import MarkerPayload

class InvariantProfile(StrEnum):
    STRICT = "strict"
    DIAGNOSTIC = "diagnostic"
    DEBT_GATE = "debt_gate"
    SUNSET_GATE = "sunset_gate"


@dataclass(frozen=True)
class MarkerRuntimeBehavior:
    throws: bool
    emits_warning: bool
    warning_limit: int = 0


@dataclass(frozen=True)
class InvariantRuntimeBehaviorConfig:
    profile: InvariantProfile = InvariantProfile.STRICT


InvariantWarningKey = tuple[str, str, str, tuple[str, ...]]


@dataclass(frozen=True)
class InvariantWarningState:
    emitted_count: int = 0
    seen_keys: frozenset[InvariantWarningKey] = frozenset()


class InvariantMarkerWarning(UserWarning):
    """Profile-scoped warning emitted by non-throwing marker behaviors."""

    def __init__(self, *, payload: MarkerPayload, profile: InvariantProfile):
        self.marker_payload = payload
        self.profile = profile
        super().__init__(
            f"[{profile.value}] {payload.marker_kind.value}: {payload.reason}"
        )


_INVARIANT_RUNTIME_BEHAVIOR_CONFIG: ContextVar[InvariantRuntimeBehaviorConfig] = ContextVar(
    "gabion_invariant_runtime_behavior_config",
    default=InvariantRuntimeBehaviorConfig(),
)

_INVARIANT_WARNING_STATE: ContextVar[InvariantWarningState] = ContextVar(
    "gabion_invariant_warning_state",
    default=InvariantWarningState(),
)

_STRICT_BEHAVIOR = MarkerRuntimeBehavior(throws=True, emits_warning=False, warning_limit=0)
_DIAGNOSTIC_BEHAVIOR = MarkerRuntimeBehavior(
    throws=False,
    emits_warning=True,
    warning_limit=1000,
)
_DEBT_GATE_NEVER = MarkerRuntimeBehavior(throws=True, emits_warning=False, warning_limit=0)
_DEBT_GATE_TODO = MarkerRuntimeBehavior(throws=True, emits_warning=True, warning_limit=50)
_DEBT_GATE_DEPRECATED = MarkerRuntimeBehavior(
    throws=False,
    emits_warning=True,
    warning_limit=50,
)
_SUNSET_GATE_NEVER = MarkerRuntimeBehavior(throws=True, emits_warning=False, warning_limit=0)
_SUNSET_GATE_TODO = MarkerRuntimeBehavior(
    throws=False,
    emits_warning=True,
    warning_limit=50,
)
_SUNSET_GATE_DEPRECATED = MarkerRuntimeBehavior(
    throws=True,
    emits_warning=True,
    warning_limit=50,
)

_PROFILE_RUNTIME_BEHAVIOR_MATRIX: dict[
    InvariantProfile, dict[str, MarkerRuntimeBehavior]
] = {
    InvariantProfile.STRICT: {
        "never": _STRICT_BEHAVIOR,
        "todo": _STRICT_BEHAVIOR,
        "deprecated": _STRICT_BEHAVIOR,
    },
    InvariantProfile.DIAGNOSTIC: {
        "never": _DIAGNOSTIC_BEHAVIOR,
        "todo": _DIAGNOSTIC_BEHAVIOR,
        "deprecated": _DIAGNOSTIC_BEHAVIOR,
    },
    InvariantProfile.DEBT_GATE: {
        "never": _DEBT_GATE_NEVER,
        "todo": _DEBT_GATE_TODO,
        "deprecated": _DEBT_GATE_DEPRECATED,
    },
    InvariantProfile.SUNSET_GATE: {
        "never": _SUNSET_GATE_NEVER,
        "todo": _SUNSET_GATE_TODO,
        "deprecated": _SUNSET_GATE_DEPRECATED,
    },
}

T = TypeVar("T")
FuncT = TypeVar("FuncT", bound=Callable[..., object])
DecoratableT = TypeVar("DecoratableT")


_INVARIANT_DECORATIONS_ATTR = "__gabion_invariant_decorations__"
_LEGACY_NEVER_STRING_REASON_DEPRECATION_CONTROL: dict[str, object] = {
    "actor": "codex",
    "rationale": "Deprecate string-only never() calls in favor of structured reasoning payloads.",
    "scope": "invariants.never.legacy_string_reason",
    "start": "2026-03-05",
    "expiry": "invariants.legacy-never-string deprecation window closure",
    "rollback_condition": "all in-repo callsites migrate to structured never() reasoning payloads",
    "evidence_links": ["docs/invariants_system_design.md"],
}


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


def resolve_marker_runtime_behavior(
    marker_kind: str,
    *,
    config: InvariantRuntimeBehaviorConfig | None = None,
) -> MarkerRuntimeBehavior:
    active_config = config or invariant_runtime_behavior_config()
    profile_matrix = _PROFILE_RUNTIME_BEHAVIOR_MATRIX[active_config.profile]
    return profile_matrix[marker_kind]


def invariant_warning_state() -> InvariantWarningState:
    return _INVARIANT_WARNING_STATE.get()


def set_invariant_warning_state(
    state: InvariantWarningState,
) -> Token[InvariantWarningState]:
    return _INVARIANT_WARNING_STATE.set(state)


def reset_invariant_warning_state(
    token: Token[InvariantWarningState],
) -> None:
    _INVARIANT_WARNING_STATE.reset(token)


def _marker_warning_key(payload: MarkerPayload) -> InvariantWarningKey:
    return (
        payload.marker_kind.value,
        payload.reasoning.summary,
        payload.reasoning.control,
        payload.reasoning.blocking_dependencies,
    )


def _advance_warning_state(
    *,
    state: InvariantWarningState,
    warning_key: InvariantWarningKey,
    warning_limit: int,
) -> tuple[InvariantWarningState, bool]:
    if warning_key in state.seen_keys:
        return state, False
    if state.emitted_count >= warning_limit:
        return state, False
    return (
        InvariantWarningState(
            emitted_count=state.emitted_count + 1,
            seen_keys=state.seen_keys | {warning_key},
        ),
        True,
    )


def _emit_invariant_marker_warning(
    *,
    payload: MarkerPayload,
    profile: InvariantProfile,
    warning_limit: int,
) -> None:
    warning_key = _marker_warning_key(payload)
    state = invariant_warning_state()
    next_state, should_emit = _advance_warning_state(
        state=state,
        warning_key=warning_key,
        warning_limit=warning_limit,
    )
    if next_state != state:
        set_invariant_warning_state(next_state)
    if not should_emit:
        return
    warnings.warn(
        InvariantMarkerWarning(payload=payload, profile=profile),
        stacklevel=3,
    )


def _normalized_invariant_marker_payload(
    marker_kind: str,
    reasoning: object = "",
    **env: object,
) -> MarkerPayload:
    from gabion.analysis.foundation.marker_protocol import (
        MarkerKind,
        never_marker_payload,
        normalize_marker_payload,
        normalize_marker_reasoning,
    )

    owner = str(env.get("owner", ""))
    expiry = str(env.get("expiry", ""))
    links = _normalized_marker_links(env.get("links", ()))
    raw_reasoning = env.get("reasoning", reasoning)
    if "reason" in env:
        reason = str(env["reason"])
    elif isinstance(reasoning, str):
        reason = reasoning
    else:
        reason = ""
    normalized_reasoning = normalize_marker_reasoning(raw_reasoning)
    if not normalized_reasoning.summary and reason:
        normalized_reasoning = normalize_marker_reasoning(reason)
    extra_env = {
        key: value
        for key, value in env.items()
        if key not in {"owner", "expiry", "links", "reasoning", "reason"}
    }
    marker_kind_enum = MarkerKind(marker_kind)
    if marker_kind_enum is MarkerKind.NEVER:
        return never_marker_payload(
            reason=reason,
            reasoning=normalized_reasoning,
            env=extra_env,
            owner=owner,
            expiry=expiry,
            links=links,
        )
    fallback_reasoning = normalized_reasoning
    if not fallback_reasoning.summary:
        fallback_reasoning = normalize_marker_reasoning(
            f"{marker_kind_enum.value}() marker reached"
        )
    return normalize_marker_payload(
        reason=reason or f"{marker_kind_enum.value}() marker reached",
        reasoning=fallback_reasoning,
        env=extra_env,
        marker_kind=marker_kind_enum,
        owner=owner,
        expiry=expiry,
        links=links,
    )


def invariant_factory(
    marker_kind: str, reasoning: object = "", **env: object
) -> MarkerPayload:
    payload = _normalized_invariant_marker_payload(marker_kind, reasoning, **env)
    runtime_config = invariant_runtime_behavior_config()
    behavior = resolve_marker_runtime_behavior(
        payload.marker_kind.value,
        config=runtime_config,
    )
    if behavior.emits_warning:
        _emit_invariant_marker_warning(
            payload=payload,
            profile=runtime_config.profile,
            warning_limit=behavior.warning_limit,
        )
    if behavior.throws:
        raise NeverThrown(payload.reason, marker_payload=payload)
    return payload


def _emit_legacy_never_string_reason_deprecation(
    reason: str,
    *,
    deprecated_fn: Callable[..., MarkerPayload] | None = None,
) -> None:
    resolved_deprecated_fn = deprecated if deprecated_fn is None else deprecated_fn
    try:
        resolved_deprecated_fn(
            "never() string-only API is deprecated; use structured reasoning",
            reasoning={
                "summary": "never() string-only API is deprecated; use structured reasoning",
                "control": str(
                    _LEGACY_NEVER_STRING_REASON_DEPRECATION_CONTROL["scope"]
                ),
                "blocking_dependencies": (
                    "migrate_never_callsites_to_structured_reasoning",
                ),
            },
            control_id=str(_LEGACY_NEVER_STRING_REASON_DEPRECATION_CONTROL["scope"]),
            legacy_api="never(reason: str)",
            replacement_api=(
                "never(reasoning={summary, control, blocking_dependencies}, ...)"
            ),
            legacy_reason=reason,
            links=[{"kind": "doc_id", "value": "invariants_system_design"}],
        )
    except NeverThrown:
        # In strict profiles, deprecated() throws; never() remains the terminal signal.
        return


def never(
    reason: str = "",
    *,
    invariant_factory_fn: Callable[..., MarkerPayload] = invariant_factory,
    emit_legacy_never_string_reason_deprecation_fn: Callable[[str], None] = (
        _emit_legacy_never_string_reason_deprecation
    ),
    **env: object,
) -> MarkerPayload:
    """Mark a code path as intentionally unreachable.

    The analysis treats this as a sink that should be proven unreachable. The
    optional env payload is metadata only; it is not evaluated at runtime.
    """
    if reason.strip() and "reasoning" not in env:
        emit_legacy_never_string_reason_deprecation_fn(reason)
    return invariant_factory_fn("never", reason=reason, **env)


def todo(
    reason: str = "",
    *,
    invariant_factory_fn: Callable[..., MarkerPayload] = invariant_factory,
    **env: object,
) -> MarkerPayload:
    """Mark a code path as intentionally pending implementation."""
    return invariant_factory_fn("todo", reason=reason, **env)


def deprecated(
    reason: str = "",
    *,
    invariant_factory_fn: Callable[..., MarkerPayload] = invariant_factory,
    **env: object,
) -> MarkerPayload:
    """Mark a code path as a deprecated/blocked semantic surface."""
    return invariant_factory_fn("deprecated", reason=reason, **env)


def invariant_decorations(target: object) -> tuple[MarkerPayload, ...]:
    raw = getattr(target, _INVARIANT_DECORATIONS_ATTR, ())
    return raw if isinstance(raw, tuple) else ()


def invariant_decorator(
    marker_kind: str,
    reason: str = "",
    **env: object,
) -> Callable[[DecoratableT], DecoratableT]:
    payload = _normalized_invariant_marker_payload(marker_kind, reason=reason, **env)

    def _decorate(target: DecoratableT) -> DecoratableT:
        existing = invariant_decorations(target)
        setattr(target, _INVARIANT_DECORATIONS_ATTR, (*existing, payload))
        return target

    return _decorate


def never_decorator(reason: str = "", **env: object) -> Callable[[DecoratableT], DecoratableT]:
    return invariant_decorator("never", reason=reason, **env)


def todo_decorator(reason: str = "", **env: object) -> Callable[[DecoratableT], DecoratableT]:
    return invariant_decorator("todo", reason=reason, **env)


def deprecated_decorator(
    reason: str = "",
    **env: object,
) -> Callable[[DecoratableT], DecoratableT]:
    return invariant_decorator("deprecated", reason=reason, **env)


def invariant_runtime_behavior_config() -> InvariantRuntimeBehaviorConfig:
    return _INVARIANT_RUNTIME_BEHAVIOR_CONFIG.get()


def set_invariant_runtime_behavior_config(
    config: InvariantRuntimeBehaviorConfig,
) -> Token[InvariantRuntimeBehaviorConfig]:
    return _INVARIANT_RUNTIME_BEHAVIOR_CONFIG.set(config)


def reset_invariant_runtime_behavior_config(
    token: Token[InvariantRuntimeBehaviorConfig],
) -> None:
    _INVARIANT_RUNTIME_BEHAVIOR_CONFIG.reset(token)


@contextmanager
def invariant_runtime_behavior_scope(config: InvariantRuntimeBehaviorConfig):
    config_token = set_invariant_runtime_behavior_config(config)
    warning_token = set_invariant_warning_state(InvariantWarningState())
    try:
        yield
    finally:
        reset_invariant_warning_state(warning_token)
        reset_invariant_runtime_behavior_config(config_token)


def require_not_none(
    value: T | None,
    *,
    reason: str = "",
    strict: bool | None = None,
    **env: object,
) -> T | None:
    if value is None:
        if strict is not False:
            never(reason or "required value is None", **env)
    return value


def decision_protocol(func: FuncT) -> FuncT:
    """Marker decorator for explicit decision-protocol control surfaces."""
    return func


def boundary_normalization(func: FuncT) -> FuncT:
    """Marker decorator for boundary normalization surfaces."""
    return func
