from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch

from gabion.analysis.aspf.aspf_visitors import (
    AspfCofibrationEvent,
    AspfOneCellEvent,
    AspfRunBoundaryEvent,
    AspfSurfaceUpdateEvent,
    AspfTraceReplayEvent,
    AspfTwoCellEvent,
)
from gabion.analysis.foundation.event_algebra import (
    CanonicalAdaptationDecision,
    CanonicalEventEnvelope,
    CanonicalRunContext,
    build_canonical_event_envelope,
    canonical_adaptation_rejected,
    canonical_adaptation_valid,
    derive_identity_projection_from_tokens,
    envelope_from_decision_or_raise,
)
from gabion.analysis.foundation.event_algebra_adapter_utils import (
    mapping_payload_to_wire_object,
    payload_sha1_digest,
)
from gabion.analysis.foundation.frozen_object_map import ObjectEntry, make_object_map
from gabion.analysis.foundation.wire_types import WireObject
from gabion.analysis.foundation.wire_types import WireValue
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never

ASPF_CANONICAL_SOURCE = "aspf.trace_replay"


@dataclass(frozen=True)
class _NormalizedReplayEvent:
    kind: str
    phase: str
    payload: WireObject
    identity_tokens: list[str]


def adapt_aspf_replay_event(
    *,
    event: AspfTraceReplayEvent,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...] = (),
) -> CanonicalAdaptationDecision:
    check_deadline()
    return _adapt_event(event, run_context=run_context, causal_refs=causal_refs)


def adapt_aspf_replay_event_or_raise(
    *,
    event: AspfTraceReplayEvent,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...] = (),
) -> CanonicalEventEnvelope:
    check_deadline()
    return envelope_from_decision_or_raise(
        adapt_aspf_replay_event(
            event=event,
            run_context=run_context,
            causal_refs=causal_refs,
        )
    )


@singledispatch
def _adapt_event(
    event: AspfTraceReplayEvent,
    *,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...],
) -> CanonicalAdaptationDecision:
    _ = run_context
    _ = causal_refs
    never("invalid aspf replay event", kind=type(event).__name__)


@_adapt_event.register
def _sd_reg_1(
    event: AspfOneCellEvent,
    *,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...],
) -> CanonicalAdaptationDecision:
    payload = mapping_payload_to_wire_object(event.payload)
    normalized = _trace_replay_event_payload(
        kind="one_cell",
        index=event.index,
        payload=payload,
    )
    return _canonical_valid_decision(
        normalized=normalized,
        run_context=run_context,
        causal_refs=causal_refs,
    )


@_adapt_event.register
def _sd_reg_2(
    event: AspfTwoCellEvent,
    *,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...],
) -> CanonicalAdaptationDecision:
    payload = mapping_payload_to_wire_object(event.payload)
    normalized = _trace_replay_event_payload(
        kind="two_cell",
        index=event.index,
        payload=payload,
    )
    return _canonical_valid_decision(
        normalized=normalized,
        run_context=run_context,
        causal_refs=causal_refs,
    )


@_adapt_event.register
def _sd_reg_3(
    event: AspfCofibrationEvent,
    *,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...],
) -> CanonicalAdaptationDecision:
    payload = mapping_payload_to_wire_object(event.payload)
    normalized = _trace_replay_event_payload(
        kind="cofibration",
        index=event.index,
        payload=payload,
    )
    return _canonical_valid_decision(
        normalized=normalized,
        run_context=run_context,
        causal_refs=causal_refs,
    )


@_adapt_event.register
def _sd_reg_4(
    event: AspfSurfaceUpdateEvent,
    *,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...],
) -> CanonicalAdaptationDecision:
    surface = _trimmed_text(event.surface)
    representative = _trimmed_text(event.representative)
    error_index = (surface == "") + ((representative == "") * 2)
    error_message = _SURFACE_UPDATE_ERRORS[error_index]
    normalized = _NormalizedReplayEvent(
        kind="surface_update",
        phase="trace_replay",
        payload=make_object_map(
            [
                ObjectEntry("surface", surface),
                ObjectEntry("representative", representative),
            ]
        ),
        identity_tokens=[
            ASPF_CANONICAL_SOURCE,
            "trace_replay",
            "surface_update",
            "surface:%s" % surface,
            "representative:%s" % representative,
        ],
    )
    return _SURFACE_UPDATE_DECISIONS[error_index == 0](
        normalized=normalized,
        run_context=run_context,
        causal_refs=causal_refs,
        error_message=error_message,
    )


@_adapt_event.register
def _sd_reg_5(
    event: AspfRunBoundaryEvent,
    *,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...],
) -> CanonicalAdaptationDecision:
    payload = mapping_payload_to_wire_object(event.payload)
    digest = payload_sha1_digest(payload)
    boundary = _trimmed_text(event.boundary)
    normalized = _NormalizedReplayEvent(
        kind="run_boundary",
        phase="run_boundary",
        payload=make_object_map(
            [
                ObjectEntry("boundary", boundary),
                ObjectEntry("payload", payload),
            ]
        ),
        identity_tokens=[
            ASPF_CANONICAL_SOURCE,
            "run_boundary",
            "run_boundary",
            "boundary:%s" % boundary,
            "payload:%s" % digest,
        ],
    )
    return _RUN_BOUNDARY_DECISIONS[boundary != ""](
        normalized=normalized,
        run_context=run_context,
        causal_refs=causal_refs,
    )


def _trace_replay_event_payload(*, kind: str, index: int, payload: WireObject) -> _NormalizedReplayEvent:
    digest = payload_sha1_digest(payload)
    return _NormalizedReplayEvent(
        kind=kind,
        phase="trace_replay",
        payload=make_object_map(
            [
                ObjectEntry("index", index),
                ObjectEntry("payload", payload),
            ]
        ),
        identity_tokens=[
            ASPF_CANONICAL_SOURCE,
            "trace_replay",
            kind,
            "index:%s" % index,
            "payload:%s" % digest,
        ],
    )


def _canonical_valid_decision(
    *,
    normalized: _NormalizedReplayEvent,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...],
) -> CanonicalAdaptationDecision:
    identity_projection = derive_identity_projection_from_tokens(
        run_context=run_context,
        tokens=(*normalized.identity_tokens,),
    )
    envelope = build_canonical_event_envelope(
        run_context=run_context,
        source=ASPF_CANONICAL_SOURCE,
        phase=normalized.phase,
        kind=normalized.kind,
        identity_projection=identity_projection,
        payload=normalized.payload,
        causal_refs=causal_refs,
    )
    return canonical_adaptation_valid(envelope)


def _rejected_surface_update_decision(
    *,
    normalized: _NormalizedReplayEvent,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...],
    error_message: str,
) -> CanonicalAdaptationDecision:
    _ = normalized
    _ = run_context
    _ = causal_refs
    return canonical_adaptation_rejected(error_message)


def _valid_surface_update_decision(
    *,
    normalized: _NormalizedReplayEvent,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...],
    error_message: str,
) -> CanonicalAdaptationDecision:
    _ = error_message
    return _canonical_valid_decision(
        normalized=normalized,
        run_context=run_context,
        causal_refs=causal_refs,
    )


def _rejected_run_boundary_decision(
    *,
    normalized: _NormalizedReplayEvent,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...],
) -> CanonicalAdaptationDecision:
    _ = normalized
    _ = run_context
    _ = causal_refs
    return canonical_adaptation_rejected(
        "run_boundary event requires a non-empty boundary identifier."
    )


_TEXT_COERCERS = [lambda _: "", lambda value: value]
_SURFACE_UPDATE_ERRORS = [
    "",
    "surface_update event requires a non-empty surface identifier.",
    "surface_update event requires a non-empty representative identifier.",
    "surface_update event requires non-empty surface and representative identifiers.",
]
_SURFACE_UPDATE_DECISIONS = [
    _rejected_surface_update_decision,
    _valid_surface_update_decision,
]
_RUN_BOUNDARY_DECISIONS = [
    _rejected_run_boundary_decision,
    _canonical_valid_decision,
]


def _trimmed_text(value: WireValue) -> str:
    return _TEXT_COERCERS[type(value) is str](value).strip()


__all__ = [
    "ASPF_CANONICAL_SOURCE",
    "adapt_aspf_replay_event",
    "adapt_aspf_replay_event_or_raise",
]
