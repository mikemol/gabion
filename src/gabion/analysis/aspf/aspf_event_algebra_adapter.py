# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

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
    CanonicalEventAdaptationError,
    CanonicalEventEnvelope,
    CanonicalRunContext,
    build_canonical_event_envelope,
    canonical_adaptation_rejected,
    canonical_adaptation_valid,
    derive_identity_projection_from_tokens,
    envelope_from_decision_or_raise,
)
from gabion.analysis.foundation.event_algebra_adapter_utils import (
    mapping_payload_to_json_object,
    payload_sha1_digest,
)
from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never

ASPF_CANONICAL_SOURCE = "aspf.trace_replay"


def adapt_aspf_replay_event(
    *,
    event: AspfTraceReplayEvent,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...] = (),
) -> CanonicalAdaptationDecision:
    check_deadline()
    try:
        kind, phase, payload, identity_tokens = _normalized_aspf_event_payload(event=event)
        identity_projection = derive_identity_projection_from_tokens(
            run_context=run_context,
            tokens=identity_tokens,
        )
        envelope = build_canonical_event_envelope(
            run_context=run_context,
            source=ASPF_CANONICAL_SOURCE,
            phase=phase,
            kind=kind,
            identity_projection=identity_projection,
            payload=payload,
            causal_refs=causal_refs,
        )
        return canonical_adaptation_valid(envelope)
    except (CanonicalEventAdaptationError, ValueError) as exc:
        return canonical_adaptation_rejected(str(exc))


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


def _normalized_aspf_event_payload(
    *,
    event: AspfTraceReplayEvent,
) -> tuple[str, str, JSONObject, tuple[str, ...]]:
    check_deadline()
    match event:
        case AspfOneCellEvent():
            payload = mapping_payload_to_json_object(event.payload)
            digest = payload_sha1_digest(payload)
            return (
                "one_cell",
                "trace_replay",
                {"index": event.index, "payload": payload},
                (
                    ASPF_CANONICAL_SOURCE,
                    "trace_replay",
                    "one_cell",
                    f"index:{event.index}",
                    f"payload:{digest}",
                ),
            )
        case AspfTwoCellEvent():
            payload = mapping_payload_to_json_object(event.payload)
            digest = payload_sha1_digest(payload)
            return (
                "two_cell",
                "trace_replay",
                {"index": event.index, "payload": payload},
                (
                    ASPF_CANONICAL_SOURCE,
                    "trace_replay",
                    "two_cell",
                    f"index:{event.index}",
                    f"payload:{digest}",
                ),
            )
        case AspfCofibrationEvent():
            payload = mapping_payload_to_json_object(event.payload)
            digest = payload_sha1_digest(payload)
            return (
                "cofibration",
                "trace_replay",
                {"index": event.index, "payload": payload},
                (
                    ASPF_CANONICAL_SOURCE,
                    "trace_replay",
                    "cofibration",
                    f"index:{event.index}",
                    f"payload:{digest}",
                ),
            )
        case AspfSurfaceUpdateEvent():
            surface = str(event.surface).strip()
            representative = str(event.representative).strip()
            if not surface:
                raise CanonicalEventAdaptationError(
                    "surface_update event requires a non-empty surface identifier."
                )
            if not representative:
                raise CanonicalEventAdaptationError(
                    "surface_update event requires a non-empty representative identifier."
                )
            payload = {"surface": surface, "representative": representative}
            return (
                "surface_update",
                "trace_replay",
                payload,
                (
                    ASPF_CANONICAL_SOURCE,
                    "trace_replay",
                    "surface_update",
                    f"surface:{surface}",
                    f"representative:{representative}",
                ),
            )
        case AspfRunBoundaryEvent():
            payload = mapping_payload_to_json_object(event.payload)
            digest = payload_sha1_digest(payload)
            boundary = str(event.boundary).strip()
            if not boundary:
                raise CanonicalEventAdaptationError(
                    "run_boundary event requires a non-empty boundary identifier."
                )
            return (
                "run_boundary",
                "run_boundary",
                {"boundary": boundary, "payload": payload},
                (
                    ASPF_CANONICAL_SOURCE,
                    "run_boundary",
                    "run_boundary",
                    f"boundary:{boundary}",
                    f"payload:{digest}",
                ),
            )
        case _:
            never("invalid aspf replay event", kind=type(event).__name__)

__all__ = [
    "ASPF_CANONICAL_SOURCE",
    "adapt_aspf_replay_event",
    "adapt_aspf_replay_event_or_raise",
]
