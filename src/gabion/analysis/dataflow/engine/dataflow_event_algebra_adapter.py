# gabion:boundary_normalization_module
# gabion:decision_protocol_module
# gabion:ambiguity_boundary_module
from __future__ import annotations

from hashlib import sha1
from typing import Callable, Mapping

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
from gabion.analysis.foundation.resume_codec import mapping_or_none
from gabion.analysis.foundation.timeout_context import check_deadline

DATAFLOW_PHASE_PROGRESS_SOURCE = "dataflow.phase_progress"
DATAFLOW_COLLECTION_PROGRESS_SOURCE = "dataflow.collection_progress"
DataflowIntegerAnchorEncoder = Callable[[str, int], tuple[str, ...]]


def adapt_dataflow_phase_progress_event(
    *,
    phase_progress: Mapping[str, object],
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...] = (),
    integer_anchor_encoder: DataflowIntegerAnchorEncoder | None = None,
) -> CanonicalAdaptationDecision:
    check_deadline()
    try:
        payload = mapping_payload_to_json_object(phase_progress)
        phase = _required_phase(payload)
        kind = _phase_kind(payload)
        identity_tokens = _phase_identity_tokens(
            payload=payload,
            phase=phase,
            kind=kind,
            integer_anchor_encoder=integer_anchor_encoder,
        )
        identity_projection = derive_identity_projection_from_tokens(
            run_context=run_context,
            tokens=identity_tokens,
        )
        envelope = build_canonical_event_envelope(
            run_context=run_context,
            source=DATAFLOW_PHASE_PROGRESS_SOURCE,
            phase=phase,
            kind=kind,
            identity_projection=identity_projection,
            payload=payload,
            causal_refs=causal_refs,
        )
        return canonical_adaptation_valid(envelope)
    except (CanonicalEventAdaptationError, ValueError) as exc:
        return canonical_adaptation_rejected(str(exc))


def adapt_dataflow_phase_progress_event_or_raise(
    *,
    phase_progress: Mapping[str, object],
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...] = (),
    integer_anchor_encoder: DataflowIntegerAnchorEncoder | None = None,
) -> CanonicalEventEnvelope:
    check_deadline()
    return envelope_from_decision_or_raise(
        adapt_dataflow_phase_progress_event(
            phase_progress=phase_progress,
            run_context=run_context,
            causal_refs=causal_refs,
            integer_anchor_encoder=integer_anchor_encoder,
        )
    )


def adapt_dataflow_collection_progress_event(
    *,
    collection_progress: Mapping[str, object],
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...] = (),
    integer_anchor_encoder: DataflowIntegerAnchorEncoder | None = None,
) -> CanonicalAdaptationDecision:
    check_deadline()
    try:
        payload = mapping_payload_to_json_object(collection_progress)
        phase = "collection"
        kind = _collection_kind(payload)
        identity_tokens = _collection_identity_tokens(
            payload=payload,
            kind=kind,
            integer_anchor_encoder=integer_anchor_encoder,
        )
        identity_projection = derive_identity_projection_from_tokens(
            run_context=run_context,
            tokens=identity_tokens,
        )
        envelope = build_canonical_event_envelope(
            run_context=run_context,
            source=DATAFLOW_COLLECTION_PROGRESS_SOURCE,
            phase=phase,
            kind=kind,
            identity_projection=identity_projection,
            payload=payload,
            causal_refs=causal_refs,
        )
        return canonical_adaptation_valid(envelope)
    except (CanonicalEventAdaptationError, ValueError) as exc:
        return canonical_adaptation_rejected(str(exc))


def adapt_dataflow_collection_progress_event_or_raise(
    *,
    collection_progress: Mapping[str, object],
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...] = (),
    integer_anchor_encoder: DataflowIntegerAnchorEncoder | None = None,
) -> CanonicalEventEnvelope:
    check_deadline()
    return envelope_from_decision_or_raise(
        adapt_dataflow_collection_progress_event(
            collection_progress=collection_progress,
            run_context=run_context,
            causal_refs=causal_refs,
            integer_anchor_encoder=integer_anchor_encoder,
        )
    )


def _required_phase(payload: Mapping[str, object]) -> str:
    check_deadline()
    phase = str(payload.get("phase", "") or "").strip()
    if not phase:
        raise CanonicalEventAdaptationError(
            "dataflow phase_progress event requires non-empty 'phase'."
        )
    return phase


def _phase_kind(payload: Mapping[str, object]) -> str:
    check_deadline()
    transition_v2 = mapping_or_none(payload.get("progress_transition_v2"))
    for candidate in (
        payload.get("event_kind"),
        (transition_v2 or {}).get("event_kind"),
    ):
        check_deadline()
        text = str(candidate or "").strip()
        if text:
            return text
    marker = str(payload.get("progress_marker", "") or "").strip()
    if marker:
        return "progress"
    raise CanonicalEventAdaptationError(
        "dataflow phase_progress event requires determinable event kind."
    )


def _phase_identity_tokens(
    *,
    payload: Mapping[str, object],
    phase: str,
    kind: str,
    integer_anchor_encoder: DataflowIntegerAnchorEncoder | None,
) -> tuple[str, ...]:
    check_deadline()
    components: list[str] = [DATAFLOW_PHASE_PROGRESS_SOURCE, phase, kind]

    event_seq = _positive_int_or_none(payload.get("event_seq"))
    if event_seq is not None:
        components.extend(
            _integer_anchor_tokens(
                key="event_seq",
                value=event_seq,
                integer_anchor_encoder=integer_anchor_encoder,
            )
        )

    marker = str(payload.get("progress_marker", "") or "").strip()
    if marker:
        components.append(f"progress_marker:{marker}")

    phase_progress_v2 = mapping_or_none(payload.get("phase_progress_v2"))
    if phase_progress_v2 is not None:
        primary_unit = str(phase_progress_v2.get("primary_unit", "") or "").strip()
        if primary_unit:
            components.append(f"primary_unit:{primary_unit}")
        dimensions = mapping_or_none(phase_progress_v2.get("dimensions"))
        if dimensions is not None:
            components.append(
                f"dimensions_digest:{_payload_digest({str(key): dimensions[key] for key in dimensions})}"
            )

    transition_v2 = mapping_or_none(payload.get("progress_transition_v2"))
    if transition_v2 is not None:
        root = mapping_or_none(transition_v2.get("root"))
        if root is not None:
            root_identity = str(root.get("identity", "") or "").strip()
            if root_identity:
                components.append(f"root_identity:{root_identity}")
        active_path_value = transition_v2.get("active_path")
        if isinstance(active_path_value, list) and active_path_value:
            active_nodes = [
                str(node).strip()
                for node in active_path_value
                if isinstance(node, str) and str(node).strip()
            ]
            if active_nodes and len(active_nodes) == len(active_path_value):
                components.append(f"active_path:{'|'.join(active_nodes)}")

    has_identity_anchor = any(
        token.startswith(
            (
                "event_seq:",
                "progress_marker:",
                "root_identity:",
                "active_path:",
                "primary_unit:",
            )
        )
        for token in components
    )
    if not has_identity_anchor:
        raise CanonicalEventAdaptationError(
            "dataflow phase_progress event missing deterministic identity anchor "
            "(event_seq/root identity/marker)."
        )
    return tuple(components)


def _collection_kind(payload: Mapping[str, object]) -> str:
    check_deadline()
    if mapping_or_none(payload.get("analysis_index_resume")) is not None:
        return "analysis_index_progress"
    if mapping_or_none(payload.get("in_progress_scan_by_path")) is not None:
        return "collection_progress"
    if payload.get("completed_paths") is not None:
        return "collection_progress"
    return "collection_progress"


def _collection_identity_tokens(
    *,
    payload: Mapping[str, object],
    kind: str,
    integer_anchor_encoder: DataflowIntegerAnchorEncoder | None,
) -> tuple[str, ...]:
    check_deadline()
    components: list[str] = [
        DATAFLOW_COLLECTION_PROGRESS_SOURCE,
        "collection",
        kind,
    ]
    event_seq = _positive_int_or_none(payload.get("event_seq"))
    if event_seq is not None:
        components.extend(
            _integer_anchor_tokens(
                key="event_seq",
                value=event_seq,
                integer_anchor_encoder=integer_anchor_encoder,
            )
        )

    analysis_index_resume = mapping_or_none(payload.get("analysis_index_resume"))
    if analysis_index_resume is not None:
        index_cache_identity = str(
            analysis_index_resume.get("index_cache_identity", "") or ""
        ).strip()
        projection_cache_identity = str(
            analysis_index_resume.get("projection_cache_identity", "") or ""
        ).strip()
        if index_cache_identity:
            components.append(f"index_cache_identity:{index_cache_identity}")
        if projection_cache_identity:
            components.append(f"projection_cache_identity:{projection_cache_identity}")
        components.append(
            f"analysis_index_resume_digest:{_payload_digest({str(key): analysis_index_resume[key] for key in analysis_index_resume})}"
        )

    in_progress = mapping_or_none(payload.get("in_progress_scan_by_path"))
    if in_progress is not None:
        components.append(
            f"in_progress_digest:{_payload_digest({str(key): in_progress[key] for key in in_progress})}"
        )

    completed_paths = payload.get("completed_paths")
    if isinstance(completed_paths, list):
        normalized_paths = [
            str(path).strip()
            for path in completed_paths
            if isinstance(path, str) and str(path).strip()
        ]
        if normalized_paths and len(normalized_paths) == len(completed_paths):
            components.append(
                f"completed_paths:{sha1('|'.join(normalized_paths).encode('utf-8')).hexdigest()}"
            )

    has_identity_anchor = any(
        token.startswith(
            (
                "event_seq:",
                "index_cache_identity:",
                "projection_cache_identity:",
                "analysis_index_resume_digest:",
                "in_progress_digest:",
                "completed_paths:",
            )
        )
        for token in components
    )
    if not has_identity_anchor:
        raise CanonicalEventAdaptationError(
            "dataflow collection_progress event missing deterministic identity anchor."
        )
    return tuple(components)


def _positive_int_or_none(value: object) -> int | None:
    check_deadline()
    if type(value) is int and value > 0:
        return int(value)
    return None


def _payload_digest(payload: Mapping[str, object]) -> str:
    return payload_sha1_digest(payload)


def _integer_anchor_tokens(
    *,
    key: str,
    value: int,
    integer_anchor_encoder: DataflowIntegerAnchorEncoder | None,
) -> tuple[str, ...]:
    check_deadline()
    key_text = str(key).strip()
    if not key_text:
        raise CanonicalEventAdaptationError(
            "integer anchor key must be non-empty."
        )
    if integer_anchor_encoder is None:
        return (f"{key_text}:{int(value)}",)
    encoded_values = integer_anchor_encoder(key_text, int(value))
    if not encoded_values:
        raise CanonicalEventAdaptationError(
            f"integer anchor encoder returned empty token tuple for '{key_text}'."
        )
    tokens: list[str] = []
    for raw_value in encoded_values:
        check_deadline()
        encoded_text = str(raw_value).strip()
        if not encoded_text:
            raise CanonicalEventAdaptationError(
                f"integer anchor encoder returned empty token for '{key_text}'."
            )
        tokens.append(f"{key_text}:{encoded_text}")
    return tuple(tokens)


__all__ = [
    "DataflowIntegerAnchorEncoder",
    "DATAFLOW_COLLECTION_PROGRESS_SOURCE",
    "DATAFLOW_PHASE_PROGRESS_SOURCE",
    "adapt_dataflow_collection_progress_event",
    "adapt_dataflow_collection_progress_event_or_raise",
    "adapt_dataflow_phase_progress_event",
    "adapt_dataflow_phase_progress_event_or_raise",
]
