# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_event_algebra_adapter
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Callable, Mapping
from gabion.json_types import JSONValue

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
from gabion.analysis.foundation.resume_codec import mapping_optional
from gabion.analysis.foundation.timeout_context import check_deadline

DATAFLOW_PHASE_PROGRESS_SOURCE = "dataflow.phase_progress"
DATAFLOW_COLLECTION_PROGRESS_SOURCE = "dataflow.collection_progress"
DataflowIntegerAnchorEncoder = Callable[[str, int], tuple[str, ...]]


@dataclass(frozen=True)
class _NormalizedPhaseProgressEvent:
    phase: str
    kind: str
    payload: Mapping[str, JSONValue]
    event_seq: int | None
    progress_marker: str | None
    primary_unit: str | None
    dimensions_digest: str | None
    root_identity: str | None
    active_nodes: tuple[str, ...]


@dataclass(frozen=True)
class _NormalizedCollectionProgressEvent:
    kind: str
    payload: Mapping[str, JSONValue]
    event_seq: int | None
    index_cache_identity: str | None
    projection_cache_identity: str | None
    analysis_index_resume_digest: str | None
    in_progress_digest: str | None
    completed_paths_digest: str | None


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
        normalized = _normalize_phase_progress_event(payload)
        identity_tokens = _phase_identity_tokens(
            normalized=normalized,
            integer_anchor_encoder=integer_anchor_encoder,
        )
        identity_projection = derive_identity_projection_from_tokens(
            run_context=run_context,
            tokens=identity_tokens,
        )
        envelope = build_canonical_event_envelope(
            run_context=run_context,
            source=DATAFLOW_PHASE_PROGRESS_SOURCE,
            phase=normalized.phase,
            kind=normalized.kind,
            identity_projection=identity_projection,
            payload=normalized.payload,
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
        normalized = _normalize_collection_progress_event(payload)
        identity_tokens = _collection_identity_tokens(
            normalized=normalized,
            integer_anchor_encoder=integer_anchor_encoder,
        )
        identity_projection = derive_identity_projection_from_tokens(
            run_context=run_context,
            tokens=identity_tokens,
        )
        envelope = build_canonical_event_envelope(
            run_context=run_context,
            source=DATAFLOW_COLLECTION_PROGRESS_SOURCE,
            phase="collection",
            kind=normalized.kind,
            identity_projection=identity_projection,
            payload=normalized.payload,
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


def _required_phase(payload: Mapping[str, JSONValue]) -> str:
    check_deadline()
    phase = str(payload.get("phase", "") or "").strip()
    if not phase:
        raise CanonicalEventAdaptationError(
            "dataflow phase_progress event requires non-empty 'phase'."
        )
    return phase


def _phase_kind(payload: Mapping[str, JSONValue]) -> str:
    check_deadline()
    transition_v2 = mapping_optional(payload.get("progress_transition_v2"))
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
    normalized: _NormalizedPhaseProgressEvent,
    integer_anchor_encoder: DataflowIntegerAnchorEncoder | None,
) -> tuple[str, ...]:
    check_deadline()
    components: list[str] = [
        DATAFLOW_PHASE_PROGRESS_SOURCE,
        normalized.phase,
        normalized.kind,
    ]

    if normalized.event_seq is not None:
        components.extend(
            _integer_anchor_tokens(
                key="event_seq",
                value=normalized.event_seq,
                integer_anchor_encoder=integer_anchor_encoder,
            )
        )

    if normalized.progress_marker is not None:
        components.append(f"progress_marker:{normalized.progress_marker}")
    if normalized.primary_unit is not None:
        components.append(f"primary_unit:{normalized.primary_unit}")
    if normalized.dimensions_digest is not None:
        components.append(f"dimensions_digest:{normalized.dimensions_digest}")
    if normalized.root_identity is not None:
        components.append(f"root_identity:{normalized.root_identity}")
    if normalized.active_nodes:
        components.append(f"active_path:{'|'.join(normalized.active_nodes)}")

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


def _collection_kind(payload: Mapping[str, JSONValue]) -> str:
    check_deadline()
    if mapping_optional(payload.get("analysis_index_resume")) is not None:
        return "analysis_index_progress"
    if mapping_optional(payload.get("in_progress_scan_by_path")) is not None:
        return "collection_progress"
    if payload.get("completed_paths") is not None:
        return "collection_progress"
    return "collection_progress"


def _collection_identity_tokens(
    *,
    normalized: _NormalizedCollectionProgressEvent,
    integer_anchor_encoder: DataflowIntegerAnchorEncoder | None,
) -> tuple[str, ...]:
    check_deadline()
    components: list[str] = [
        DATAFLOW_COLLECTION_PROGRESS_SOURCE,
        "collection",
        normalized.kind,
    ]
    if normalized.event_seq is not None:
        components.extend(
            _integer_anchor_tokens(
                key="event_seq",
                value=normalized.event_seq,
                integer_anchor_encoder=integer_anchor_encoder,
            )
        )
    if normalized.index_cache_identity is not None:
        components.append(f"index_cache_identity:{normalized.index_cache_identity}")
    if normalized.projection_cache_identity is not None:
        components.append(
            f"projection_cache_identity:{normalized.projection_cache_identity}"
        )
    if normalized.analysis_index_resume_digest is not None:
        components.append(
            f"analysis_index_resume_digest:{normalized.analysis_index_resume_digest}"
        )
    if normalized.in_progress_digest is not None:
        components.append(f"in_progress_digest:{normalized.in_progress_digest}")
    if normalized.completed_paths_digest is not None:
        components.append(f"completed_paths:{normalized.completed_paths_digest}")

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


def _normalize_phase_progress_event(
    payload: Mapping[str, JSONValue],
) -> _NormalizedPhaseProgressEvent:
    check_deadline()
    phase = _required_phase(payload)
    kind = _phase_kind(payload)
    marker = _nonempty_text(payload.get("progress_marker"))
    phase_progress_v2 = mapping_optional(payload.get("phase_progress_v2"))
    primary_unit = None
    dimensions_digest = None
    if phase_progress_v2 is not None:
        primary_unit = _nonempty_text(phase_progress_v2.get("primary_unit"))
        dimensions = mapping_optional(phase_progress_v2.get("dimensions"))
        if dimensions is not None:
            dimensions_digest = _payload_digest(
                {str(key): dimensions[key] for key in dimensions}
            )
    transition_v2 = mapping_optional(payload.get("progress_transition_v2"))
    root_identity = None
    active_nodes: tuple[str, ...] = ()
    if transition_v2 is not None:
        root = mapping_optional(transition_v2.get("root"))
        if root is not None:
            root_identity = _nonempty_text(root.get("identity"))
        active_nodes = tuple(
            _normalized_nonempty_str_list(transition_v2.get("active_path"))
        )
    return _NormalizedPhaseProgressEvent(
        phase=phase,
        kind=kind,
        payload=payload,
        event_seq=_positive_int_optional(payload.get("event_seq")),
        progress_marker=marker,
        primary_unit=primary_unit,
        dimensions_digest=dimensions_digest,
        root_identity=root_identity,
        active_nodes=active_nodes,
    )


def _normalize_collection_progress_event(
    payload: Mapping[str, JSONValue],
) -> _NormalizedCollectionProgressEvent:
    check_deadline()
    analysis_index_resume = mapping_optional(payload.get("analysis_index_resume"))
    in_progress = mapping_optional(payload.get("in_progress_scan_by_path"))
    completed_paths = _normalized_nonempty_str_list(payload.get("completed_paths"))
    return _NormalizedCollectionProgressEvent(
        kind=_collection_kind(payload),
        payload=payload,
        event_seq=_positive_int_optional(payload.get("event_seq")),
        index_cache_identity=(
            _nonempty_text(analysis_index_resume.get("index_cache_identity"))
            if analysis_index_resume is not None
            else None
        ),
        projection_cache_identity=(
            _nonempty_text(analysis_index_resume.get("projection_cache_identity"))
            if analysis_index_resume is not None
            else None
        ),
        analysis_index_resume_digest=(
            _payload_digest({str(key): analysis_index_resume[key] for key in analysis_index_resume})
            if analysis_index_resume is not None
            else None
        ),
        in_progress_digest=(
            _payload_digest({str(key): in_progress[key] for key in in_progress})
            if in_progress is not None
            else None
        ),
        completed_paths_digest=(
            sha1("|".join(completed_paths).encode("utf-8")).hexdigest()
            if completed_paths
            else None
        ),
    )


def _positive_int_optional(value: JSONValue | None) -> int | None:
    check_deadline()
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _normalized_nonempty_str_list(value: JSONValue | None) -> list[str]:
    check_deadline()
    if isinstance(value, list) and value:
        normalized_items: list[str] = []
        for item in value:
            check_deadline()
            if not isinstance(item, str):
                return []
            normalized_item = item.strip()
            if not normalized_item:
                return []
            normalized_items.append(normalized_item)
        if len(normalized_items) == len(value):
            return normalized_items
        return []
    return []


def _nonempty_text(value: JSONValue | None) -> str | None:
    check_deadline()
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if normalized:
        return normalized
    return None


def _payload_digest(payload: Mapping[str, JSONValue]) -> str:
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
