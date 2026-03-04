from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Mapping, TypeAlias

from gabion.analysis.foundation.event_algebra_adapter_utils import (
    mapping_payload_to_json_object,
    payload_sha1_digest,
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
from gabion.analysis.foundation.json_types import JSONObject


@dataclass(frozen=True)
class NodeDiscovered:
    node_id: str
    module_path: str
    label: str = ""


@dataclass(frozen=True)
class EdgeFormed:
    src_node_id: str
    dst_node_id: str
    relation: str


@dataclass(frozen=True)
class ComponentSealed:
    component_id: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class StreamTerminated:
    reason: str
    total_events: int


@dataclass(frozen=True)
class NameInterned:
    namespace: str
    token: str
    atom_id: int


TranscriptFixtureEvent: TypeAlias = (
    NodeDiscovered
    | EdgeFormed
    | ComponentSealed
    | StreamTerminated
    | NameInterned
)

TRANSCRIPT_FIXTURE_SOURCE = "transcript.scout"
TRANSCRIPT_FIXTURE_PHASE = "scout"


def adapt_transcript_fixture_event(
    *,
    event: TranscriptFixtureEvent,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...] = (),
) -> CanonicalAdaptationDecision:
    try:
        kind, payload, tokens = _fixture_payload(event)
        projection = derive_identity_projection_from_tokens(
            run_context=run_context,
            tokens=tokens,
        )
        envelope = build_canonical_event_envelope(
            run_context=run_context,
            source=TRANSCRIPT_FIXTURE_SOURCE,
            phase=TRANSCRIPT_FIXTURE_PHASE,
            kind=kind,
            identity_projection=projection,
            payload=payload,
            causal_refs=causal_refs,
        )
        return canonical_adaptation_valid(envelope)
    except (CanonicalEventAdaptationError, ValueError) as exc:
        return canonical_adaptation_rejected(str(exc))


def adapt_transcript_fixture_event_or_raise(
    *,
    event: TranscriptFixtureEvent,
    run_context: CanonicalRunContext,
    causal_refs: tuple[str, ...] = (),
) -> CanonicalEventEnvelope:
    return envelope_from_decision_or_raise(
        adapt_transcript_fixture_event(
            event=event,
            run_context=run_context,
            causal_refs=causal_refs,
        )
    )


def _fixture_payload(
    event: TranscriptFixtureEvent,
) -> tuple[str, JSONObject, tuple[str, ...]]:
    match event:
        case NodeDiscovered():
            node_id = event.node_id.strip()
            module_path = event.module_path.strip()
            if not node_id or not module_path:
                raise CanonicalEventAdaptationError(
                    "NodeDiscovered requires non-empty node_id and module_path."
                )
            payload = mapping_payload_to_json_object(
                {
                "node_id": node_id,
                "module_path": module_path,
                "label": str(event.label).strip(),
                }
            )
            return (
                "node_discovered",
                payload,
                (
                    TRANSCRIPT_FIXTURE_SOURCE,
                    "node_discovered",
                    f"node_id:{node_id}",
                    f"module:{module_path}",
                ),
            )
        case EdgeFormed():
            src_node = event.src_node_id.strip()
            dst_node = event.dst_node_id.strip()
            relation = event.relation.strip()
            if not src_node or not dst_node or not relation:
                raise CanonicalEventAdaptationError(
                    "EdgeFormed requires non-empty src/dst/relation."
                )
            payload = mapping_payload_to_json_object(
                {
                "src_node_id": src_node,
                "dst_node_id": dst_node,
                "relation": relation,
                }
            )
            return (
                "edge_formed",
                payload,
                (
                    TRANSCRIPT_FIXTURE_SOURCE,
                    "edge_formed",
                    f"src:{src_node}",
                    f"dst:{dst_node}",
                    f"relation:{relation}",
                ),
            )
        case ComponentSealed():
            component_id = event.component_id.strip()
            members = tuple(str(member).strip() for member in event.members if str(member).strip())
            if not component_id or not members:
                raise CanonicalEventAdaptationError(
                    "ComponentSealed requires non-empty component_id and members."
                )
            payload = mapping_payload_to_json_object(
                {"component_id": component_id, "members": list(members)}
            )
            members_hash = sha1("|".join(members).encode("utf-8")).hexdigest()
            return (
                "component_sealed",
                payload,
                (
                    TRANSCRIPT_FIXTURE_SOURCE,
                    "component_sealed",
                    f"component:{component_id}",
                    f"members:{members_hash}",
                ),
            )
        case StreamTerminated():
            reason = event.reason.strip()
            if not reason:
                raise CanonicalEventAdaptationError(
                    "StreamTerminated requires a non-empty reason."
                )
            payload = mapping_payload_to_json_object(
                {
                "reason": reason,
                "total_events": int(event.total_events),
                }
            )
            return (
                "stream_terminated",
                payload,
                (
                    TRANSCRIPT_FIXTURE_SOURCE,
                    "stream_terminated",
                    f"reason:{reason}",
                    f"total_events:{int(event.total_events)}",
                ),
            )
        case NameInterned():
            namespace = event.namespace.strip()
            token = event.token.strip()
            atom_id = int(event.atom_id)
            if not namespace or not token or atom_id <= 0:
                raise CanonicalEventAdaptationError(
                    "NameInterned requires namespace/token and positive atom_id."
                )
            payload = mapping_payload_to_json_object(
                {
                "namespace": namespace,
                "token": token,
                "atom_id": atom_id,
                }
            )
            digest = payload_sha1_digest(payload)
            return (
                "name_interned",
                payload,
                (
                    TRANSCRIPT_FIXTURE_SOURCE,
                    "name_interned",
                    f"namespace:{namespace}",
                    f"digest:{digest}",
                ),
            )
        case _:
            raise CanonicalEventAdaptationError(
                f"Unsupported transcript fixture event type: {type(event).__name__}"
            )
