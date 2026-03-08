# gabion:ambiguity_boundary_module
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import json
import threading
from typing import Mapping, cast
from gabion.json_types import JSONObject, JSONValue

from gabion.analysis.foundation.identity_space import (
    GlobalIdentitySpace,
    IdentityNamespace,
    IdentityPath,
    IdentityProjection,
)
from gabion.analysis.foundation.json_types import JSONObject, JSONValue
from gabion.analysis.foundation.resume_codec import mapping_or_none, sequence_or_none
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never
from gabion.runtime import stable_encode

CANONICAL_EVENT_SCHEMA_VERSION = 1
EVENT_IDENTITY_NAMESPACE = str(IdentityNamespace.PATH)


class CanonicalAdaptationKind(StrEnum):
    VALID = "valid"
    REJECTED = "rejected"


class CanonicalEventAdaptationError(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalEventEnvelope:
    schema_version: int
    sequence: int
    run_id: str
    source: str
    phase: str
    kind: str
    identity_projection: IdentityProjection
    payload: JSONObject
    causal_refs: tuple[str, ...]
    event_id: str


@dataclass
class GlobalEventSequencer:
    _next_sequence: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def next_sequence(self) -> int:
        check_deadline()
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
        return sequence


@dataclass(frozen=True)
class CanonicalRunContext:
    run_id: str
    sequencer: GlobalEventSequencer
    identity_space: GlobalIdentitySpace


@dataclass(frozen=True)
class CanonicalAdaptationDecision:
    kind: CanonicalAdaptationKind
    envelope: object = None
    reason: str = ""


def canonical_event_id(*, run_id: str, sequence: int) -> str:
    check_deadline()
    run_id_text = str(run_id).strip()
    if not run_id_text:
        raise CanonicalEventAdaptationError("run_id must be non-empty.")
    if sequence <= 0:
        raise CanonicalEventAdaptationError("sequence must be positive.")
    return f"{run_id_text}:{sequence}"


def canonical_adaptation_valid(
    envelope: CanonicalEventEnvelope,
) -> CanonicalAdaptationDecision:
    check_deadline()
    return CanonicalAdaptationDecision(
        kind=CanonicalAdaptationKind.VALID,
        envelope=envelope,
    )


def canonical_adaptation_rejected(reason: str) -> CanonicalAdaptationDecision:
    check_deadline()
    return CanonicalAdaptationDecision(
        kind=CanonicalAdaptationKind.REJECTED,
        reason=str(reason),
    )


def envelope_from_decision_or_raise(
    decision: CanonicalAdaptationDecision,
) -> CanonicalEventEnvelope:
    check_deadline()
    match decision.kind:
        case CanonicalAdaptationKind.VALID:
            match decision.envelope:
                case CanonicalEventEnvelope() as envelope:
                    return envelope
                case _:
                    raise CanonicalEventAdaptationError(
                        "valid adaptation decision missing canonical envelope."
                    )
        case CanonicalAdaptationKind.REJECTED:
            raise CanonicalEventAdaptationError(
                str(decision.reason) or "canonical adaptation rejected."
            )
        case _:
            never("invalid canonical adaptation decision kind", kind=decision.kind)


def derive_identity_projection_from_tokens(
    *,
    run_context: CanonicalRunContext,
    tokens: tuple[str, ...],
    namespace: str = EVENT_IDENTITY_NAMESPACE,
) -> IdentityProjection:
    check_deadline()
    normalized_tokens: list[str] = []
    for token in tokens:
        check_deadline()
        token_text = str(token).strip()
        if not token_text:
            raise CanonicalEventAdaptationError(
                "identity derivation requires non-empty deterministic tokens."
            )
        normalized_tokens.append(token_text)
    if not normalized_tokens:
        raise CanonicalEventAdaptationError(
            "identity derivation requires at least one token."
        )
    path = run_context.identity_space.intern_path(
        namespace=namespace,
        tokens=tuple(normalized_tokens),
    )
    return run_context.identity_space.project(path=path)


def build_canonical_event_envelope(
    *,
    run_context: CanonicalRunContext,
    source: str,
    phase: str,
    kind: str,
    identity_projection: IdentityProjection,
    payload: Mapping[str, JSONValue],
    causal_refs: tuple[str, ...] = (),
) -> CanonicalEventEnvelope:
    check_deadline()
    source_text = str(source).strip()
    phase_text = str(phase).strip()
    kind_text = str(kind).strip()
    if not source_text:
        raise CanonicalEventAdaptationError("canonical envelope source must be non-empty.")
    if not phase_text:
        raise CanonicalEventAdaptationError("canonical envelope phase must be non-empty.")
    if not kind_text:
        raise CanonicalEventAdaptationError("canonical envelope kind must be non-empty.")
    sequence = run_context.sequencer.next_sequence()
    event_id = canonical_event_id(run_id=run_context.run_id, sequence=sequence)
    normalized_causal_refs = tuple(
        ref_text
        for ref_text in (str(item).strip() for item in causal_refs)
        if ref_text
    )
    normalized_payload: JSONObject = {
        str(key): payload[key] for key in payload
    }
    return CanonicalEventEnvelope(
        schema_version=CANONICAL_EVENT_SCHEMA_VERSION,
        sequence=sequence,
        run_id=str(run_context.run_id),
        source=source_text,
        phase=phase_text,
        kind=kind_text,
        identity_projection=identity_projection,
        payload=normalized_payload,
        causal_refs=normalized_causal_refs,
        event_id=event_id,
    )


def canonical_event_to_json_object(envelope: CanonicalEventEnvelope) -> JSONObject:
    check_deadline()
    return {
        "schema_version": int(envelope.schema_version),
        "sequence": int(envelope.sequence),
        "run_id": str(envelope.run_id),
        "source": str(envelope.source),
        "phase": str(envelope.phase),
        "kind": str(envelope.kind),
        "identity_projection": _identity_projection_to_json_object(
            envelope.identity_projection
        ),
        "payload": dict(envelope.payload),
        "causal_refs": list(envelope.causal_refs),
        "event_id": str(envelope.event_id),
    }


def encode_canonical_event_json(envelope: CanonicalEventEnvelope) -> str:
    check_deadline()
    return stable_encode.stable_compact_text(canonical_event_to_json_object(envelope))


def decode_canonical_event_json(raw: str) -> CanonicalEventEnvelope:
    check_deadline()
    try:
        loaded = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise CanonicalEventAdaptationError("invalid canonical event json payload.") from exc
    match loaded:
        case dict() as loaded_map:
            payload = {str(key): loaded_map[key] for key in loaded_map}
        case _:
            raise CanonicalEventAdaptationError(
                "canonical event json payload must decode to an object."
            )
    schema_version = _required_non_negative_int(payload, "schema_version")
    if schema_version != CANONICAL_EVENT_SCHEMA_VERSION:
        raise CanonicalEventAdaptationError(
            f"unsupported canonical event schema_version: {schema_version}"
        )
    sequence = _required_positive_int(payload, "sequence")
    run_id = _required_non_empty_text(payload, "run_id")
    source = _required_non_empty_text(payload, "source")
    phase = _required_non_empty_text(payload, "phase")
    kind = _required_non_empty_text(payload, "kind")
    identity_projection = _identity_projection_from_json_object(
        _required_mapping(payload, "identity_projection")
    )
    event_payload = _required_mapping(payload, "payload")
    causal_refs = _required_text_sequence(payload, "causal_refs")
    event_id = _required_non_empty_text(payload, "event_id")
    expected_event_id = canonical_event_id(run_id=run_id, sequence=sequence)
    if event_id != expected_event_id:
        raise CanonicalEventAdaptationError(
            f"canonical event id mismatch: expected {expected_event_id}, got {event_id}"
        )
    return CanonicalEventEnvelope(
        schema_version=schema_version,
        sequence=sequence,
        run_id=run_id,
        source=source,
        phase=phase,
        kind=kind,
        identity_projection=identity_projection,
        payload=cast(JSONObject, event_payload),
        causal_refs=causal_refs,
        event_id=event_id,
    )


def _identity_projection_to_json_object(projection: IdentityProjection) -> JSONObject:
    check_deadline()
    return {
        "basis_path": {
            "namespace": projection.basis_path.namespace,
            "atoms": list(projection.basis_path.atoms),
        },
        "prime_product": projection.prime_product,
        "digest_alias": projection.digest_alias,
        "witness": dict(projection.witness),
    }


def _identity_projection_from_json_object(
    payload: Mapping[str, JSONValue],
) -> IdentityProjection:
    check_deadline()
    basis_path_payload = _required_mapping(payload, "basis_path")
    namespace = _required_non_empty_text(basis_path_payload, "namespace")
    atoms_raw = sequence_or_none(basis_path_payload.get("atoms"))
    match atoms_raw:
        case list() as atoms_list:
            normalized_atoms: list[int] = []
            for value in atoms_list:
                check_deadline()
                match value:
                    case bool():
                        raise CanonicalEventAdaptationError(
                            "identity basis path atoms must be positive integers."
                        )
                    case int() as atom if atom > 0:
                        normalized_atoms.append(atom)
                    case _:
                        raise CanonicalEventAdaptationError(
                            "identity basis path atoms must be positive integers."
                        )
        case _:
            raise CanonicalEventAdaptationError(
                "identity basis path atoms must be a sequence."
            )
    if not normalized_atoms:
        raise CanonicalEventAdaptationError(
            "identity basis path atoms must be non-empty."
        )
    prime_product = _required_positive_int(payload, "prime_product")
    digest_alias = _required_non_empty_text(payload, "digest_alias")
    witness = _required_mapping(payload, "witness")
    return IdentityProjection(
        basis_path=IdentityPath(namespace=namespace, atoms=tuple(normalized_atoms)),
        prime_product=prime_product,
        digest_alias=digest_alias,
        witness={str(key): witness[key] for key in witness},
    )


def _required_mapping(payload: Mapping[str, JSONValue], key: str) -> JSONObject:
    check_deadline()
    mapping = mapping_or_none(payload.get(key))
    match mapping:
        case dict() as mapping_value:
            return cast(JSONObject, {str(item): mapping_value[item] for item in mapping_value})
        case _:
            raise CanonicalEventAdaptationError(
                f"canonical event field '{key}' must be an object."
            )


def _required_non_empty_text(payload: Mapping[str, JSONValue], key: str) -> str:
    check_deadline()
    value = payload.get(key)
    match value:
        case str() as text:
            normalized = text.strip()
            if normalized:
                return normalized
            raise CanonicalEventAdaptationError(
                f"canonical event field '{key}' must be non-empty text."
            )
        case _:
            raise CanonicalEventAdaptationError(
                f"canonical event field '{key}' must be text."
            )


def _required_non_negative_int(payload: Mapping[str, JSONValue], key: str) -> int:
    check_deadline()
    value = payload.get(key)
    match value:
        case bool():
            raise CanonicalEventAdaptationError(
                f"canonical event field '{key}' must be a non-negative integer."
            )
        case int() as parsed if parsed >= 0:
            return parsed
        case _:
            raise CanonicalEventAdaptationError(
                f"canonical event field '{key}' must be a non-negative integer."
            )


def _required_positive_int(payload: Mapping[str, JSONValue], key: str) -> int:
    check_deadline()
    value = payload.get(key)
    match value:
        case bool():
            raise CanonicalEventAdaptationError(
                f"canonical event field '{key}' must be a positive integer."
            )
        case int() as parsed if parsed > 0:
            return parsed
        case _:
            raise CanonicalEventAdaptationError(
                f"canonical event field '{key}' must be a positive integer."
            )


def _required_text_sequence(payload: Mapping[str, JSONValue], key: str) -> tuple[str, ...]:
    check_deadline()
    raw = sequence_or_none(payload.get(key))
    match raw:
        case list() as items:
            refs: list[str] = []
            for item in items:
                check_deadline()
                match item:
                    case str() as text:
                        normalized = text.strip()
                        if not normalized:
                            raise CanonicalEventAdaptationError(
                                f"canonical event field '{key}' may not contain empty refs."
                            )
                        refs.append(normalized)
                    case _:
                        raise CanonicalEventAdaptationError(
                            f"canonical event field '{key}' must contain text refs."
                        )
            return tuple(refs)
        case _:
            raise CanonicalEventAdaptationError(
                f"canonical event field '{key}' must be a sequence."
            )


__all__ = [
    "CANONICAL_EVENT_SCHEMA_VERSION",
    "EVENT_IDENTITY_NAMESPACE",
    "CanonicalAdaptationDecision",
    "CanonicalAdaptationKind",
    "CanonicalEventAdaptationError",
    "CanonicalEventEnvelope",
    "CanonicalRunContext",
    "GlobalEventSequencer",
    "build_canonical_event_envelope",
    "canonical_adaptation_rejected",
    "canonical_adaptation_valid",
    "canonical_event_id",
    "canonical_event_to_json_object",
    "decode_canonical_event_json",
    "derive_identity_projection_from_tokens",
    "encode_canonical_event_json",
    "envelope_from_decision_or_raise",
]
