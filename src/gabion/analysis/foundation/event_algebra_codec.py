# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
import json
from typing import Iterator, Mapping

from gabion.analysis.foundation.event_algebra import (
    CanonicalEventEnvelope,
    decode_canonical_event_json,
)
from gabion.analysis.foundation.identity_space import IdentityProjection
from gabion.analysis.foundation.json_types import JSONObject, JSONValue
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.runtime import stable_encode


class CanonicalEventProtoDecodeError(ValueError):
    pass


@dataclass(frozen=True)
class _ProtobufWireFields:
    varints: dict[int, list[int]]
    bytes_fields: dict[int, list[bytes]]


@dataclass(frozen=True)
class _VarintDecode:
    value: int
    cursor: int


@dataclass(frozen=True)
class _IdentityProjectionPayload:
    namespace: str
    atoms: list[int]
    prime_product: int
    digest_alias: str
    witness: JSONObject


def _encode_varint(value: int) -> bytes:
    check_deadline()
    if value < 0:
        raise ValueError("varint cannot encode negative values")
    pieces = bytearray()
    pending = value
    while True:
        chunk = pending & 0x7F
        pending >>= 7
        if pending:
            pieces.append(chunk | 0x80)
        else:
            pieces.append(chunk)
            break
    return bytes(pieces)


def _decode_varint(buffer: bytes, offset: int) -> _VarintDecode:
    check_deadline()
    shift = 0
    value = 0
    cursor = offset
    while cursor < len(buffer):
        byte = buffer[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return _VarintDecode(value=value, cursor=cursor)
        shift += 7
        if shift > 63:
            break
    raise CanonicalEventProtoDecodeError("invalid varint payload")


def _encode_length_delimited(field_number: int, payload: bytes) -> bytes:
    check_deadline()
    tag = (field_number << 3) | 2
    return _encode_varint(tag) + _encode_varint(len(payload)) + payload


def _encode_uint64(field_number: int, value: int) -> bytes:
    check_deadline()
    tag = field_number << 3
    return _encode_varint(tag) + _encode_varint(value)


def _parse_wire_fields(payload: bytes) -> _ProtobufWireFields:
    check_deadline()
    varints: dict[int, list[int]] = {}
    bytes_fields: dict[int, list[bytes]] = {}
    offset = 0
    while offset < len(payload):
        decoded_tag = _decode_varint(payload, offset)
        tag = decoded_tag.value
        offset = decoded_tag.cursor
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:
            decoded_value = _decode_varint(payload, offset)
            value = decoded_value.value
            offset = decoded_value.cursor
            varints.setdefault(field_number, []).append(value)
            continue
        if wire_type == 2:
            decoded_size = _decode_varint(payload, offset)
            size = decoded_size.value
            offset = decoded_size.cursor
            end = offset + size
            if end > len(payload):
                raise CanonicalEventProtoDecodeError(
                    "declared field length exceeds payload"
                )
            bytes_fields.setdefault(field_number, []).append(payload[offset:end])
            offset = end
            continue
        raise CanonicalEventProtoDecodeError(f"unsupported wire type: {wire_type}")
    return _ProtobufWireFields(varints=varints, bytes_fields=bytes_fields)


def _required_single_varint(
    fields: _ProtobufWireFields,
    *,
    field_number: int,
    field_name: str,
) -> int:
    check_deadline()
    values = fields.varints.get(field_number)
    if values is None or len(values) != 1:
        raise CanonicalEventProtoDecodeError(
            f"canonical event proto requires exactly one '{field_name}' field"
        )
    return int(values[0])


def _required_single_bytes(
    fields: _ProtobufWireFields,
    *,
    field_number: int,
    field_name: str,
) -> bytes:
    check_deadline()
    values = fields.bytes_fields.get(field_number)
    if values is None or len(values) != 1:
        raise CanonicalEventProtoDecodeError(
            f"canonical event proto requires exactly one '{field_name}' field"
        )
    return values[0]


def _repeated_bytes(
    fields: _ProtobufWireFields,
    *,
    field_number: int,
) -> Iterator[bytes]:
    check_deadline()
    return iter(fields.bytes_fields.get(field_number, ()))


def _decode_utf8_text(payload: bytes, *, field_name: str) -> str:
    check_deadline()
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanonicalEventProtoDecodeError(
            f"canonical event proto field '{field_name}' has invalid utf-8 encoding"
        ) from exc
    return decoded


def _decode_json_object_entries(payload: bytes, *, field_name: str) -> Iterator[tuple[str, JSONValue]]:
    check_deadline()
    text = _decode_utf8_text(payload, field_name=field_name)
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CanonicalEventProtoDecodeError(
            f"canonical event proto field '{field_name}' is not valid json"
        ) from exc
    match loaded:
        case dict() as loaded_map:
            return map(
                lambda item: (str(item[0]), item[1]),
                loaded_map.items(),
            )
        case _:
            raise CanonicalEventProtoDecodeError(
                f"canonical event proto field '{field_name}' must decode to an object"
            )


def _encode_identity_projection_proto(identity: IdentityProjection) -> bytes:
    check_deadline()
    witness_json = stable_encode.stable_compact_text(identity.witness).encode("utf-8")
    prefix_fields = (
        _encode_length_delimited(1, identity.basis_path.namespace.encode("utf-8")),
    )
    atom_fields = map(
        lambda atom: _encode_uint64(2, int(atom)),
        identity.basis_path.atoms,
    )
    suffix_fields = (
        _encode_uint64(3, int(identity.prime_product)),
        _encode_length_delimited(4, identity.digest_alias.encode("utf-8")),
        _encode_length_delimited(5, witness_json),
    )
    return b"".join(chain(prefix_fields, atom_fields, suffix_fields))


def _decode_identity_projection_proto(payload: bytes) -> _IdentityProjectionPayload:
    check_deadline()
    fields = _parse_wire_fields(payload)
    namespace = _decode_utf8_text(
        _required_single_bytes(
            fields,
            field_number=1,
            field_name="identity_projection.namespace",
        ),
        field_name="identity_projection.namespace",
    )
    atoms = list(map(int, fields.varints.get(2, ())))
    if not atoms:
        raise CanonicalEventProtoDecodeError(
            "canonical event proto identity_projection.atoms must be non-empty"
        )
    prime_product = _required_single_varint(
        fields,
        field_number=3,
        field_name="identity_projection.prime_product",
    )
    digest_alias = _decode_utf8_text(
        _required_single_bytes(
            fields,
            field_number=4,
            field_name="identity_projection.digest_alias",
        ),
        field_name="identity_projection.digest_alias",
    )
    witness = dict(
        _decode_json_object_entries(
            _required_single_bytes(
                fields,
                field_number=5,
                field_name="identity_projection.witness_json",
            ),
            field_name="identity_projection.witness_json",
        )
    )
    return _IdentityProjectionPayload(
        namespace=namespace,
        atoms=atoms,
        prime_product=prime_product,
        digest_alias=digest_alias,
        witness=witness,
    )


def encode_canonical_event_proto(envelope: CanonicalEventEnvelope) -> bytes:
    check_deadline()
    payload_json = stable_encode.stable_compact_text(envelope.payload).encode("utf-8")
    identity_payload = _encode_identity_projection_proto(envelope.identity_projection)
    prefix_fields = (
        _encode_uint64(1, int(envelope.schema_version)),
        _encode_uint64(2, int(envelope.sequence)),
        _encode_length_delimited(3, envelope.run_id.encode("utf-8")),
        _encode_length_delimited(4, envelope.source.encode("utf-8")),
        _encode_length_delimited(5, envelope.phase.encode("utf-8")),
        _encode_length_delimited(6, envelope.kind.encode("utf-8")),
        _encode_length_delimited(7, identity_payload),
        _encode_length_delimited(8, payload_json),
    )
    causal_fields = map(
        lambda causal_ref: _encode_length_delimited(9, causal_ref.encode("utf-8")),
        envelope.causal_refs,
    )
    suffix_fields = (
        _encode_length_delimited(10, envelope.event_id.encode("utf-8")),
    )
    return b"".join(chain(prefix_fields, causal_fields, suffix_fields))


def decode_canonical_event_proto(payload: bytes) -> CanonicalEventEnvelope:
    check_deadline()
    fields = _parse_wire_fields(payload)
    schema_version = _required_single_varint(
        fields,
        field_number=1,
        field_name="schema_version",
    )
    sequence = _required_single_varint(fields, field_number=2, field_name="sequence")
    run_id = _decode_utf8_text(
        _required_single_bytes(fields, field_number=3, field_name="run_id"),
        field_name="run_id",
    )
    source = _decode_utf8_text(
        _required_single_bytes(fields, field_number=4, field_name="source"),
        field_name="source",
    )
    phase = _decode_utf8_text(
        _required_single_bytes(fields, field_number=5, field_name="phase"),
        field_name="phase",
    )
    kind = _decode_utf8_text(
        _required_single_bytes(fields, field_number=6, field_name="kind"),
        field_name="kind",
    )
    decoded_identity_projection = _decode_identity_projection_proto(
        _required_single_bytes(
            fields,
            field_number=7,
            field_name="identity_projection",
        )
    )
    payload_json = dict(
        _decode_json_object_entries(
            _required_single_bytes(fields, field_number=8, field_name="payload_json"),
            field_name="payload_json",
        )
    )
    causal_refs = tuple(
        map(
            lambda raw: _decode_utf8_text(raw, field_name="causal_refs"),
            _repeated_bytes(fields, field_number=9),
        )
    )
    event_id = _decode_utf8_text(
        _required_single_bytes(fields, field_number=10, field_name="event_id"),
        field_name="event_id",
    )
    json_payload: Mapping[str, object] = {
        "schema_version": int(schema_version),
        "sequence": int(sequence),
        "run_id": run_id,
        "source": source,
        "phase": phase,
        "kind": kind,
        "identity_projection": {
            "basis_path": {
                "namespace": decoded_identity_projection.namespace,
                "atoms": decoded_identity_projection.atoms,
            },
            "prime_product": decoded_identity_projection.prime_product,
            "digest_alias": decoded_identity_projection.digest_alias,
            "witness": decoded_identity_projection.witness,
        },
        "payload": payload_json,
        "causal_refs": list(causal_refs),
        "event_id": event_id,
    }
    return decode_canonical_event_json(stable_encode.stable_compact_text(json_payload))


__all__ = [
    "CanonicalEventProtoDecodeError",
    "decode_canonical_event_proto",
    "encode_canonical_event_proto",
]
