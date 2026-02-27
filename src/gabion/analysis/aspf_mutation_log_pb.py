"""Generated protobuf wire encoders/decoders for ASPF mutation log envelopes.

This file is generated from the in-38 archive projection contract schema.
Do not hand-edit.
"""

from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class PbMutationRecord:
    op_id: str
    op_kind: str
    payload_json: str


@dataclass(frozen=True)
class PbMutationSnapshot:
    seq: int
    state_json: str


@dataclass(frozen=True)
class PbEventEnvelope:
    schema_version: int
    sequence: int
    run_id: str
    record: PbMutationRecord


@dataclass(frozen=True)
class PbSnapshotEnvelope:
    schema_version: int
    run_id: str
    replay_cursor: int
    snapshot: PbMutationSnapshot


@dataclass(frozen=True)
class PbArchiveManifest:
    schema_version: int
    projection_version: int
    run_id: str
    event_sequences: tuple[int, ...]
    snapshot_sequences: tuple[int, ...]


@dataclass(frozen=True)
class PbCommitMarker:
    schema_version: int
    run_id: str
    last_durable_sequence: int


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    current = value
    while current > 0x7F:
        out.append((current & 0x7F) | 0x80)
        current >>= 7
    out.append(current)
    return bytes(out)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    index = offset
    while True:
        if index >= len(data):
            raise ValueError("truncated varint")
        byte = data[index]
        result |= (byte & 0x7F) << shift
        index += 1
        if byte < 0x80:
            return result, index
        shift += 7
        if shift >= 64:
            raise ValueError("varint too large")


def _key(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_uint64(field_number: int, value: int) -> bytes:
    return _key(field_number, 0) + _encode_varint(value)


def _encode_string(field_number: int, value: str) -> bytes:
    raw = value.encode("utf-8")
    return _key(field_number, 2) + _encode_varint(len(raw)) + raw


def _encode_message(field_number: int, payload: bytes) -> bytes:
    return _key(field_number, 2) + _encode_varint(len(payload)) + payload


def _read_length_delimited(data: bytes, offset: int) -> tuple[bytes, int]:
    size, index = _decode_varint(data, offset)
    end = index + size
    if end > len(data):
        raise ValueError("truncated length-delimited field")
    return data[index:end], end


def _skip(data: bytes, offset: int, wire_type: int) -> int:
    if wire_type == 0:
        _, index = _decode_varint(data, offset)
        return index
    if wire_type == 2:
        _, index = _read_length_delimited(data, offset)
        return index
    raise ValueError(f"unsupported wire type: {wire_type}")


def serialize_mutation_record(record: PbMutationRecord) -> bytes:
    return b"".join(
        [
            _encode_string(1, record.op_id),
            _encode_string(2, record.op_kind),
            _encode_string(3, record.payload_json),
        ]
    )


def parse_mutation_record(data: bytes) -> PbMutationRecord:
    op_id = ""
    op_kind = ""
    payload_json = json.dumps({}, sort_keys=True, separators=(",", ":"))
    index = 0
    while index < len(data):
        key, index = _decode_varint(data, index)
        field = key >> 3
        wire_type = key & 0b111
        if field in {1, 2, 3} and wire_type == 2:
            raw, index = _read_length_delimited(data, index)
            text = raw.decode("utf-8")
            if field == 1:
                op_id = text
            elif field == 2:
                op_kind = text
            else:
                payload_json = text
            continue
        index = _skip(data, index, wire_type)
    return PbMutationRecord(op_id=op_id, op_kind=op_kind, payload_json=payload_json)


def serialize_mutation_snapshot(snapshot: PbMutationSnapshot) -> bytes:
    return b"".join([_encode_uint64(1, snapshot.seq), _encode_string(2, snapshot.state_json)])


def parse_mutation_snapshot(data: bytes) -> PbMutationSnapshot:
    seq = 0
    state_json = json.dumps({}, sort_keys=True, separators=(",", ":"))
    index = 0
    while index < len(data):
        key, index = _decode_varint(data, index)
        field = key >> 3
        wire_type = key & 0b111
        if field == 1 and wire_type == 0:
            seq, index = _decode_varint(data, index)
            continue
        if field == 2 and wire_type == 2:
            raw, index = _read_length_delimited(data, index)
            state_json = raw.decode("utf-8")
            continue
        index = _skip(data, index, wire_type)
    return PbMutationSnapshot(seq=seq, state_json=state_json)


def serialize_event_envelope(envelope: PbEventEnvelope) -> bytes:
    return b"".join(
        [
            _encode_uint64(1, envelope.schema_version),
            _encode_uint64(2, envelope.sequence),
            _encode_string(3, envelope.run_id),
            _encode_message(4, serialize_mutation_record(envelope.record)),
        ]
    )


def parse_event_envelope(data: bytes) -> PbEventEnvelope:
    schema_version = 0
    sequence = 0
    run_id = ""
    record = PbMutationRecord(op_id="", op_kind="", payload_json=json.dumps({}, sort_keys=True, separators=(",", ":")))
    index = 0
    while index < len(data):
        key, index = _decode_varint(data, index)
        field = key >> 3
        wire_type = key & 0b111
        if field in {1, 2} and wire_type == 0:
            value, index = _decode_varint(data, index)
            if field == 1:
                schema_version = value
            else:
                sequence = value
            continue
        if field == 3 and wire_type == 2:
            raw, index = _read_length_delimited(data, index)
            run_id = raw.decode("utf-8")
            continue
        if field == 4 and wire_type == 2:
            raw, index = _read_length_delimited(data, index)
            record = parse_mutation_record(raw)
            continue
        index = _skip(data, index, wire_type)
    return PbEventEnvelope(
        schema_version=schema_version,
        sequence=sequence,
        run_id=run_id,
        record=record,
    )


def serialize_snapshot_envelope(envelope: PbSnapshotEnvelope) -> bytes:
    return b"".join(
        [
            _encode_uint64(1, envelope.schema_version),
            _encode_string(2, envelope.run_id),
            _encode_uint64(3, envelope.replay_cursor),
            _encode_message(4, serialize_mutation_snapshot(envelope.snapshot)),
        ]
    )


def parse_snapshot_envelope(data: bytes) -> PbSnapshotEnvelope:
    schema_version = 0
    run_id = ""
    replay_cursor = 0
    snapshot = PbMutationSnapshot(seq=0, state_json=json.dumps({}, sort_keys=True, separators=(",", ":")))
    index = 0
    while index < len(data):
        key, index = _decode_varint(data, index)
        field = key >> 3
        wire_type = key & 0b111
        if field in {1, 3} and wire_type == 0:
            value, index = _decode_varint(data, index)
            if field == 1:
                schema_version = value
            else:
                replay_cursor = value
            continue
        if field == 2 and wire_type == 2:
            raw, index = _read_length_delimited(data, index)
            run_id = raw.decode("utf-8")
            continue
        if field == 4 and wire_type == 2:
            raw, index = _read_length_delimited(data, index)
            snapshot = parse_mutation_snapshot(raw)
            continue
        index = _skip(data, index, wire_type)
    return PbSnapshotEnvelope(
        schema_version=schema_version,
        run_id=run_id,
        replay_cursor=replay_cursor,
        snapshot=snapshot,
    )


def serialize_archive_manifest(manifest: PbArchiveManifest) -> bytes:
    payload = [
        _encode_uint64(1, manifest.schema_version),
        _encode_uint64(2, manifest.projection_version),
        _encode_string(3, manifest.run_id),
    ]
    payload.extend(_encode_uint64(4, seq) for seq in manifest.event_sequences)
    payload.extend(_encode_uint64(5, seq) for seq in manifest.snapshot_sequences)
    return b"".join(payload)


def parse_archive_manifest(data: bytes) -> PbArchiveManifest:
    schema_version = 0
    projection_version = 0
    run_id = ""
    event_sequences: list[int] = []
    snapshot_sequences: list[int] = []
    index = 0
    while index < len(data):
        key, index = _decode_varint(data, index)
        field = key >> 3
        wire_type = key & 0b111
        if field in {1, 2, 4, 5} and wire_type == 0:
            value, index = _decode_varint(data, index)
            if field == 1:
                schema_version = value
            elif field == 2:
                projection_version = value
            elif field == 4:
                event_sequences.append(value)
            else:
                snapshot_sequences.append(value)
            continue
        if field == 3 and wire_type == 2:
            raw, index = _read_length_delimited(data, index)
            run_id = raw.decode("utf-8")
            continue
        index = _skip(data, index, wire_type)
    return PbArchiveManifest(
        schema_version=schema_version,
        projection_version=projection_version,
        run_id=run_id,
        event_sequences=tuple(event_sequences),
        snapshot_sequences=tuple(snapshot_sequences),
    )


def serialize_commit_marker(commit: PbCommitMarker) -> bytes:
    return b"".join(
        [
            _encode_uint64(1, commit.schema_version),
            _encode_string(2, commit.run_id),
            _encode_uint64(3, commit.last_durable_sequence),
        ]
    )


def parse_commit_marker(data: bytes) -> PbCommitMarker:
    schema_version = 0
    run_id = ""
    last_durable_sequence = 0
    index = 0
    while index < len(data):
        key, index = _decode_varint(data, index)
        field = key >> 3
        wire_type = key & 0b111
        if field in {1, 3} and wire_type == 0:
            value, index = _decode_varint(data, index)
            if field == 1:
                schema_version = value
            else:
                last_durable_sequence = value
            continue
        if field == 2 and wire_type == 2:
            raw, index = _read_length_delimited(data, index)
            run_id = raw.decode("utf-8")
            continue
        index = _skip(data, index, wire_type)
    return PbCommitMarker(
        schema_version=schema_version,
        run_id=run_id,
        last_durable_sequence=last_durable_sequence,
    )
