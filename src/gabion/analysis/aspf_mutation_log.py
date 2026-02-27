# gabion:decision_protocol_module
# gabion:boundary_normalization_module
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile
from typing import Mapping

from gabion.analysis.json_types import JSONObject, JSONValue
from gabion.analysis.resume_codec import mapping_or_none, sequence_or_none


@dataclass(frozen=True)
class AspfMutationRecord:
    op_id: str
    op_kind: str
    payload: JSONObject


@dataclass(frozen=True)
class AspfMutationSnapshot:
    seq: int
    state: JSONObject


@dataclass(frozen=True)
class AspfShadowReplayResult:
    equivalent: bool
    expected: JSONObject
    replayed: JSONObject
    tail_length: int


@dataclass(frozen=True)
class EventEnvelope:
    sequence: int
    run_id: str
    record: AspfMutationRecord


@dataclass(frozen=True)
class SnapshotEnvelope:
    run_id: str
    replay_cursor: int
    snapshot: AspfMutationSnapshot


@dataclass(frozen=True)
class ArchiveManifest:
    schema_version: int
    projection_version: int
    run_id: str
    event_sequences: tuple[int, ...]
    snapshot_sequences: tuple[int, ...]


@dataclass(frozen=True)
class CommitMarker:
    run_id: str
    last_durable_sequence: int


@dataclass(frozen=True)
class SnapshotTailReplayResult:
    state: JSONObject
    ignored_tail_count: int
    equivalent_to_json_replay: bool


@dataclass(frozen=True)
class ShadowWriteParityResult:
    enabled: bool
    equivalent: bool
    archive_replay: JSONObject
    json_replay: JSONObject


class ProtobufDecodeError(ValueError):
    pass


@dataclass(frozen=True)
class ProtobufWireFields:
    varints: dict[int, int]
    bytes_fields: dict[int, bytes]


def _canonical_json_bytes(payload: Mapping[str, JSONValue]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_varint(value: int) -> bytes:
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


def _decode_varint(buffer: bytes, offset: int) -> tuple[int, int]:
    shift = 0
    value = 0
    cursor = offset
    while cursor < len(buffer):
        byte = buffer[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, cursor
        shift += 7
        if shift > 63:
            break
    raise ProtobufDecodeError("invalid varint payload")


def _encode_length_delimited(field_number: int, payload: bytes) -> bytes:
    tag = (field_number << 3) | 2
    return _encode_varint(tag) + _encode_varint(len(payload)) + payload


def _encode_uint64(field_number: int, value: int) -> bytes:
    tag = field_number << 3
    return _encode_varint(tag) + _encode_varint(value)


def _parse_wire_fields(payload: bytes) -> ProtobufWireFields:
    varints: dict[int, int] = {}
    bytes_fields: dict[int, bytes] = {}
    offset = 0
    while offset < len(payload):
        tag, offset = _decode_varint(payload, offset)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:
            value, offset = _decode_varint(payload, offset)
            varints[field_number] = value
            continue
        if wire_type == 2:
            size, offset = _decode_varint(payload, offset)
            end = offset + size
            if end > len(payload):
                raise ProtobufDecodeError("declared field length exceeds payload")
            bytes_fields[field_number] = payload[offset:end]
            offset = end
            continue
        raise ProtobufDecodeError(f"unsupported wire type: {wire_type}")
    return ProtobufWireFields(varints=varints, bytes_fields=bytes_fields)


def _json_bytes_to_object(payload: bytes) -> JSONObject:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtobufDecodeError("invalid json payload") from exc
    match value:
        case dict() as value_map:
            return value_map
        case _:
            raise ProtobufDecodeError("json payload must decode to an object")


def encode_event_envelope_proto(envelope: EventEnvelope) -> bytes:
    record_payload = _canonical_json_bytes(
        {
            "op_id": envelope.record.op_id,
            "op_kind": envelope.record.op_kind,
            "payload": envelope.record.payload,
        }
    )
    return b"".join(
        (
            _encode_uint64(1, envelope.sequence),
            _encode_length_delimited(2, envelope.run_id.encode("utf-8")),
            _encode_length_delimited(3, record_payload),
        )
    )


def decode_event_envelope_proto(payload: bytes) -> EventEnvelope:
    fields = _parse_wire_fields(payload)
    sequence = int(fields.varints.get(1, 0) or 0)
    run_id_field = fields.bytes_fields.get(2)
    record_field = fields.bytes_fields.get(3)
    match (run_id_field, record_field):
        case (bytes(), bytes()):
            pass
        case _:
            raise ProtobufDecodeError("event envelope missing required fields")
    try:
        run_id = run_id_field.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtobufDecodeError("invalid run_id encoding") from exc
    record_payload = _json_bytes_to_object(record_field)
    return EventEnvelope(
        sequence=sequence,
        run_id=run_id,
        record=AspfMutationRecord(
            op_id=str(record_payload.get("op_id", "") or ""),
            op_kind=str(record_payload.get("op_kind", "") or ""),
            payload=(
                dict(payload_map)
                if (payload_map := mapping_or_none(record_payload.get("payload"))) is not None
                else {}
            ),
        ),
    )


def encode_snapshot_envelope_proto(envelope: SnapshotEnvelope) -> bytes:
    snapshot_payload = _canonical_json_bytes(
        {
            "seq": envelope.snapshot.seq,
            "state": envelope.snapshot.state,
        }
    )
    return b"".join(
        (
            _encode_length_delimited(1, envelope.run_id.encode("utf-8")),
            _encode_uint64(2, envelope.replay_cursor),
            _encode_length_delimited(3, snapshot_payload),
        )
    )


def decode_snapshot_envelope_proto(payload: bytes) -> SnapshotEnvelope:
    fields = _parse_wire_fields(payload)
    run_id_field = fields.bytes_fields.get(1)
    replay_cursor = int(fields.varints.get(2, 0) or 0)
    snapshot_field = fields.bytes_fields.get(3)
    match (run_id_field, snapshot_field):
        case (bytes(), bytes()):
            pass
        case _:
            raise ProtobufDecodeError("snapshot envelope missing required fields")
    try:
        run_id = run_id_field.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtobufDecodeError("invalid run_id encoding") from exc
    snapshot_payload = _json_bytes_to_object(snapshot_field)
    return SnapshotEnvelope(
        run_id=run_id,
        replay_cursor=replay_cursor,
        snapshot=AspfMutationSnapshot(
            seq=int(snapshot_payload.get("seq", 0) or 0),
            state=(
                dict(state_map)
                if (state_map := mapping_or_none(snapshot_payload.get("state"))) is not None
                else {}
            ),
        ),
    )


def encode_archive_manifest_proto(manifest: ArchiveManifest) -> bytes:
    return _encode_length_delimited(1, _canonical_json_bytes(_manifest_payload(manifest)))


def decode_archive_manifest_proto(payload: bytes) -> ArchiveManifest:
    fields = _parse_wire_fields(payload)
    body = fields.bytes_fields.get(1)
    match body:
        case bytes():
            pass
        case _:
            raise ProtobufDecodeError("manifest envelope missing payload")
    data = _json_bytes_to_object(body)
    return ArchiveManifest(
        schema_version=int(data.get("schema_version", 0) or 0),
        projection_version=int(data.get("projection_version", 0) or 0),
        run_id=str(data.get("run_id", "") or ""),
        event_sequences=tuple(int(value) for value in list(data.get("event_sequences", []))),
        snapshot_sequences=tuple(int(value) for value in list(data.get("snapshot_sequences", []))),
    )


def encode_commit_marker_proto(commit: CommitMarker) -> bytes:
    return b"".join(
        (
            _encode_length_delimited(1, commit.run_id.encode("utf-8")),
            _encode_uint64(2, commit.last_durable_sequence),
        )
    )


def decode_commit_marker_proto(payload: bytes) -> CommitMarker:
    fields = _parse_wire_fields(payload)
    run_id_field = fields.bytes_fields.get(1)
    if run_id_field is None:
        raise ProtobufDecodeError("commit marker missing run_id")
    try:
        run_id = run_id_field.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtobufDecodeError("invalid run_id encoding") from exc
    return CommitMarker(
        run_id=run_id,
        last_durable_sequence=int(fields.varints.get(2, 0) or 0),
    )


def _manifest_payload(manifest: ArchiveManifest) -> JSONObject:
    return {
        "schema_version": manifest.schema_version,
        "projection_version": manifest.projection_version,
        "run_id": manifest.run_id,
        "event_sequences": list(manifest.event_sequences),
        "snapshot_sequences": list(manifest.snapshot_sequences),
    }


def project_archive_filesystem(
    *,
    root_dir: Path,
    manifest: ArchiveManifest,
    events: list[EventEnvelope],
    snapshots: list[SnapshotEnvelope],
    commit: CommitMarker,
) -> None:
    manifest_dir = root_dir / "001_manifest"
    events_dir = root_dir / "010_events"
    snapshots_dir = root_dir / "020_snapshots"
    commit_dir = root_dir / "099_commit"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    commit_dir.mkdir(parents=True, exist_ok=True)

    (manifest_dir / "manifest.pb").write_bytes(encode_archive_manifest_proto(manifest))

    for envelope in sorted(events, key=lambda item: item.sequence):
        stem = f"{envelope.sequence:012d}"
        encoded = encode_event_envelope_proto(envelope)
        (events_dir / f"{stem}.data.pb").write_bytes(encoded)
        (events_dir / f"{stem}.crc").write_text(sha256(encoded).hexdigest(), encoding="utf-8")

    for envelope in sorted(snapshots, key=lambda item: item.snapshot.seq):
        stem = f"{envelope.snapshot.seq:012d}"
        encoded = encode_snapshot_envelope_proto(envelope)
        (snapshots_dir / f"{stem}.data.pb").write_bytes(encoded)

    (commit_dir / "commit.pb").write_bytes(encode_commit_marker_proto(commit))


def load_projected_archive(
    *, root_dir: Path
) -> tuple[ArchiveManifest, list[EventEnvelope], list[SnapshotEnvelope], CommitMarker]:
    manifest = decode_archive_manifest_proto((root_dir / "001_manifest" / "manifest.pb").read_bytes())
    commit = decode_commit_marker_proto((root_dir / "099_commit" / "commit.pb").read_bytes())

    events: list[EventEnvelope] = []
    for entry in sorted(
        (root_dir / "010_events").glob("*.data.pb"), key=lambda path: path.name
    ):
        encoded = entry.read_bytes()
        crc = (entry.with_suffix("").with_suffix(".crc")).read_text(encoding="utf-8").strip()
        if sha256(encoded).hexdigest() != crc:
            raise ProtobufDecodeError(f"event checksum mismatch: {entry.name}")
        events.append(decode_event_envelope_proto(encoded))

    snapshots: list[SnapshotEnvelope] = []
    for entry in sorted(
        (root_dir / "020_snapshots").glob("*.data.pb"), key=lambda path: path.name
    ):
        snapshots.append(decode_snapshot_envelope_proto(entry.read_bytes()))

    return manifest, events, snapshots, commit


def package_archive_tar(*, root_dir: Path, tar_path: Path) -> None:
    entries: list[Path] = []
    for entry in root_dir.rglob("*"):
        if entry == tar_path:
            continue
        entries.append(entry)
    entries.sort(key=lambda path: path.relative_to(root_dir).as_posix())

    with tarfile.open(tar_path, "w") as archive:
        for entry in entries:
            rel = entry.relative_to(root_dir).as_posix()
            tar_info = archive.gettarinfo(str(entry), arcname=rel)
            tar_info.uid = 0
            tar_info.gid = 0
            tar_info.uname = ""
            tar_info.gname = ""
            tar_info.mtime = 0
            if entry.is_dir():
                tar_info.mode = 0o755
                archive.addfile(tar_info)
                continue
            tar_info.mode = 0o644
            archive.addfile(tar_info, BytesIO(entry.read_bytes()))


def _record_from_event_envelope(event: EventEnvelope) -> AspfMutationRecord:
    return AspfMutationRecord(
        op_id=event.record.op_id,
        op_kind=event.record.op_kind,
        payload=dict(event.record.payload),
    )


def replay_from_snapshot_and_committed_tail(
    *,
    snapshot: SnapshotEnvelope,
    events: list[EventEnvelope],
    commit: CommitMarker,
) -> SnapshotTailReplayResult:
    committed_tail = [
        event
        for event in sorted(events, key=lambda item: item.sequence)
        if snapshot.replay_cursor < event.sequence <= commit.last_durable_sequence
    ]
    ignored_tail_count = len(
        [event for event in events if event.sequence > commit.last_durable_sequence]
    )
    replay_records = [_record_from_event_envelope(event) for event in committed_tail]
    archive_records = [
        decode_event_envelope_proto(encode_event_envelope_proto(event)).record
        for event in committed_tail
    ]
    archive_replay = replay_tail(snapshot.snapshot, archive_records)
    json_replay = replay_tail(snapshot.snapshot, replay_records)
    equivalent = json.dumps(archive_replay, sort_keys=True) == json.dumps(json_replay, sort_keys=True)
    return SnapshotTailReplayResult(
        state=archive_replay,
        ignored_tail_count=ignored_tail_count,
        equivalent_to_json_replay=equivalent,
    )


def replay_from_projected_archive(*, root_dir: Path) -> SnapshotTailReplayResult:
    _manifest, events, snapshots, commit = load_projected_archive(root_dir=root_dir)
    if not snapshots:
        raise ProtobufDecodeError("archive does not contain a snapshot")
    latest_snapshot = max(snapshots, key=lambda envelope: envelope.snapshot.seq)
    return replay_from_snapshot_and_committed_tail(
        snapshot=latest_snapshot,
        events=events,
        commit=commit,
    )


def shadow_write_parity(
    *,
    enabled: bool,
    snapshot: SnapshotEnvelope,
    events: list[EventEnvelope],
    commit: CommitMarker,
) -> ShadowWriteParityResult:
    replay_result = replay_from_snapshot_and_committed_tail(
        snapshot=snapshot,
        events=events,
        commit=commit,
    )
    json_tail = [
        _record_from_event_envelope(event)
        for event in sorted(events, key=lambda item: item.sequence)
        if snapshot.replay_cursor < event.sequence <= commit.last_durable_sequence
    ]
    json_replay = replay_tail(snapshot.snapshot, json_tail)
    equivalent = json.dumps(replay_result.state, sort_keys=True) == json.dumps(json_replay, sort_keys=True)
    return ShadowWriteParityResult(
        enabled=enabled,
        equivalent=(equivalent if enabled else True),
        archive_replay=replay_result.state,
        json_replay=json_replay,
    )


def replay_state_hash(*, replay_state: Mapping[str, JSONValue]) -> str:
    return sha256(_canonical_json_bytes(dict(replay_state))).hexdigest()


def apply_mutation(state: JSONObject, record: AspfMutationRecord) -> JSONObject:
    next_state: JSONObject = dict(state)
    if record.op_kind == "set":
        key = str(record.payload.get("key", ""))
        if key:
            next_state[key] = record.payload.get("value")
    elif record.op_kind == "delete":
        key = str(record.payload.get("key", ""))
        if key:
            next_state.pop(key, None)
    else:
        unknown_ops = sequence_or_none(next_state.get("_unknown_ops"), allow_str=False)
        ops = list(unknown_ops) if unknown_ops is not None else []
        ops.append(record.op_kind)
        next_state["_unknown_ops"] = ops
    return next_state


def snapshot_state(state: Mapping[str, JSONValue], *, seq: int) -> AspfMutationSnapshot:
    return AspfMutationSnapshot(seq=seq, state=dict(state))


def replay_tail(snapshot: AspfMutationSnapshot, tail: list[AspfMutationRecord]) -> JSONObject:
    state = dict(snapshot.state)
    for record in tail:
        state = apply_mutation(state, record)
    return state


def shadow_replay_equivalence(
    *,
    live_state: Mapping[str, JSONValue],
    snapshot: AspfMutationSnapshot,
    tail: list[AspfMutationRecord],
) -> AspfShadowReplayResult:
    replayed = replay_tail(snapshot, tail)
    expected = dict(live_state)
    equivalent = json.dumps(expected, sort_keys=True) == json.dumps(replayed, sort_keys=True)
    return AspfShadowReplayResult(
        equivalent=equivalent,
        expected=expected,
        replayed=replayed,
        tail_length=len(tail),
    )
