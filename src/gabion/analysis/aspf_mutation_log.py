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
from gabion.analysis.aspf_mutation_log_pb import (
    PbArchiveManifest,
    PbCommitMarker,
    PbEventEnvelope,
    PbMutationRecord,
    PbMutationSnapshot,
    PbSnapshotEnvelope,
    parse_archive_manifest,
    parse_commit_marker,
    parse_event_envelope,
    parse_snapshot_envelope,
    serialize_archive_manifest,
    serialize_commit_marker,
    serialize_event_envelope,
    serialize_snapshot_envelope,
)
from gabion.analysis.resume_codec import sequence_or_none


CURRENT_ARCHIVE_SCHEMA_VERSION = 2
LEGACY_ARCHIVE_SCHEMA_VERSION = 1


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
    schema_version: int = CURRENT_ARCHIVE_SCHEMA_VERSION


@dataclass(frozen=True)
class SnapshotEnvelope:
    run_id: str
    replay_cursor: int
    snapshot: AspfMutationSnapshot
    schema_version: int = CURRENT_ARCHIVE_SCHEMA_VERSION


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
    schema_version: int = CURRENT_ARCHIVE_SCHEMA_VERSION


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


def _canonical_json_bytes(payload: Mapping[str, JSONValue]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _event_envelope_payload(envelope: EventEnvelope) -> JSONObject:
    return {
        "schema_version": envelope.schema_version,
        "sequence": envelope.sequence,
        "run_id": envelope.run_id,
        "record": {
            "op_id": envelope.record.op_id,
            "op_kind": envelope.record.op_kind,
            "payload": envelope.record.payload,
        },
    }


def _snapshot_envelope_payload(envelope: SnapshotEnvelope) -> JSONObject:
    return {
        "schema_version": envelope.schema_version,
        "run_id": envelope.run_id,
        "replay_cursor": envelope.replay_cursor,
        "snapshot": {
            "seq": envelope.snapshot.seq,
            "state": envelope.snapshot.state,
        },
    }


def _manifest_payload(manifest: ArchiveManifest) -> JSONObject:
    return {
        "schema_version": manifest.schema_version,
        "projection_version": manifest.projection_version,
        "run_id": manifest.run_id,
        "event_sequences": list(manifest.event_sequences),
        "snapshot_sequences": list(manifest.snapshot_sequences),
    }


def _commit_payload(commit: CommitMarker) -> JSONObject:
    return {
        "schema_version": commit.schema_version,
        "run_id": commit.run_id,
        "last_durable_sequence": commit.last_durable_sequence,
    }


def _protobuf_event_bytes(envelope: EventEnvelope) -> bytes:
    return serialize_event_envelope(
        PbEventEnvelope(
            schema_version=envelope.schema_version,
            sequence=envelope.sequence,
            run_id=envelope.run_id,
            record=PbMutationRecord(
                op_id=envelope.record.op_id,
                op_kind=envelope.record.op_kind,
                payload_json=_canonical_json_bytes(envelope.record.payload).decode("utf-8"),
            ),
        )
    )


def _protobuf_snapshot_bytes(envelope: SnapshotEnvelope) -> bytes:
    return serialize_snapshot_envelope(
        PbSnapshotEnvelope(
            schema_version=envelope.schema_version,
            run_id=envelope.run_id,
            replay_cursor=envelope.replay_cursor,
            snapshot=PbMutationSnapshot(
                seq=envelope.snapshot.seq,
                state_json=_canonical_json_bytes(envelope.snapshot.state).decode("utf-8"),
            ),
        )
    )


def _protobuf_manifest_bytes(manifest: ArchiveManifest) -> bytes:
    return serialize_archive_manifest(
        PbArchiveManifest(
            schema_version=manifest.schema_version,
            projection_version=manifest.projection_version,
            run_id=manifest.run_id,
            event_sequences=manifest.event_sequences,
            snapshot_sequences=manifest.snapshot_sequences,
        )
    )


def _protobuf_commit_bytes(commit: CommitMarker) -> bytes:
    return serialize_commit_marker(
        PbCommitMarker(
            schema_version=commit.schema_version,
            run_id=commit.run_id,
            last_durable_sequence=commit.last_durable_sequence,
        )
    )


def decode_manifest_compat(raw: bytes) -> ArchiveManifest:
    if raw.startswith(b"{"):
        payload = json.loads(raw.decode("utf-8"))
        return ArchiveManifest(
            schema_version=int(payload.get("schema_version", LEGACY_ARCHIVE_SCHEMA_VERSION)),
            projection_version=int(payload.get("projection_version", 1)),
            run_id=str(payload.get("run_id", "")),
            event_sequences=tuple(int(item) for item in sequence_or_none(payload.get("event_sequences"), allow_str=False) or []),
            snapshot_sequences=tuple(int(item) for item in sequence_or_none(payload.get("snapshot_sequences"), allow_str=False) or []),
        )
    parsed = parse_archive_manifest(raw)
    return ArchiveManifest(
        schema_version=parsed.schema_version,
        projection_version=parsed.projection_version,
        run_id=parsed.run_id,
        event_sequences=parsed.event_sequences,
        snapshot_sequences=parsed.snapshot_sequences,
    )


def decode_event_compat(raw: bytes) -> EventEnvelope:
    if raw.startswith(b"{"):
        payload = json.loads(raw.decode("utf-8"))
        record = payload.get("record", {})
        return EventEnvelope(
            schema_version=int(payload.get("schema_version", LEGACY_ARCHIVE_SCHEMA_VERSION)),
            sequence=int(payload.get("sequence", 0)),
            run_id=str(payload.get("run_id", "")),
            record=AspfMutationRecord(
                op_id=str(record.get("op_id", "")),
                op_kind=str(record.get("op_kind", "")),
                payload=dict(record.get("payload", {})),
            ),
        )
    parsed = parse_event_envelope(raw)
    return EventEnvelope(
        schema_version=parsed.schema_version,
        sequence=parsed.sequence,
        run_id=parsed.run_id,
        record=AspfMutationRecord(
            op_id=parsed.record.op_id,
            op_kind=parsed.record.op_kind,
            payload=json.loads(parsed.record.payload_json),
        ),
    )


def decode_commit_compat(raw: bytes) -> CommitMarker:
    if raw.startswith(b"{"):
        payload = json.loads(raw.decode("utf-8"))
        return CommitMarker(
            schema_version=int(payload.get("schema_version", LEGACY_ARCHIVE_SCHEMA_VERSION)),
            run_id=str(payload.get("run_id", "")),
            last_durable_sequence=int(payload.get("last_durable_sequence", 0)),
        )
    parsed = parse_commit_marker(raw)
    return CommitMarker(
        schema_version=parsed.schema_version,
        run_id=parsed.run_id,
        last_durable_sequence=parsed.last_durable_sequence,
    )


def decode_snapshot_compat(raw: bytes) -> SnapshotEnvelope:
    if raw.startswith(b"{"):
        payload = json.loads(raw.decode("utf-8"))
        snapshot = payload.get("snapshot", {})
        return SnapshotEnvelope(
            schema_version=int(payload.get("schema_version", LEGACY_ARCHIVE_SCHEMA_VERSION)),
            run_id=str(payload.get("run_id", "")),
            replay_cursor=int(payload.get("replay_cursor", 0)),
            snapshot=AspfMutationSnapshot(
                seq=int(snapshot.get("seq", 0)),
                state=dict(snapshot.get("state", {})),
            ),
        )
    parsed = parse_snapshot_envelope(raw)
    return SnapshotEnvelope(
        schema_version=parsed.schema_version,
        run_id=parsed.run_id,
        replay_cursor=parsed.replay_cursor,
        snapshot=AspfMutationSnapshot(seq=parsed.snapshot.seq, state=json.loads(parsed.snapshot.state_json)),
    )


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

    (manifest_dir / "manifest.pb").write_bytes(_protobuf_manifest_bytes(manifest))

    for envelope in sorted(events, key=lambda item: item.sequence):
        stem = f"{envelope.sequence:012d}"
        encoded = _protobuf_event_bytes(envelope)
        (events_dir / f"{stem}.data.pb").write_bytes(encoded)
        (events_dir / f"{stem}.crc").write_text(sha256(encoded).hexdigest(), encoding="utf-8")

    for envelope in sorted(snapshots, key=lambda item: item.snapshot.seq):
        stem = f"{envelope.snapshot.seq:012d}"
        encoded = _protobuf_snapshot_bytes(envelope)
        (snapshots_dir / f"{stem}.data.pb").write_bytes(encoded)

    (commit_dir / "commit.pb").write_bytes(_protobuf_commit_bytes(commit))


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
    archive_replay = replay_tail(snapshot.snapshot, replay_records)
    json_replay = replay_tail(snapshot.snapshot, replay_records)
    equivalent = json.dumps(archive_replay, sort_keys=True) == json.dumps(json_replay, sort_keys=True)
    return SnapshotTailReplayResult(
        state=archive_replay,
        ignored_tail_count=ignored_tail_count,
        equivalent_to_json_replay=equivalent,
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
