from __future__ import annotations

from pathlib import Path

import pytest

from gabion.analysis import aspf_mutation_log
from gabion.analysis.aspf_mutation_log import (
    ArchiveManifest,
    AspfMutationRecord,
    CommitMarker,
    EventEnvelope,
    ProtobufDecodeError,
    SnapshotEnvelope,
    apply_mutation,
    load_projected_archive,
    package_archive_tar,
    project_archive_filesystem,
    replay_from_projected_archive,
    replay_from_snapshot_and_committed_tail,
    replay_state_hash,
    shadow_replay_equivalence,
    shadow_write_parity,
    snapshot_state,
)


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_shadow_replay_equivalence_passes_for_matching_live_state
def test_shadow_replay_equivalence_passes_for_matching_live_state() -> None:
    snapshot = snapshot_state({"a": 1}, seq=1)
    tail = [AspfMutationRecord(op_id="2", op_kind="set", payload={"key": "b", "value": 2})]
    result = shadow_replay_equivalence(live_state={"a": 1, "b": 2}, snapshot=snapshot, tail=tail)
    assert result.equivalent is True
    assert result.tail_length == 1


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_shadow_replay_equivalence_detects_divergence
def test_shadow_replay_equivalence_detects_divergence() -> None:
    snapshot = snapshot_state({"a": 1}, seq=1)
    tail = [AspfMutationRecord(op_id="2", op_kind="delete", payload={"key": "a"})]
    result = shadow_replay_equivalence(live_state={"a": 1}, snapshot=snapshot, tail=tail)
    assert result.equivalent is False


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_apply_mutation_unknown_ops_and_empty_keys
def test_apply_mutation_unknown_ops_and_empty_keys() -> None:
    state = {"a": 1, "_unknown_ops": "legacy"}

    no_set = apply_mutation(
        state,
        AspfMutationRecord(op_id="1", op_kind="set", payload={"key": "", "value": 2}),
    )
    assert no_set["a"] == 1

    no_delete = apply_mutation(
        no_set,
        AspfMutationRecord(op_id="2", op_kind="delete", payload={"key": ""}),
    )
    assert no_delete["a"] == 1

    unknown = apply_mutation(
        no_delete,
        AspfMutationRecord(op_id="3", op_kind="noop", payload={}),
    )
    assert unknown["_unknown_ops"] == ["noop"]


def _sample_snapshot_and_events() -> tuple[SnapshotEnvelope, list[EventEnvelope], CommitMarker]:
    snapshot = SnapshotEnvelope(
        run_id="run-1",
        replay_cursor=1,
        snapshot=snapshot_state({"a": 1}, seq=1),
    )
    events = [
        EventEnvelope(
            sequence=2,
            run_id="run-1",
            record=AspfMutationRecord(op_id="2", op_kind="set", payload={"key": "b", "value": 2}),
        ),
        EventEnvelope(
            sequence=3,
            run_id="run-1",
            record=AspfMutationRecord(op_id="3", op_kind="set", payload={"key": "c", "value": 3}),
        ),
    ]
    commit = CommitMarker(run_id="run-1", last_durable_sequence=2)
    return snapshot, events, commit


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_replay_ignores_uncommitted_tail_and_preserves_json_equivalence
def test_replay_ignores_uncommitted_tail_and_preserves_json_equivalence() -> None:
    snapshot, events, commit = _sample_snapshot_and_events()
    result = replay_from_snapshot_and_committed_tail(snapshot=snapshot, events=events, commit=commit)
    assert result.state == {"a": 1, "b": 2}
    assert result.ignored_tail_count == 1
    assert result.equivalent_to_json_replay is True


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_shadow_write_parity_matches_legacy_json_replay
def test_shadow_write_parity_matches_legacy_json_replay() -> None:
    snapshot, events, commit = _sample_snapshot_and_events()
    parity = shadow_write_parity(enabled=True, snapshot=snapshot, events=events, commit=commit)
    assert parity.enabled is True
    assert parity.equivalent is True
    assert parity.archive_replay == parity.json_replay == {"a": 1, "b": 2}


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_filesystem_projection_and_tar_packaging_are_deterministic
def test_filesystem_projection_and_tar_packaging_are_deterministic(tmp_path: Path) -> None:
    snapshot, events, commit = _sample_snapshot_and_events()
    manifest = ArchiveManifest(
        schema_version=1,
        projection_version=1,
        run_id="run-1",
        event_sequences=(2, 3),
        snapshot_sequences=(1,),
    )

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    project_archive_filesystem(
        root_dir=first_root,
        manifest=manifest,
        events=events,
        snapshots=[snapshot],
        commit=commit,
    )
    project_archive_filesystem(
        root_dir=second_root,
        manifest=manifest,
        events=list(reversed(events)),
        snapshots=[snapshot],
        commit=commit,
    )

    first_tar = first_root / "archive.tar"
    second_tar = second_root / "archive.tar"
    package_archive_tar(root_dir=first_root, tar_path=first_tar)
    package_archive_tar(root_dir=second_root, tar_path=second_tar)

    assert first_tar.read_bytes() == second_tar.read_bytes()


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_replay_state_hash_stable_across_repeated_archive_replays
def test_replay_state_hash_stable_across_repeated_archive_replays() -> None:
    snapshot, events, commit = _sample_snapshot_and_events()
    first = replay_from_snapshot_and_committed_tail(snapshot=snapshot, events=events, commit=commit)
    second = replay_from_snapshot_and_committed_tail(snapshot=snapshot, events=events, commit=commit)
    assert replay_state_hash(replay_state=first.state) == replay_state_hash(replay_state=second.state)


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_snapshot_tail_replay_loader_matches_json_checkpoint_replay
def test_snapshot_tail_replay_loader_matches_json_checkpoint_replay(tmp_path: Path) -> None:
    snapshot, events, commit = _sample_snapshot_and_events()
    manifest = ArchiveManifest(
        schema_version=1,
        projection_version=1,
        run_id="run-1",
        event_sequences=(2, 3),
        snapshot_sequences=(1,),
    )
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    project_archive_filesystem(
        root_dir=archive_root,
        manifest=manifest,
        events=events,
        snapshots=[snapshot],
        commit=commit,
    )

    loaded_manifest, loaded_events, loaded_snapshots, loaded_commit = load_projected_archive(root_dir=archive_root)
    loader_replay = replay_from_projected_archive(root_dir=archive_root)
    json_replay = replay_from_snapshot_and_committed_tail(
        snapshot=max(loaded_snapshots, key=lambda item: item.snapshot.seq),
        events=loaded_events,
        commit=loaded_commit,
    )

    assert loaded_manifest == manifest
    assert loader_replay.state == json_replay.state
    assert loader_replay.equivalent_to_json_replay is True


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_crash_recovery_detects_corrupted_tail_entry
def test_crash_recovery_detects_corrupted_tail_entry(tmp_path: Path) -> None:
    snapshot, events, commit = _sample_snapshot_and_events()
    manifest = ArchiveManifest(
        schema_version=1,
        projection_version=1,
        run_id="run-1",
        event_sequences=(2, 3),
        snapshot_sequences=(1,),
    )
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    project_archive_filesystem(
        root_dir=archive_root,
        manifest=manifest,
        events=events,
        snapshots=[snapshot],
        commit=commit,
    )

    event_file = archive_root / "010_events" / "000000000002.data.pb"
    event_file.write_bytes(event_file.read_bytes() + b"\x00")

    try:
        load_projected_archive(root_dir=archive_root)
    except ProtobufDecodeError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("expected checksum mismatch to fail archive load")


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_replay_determinism_hash_stable_across_projection_and_tar_runs
def test_replay_determinism_hash_stable_across_projection_and_tar_runs(tmp_path: Path) -> None:
    snapshot, events, commit = _sample_snapshot_and_events()
    manifest = ArchiveManifest(
        schema_version=1,
        projection_version=1,
        run_id="run-1",
        event_sequences=(2, 3),
        snapshot_sequences=(1,),
    )

    hashes: list[str] = []
    tar_payloads: list[bytes] = []
    for index in (1, 2):
        archive_root = tmp_path / f"run-{index}"
        archive_root.mkdir()
        project_archive_filesystem(
            root_dir=archive_root,
            manifest=manifest,
            events=list(reversed(events)) if index == 2 else events,
            snapshots=[snapshot],
            commit=commit,
        )
        replay = replay_from_projected_archive(root_dir=archive_root)
        hashes.append(replay_state_hash(replay_state=replay.state))

        tar_path = archive_root / "archive.tar"
        package_archive_tar(root_dir=archive_root, tar_path=tar_path)
        tar_payloads.append(tar_path.read_bytes())

    assert hashes[0] == hashes[1]
    assert tar_payloads[0] == tar_payloads[1]


def test_protobuf_varint_and_wire_error_branches() -> None:
    with pytest.raises(ValueError):
        aspf_mutation_log._encode_varint(-1)

    encoded = aspf_mutation_log._encode_varint(300)
    decoded, cursor = aspf_mutation_log._decode_varint(encoded, 0)
    assert decoded == 300
    assert cursor == len(encoded)

    with pytest.raises(ProtobufDecodeError):
        aspf_mutation_log._decode_varint(bytes([0x80] * 10), 0)
    with pytest.raises(ProtobufDecodeError):
        aspf_mutation_log._decode_varint(b"\x80", 0)

    with pytest.raises(ProtobufDecodeError):
        aspf_mutation_log._parse_wire_fields(b"\x0a\x05hi")

    with pytest.raises(ProtobufDecodeError):
        aspf_mutation_log._parse_wire_fields(b"\x09")


def test_protobuf_json_decode_error_branches() -> None:
    with pytest.raises(ProtobufDecodeError):
        aspf_mutation_log._json_bytes_to_object(b"\xff")
    with pytest.raises(ProtobufDecodeError):
        aspf_mutation_log._json_bytes_to_object(b"[]")


def test_event_snapshot_manifest_and_commit_decode_error_branches() -> None:
    with pytest.raises(ProtobufDecodeError):
        aspf_mutation_log.decode_event_envelope_proto(
            aspf_mutation_log._encode_uint64(1, 1)
        )
    with pytest.raises(ProtobufDecodeError):
        aspf_mutation_log.decode_event_envelope_proto(
            b"".join(
                (
                    aspf_mutation_log._encode_uint64(1, 1),
                    aspf_mutation_log._encode_length_delimited(2, b"\xff"),
                    aspf_mutation_log._encode_length_delimited(3, b"{}"),
                )
            )
        )

    with pytest.raises(ProtobufDecodeError):
        aspf_mutation_log.decode_snapshot_envelope_proto(
            aspf_mutation_log._encode_uint64(2, 1)
        )
    with pytest.raises(ProtobufDecodeError):
        aspf_mutation_log.decode_snapshot_envelope_proto(
            b"".join(
                (
                    aspf_mutation_log._encode_length_delimited(1, b"\xff"),
                    aspf_mutation_log._encode_uint64(2, 1),
                    aspf_mutation_log._encode_length_delimited(3, b'{"seq":1,"state":{}}'),
                )
            )
        )

    with pytest.raises(ProtobufDecodeError):
        aspf_mutation_log.decode_archive_manifest_proto(b"")

    with pytest.raises(ProtobufDecodeError):
        aspf_mutation_log.decode_commit_marker_proto(b"")
    with pytest.raises(ProtobufDecodeError):
        aspf_mutation_log.decode_commit_marker_proto(
            aspf_mutation_log._encode_length_delimited(1, b"\xff")
        )


def test_package_archive_tar_skips_target_path_when_preexisting(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "payload.txt").write_text("payload", encoding="utf-8")
    tar_path = root / "archive.tar"
    tar_path.write_text("stale", encoding="utf-8")
    package_archive_tar(root_dir=root, tar_path=tar_path)
    assert tar_path.exists()
    assert tar_path.stat().st_size > 0


def test_replay_from_projected_archive_requires_snapshot(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    project_archive_filesystem(
        root_dir=archive_root,
        manifest=ArchiveManifest(
            schema_version=1,
            projection_version=1,
            run_id="run-1",
            event_sequences=(),
            snapshot_sequences=(),
        ),
        events=[],
        snapshots=[],
        commit=CommitMarker(run_id="run-1", last_durable_sequence=0),
    )
    with pytest.raises(ProtobufDecodeError):
        replay_from_projected_archive(root_dir=archive_root)
