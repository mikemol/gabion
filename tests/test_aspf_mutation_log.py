from __future__ import annotations

from pathlib import Path

from gabion.analysis.aspf_mutation_log import (
    ArchiveManifest,
    AspfMutationRecord,
    CURRENT_ARCHIVE_SCHEMA_VERSION,
    CommitMarker,
    decode_commit_compat,
    decode_event_compat,
    decode_manifest_compat,
    decode_snapshot_compat,
    EventEnvelope,
    LEGACY_ARCHIVE_SCHEMA_VERSION,
    SnapshotEnvelope,
    apply_mutation,
    package_archive_tar,
    project_archive_filesystem,
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


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_projection_writes_protobuf_payloads_with_schema_version
def test_projection_writes_protobuf_payloads_with_schema_version(tmp_path: Path) -> None:
    snapshot, events, commit = _sample_snapshot_and_events()
    manifest = ArchiveManifest(
        schema_version=CURRENT_ARCHIVE_SCHEMA_VERSION,
        projection_version=1,
        run_id="run-1",
        event_sequences=(2, 3),
        snapshot_sequences=(1,),
    )
    root = tmp_path / "archive"
    root.mkdir()
    project_archive_filesystem(
        root_dir=root,
        manifest=manifest,
        events=events,
        snapshots=[snapshot],
        commit=commit,
    )

    manifest_raw = (root / "001_manifest" / "manifest.pb").read_bytes()
    event_raw = (root / "010_events" / "000000000002.data.pb").read_bytes()
    snapshot_raw = (root / "020_snapshots" / "000000000001.data.pb").read_bytes()
    commit_raw = (root / "099_commit" / "commit.pb").read_bytes()

    assert not manifest_raw.startswith(b"{")
    assert not event_raw.startswith(b"{")
    assert not snapshot_raw.startswith(b"{")
    assert not commit_raw.startswith(b"{")

    decoded_manifest = decode_manifest_compat(manifest_raw)
    decoded_event = decode_event_compat(event_raw)
    decoded_snapshot = decode_snapshot_compat(snapshot_raw)
    decoded_commit = decode_commit_compat(commit_raw)

    assert decoded_manifest.schema_version == CURRENT_ARCHIVE_SCHEMA_VERSION
    assert decoded_event.schema_version == CURRENT_ARCHIVE_SCHEMA_VERSION
    assert decoded_snapshot.schema_version == CURRENT_ARCHIVE_SCHEMA_VERSION
    assert decoded_commit.schema_version == CURRENT_ARCHIVE_SCHEMA_VERSION


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_schema_version_migration_and_legacy_json_compatibility
def test_schema_version_migration_and_legacy_json_compatibility() -> None:
    legacy_event_raw = (
        b'{"sequence":2,"run_id":"run-legacy","record":{"op_id":"2","op_kind":"set","payload":{"key":"b","value":2}}}'
    )
    legacy_snapshot_raw = (
        b'{"run_id":"run-legacy","replay_cursor":1,"snapshot":{"seq":1,"state":{"a":1}}}'
    )
    legacy_manifest_raw = (
        b'{"projection_version":1,"run_id":"run-legacy","event_sequences":[2],"snapshot_sequences":[1]}'
    )
    legacy_commit_raw = b'{"run_id":"run-legacy","last_durable_sequence":2}'

    event = decode_event_compat(legacy_event_raw)
    snapshot = decode_snapshot_compat(legacy_snapshot_raw)
    manifest = decode_manifest_compat(legacy_manifest_raw)
    commit = decode_commit_compat(legacy_commit_raw)

    assert event.schema_version == LEGACY_ARCHIVE_SCHEMA_VERSION
    assert snapshot.schema_version == LEGACY_ARCHIVE_SCHEMA_VERSION
    assert manifest.schema_version == LEGACY_ARCHIVE_SCHEMA_VERSION
    assert commit.schema_version == LEGACY_ARCHIVE_SCHEMA_VERSION

    replay = replay_from_snapshot_and_committed_tail(snapshot=snapshot, events=[event], commit=commit)
    assert replay.state == {"a": 1, "b": 2}
    assert replay.equivalent_to_json_replay is True


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_replay_state_hash_stable_across_repeated_archive_replays
def test_replay_state_hash_stable_across_repeated_archive_replays() -> None:
    snapshot, events, commit = _sample_snapshot_and_events()
    first = replay_from_snapshot_and_committed_tail(snapshot=snapshot, events=events, commit=commit)
    second = replay_from_snapshot_and_committed_tail(snapshot=snapshot, events=events, commit=commit)
    assert replay_state_hash(replay_state=first.state) == replay_state_hash(replay_state=second.state)
