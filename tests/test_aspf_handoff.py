from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling import aspf_handoff


# gabion:evidence E:call_footprint::tests/test_aspf_handoff.py::test_prepare_step_uses_cumulative_success_chain::aspf_handoff.py::gabion.tooling.aspf_handoff.prepare_step::aspf_handoff.py::gabion.tooling.aspf_handoff.record_step
def test_prepare_step_uses_cumulative_success_chain(tmp_path: Path) -> None:
    root = tmp_path
    manifest_path = root / "artifacts/out/aspf_handoff_manifest.json"
    state_root = root / "artifacts/out/aspf_state"
    session_id = "session-test"
    checkpoint = root / "artifacts/audit_reports/checkpoint.json"

    step1 = aspf_handoff.prepare_step(
        root=root,
        session_id=session_id,
        step_id="check.run",
        command_profile="check.run",
        resume_checkpoint_path=checkpoint,
        manifest_path=manifest_path,
        state_root=state_root,
    )
    assert step1.sequence == 1
    assert step1.import_state_paths == ()
    assert aspf_handoff.record_step(
        manifest_path=manifest_path,
        session_id=session_id,
        sequence=step1.sequence,
        status="success",
        exit_code=0,
        analysis_state="succeeded",
    )

    step2 = aspf_handoff.prepare_step(
        root=root,
        session_id=session_id,
        step_id="check.annotation-drift.delta",
        command_profile="check.annotation-drift.delta",
        resume_checkpoint_path=checkpoint,
        manifest_path=manifest_path,
        state_root=state_root,
    )
    assert step2.sequence == 2
    assert step2.import_state_paths == (step1.state_path,)
    assert aspf_handoff.record_step(
        manifest_path=manifest_path,
        session_id=session_id,
        sequence=step2.sequence,
        status="failed",
        exit_code=2,
        analysis_state="failed",
    )

    step3 = aspf_handoff.prepare_step(
        root=root,
        session_id=session_id,
        step_id="check.ambiguity.delta",
        command_profile="check.ambiguity.delta",
        resume_checkpoint_path=checkpoint,
        manifest_path=manifest_path,
        state_root=state_root,
    )
    assert step3.sequence == 3
    assert step3.import_state_paths == (step1.state_path,)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    assert isinstance(entries, list)
    assert [entry.get("status") for entry in entries] == ["success", "failed", "started"]


# gabion:evidence E:call_footprint::tests/test_aspf_handoff.py::test_prepare_step_resets_manifest_when_session_changes::aspf_handoff.py::gabion.tooling.aspf_handoff.prepare_step
def test_prepare_step_resets_manifest_when_session_changes(tmp_path: Path) -> None:
    root = tmp_path
    manifest_path = root / "manifest.json"
    state_root = root / "state"

    step1 = aspf_handoff.prepare_step(
        root=root,
        session_id="session-a",
        step_id="one",
        command_profile="check.run",
        resume_checkpoint_path=None,
        manifest_path=manifest_path,
        state_root=state_root,
    )
    assert aspf_handoff.record_step(
        manifest_path=manifest_path,
        session_id="session-a",
        sequence=step1.sequence,
        status="success",
        exit_code=0,
        analysis_state="succeeded",
    )

    step2 = aspf_handoff.prepare_step(
        root=root,
        session_id="session-b",
        step_id="two",
        command_profile="check.run",
        resume_checkpoint_path=None,
        manifest_path=manifest_path,
        state_root=state_root,
    )
    assert step2.sequence == 1
    assert step2.import_state_paths == ()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("session_id") == "session-b"
    entries = manifest.get("entries")
    assert isinstance(entries, list)
    assert len(entries) == 1


def test_prepare_step_manifest_paths_are_relative_and_portable(tmp_path: Path) -> None:
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    manifest_rel = Path("artifacts/out/aspf_handoff_manifest.json")
    state_root_rel = Path("artifacts/out/aspf_state")
    session_id = "session-portable"

    step_a = aspf_handoff.prepare_step(
        root=root_a,
        session_id=session_id,
        step_id="check.run",
        command_profile="check.run",
        manifest_path=manifest_rel,
        state_root=state_root_rel,
    )
    step_a.state_path.parent.mkdir(parents=True, exist_ok=True)
    step_a.state_path.write_text("{}", encoding="utf-8")
    assert aspf_handoff.record_step(
        manifest_path=step_a.manifest_path,
        session_id=session_id,
        sequence=step_a.sequence,
        status="success",
        exit_code=0,
        analysis_state="succeeded",
    )

    manifest_a = json.loads(step_a.manifest_path.read_text(encoding="utf-8"))
    entries_a = manifest_a.get("entries")
    assert isinstance(entries_a, list)
    assert isinstance(entries_a[0], dict)
    state_path_ref = entries_a[0].get("state_path")
    assert isinstance(state_path_ref, str)
    assert not Path(state_path_ref).is_absolute()

    manifest_b_path = root_b / manifest_rel
    manifest_b_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_b_path.write_text(
        json.dumps(manifest_a, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    portable_state_path = root_b / state_path_ref
    portable_state_path.parent.mkdir(parents=True, exist_ok=True)
    portable_state_path.write_text("{}", encoding="utf-8")

    step_b = aspf_handoff.prepare_step(
        root=root_b,
        session_id=session_id,
        step_id="check.next",
        command_profile="check.run",
        manifest_path=manifest_rel,
        state_root=state_root_rel,
    )
    assert step_b.import_state_paths == (portable_state_path.resolve(),)
