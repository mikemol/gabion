from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from gabion.exceptions import NeverThrown
from gabion.tooling import aspf_lifecycle


@dataclass(frozen=True)
class _Prepared:
    sequence: int
    session_id: str
    step_id: str
    command_profile: str
    state_path: Path
    delta_path: Path
    import_state_paths: tuple[Path, ...]
    manifest_path: Path
    started_at_utc: str


def test_resume_import_policy_contract() -> None:
    disabled = aspf_lifecycle.AspfLifecycleConfig(
        enabled=False,
        root=Path("."),
        session_id="session",
        manifest_path=Path("manifest.json"),
        state_root=Path("state"),
    )
    assert aspf_lifecycle.resume_import_policy(config=disabled) == "success_only"

    enabled = aspf_lifecycle.AspfLifecycleConfig(
        enabled=True,
        root=Path("."),
        session_id="session",
        manifest_path=Path("manifest.json"),
        state_root=Path("state"),
        resume_import_policy="success_or_resumable_timeout",
    )
    assert (
        aspf_lifecycle.resume_import_policy(config=enabled)
        == "success_or_resumable_timeout"
    )

    invalid = aspf_lifecycle.AspfLifecycleConfig(
        enabled=True,
        root=Path("."),
        session_id="session",
        manifest_path=Path("manifest.json"),
        state_root=Path("state"),
        resume_import_policy="bad-policy",  # type: ignore[arg-type]
    )
    with pytest.raises(NeverThrown):
        aspf_lifecycle.resume_import_policy(config=invalid)


def test_run_with_aspf_lifecycle_disabled_short_circuit() -> None:
    result = aspf_lifecycle.run_with_aspf_lifecycle(
        config=None,
        step_id="step",
        command_profile="profile",
        command=["python", "-m", "gabion", "check", "delta-bundle"],
        run_command_fn=lambda _command: 0,
        analysis_state_from_state_path_fn=lambda _path: "unused",
    )
    assert result.exit_code == 0
    assert result.status == "success"
    assert result.analysis_state == "succeeded"
    assert result.session_id is None
    assert result.sequence is None


def test_run_with_aspf_lifecycle_enabled_records_state(tmp_path: Path) -> None:
    recorded: dict[str, object] = {}
    state_path = tmp_path / "aspf_state.snapshot.json"
    state_path.write_text('{"analysis_state":"timed_out_progress_resume"}\n', encoding="utf-8")
    prepared = _Prepared(
        sequence=3,
        session_id="session-1",
        step_id="step",
        command_profile="profile",
        state_path=state_path,
        delta_path=tmp_path / "aspf_state.delta.jsonl",
        import_state_paths=(tmp_path / "import.snapshot.json",),
        manifest_path=tmp_path / "aspf_manifest.json",
        started_at_utc="2026-01-01T00:00:00Z",
    )

    def _record_step_fn(**kwargs):
        recorded.update(kwargs)
        return True

    config = aspf_lifecycle.AspfLifecycleConfig(
        enabled=True,
        root=Path("."),
        session_id="session-1",
        manifest_path=Path("manifest.json"),
        state_root=Path("state"),
        resume_import_policy="success_or_resumable_timeout",
    )
    result = aspf_lifecycle.run_with_aspf_lifecycle(
        config=config,
        step_id="step",
        command_profile="profile",
        command=["python", "-m", "gabion", "check", "delta-bundle"],
        run_command_fn=lambda _command: 1,
        analysis_state_from_state_path_fn=lambda _path: "timed_out_progress_resume",
        prepare_step_fn=lambda **_kwargs: prepared,
        aspf_cli_args_fn=lambda _step: ["--aspf-state-json", str(state_path)],
        record_step_fn=_record_step_fn,
    )
    assert result.exit_code == 1
    assert result.status == "failed"
    assert result.analysis_state == "timed_out_progress_resume"
    assert result.command_with_aspf[-2:] == ("--aspf-state-json", str(state_path))
    assert recorded["session_id"] == "session-1"
    assert recorded["sequence"] == 3
    assert recorded["status"] == "failed"


def test_run_with_aspf_lifecycle_enabled_missing_state_defaults_to_exit_status() -> None:
    state_path = Path("/tmp/nonexistent.snapshot.json")
    prepared = _Prepared(
        sequence=1,
        session_id="session-1",
        step_id="step",
        command_profile="profile",
        state_path=state_path,
        delta_path=Path("/tmp/nonexistent.delta.jsonl"),
        import_state_paths=(),
        manifest_path=Path("/tmp/aspf_manifest.json"),
        started_at_utc="2026-01-01T00:00:00Z",
    )
    config = aspf_lifecycle.AspfLifecycleConfig(
        enabled=True,
        root=Path("."),
        session_id="session-1",
        manifest_path=Path("manifest.json"),
        state_root=Path("state"),
    )
    result = aspf_lifecycle.run_with_aspf_lifecycle(
        config=config,
        step_id="step",
        command_profile="profile",
        command=["python", "-m", "gabion", "check", "delta-bundle"],
        run_command_fn=lambda _command: 0,
        analysis_state_from_state_path_fn=lambda _path: "ignored",
        prepare_step_fn=lambda **_kwargs: prepared,
        aspf_cli_args_fn=lambda _step: [],
        record_step_fn=lambda **_kwargs: True,
    )
    assert result.analysis_state == "succeeded"


def test_run_with_aspf_lifecycle_record_failure_raises() -> None:
    prepared = _Prepared(
        sequence=1,
        session_id="session-1",
        step_id="step",
        command_profile="profile",
        state_path=Path("/tmp/nonexistent.snapshot.json"),
        delta_path=Path("/tmp/nonexistent.delta.jsonl"),
        import_state_paths=(),
        manifest_path=Path("/tmp/aspf_manifest.json"),
        started_at_utc="2026-01-01T00:00:00Z",
    )
    config = aspf_lifecycle.AspfLifecycleConfig(
        enabled=True,
        root=Path("."),
        session_id="session-1",
        manifest_path=Path("manifest.json"),
        state_root=Path("state"),
    )
    with pytest.raises(NeverThrown):
        aspf_lifecycle.run_with_aspf_lifecycle(
            config=config,
            step_id="step",
            command_profile="profile",
            command=["python", "-m", "gabion", "check", "delta-bundle"],
            run_command_fn=lambda _command: 1,
            analysis_state_from_state_path_fn=lambda _path: "failed",
            prepare_step_fn=lambda **_kwargs: prepared,
            aspf_cli_args_fn=lambda _step: [],
            record_step_fn=lambda **_kwargs: False,
        )
