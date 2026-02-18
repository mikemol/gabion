from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import refresh_baselines


def test_run_check_adds_timeout_progress_and_resume_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[refresh_baselines._RefreshCommand] = []

    def _capture(run: refresh_baselines._RefreshCommand) -> None:
        captured.append(run)

    monkeypatch.setattr(refresh_baselines, "_run_refresh_command", _capture)
    monkeypatch.setenv("GABION_LSP_TIMEOUT_TICKS", "77")

    refresh_baselines._run_check("--emit-ambiguity-delta", timeout=3)

    run = captured[0]
    assert "--emit-timeout-progress-report" in run.cmd
    assert "--resume-checkpoint" in run.cmd
    assert "--resume-on-timeout" in run.cmd
    checkpoint_index = run.cmd.index("--resume-checkpoint") + 1
    assert run.cmd[checkpoint_index] == str(
        refresh_baselines.RESUME_CHECKPOINT_DIR / "--emit-ambiguity-delta.json"
    )


def test_run_refresh_command_timeout_writes_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    def _raise_timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd=[sys.executable, "-m", "gabion"], timeout=9)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    run = refresh_baselines._RefreshCommand(
        label="timeout-case",
        cmd=[sys.executable, "-m", "gabion", "check"],
        env={"GABION_LSP_TIMEOUT_MS": "9000", "GABION_DIRECT_RUN": "1"},
        timeout=9,
        expected_artifacts=["artifacts/out/test_obsolescence_delta.json"],
    )

    with pytest.raises(subprocess.TimeoutExpired):
        refresh_baselines._run_refresh_command(run)

    payload = json.loads(refresh_baselines.FAILURE_ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert payload["failure_type"] == "TimeoutExpired"
    assert payload["exit_code"] is None
    assert payload["expected_artifact_paths"] == [
        "artifacts/out/test_obsolescence_delta.json"
    ]
    stderr = capsys.readouterr().err
    assert "Refresh baseline subprocess failed" in stderr
    assert "GABION_DIRECT_RUN=1" in stderr


def test_run_refresh_command_called_process_error_writes_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    def _raise_called_process_error(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(returncode=21, cmd=["python", "-m", "gabion"])

    monkeypatch.setattr(subprocess, "run", _raise_called_process_error)
    run = refresh_baselines._RefreshCommand(
        label="cpe-case",
        cmd=["python", "-m", "gabion", "check"],
        env={"GABION_DIRECT_RUN": "1", "GABION_LSP_TIMEOUT_SECONDS": "7"},
        timeout=7,
        expected_artifacts=["baselines/ambiguity_baseline.json"],
    )

    with pytest.raises(subprocess.CalledProcessError):
        refresh_baselines._run_refresh_command(run)

    payload = json.loads(refresh_baselines.FAILURE_ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert payload["failure_type"] == "CalledProcessError"
    assert payload["exit_code"] == 21
    assert payload["env_timeout_settings"]["GABION_LSP_TIMEOUT_SECONDS"] == "7"


def test_run_refresh_command_success_clears_stale_failure_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    refresh_baselines.FAILURE_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    refresh_baselines.FAILURE_ARTIFACT_PATH.write_text("stale", encoding="utf-8")

    def _ok(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["python"], returncode=0)

    monkeypatch.setattr(subprocess, "run", _ok)
    run = refresh_baselines._RefreshCommand(
        label="ok-case",
        cmd=["python", "-m", "gabion", "check"],
        env={"GABION_DIRECT_RUN": "1"},
        timeout=2,
        expected_artifacts=[],
    )

    refresh_baselines._run_refresh_command(run)

    assert not refresh_baselines.FAILURE_ARTIFACT_PATH.exists()
