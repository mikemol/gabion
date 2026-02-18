from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import refresh_baselines


def test_refresh_baselines_writes_failure_artifact_on_check_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(refresh_baselines.sys, "argv", ["refresh_baselines.py", "--docflow"])
    monkeypatch.setenv("GABION_LSP_TIMEOUT_SECONDS", "17")

    failing_command: list[str] = []

    def _failing_run(
        cmd: list[str],
        *,
        check: bool,
        timeout: int | None,
        env: dict[str, str],
    ) -> None:
        del check, timeout, env
        nonlocal failing_command
        failing_command = cmd
        raise subprocess.CalledProcessError(returncode=9, cmd=cmd)

    monkeypatch.setattr(refresh_baselines.subprocess, "run", _failing_run)

    with pytest.raises(refresh_baselines.RefreshBaselinesSubprocessFailure) as failure_info:
        refresh_baselines.main()

    artifact_path = refresh_baselines._write_failure_artifact(failure_info.value)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact_path == refresh_baselines.FAILURE_ARTIFACT_PATH
    assert payload["attempted_command"] == failing_command
    assert payload["attempted_flags"] == []
    assert payload["exit_code"] == 9
    assert payload["timeout_settings"]["cli_timeout_seconds"] is None
    assert payload["timeout_settings"]["env"]["GABION_LSP_TIMEOUT_SECONDS"] == "17"
    assert payload["expected_artifacts"][str(refresh_baselines.DOCFLOW_DELTA_PATH)] is False
    assert payload["expected_artifacts"][str(refresh_baselines.DOCFLOW_CURRENT_PATH)] is False


def test_refresh_baselines_clears_stale_failure_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(refresh_baselines.sys, "argv", ["refresh_baselines.py", "--docflow"])
    stale_artifact = refresh_baselines.FAILURE_ARTIFACT_PATH
    stale_artifact.parent.mkdir(parents=True, exist_ok=True)
    stale_artifact.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(refresh_baselines, "_guard_obsolescence_delta", lambda timeout: None)
    monkeypatch.setattr(refresh_baselines, "_guard_annotation_drift_delta", lambda timeout: None)
    monkeypatch.setattr(refresh_baselines, "_guard_ambiguity_delta", lambda timeout: None)

    def _guard_docflow_with_current(timeout: int | None) -> None:
        del timeout
        refresh_baselines.DOCFLOW_CURRENT_PATH.parent.mkdir(parents=True, exist_ok=True)
        refresh_baselines.DOCFLOW_CURRENT_PATH.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        refresh_baselines,
        "_guard_docflow_delta",
        _guard_docflow_with_current,
    )

    exit_code = refresh_baselines.main()

    assert exit_code == 0
    assert not stale_artifact.exists()
