from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts import refresh_baselines
from tests.env_helpers import env_scope as _env_scope


@contextmanager
def _cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


# gabion:evidence E:call_footprint::tests/test_refresh_baselines_failure_report.py::test_refresh_baselines_writes_failure_artifact_on_check_failure::env_helpers.py::tests.env_helpers.env_scope::refresh_baselines.py::scripts.refresh_baselines._run_docflow_delta_emit::refresh_baselines.py::scripts.refresh_baselines._write_failure_artifact::refresh_baselines.py::scripts.refresh_baselines.main::test_refresh_baselines_failure_report.py::tests.test_refresh_baselines_failure_report._cwd
def test_refresh_baselines_writes_failure_artifact_on_check_failure(
    tmp_path: Path,
) -> None:
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

    def _guard_docflow_failure(
        timeout: int | None,
        timeout_env: refresh_baselines._RefreshLspTimeoutEnv,
    ) -> None:
        refresh_baselines._run_docflow_delta_emit(
            timeout,
            timeout_env,
            run_fn=_failing_run,
        )

    with _cwd(tmp_path):
        with _env_scope({"GABION_LSP_TIMEOUT_SECONDS": "17"}):
            with pytest.raises(refresh_baselines.RefreshBaselinesSubprocessFailure) as failure_info:
                refresh_baselines.main(
                    ["--docflow"],
                    deadline_scope_factory=lambda: refresh_baselines.deadline_scope_from_lsp_env(
                        default_budget=refresh_baselines.DeadlineBudget(
                            ticks=10,
                            tick_ns=1_000_000,
                        )
                    ),
                    guard_obsolescence_delta_fn=lambda *args, **kwargs: None,
                    guard_annotation_drift_delta_fn=lambda *args, **kwargs: None,
                    guard_ambiguity_delta_fn=lambda *args, **kwargs: None,
                    guard_docflow_delta_fn=_guard_docflow_failure,
                )

            artifact_path = refresh_baselines._write_failure_artifact(failure_info.value)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))

            assert artifact_path == refresh_baselines.FAILURE_ARTIFACT_PATH
            assert payload["attempted_command"] == failing_command
            assert payload["attempted_flags"] == ["--timeout"]
            assert payload["exit_code"] == 9
            assert payload["timeout_settings"]["cli_timeout_seconds"] is None
            assert payload["timeout_settings"]["env"]["GABION_LSP_TIMEOUT_SECONDS"] == "17"
            assert payload["expected_artifacts"][str(refresh_baselines.DOCFLOW_DELTA_PATH)] is False
            assert payload["expected_artifacts"][str(refresh_baselines.DOCFLOW_CURRENT_PATH)] is False


# gabion:evidence E:call_footprint::tests/test_refresh_baselines_failure_report.py::test_refresh_baselines_clears_stale_failure_artifact::refresh_baselines.py::scripts.refresh_baselines.main::test_refresh_baselines_failure_report.py::tests.test_refresh_baselines_failure_report._cwd
def test_refresh_baselines_clears_stale_failure_artifact(
    tmp_path: Path,
) -> None:
    with _cwd(tmp_path):
        stale_artifact = refresh_baselines.FAILURE_ARTIFACT_PATH
        stale_artifact.parent.mkdir(parents=True, exist_ok=True)
        stale_artifact.write_text("{}", encoding="utf-8")

        def _guard_docflow_with_current(
            timeout: int | None,
            timeout_env: refresh_baselines._RefreshLspTimeoutEnv,
        ) -> None:
            del timeout, timeout_env
            refresh_baselines.DOCFLOW_CURRENT_PATH.parent.mkdir(parents=True, exist_ok=True)
            refresh_baselines.DOCFLOW_CURRENT_PATH.write_text("{}\n", encoding="utf-8")

        exit_code = refresh_baselines.main(
            ["--docflow"],
            deadline_scope_factory=lambda: refresh_baselines.deadline_scope_from_lsp_env(
                default_budget=refresh_baselines.DeadlineBudget(
                    ticks=10,
                    tick_ns=1_000_000,
                )
            ),
            guard_obsolescence_delta_fn=lambda *args, **kwargs: None,
            guard_annotation_drift_delta_fn=lambda *args, **kwargs: None,
            guard_ambiguity_delta_fn=lambda *args, **kwargs: None,
            guard_docflow_delta_fn=_guard_docflow_with_current,
        )

        assert exit_code == 0
        assert not stale_artifact.exists()
