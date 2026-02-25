from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import refresh_baselines


# gabion:evidence E:call_footprint::tests/test_refresh_baselines_run_check.py::test_run_check_includes_timeout_diagnostics_flags::refresh_baselines.py::scripts.refresh_baselines._refresh_lsp_timeout_env::refresh_baselines.py::scripts.refresh_baselines._run_check
def test_run_check_includes_timeout_diagnostics_flags() -> None:
    captured: dict[str, object] = {}
    timeout_env = refresh_baselines._refresh_lsp_timeout_env(None, None)

    # dataflow-bundle: check, cmd, env, timeout
    def _fake_run(cmd, *, check, timeout, env):
        captured["cmd"] = cmd
        captured["check"] = check
        captured["timeout"] = timeout
        captured["env"] = env

    refresh_baselines._run_check(
        [
            "obsolescence",
            "delta",
            "--baseline",
            str(refresh_baselines.OBSOLESCENCE_BASELINE_PATH),
        ],
        timeout=17,
        timeout_env=timeout_env,
        resume_on_timeout=1,
        resume_checkpoint=Path("artifacts/out/refresh_baselines_resume.json"),
        extra=["--foo", "bar"],
        run_fn=_fake_run,
    )

    assert captured["check"] is True
    assert captured["timeout"] == 17
    env = captured["env"]
    assert isinstance(env, dict)

    cmd = captured["cmd"]
    assert cmd[:7] == [sys.executable, "-m", "gabion", "--timeout", "120000000000ns", "check", "obsolescence"]
    assert cmd[7] == "delta"
    assert "--timeout-progress-report" in cmd
    resume_idx = cmd.index("--resume-on-timeout")
    assert cmd[resume_idx + 1] == "1"
    checkpoint_idx = cmd.index("--resume-checkpoint")
    assert cmd[checkpoint_idx + 1] == "artifacts/out/refresh_baselines_resume.json"
    assert cmd[-2:] == ["--foo", "bar"]


# gabion:evidence E:call_footprint::tests/test_refresh_baselines_run_check.py::test_run_check_formats_called_process_error::refresh_baselines.py::scripts.refresh_baselines._refresh_lsp_timeout_env::refresh_baselines.py::scripts.refresh_baselines._run_check
def test_run_check_formats_called_process_error() -> None:
    timeout_env = refresh_baselines._refresh_lsp_timeout_env(None, None)

    def _raise_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(returncode=2, cmd=["gabion", "check"])

    with pytest.raises(refresh_baselines.RefreshBaselinesSubprocessFailure) as exc_info:
        refresh_baselines._run_check(
            [
                "ambiguity",
                "baseline-write",
                "--baseline",
                str(refresh_baselines.AMBIGUITY_BASELINE_PATH),
            ],
            timeout=None,
            timeout_env=timeout_env,
            resume_on_timeout=2,
            resume_checkpoint=None,
            run_fn=_raise_run,
        )
    failure = exc_info.value
    assert failure.exit_code == 2
    assert str(refresh_baselines.DEFAULT_TIMEOUT_PROGRESS_PATH) in {
        str(path) for path in failure.expected_artifacts
    }
    assert str(refresh_baselines.DEFAULT_DEADLINE_PROFILE_PATH) in {
        str(path) for path in failure.expected_artifacts
    }
    assert str(refresh_baselines.DEFAULT_CHECK_REPORT_PATH) in {
        str(path) for path in failure.expected_artifacts
    }
