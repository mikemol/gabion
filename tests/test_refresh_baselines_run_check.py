from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import refresh_baselines


def test_run_check_includes_timeout_diagnostics_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cmd, *, check, timeout, env):
        captured["cmd"] = cmd
        captured["check"] = check
        captured["timeout"] = timeout
        captured["env"] = env

    monkeypatch.setattr(refresh_baselines.subprocess, "run", _fake_run)

    refresh_baselines._run_check(
        "--emit-test-obsolescence-delta",
        timeout=17,
        resume_on_timeout=1,
        resume_checkpoint=Path("artifacts/out/refresh_baselines_resume.json"),
        extra=["--foo", "bar"],
    )

    assert captured["check"] is True
    assert captured["timeout"] == 17
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["GABION_DIRECT_RUN"] == "1"

    cmd = captured["cmd"]
    assert cmd[:5] == [sys.executable, "-m", "gabion", "check", "--no-fail-on-violations"]
    assert "--emit-timeout-progress-report" in cmd
    resume_idx = cmd.index("--resume-on-timeout")
    assert cmd[resume_idx + 1] == "1"
    checkpoint_idx = cmd.index("--resume-checkpoint")
    assert cmd[checkpoint_idx + 1] == "artifacts/out/refresh_baselines_resume.json"
    assert cmd[-2:] == ["--foo", "bar"]


def test_run_check_formats_called_process_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(returncode=2, cmd=["gabion", "check"])

    monkeypatch.setattr(refresh_baselines.subprocess, "run", _raise_run)

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        refresh_baselines._run_check(
            "--write-ambiguity-baseline",
            timeout=None,
            resume_on_timeout=2,
            resume_checkpoint=None,
        )

    notes = getattr(exc_info.value, "__notes__", [])
    assert notes
    message = "\n".join(notes)
    assert "Command:" in message
    assert "Exit code: 2" in message
    assert str(refresh_baselines.DEFAULT_TIMEOUT_PROGRESS_PATH) in message
    assert str(refresh_baselines.DEFAULT_DEADLINE_PROFILE_PATH) in message
    assert str(refresh_baselines.DEFAULT_CHECK_REPORT_PATH) in message
