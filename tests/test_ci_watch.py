from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import ci_watch


# gabion:evidence E:call_footprint::tests/test_ci_watch.py::test_ci_watch_collects_failure_artifacts_and_logs::ci_watch.py::main
def test_ci_watch_collects_failure_artifacts_and_logs(
    monkeypatch, tmp_path: Path
) -> None:
    run_id = "12345"
    commands: list[tuple[list[str], bool, bool, bool]] = []

    def _fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        commands.append((cmd, check, capture_output, text))
        if cmd[:3] == ["gh", "run", "watch"]:
            return subprocess.CompletedProcess(cmd, 1, "", "")
        if cmd[:4] == ["gh", "run", "view", run_id] and "--json" in cmd:
            return subprocess.CompletedProcess(cmd, 0, '{"status":"completed"}', "")
        if cmd[:4] == ["gh", "run", "view", run_id] and "--log-failed" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "FAILED LOG", "")
        if cmd[:3] == ["gh", "run", "download"]:
            return subprocess.CompletedProcess(cmd, 0, "download ok", "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ci_watch.py",
            "--run-id",
            run_id,
            "--artifact-output-root",
            str(tmp_path),
            "--artifact-name",
            "test-runs",
            "--artifact-name",
            "controller-drift-gate-history",
        ],
    )

    rc = ci_watch.main()
    assert rc == 1

    run_root = tmp_path / f"run_{run_id}"
    assert (run_root / "run.json").read_text(encoding="utf-8") == '{"status":"completed"}'
    assert (run_root / "failed.log").read_text(encoding="utf-8") == "FAILED LOG"
    assert (run_root / "download.stdout.log").read_text(encoding="utf-8") == "download ok"

    download_commands = [command for command, _, _, _ in commands if command[:3] == ["gh", "run", "download"]]
    assert len(download_commands) == 1
    assert download_commands[0].count("--name") == 2
    assert "test-runs" in download_commands[0]
    assert "controller-drift-gate-history" in download_commands[0]


# gabion:evidence E:call_footprint::tests/test_ci_watch.py::test_ci_watch_failure_can_skip_artifact_collection::ci_watch.py::main
def test_ci_watch_failure_can_skip_artifact_collection(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def _fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        if cmd[:3] == ["gh", "run", "watch"]:
            return subprocess.CompletedProcess(cmd, 1, "", "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ci_watch.py",
            "--run-id",
            "98765",
            "--artifact-output-root",
            str(tmp_path),
            "--no-download-artifacts-on-failure",
        ],
    )

    rc = ci_watch.main()
    assert rc == 1
    assert commands == [["gh", "run", "watch", "98765", "--exit-status"]]
    assert list(tmp_path.iterdir()) == []


# gabion:evidence E:call_footprint::tests/test_ci_watch.py::test_ci_watch_success_does_not_collect_failure_artifacts::ci_watch.py::main
def test_ci_watch_success_does_not_collect_failure_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []

    def _fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        if cmd[:3] == ["gh", "run", "watch"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ci_watch.py",
            "--run-id",
            "55555",
            "--artifact-output-root",
            str(tmp_path),
        ],
    )

    rc = ci_watch.main()
    assert rc == 0
    assert commands == [["gh", "run", "watch", "55555", "--exit-status"]]
    assert list(tmp_path.iterdir()) == []
