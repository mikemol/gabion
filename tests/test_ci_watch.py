from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts import ci_watch


def _make_deps(
    *,
    run_handler,
    stderr_messages: list[str],
) -> ci_watch.CiWatchDeps:
    def _run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return run_handler(
            cmd,
            check=check,
            capture_output=capture_output,
            text=text,
        )

    return ci_watch.CiWatchDeps(
        run=_run,
        print_err=stderr_messages.append,
    )


# gabion:evidence E:call_footprint::tests/test_ci_watch.py::test_ci_watch_collects_failure_artifacts_and_logs::ci_watch.py::main
def test_ci_watch_collects_failure_artifacts_and_logs(tmp_path: Path) -> None:
    run_id = "12345"
    commands: list[tuple[list[str], bool, bool, bool]] = []
    stderr_messages: list[str] = []
    summary_json = tmp_path / "ci_watch_summary.json"

    run_payload = json.dumps(
        {
            "status": "completed",
            "conclusion": "failure",
            "jobs": [
                {
                    "databaseId": 11,
                    "name": "audit",
                    "status": "completed",
                    "conclusion": "failure",
                    "url": "https://example.invalid/audit",
                    "steps": [
                        {
                            "number": 1,
                            "name": "setup",
                            "status": "completed",
                            "conclusion": "success",
                        },
                        {
                            "number": 14,
                            "name": "Policy check (no monkeypatch)",
                            "status": "completed",
                            "conclusion": "failure",
                        },
                    ],
                },
                {
                    "databaseId": 22,
                    "name": "dataflow-grammar",
                    "status": "completed",
                    "conclusion": "success",
                    "url": "https://example.invalid/dataflow",
                    "steps": [],
                },
            ],
        }
    )

    def _fake_run_handler(
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
            return subprocess.CompletedProcess(cmd, 0, run_payload, "")
        if cmd[:4] == ["gh", "run", "view", run_id] and "--log-failed" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "FAILED LOG", "")
        if cmd[:3] == ["gh", "run", "download"]:
            artifacts_dir = Path(cmd[cmd.index("--dir") + 1])
            (artifacts_dir / "test-runs").mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "test-runs" / "payload.txt").write_text(
                "ok",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, "download ok", "")
        raise AssertionError(f"unexpected command: {cmd}")

    deps = _make_deps(
        run_handler=_fake_run_handler,
        stderr_messages=stderr_messages,
    )
    rc = ci_watch.main(
        [
            "--run-id",
            run_id,
            "--artifact-output-root",
            str(tmp_path),
            "--summary-json",
            str(summary_json),
            "--artifact-name",
            "test-runs",
            "--artifact-name",
            "controller-drift-gate-history",
        ],
        deps=deps,
    )
    assert rc == 1

    run_root = tmp_path / f"run_{run_id}"
    assert (run_root / "run.json").read_text(encoding="utf-8") == run_payload
    assert (run_root / "failed.log").read_text(encoding="utf-8") == "FAILED LOG"
    assert (run_root / "download.stdout.log").read_text(encoding="utf-8") == "download ok"
    assert (run_root / "failed_jobs.json").is_file()
    assert (run_root / "failed_steps.json").is_file()
    assert (run_root / "collection_status.json").is_file()
    assert (run_root / "artifacts_manifest.json").is_file()

    failed_jobs = json.loads((run_root / "failed_jobs.json").read_text(encoding="utf-8"))
    failed_steps = json.loads((run_root / "failed_steps.json").read_text(encoding="utf-8"))
    collection_status = json.loads(
        (run_root / "collection_status.json").read_text(encoding="utf-8")
    )
    artifact_manifest = json.loads(
        (run_root / "artifacts_manifest.json").read_text(encoding="utf-8")
    )
    summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert len(failed_jobs) == 1
    assert failed_jobs[0]["name"] == "audit"
    assert len(failed_steps) == 1
    assert failed_steps[0]["step_name"] == "Policy check (no monkeypatch)"
    assert collection_status["collection_success"] is True
    assert collection_status["mandatory_failures"] == []
    assert artifact_manifest["artifact_dirs"] == ["test-runs"]
    assert artifact_manifest["files"] == ["test-runs/payload.txt"]
    assert summary_payload["run_id"] == run_id
    assert summary_payload["watch_exit_code"] == 1
    assert summary_payload["exit_code"] == 1
    assert summary_payload["artifact_output_root"] == str(tmp_path)
    assert summary_payload["collection"]["run_root"] == str(run_root)
    assert summary_payload["collection"]["status"]["collection_success"] is True

    download_commands = [command for command, _, _, _ in commands if command[:3] == ["gh", "run", "download"]]
    assert len(download_commands) == 1
    assert download_commands[0].count("--name") == 2
    assert "test-runs" in download_commands[0]
    assert "controller-drift-gate-history" in download_commands[0]
    assert stderr_messages == [
        f"ci_watch: run {run_id} failed; collected artifacts under {run_root}"
    ]


# gabion:evidence E:call_footprint::tests/test_ci_watch.py::test_ci_watch_failure_can_skip_artifact_collection::ci_watch.py::main
def test_ci_watch_failure_can_skip_artifact_collection(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    stderr_messages: list[str] = []

    def _fake_run_handler(
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

    deps = _make_deps(
        run_handler=_fake_run_handler,
        stderr_messages=stderr_messages,
    )
    rc = ci_watch.main(
        [
            "--run-id",
            "98765",
            "--artifact-output-root",
            str(tmp_path),
            "--no-download-artifacts-on-failure",
        ],
        deps=deps,
    )
    assert rc == 1
    assert commands == [["gh", "run", "watch", "98765", "--exit-status"]]
    assert list(tmp_path.iterdir()) == []
    assert stderr_messages == []


# gabion:evidence E:call_footprint::tests/test_ci_watch.py::test_ci_watch_success_does_not_collect_failure_artifacts::ci_watch.py::main
def test_ci_watch_success_does_not_collect_failure_artifacts(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    stderr_messages: list[str] = []

    def _fake_run_handler(
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

    deps = _make_deps(
        run_handler=_fake_run_handler,
        stderr_messages=stderr_messages,
    )
    rc = ci_watch.main(
        [
            "--run-id",
            "55555",
            "--artifact-output-root",
            str(tmp_path),
        ],
        deps=deps,
    )
    assert rc == 0
    assert commands == [["gh", "run", "watch", "55555", "--exit-status"]]
    assert list(tmp_path.iterdir()) == []
    assert stderr_messages == []


# gabion:evidence E:call_footprint::tests/test_ci_watch.py::test_ci_watch_collection_failures_return_strict_nonzero::ci_watch.py::main
def test_ci_watch_collection_failures_return_strict_nonzero(tmp_path: Path) -> None:
    run_id = "90000"
    stderr_messages: list[str] = []

    def _fake_run_handler(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gh", "run", "watch"]:
            return subprocess.CompletedProcess(cmd, 1, "", "")
        if cmd[:4] == ["gh", "run", "view", run_id] and "--json" in cmd:
            payload = json.dumps({"status": "completed", "conclusion": "failure", "jobs": []})
            return subprocess.CompletedProcess(cmd, 0, payload, "")
        if cmd[:4] == ["gh", "run", "view", run_id] and "--log-failed" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:3] == ["gh", "run", "download"]:
            return subprocess.CompletedProcess(cmd, 1, "", "download failed")
        raise AssertionError(f"unexpected command: {cmd}")

    deps = _make_deps(
        run_handler=_fake_run_handler,
        stderr_messages=stderr_messages,
    )
    rc = ci_watch.main(
        [
            "--run-id",
            run_id,
            "--artifact-output-root",
            str(tmp_path),
        ],
        deps=deps,
    )
    assert rc == 2
    run_root = tmp_path / f"run_{run_id}"
    assert (run_root / "download.stderr.log").read_text(encoding="utf-8") == "download failed"
    collection_status = json.loads(
        (run_root / "collection_status.json").read_text(encoding="utf-8")
    )
    assert collection_status["collection_success"] is False
    assert collection_status["mandatory_failures"] == ["download"]
    assert stderr_messages == [
        "ci_watch: run "
        f"{run_id} failed; collection had failures (download) under {run_root}"
    ]
