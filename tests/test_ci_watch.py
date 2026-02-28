from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gabion.tooling import ci_watch as tooling_ci_watch
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


def test_collection_status_mandatory_failures_respects_log_toggle() -> None:
    status = tooling_ci_watch.CollectionStatus(
        run_view_json_rc=1,
        log_failed_rc=9,
        download_rc=2,
        failed_job_count=0,
        failed_step_count=0,
        artifact_file_count=0,
    )
    assert status.mandatory_failures(collect_failed_logs=True) == [
        "run_view_json",
        "log_failed",
        "download",
    ]
    assert status.mandatory_failures(collect_failed_logs=False) == [
        "run_view_json",
        "download",
    ]


def test_status_watch_options_to_argv_and_summary_without_collection(
    tmp_path: Path,
) -> None:
    options = tooling_ci_watch.StatusWatchOptions(
        branch="next",
        run_id=None,
        status="completed",
        workflow="ci.yml",
        prefer_active=False,
        download_artifacts_on_failure=False,
        artifact_output_root=tmp_path / "artifacts",
        artifact_names=("test-runs",),
        collect_failed_logs=False,
        summary_json=None,
    )
    argv = options.to_argv()
    assert argv == [
        "--branch",
        "next",
        "--status",
        "completed",
        "--workflow",
        "ci.yml",
        "--no-prefer-active",
        "--no-download-artifacts-on-failure",
        "--artifact-output-root",
        str(tmp_path / "artifacts"),
        "--artifact-name",
        "test-runs",
        "--no-collect-failed-logs",
    ]
    result = tooling_ci_watch.StatusWatchResult(
        run_id="100",
        watch_exit_code=0,
        exit_code=0,
        artifact_output_root=tmp_path / "artifacts",
        collection=None,
    )
    payload = result.summary_payload(options=options)
    assert payload["collection"] is None
    options_no_workflow = tooling_ci_watch.StatusWatchOptions(
        branch="stage",
        run_id=None,
        status=None,
        workflow=None,
        prefer_active=True,
        download_artifacts_on_failure=False,
        collect_failed_logs=True,
    )
    argv_no_workflow = options_no_workflow.to_argv()
    assert "--workflow" not in argv_no_workflow


def test_decode_and_failure_extractors_handle_invalid_payload_shapes() -> None:
    assert tooling_ci_watch._decode_json_dict("{not-json") == {}
    assert tooling_ci_watch._decode_json_dict("[]") == {}
    payload = {
        "jobs": [
            "bad",
            {"conclusion": "success"},
            {"conclusion": "failure", "name": "audit", "status": "completed", "url": "u"},
        ]
    }
    failed_jobs = tooling_ci_watch._failed_jobs(payload)
    assert failed_jobs == [
        {
            "databaseId": None,
            "name": "audit",
            "status": "completed",
            "conclusion": "failure",
            "url": "u",
        }
    ]
    failed_steps = tooling_ci_watch._failed_steps(
        {
            "jobs": [
                "bad",
                {"name": "audit", "steps": "bad"},
                {
                    "name": "audit",
                    "databaseId": 7,
                    "url": "u",
                    "steps": [
                        "bad",
                        {"conclusion": "success"},
                        {"number": 3, "name": "step", "status": "completed", "conclusion": "failure"},
                    ],
                },
            ]
        }
    )
    assert failed_steps == [
        {
            "job_name": "audit",
            "job_databaseId": 7,
            "step_number": 3,
            "step_name": "step",
            "status": "completed",
            "conclusion": "failure",
            "job_url": "u",
        }
    ]
    assert tooling_ci_watch._failed_jobs({"jobs": "bad"}) == []
    assert tooling_ci_watch._failed_steps({"jobs": "bad"}) == []


def test_find_run_id_and_no_result_path() -> None:
    stderr_messages: list[str] = []
    commands: list[list[str]] = []

    def _fake_run_handler(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        if "--status" in cmd and cmd[cmd.index("--status") + 1] == "completed":
            return subprocess.CompletedProcess(cmd, 0, json.dumps([]), "")
        return subprocess.CompletedProcess(
            cmd,
            0,
            json.dumps([{"databaseId": 44, "status": "in_progress"}]),
            "",
        )

    deps = _make_deps(run_handler=_fake_run_handler, stderr_messages=stderr_messages)
    assert (
        tooling_ci_watch._find_run_id(deps, "stage", "in_progress", "ci")
        == "44"
    )
    assert tooling_ci_watch._find_run_id(deps, "stage", "completed", "ci") is None
    assert any("--workflow" in command for command in commands)


def test_run_watch_uses_prefer_active_then_fallback_lookup() -> None:
    stderr_messages: list[str] = []
    statuses_queried: list[str] = []

    def _fake_run_handler(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gh", "run", "list"]:
            status_value = cmd[cmd.index("--status") + 1]
            statuses_queried.append(status_value)
            if status_value == "pending":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    json.dumps([{"databaseId": 77, "status": "pending"}]),
                    "",
                )
            return subprocess.CompletedProcess(cmd, 0, json.dumps([]), "")
        if cmd[:3] == ["gh", "run", "watch"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected command: {cmd}")

    deps = _make_deps(run_handler=_fake_run_handler, stderr_messages=stderr_messages)
    result = tooling_ci_watch.run_watch(
        options=tooling_ci_watch.StatusWatchOptions(
            branch="stage",
            run_id=None,
            status="completed",
            workflow="ci",
            prefer_active=True,
            download_artifacts_on_failure=False,
        ),
        deps=deps,
    )
    assert result.exit_code == 0
    assert result.run_id == "77"
    assert statuses_queried[:5] == [
        "in_progress",
        "queued",
        "requested",
        "waiting",
        "pending",
    ]


def test_run_watch_raises_when_no_runs_found() -> None:
    stderr_messages: list[str] = []

    def _fake_run_handler(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gh", "run", "list"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps([]), "")
        raise AssertionError(f"unexpected command: {cmd}")

    deps = _make_deps(run_handler=_fake_run_handler, stderr_messages=stderr_messages)
    with pytest.raises(SystemExit, match="No runs found for branch stage"):
        tooling_ci_watch.run_watch(
            options=tooling_ci_watch.StatusWatchOptions(
                branch="stage",
                prefer_active=True,
                download_artifacts_on_failure=False,
            ),
            deps=deps,
        )


def test_run_watch_can_skip_prefer_active_and_use_fallback_status() -> None:
    stderr_messages: list[str] = []
    statuses_queried: list[str] = []

    def _fake_run_handler(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gh", "run", "list"]:
            status_value = cmd[cmd.index("--status") + 1]
            statuses_queried.append(status_value)
            return subprocess.CompletedProcess(
                cmd,
                0,
                json.dumps([{"databaseId": 66, "status": status_value}]),
                "",
            )
        if cmd[:3] == ["gh", "run", "watch"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected command: {cmd}")

    deps = _make_deps(run_handler=_fake_run_handler, stderr_messages=stderr_messages)
    result = tooling_ci_watch.run_watch(
        options=tooling_ci_watch.StatusWatchOptions(
            branch="stage",
            run_id=None,
            status="completed",
            workflow=None,
            prefer_active=False,
            download_artifacts_on_failure=False,
        ),
        deps=deps,
    )
    assert result.run_id == "66"
    assert statuses_queried == ["completed"]


def test_collect_failure_artifacts_can_skip_failed_logs(tmp_path: Path) -> None:
    run_id = "1001"
    stderr_messages: list[str] = []

    def _fake_run_handler(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:4] == ["gh", "run", "view", run_id] and "--json" in cmd:
            run_payload = json.dumps({"jobs": []})
            return subprocess.CompletedProcess(cmd, 0, run_payload, "")
        if cmd[:3] == ["gh", "run", "download"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected command: {cmd}")

    deps = _make_deps(run_handler=_fake_run_handler, stderr_messages=stderr_messages)
    result = tooling_ci_watch._collect_failure_artifacts(
        deps,
        run_id=run_id,
        output_root=tmp_path,
        artifact_names=[],
        collect_failed_logs=False,
    )
    assert result.status.log_failed_rc is None
    assert (result.run_root / "failed.log").read_text(encoding="utf-8") == ""


def test_run_watch_uses_default_deps() -> None:
    def _fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        assert cmd[:3] == ["gh", "run", "watch"]
        return subprocess.CompletedProcess(cmd, 0, "", "")

    original_run = subprocess.run
    try:
        subprocess.run = _fake_run  # type: ignore[assignment]
        result = tooling_ci_watch.run_watch(
            options=tooling_ci_watch.StatusWatchOptions(
                run_id="55",
                download_artifacts_on_failure=False,
            ),
            deps=None,
        )
    finally:
        subprocess.run = original_run  # type: ignore[assignment]
    assert result.exit_code == 0


def test_default_print_err_writes_stderr(capsys: Any) -> None:
    tooling_ci_watch._default_print_err("error-line")
    captured = capsys.readouterr()
    assert "error-line" in captured.err


def test_ci_watch_module_entrypoint_executes() -> None:
    original_argv = list(sys.argv)

    def _fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gh", "run", "watch"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected command: {cmd}")

    original_run = subprocess.run
    try:
        subprocess.run = _fake_run  # type: ignore[assignment]
        sys.argv = [
            "ci_watch.py",
            "--run-id",
            "88",
            "--no-download-artifacts-on-failure",
        ]
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("gabion.tooling.ci_watch", run_name="__main__")
        assert exc.value.code == 0
    finally:
        subprocess.run = original_run  # type: ignore[assignment]
        sys.argv = original_argv
