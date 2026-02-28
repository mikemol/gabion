from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


_GUARD_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "ci_observability_guard.py"
)


def _run_guard(
    *,
    tmp_path: Path,
    label: str,
    max_gap_seconds: float,
    max_wall_seconds: float,
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    artifact_path = tmp_path / "artifacts" / "audit_reports" / "observability_violations.json"
    args = [
        sys.executable,
        str(_GUARD_SCRIPT),
        "--label",
        label,
        "--max-gap-seconds",
        str(max_gap_seconds),
        "--max-wall-seconds",
        str(max_wall_seconds),
        "--artifact-path",
        str(artifact_path),
        "--",
        *command,
    ]
    return subprocess.run(
        args,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


def _violations(tmp_path: Path) -> list[dict[str, object]]:
    artifact_path = tmp_path / "artifacts" / "audit_reports" / "observability_violations.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    violations = payload.get("violations")
    assert isinstance(violations, list)
    return [entry for entry in violations if isinstance(entry, dict)]


# gabion:evidence E:call_footprint::tests/test_ci_observability_guard.py::test_guard_fails_on_meaningful_gap_and_writes_artifact::ci_observability_guard.py::main
def test_guard_fails_on_meaningful_gap_and_writes_artifact(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import time; "
            "print('progress start', flush=True); "
            "time.sleep(0.30); "
            "print('progress done', flush=True)"
        ),
    ]
    result = _run_guard(
        tmp_path=tmp_path,
        label="gap_case",
        max_gap_seconds=0.10,
        max_wall_seconds=5.0,
        command=command,
    )

    assert result.returncode != 0
    violations = _violations(tmp_path)
    assert violations
    latest = violations[-1]
    assert latest.get("label") == "gap_case"
    assert latest.get("reason") == "max_gap_meaningful_line_exceeded"
    assert isinstance(latest.get("measured_gap_seconds"), float | int)


# gabion:evidence E:call_footprint::tests/test_ci_observability_guard.py::test_guard_excludes_heartbeat_lines_from_meaningful_gap_metric::ci_observability_guard.py::main
def test_guard_excludes_heartbeat_lines_from_meaningful_gap_metric(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import time; "
            "print('progress start', flush=True); "
            "time.sleep(0.06); "
            "print('heartbeat tick 1', flush=True); "
            "time.sleep(0.06); "
            "print('heartbeat tick 2', flush=True); "
            "time.sleep(0.06); "
            "print('progress done', flush=True)"
        ),
    ]
    result = _run_guard(
        tmp_path=tmp_path,
        label="heartbeat_case",
        max_gap_seconds=0.05,
        max_wall_seconds=5.0,
        command=command,
    )

    assert result.returncode != 0
    latest = _violations(tmp_path)[-1]
    assert latest.get("label") == "heartbeat_case"
    assert latest.get("reason") == "max_gap_meaningful_line_exceeded"


# gabion:evidence E:call_footprint::tests/test_ci_observability_guard.py::test_guard_enforces_wall_timeout::ci_observability_guard.py::main
def test_guard_enforces_wall_timeout(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import time; print('start', flush=True); time.sleep(0.5)",
    ]
    result = _run_guard(
        tmp_path=tmp_path,
        label="wall_case",
        max_gap_seconds=10.0,
        max_wall_seconds=0.15,
        command=command,
    )

    assert result.returncode != 0
    latest = _violations(tmp_path)[-1]
    assert latest.get("label") == "wall_case"
    assert latest.get("reason") == "max_wall_timeout"


# gabion:evidence E:call_footprint::tests/test_ci_observability_guard.py::test_guard_passes_when_meaningful_output_remains_below_gap::ci_observability_guard.py::main
def test_guard_passes_when_meaningful_output_remains_below_gap(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import time; "
            "print('progress 1', flush=True); "
            "time.sleep(0.02); "
            "print('progress 2', flush=True); "
            "time.sleep(0.02); "
            "print('progress 3', flush=True)"
        ),
    ]
    result = _run_guard(
        tmp_path=tmp_path,
        label="pass_case",
        max_gap_seconds=0.20,
        max_wall_seconds=5.0,
        command=command,
    )

    assert result.returncode == 0
    artifact_path = tmp_path / "artifacts" / "audit_reports" / "observability_violations.json"
    assert not artifact_path.exists()


# gabion:evidence E:call_footprint::tests/test_ci_observability_guard.py::test_guard_counts_in_place_progress_chunks_without_newlines::ci_observability_guard.py::main
def test_guard_counts_in_place_progress_chunks_without_newlines(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys,time; "
            "sys.stdout.write('progress '); sys.stdout.flush(); "
            "time.sleep(0.03); "
            "sys.stdout.write('.'); sys.stdout.flush(); "
            "time.sleep(0.03); "
            "sys.stdout.write('.'); sys.stdout.flush(); "
            "time.sleep(0.03); "
            "sys.stdout.write('.\\n'); sys.stdout.flush()"
        ),
    ]
    result = _run_guard(
        tmp_path=tmp_path,
        label="chunk_progress_case",
        max_gap_seconds=0.15,
        max_wall_seconds=5.0,
        command=command,
    )

    assert result.returncode == 0
    artifact_path = tmp_path / "artifacts" / "audit_reports" / "observability_violations.json"
    assert not artifact_path.exists()


# gabion:evidence E:call_footprint::tests/test_ci_observability_guard.py::test_guard_allows_cleanup_after_terminal_progress_row::ci_observability_guard.py::main
def test_guard_allows_cleanup_after_terminal_progress_row(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import time; "
            "print('| ts | progress | post | complete |', flush=True); "
            "time.sleep(0.25); "
            "print('done', flush=True)"
        ),
    ]
    result = _run_guard(
        tmp_path=tmp_path,
        label="terminal_cleanup_case",
        max_gap_seconds=0.10,
        max_wall_seconds=5.0,
        command=command,
    )

    assert result.returncode == 0
    artifact_path = tmp_path / "artifacts" / "audit_reports" / "observability_violations.json"
    assert not artifact_path.exists()
