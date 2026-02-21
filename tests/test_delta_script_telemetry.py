from __future__ import annotations

import time
from pathlib import Path

from scripts import delta_state_emit
from scripts import delta_triplets


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_state_emit_main_emits_progress_telemetry_and_deduplicates::delta_state_emit.py::scripts.delta_state_emit.main
def test_delta_state_emit_main_emits_progress_telemetry_and_deduplicates(
    tmp_path: Path,
) -> None:
    expected_state_paths = (
        tmp_path / "artifacts/out/test_obsolescence_state.json",
        tmp_path / "artifacts/out/test_annotation_drift.json",
        tmp_path / "artifacts/out/ambiguity_state.json",
    )
    for path in expected_state_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    lines: list[str] = []

    def _fake_run_command_direct(
        _request,
        *,
        root: Path,
        notification_callback=None,
    ) -> dict[str, object]:
        assert root == tmp_path
        assert callable(notification_callback)
        post_progress = {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "phase": "post",
                    "work_done": 0,
                    "work_total": 2,
                    "completed_files": 20,
                    "remaining_files": 5,
                    "total_files": 25,
                    "analysis_state": "analysis_post_in_progress",
                },
            },
        }
        post_done = {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "phase": "post",
                    "work_done": 2,
                    "work_total": 2,
                    "completed_files": 25,
                    "remaining_files": 0,
                    "total_files": 25,
                    "analysis_state": "succeeded",
                    "classification": "succeeded",
                    "done": True,
                },
            },
        }
        notification_callback(post_progress)
        notification_callback(post_progress)
        notification_callback(post_done)
        return {"exit_code": 0}

    exit_code = delta_state_emit.main(
        run_command_direct_fn=_fake_run_command_direct,
        print_fn=lines.append,
        monotonic_fn=time.monotonic,
        expected_state_paths=expected_state_paths,
        root_path=tmp_path,
    )

    assert exit_code == 0
    timeline_headers = [line for line in lines if line.startswith("| ts_utc |")]
    timeline_rows = [
        line
        for line in lines
        if line.startswith("| ")
        and not line.startswith("| ---")
        and not line.startswith("| ts_utc |")
    ]
    assert len(timeline_headers) == 1
    assert len(timeline_rows) == 2
    assert any("| post |" in line and "0/2" in line for line in timeline_rows)
    assert any("| post |" in line and "2/2" in line for line in timeline_rows)
    assert any("state artifacts ready" in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_state_emit_main_fails_when_expected_outputs_missing::delta_state_emit.py::scripts.delta_state_emit.main
def test_delta_state_emit_main_fails_when_expected_outputs_missing(tmp_path: Path) -> None:
    lines: list[str] = []

    def _fake_run_command_direct(_request, *, root: Path, notification_callback=None):
        _ = notification_callback
        assert root == tmp_path
        return {"exit_code": 0}

    missing_path = tmp_path / "artifacts/out/missing_state.json"
    exit_code = delta_state_emit.main(
        run_command_direct_fn=_fake_run_command_direct,
        print_fn=lines.append,
        monotonic_fn=time.monotonic,
        expected_state_paths=(missing_path,),
        root_path=tmp_path,
    )

    assert exit_code == 1
    assert any("missing expected state artifacts" in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_run_triplet_emits_step_telemetry_and_stops_after_emit_failure::delta_triplets.py::scripts.delta_triplets._run_triplet
def test_run_triplet_emits_step_telemetry_and_stops_after_emit_failure() -> None:
    lines: list[str] = []
    steps_seen: list[str] = []

    def _fake_run_step(**kwargs: object) -> int:
        step = str(kwargs["step"])
        steps_seen.append(step)
        return 2 if step.endswith("_emit.py") else 0

    exit_code = delta_triplets._run_triplet(
        "sample",
        ["scripts/sample_emit.py", "scripts/sample_gate.py"],
        run_step_fn=_fake_run_step,
        print_fn=lines.append,
        step_heartbeat_seconds=0.0,
    )

    assert exit_code == 2
    assert steps_seen == ["scripts/sample_emit.py"]
    assert any("sample triplet start: steps=2" in line for line in lines)
    assert any("sample step start 1/2: scripts/sample_emit.py" in line for line in lines)
    assert any("sample step failed: scripts/sample_emit.py (exit 2)" in line for line in lines)
    assert any("sample triplet aborting remaining steps because emit failed." in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_triplets_main_emits_pending_heartbeat::delta_triplets.py::scripts.delta_triplets.main
def test_delta_triplets_main_emits_pending_heartbeat() -> None:
    lines: list[str] = []

    def _slow_triplet(_name: str, _steps: list[str]) -> int:
        time.sleep(0.05)
        return 0

    exit_code = delta_triplets.main(
        triplets={"left": ["scripts/a.py"], "right": ["scripts/b.py"]},
        run_triplet_fn=_slow_triplet,
        print_fn=lines.append,
        pending_heartbeat_seconds=0.01,
    )

    assert exit_code == 0
    assert any(line.startswith("delta_triplets heartbeat: pending=") for line in lines)
    assert any("delta_triplets: complete failures=0 total=2" in line for line in lines)
