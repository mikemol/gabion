from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from gabion.commands import progress_contract as progress_timeline
from gabion.tooling import delta_state_emit
from gabion.tooling import delta_triplets
from tests.env_helpers import env_scope


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_state_emit_main_emits_progress_telemetry_and_deduplicates::delta_state_emit.py::gabion.tooling.delta_state_emit.main
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


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_state_emit_main_fails_when_expected_outputs_missing::delta_state_emit.py::gabion.tooling.delta_state_emit.main
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


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_run_triplet_emits_step_telemetry_and_stops_after_emit_failure::delta_triplets.py::gabion.tooling.delta_triplets._run_triplet
def test_run_triplet_emits_step_telemetry_and_stops_after_emit_failure() -> None:
    lines: list[str] = []
    steps_seen: list[str] = []

    def _fake_run_step(**kwargs: object) -> int:
        step = kwargs["step"]
        assert isinstance(step, delta_triplets.StepSpec)
        steps_seen.append(step.id)
        return 2 if step.kind == "emit" else 0

    exit_code = delta_triplets._run_triplet(
        "sample",
        [
            delta_triplets.StepSpec(
                id="sample_emit",
                label="sample_emit",
                kind="emit",
                run=lambda: 0,
            ),
            delta_triplets.StepSpec(
                id="sample_gate",
                label="sample_gate",
                kind="gate",
                run=lambda: 0,
            ),
        ],
        run_step_fn=_fake_run_step,
        print_fn=lines.append,
        step_heartbeat_seconds=0.0,
    )

    assert exit_code == 2
    assert steps_seen == ["sample_emit"]
    assert any("sample triplet start: steps=2" in line for line in lines)
    assert any("sample step start 1/2: sample_emit" in line for line in lines)
    assert any("sample step failed: sample_emit (exit 2)" in line for line in lines)
    assert any("sample triplet aborting remaining steps because emit failed." in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_triplets_main_emits_pending_heartbeat::delta_triplets.py::gabion.tooling.delta_triplets.main
def test_delta_triplets_main_emits_pending_heartbeat() -> None:
    lines: list[str] = []

    def _slow_triplet(_name: str, _steps: list[delta_triplets.StepSpec]) -> int:
        time.sleep(0.05)
        return 0

    exit_code = delta_triplets.main(
        triplets={
            "left": (
                delta_triplets.StepSpec(
                    id="left_step",
                    label="left_step",
                    kind="gate",
                    run=lambda: 0,
                ),
            ),
            "right": (
                delta_triplets.StepSpec(
                    id="right_step",
                    label="right_step",
                    kind="gate",
                    run=lambda: 0,
                ),
            ),
        },
        run_triplet_fn=_slow_triplet,
        print_fn=lines.append,
        pending_heartbeat_seconds=0.01,
    )

    assert exit_code == 0
    assert any(line.startswith("delta_triplets heartbeat: pending=") for line in lines)
    assert any("delta_triplets: complete failures=0 total=2" in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_state_emit_helper_branches_and_nonzero_exit::delta_state_emit.py::gabion.tooling.delta_state_emit._emit_phase_progress_line::delta_state_emit.py::gabion.tooling.delta_state_emit.main
def test_delta_state_emit_helper_branches_and_nonzero_exit(
    tmp_path: Path,
) -> None:
    dims_summary = delta_state_emit._phase_progress_dimensions_summary(
        {
            "dimensions": {
                "sites": {"done": 5, "total": 3},
                "skip": {"done": "x", "total": 1},
            }
        }
    )
    assert dims_summary == "sites=3/3"

    row = delta_state_emit._phase_timeline_row_from_phase_progress(
        {
            "ts_utc": "now",
            "phase": "forest",
            "analysis_state": "analysis_forest_in_progress",
            "classification": "analysis_forest_in_progress",
            "event_kind": "progress",
            "event_seq": 7,
            "phase_progress_v2": {"primary_unit": "items"},
            "work_done": 1,
            "work_total": 2,
            "resume_checkpoint": {"checkpoint_path": "", "status": "checkpoint_loaded"},
            "stale_for_s": 1.0,
        }
    )
    assert "7" in row
    assert "items" in row
    assert "reused_files=unknown" in row

    with env_scope(
        {
            "GABION_LSP_TIMEOUT_TICKS": "bad",
            "GABION_LSP_TIMEOUT_TICK_NS": "bad",
        }
    ):
        assert delta_state_emit._timeout_ticks() == int(delta_state_emit._DEFAULT_TIMEOUT_TICKS)
        assert delta_state_emit._timeout_tick_ns() == int(delta_state_emit._DEFAULT_TIMEOUT_TICK_NS)

    assert (
        delta_state_emit._phase_progress_from_notification({"method": "x"}) is None
    )
    assert (
        delta_state_emit._phase_progress_from_notification(
            {"method": "$/progress", "params": {"token": "bad"}}
        )
        is None
    )
    assert (
        delta_state_emit._phase_progress_from_notification(
            {"method": "$/progress", "params": {"token": "gabion.dataflowAudit/progress-v1", "value": {}}}
        )
        is None
    )

    lines: list[str] = []
    delta_state_emit._emit_phase_progress_line({}, print_fn=lines.append)
    assert not lines
    delta_state_emit._emit_phase_progress_line(
        {
            "phase": "post",
            "phase_timeline_header": "",
            "phase_timeline_row": "",
            "work_done": 0,
            "work_total": 1,
        },
        print_fn=lines.append,
    )
    assert any(line.startswith("| ts_utc |") for line in lines)

    expected_state_path = tmp_path / "state.json"
    expected_state_path.write_text("{}", encoding="utf-8")

    def _fail_run(_request, *, root: Path, notification_callback=None):
        _ = notification_callback
        assert root == tmp_path
        return {"exit_code": 4}

    exit_code = delta_state_emit.main(
        run_command_direct_fn=_fail_run,
        print_fn=lines.append,
        monotonic_fn=time.monotonic,
        expected_state_paths=(expected_state_path,),
        root_path=tmp_path,
    )
    assert exit_code == 4
    assert any("failed (exit 4)" in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_triplets_edge_paths::delta_triplets.py::gabion.tooling.delta_triplets._run_step_callable::delta_triplets.py::gabion.tooling.delta_triplets.main
def test_delta_triplets_edge_paths() -> None:
    assert delta_triplets._heartbeat_seconds("UNSET_ENV_FOR_TEST", 3.0) == 3.0
    with env_scope({"GABION_TRIPLET_TEST_HEARTBEAT": "bad"}):
        assert (
            delta_triplets._heartbeat_seconds("GABION_TRIPLET_TEST_HEARTBEAT", 2.0)
            == 2.0
        )

    lines: list[str] = []

    def _raise() -> int:
        raise RuntimeError("boom")

    step = delta_triplets.StepSpec(
        id="x",
        label="x",
        kind="gate",
        run=_raise,
    )
    assert (
        delta_triplets._run_step_callable(
            name="trip",
            step=step,
            step_index=1,
            step_total=1,
            step_heartbeat_seconds=0.0,
            print_fn=lines.append,
        )
        == 1
    )
    assert any("step failed: x (boom)" in line for line in lines)

    assert delta_triplets.main(triplets={}, print_fn=lines.append) == 0
    assert any("no triplets configured" in line for line in lines)

    def _crash_triplet(_name: str, _steps: list[delta_triplets.StepSpec]) -> int:
        raise RuntimeError("triplet crashed")

    exit_code = delta_triplets.main(
        triplets={
            "only": (
                delta_triplets.StepSpec(
                    id="only",
                    label="only",
                    kind="gate",
                    run=lambda: 0,
                ),
            )
        },
        run_triplet_fn=_crash_triplet,
        print_fn=lines.append,
        pending_heartbeat_seconds=0.0,
    )
    assert exit_code == 1
    assert any("triplet crashed" in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_state_emit_notification_and_payload_edge_branches::delta_state_emit.py::gabion.tooling.delta_state_emit._phase_progress_from_notification::delta_state_emit.py::gabion.tooling.delta_state_emit.main
def test_delta_state_emit_notification_and_payload_edge_branches(
    tmp_path: Path,
) -> None:
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with env_scope(
            {
                "GABION_LSP_TIMEOUT_TICKS": "1",
                "GABION_LSP_TIMEOUT_TICK_NS": "1",
            }
        ):
            payload = delta_state_emit._build_payload()
        assert payload["resume_checkpoint"] is False
    finally:
        os.chdir(previous_cwd)

    assert (
        delta_state_emit._phase_progress_from_notification(
            {"method": "$/progress", "params": []}
        )
        is None
    )
    assert (
        delta_state_emit._phase_progress_from_notification(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": [],
                },
            }
        )
        is None
    )
    normalized = delta_state_emit._phase_progress_from_notification(
        {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "phase": "forest",
                    "work_done": "1",
                    "work_total": "2",
                    "completed_files": "3",
                    "remaining_files": "4",
                    "total_files": "5",
                },
            },
        }
    )
    assert normalized is not None
    assert normalized["work_done"] is None
    assert normalized["work_total"] is None
    assert normalized["completed_files"] is None
    assert normalized["remaining_files"] is None
    assert normalized["total_files"] is None

    summary = delta_state_emit._phase_progress_dimensions_summary(
        {
            "dimensions": {
                "good": {"done": 1, "total": 2},
                "bad": "value",
            }
        }
    )
    assert summary == "good=1/2"

    row = delta_state_emit._phase_timeline_row_from_phase_progress(
        {
            "phase": "collection",
            "phase_progress_v2": {
                "primary_unit": "files",
                "primary_done": 5,
                "primary_total": 3,
            },
            "resume_checkpoint": {
                "checkpoint_path": "x.json",
                "status": "checkpoint_loaded",
                "reused_files": 1,
                "total_files": 3,
            },
        }
    )
    assert "3/3 files" in row
    assert "reused_files=1/3" in row

    expected_state_path = tmp_path / "state.json"
    expected_state_path.parent.mkdir(parents=True, exist_ok=True)
    expected_state_path.write_text("{}", encoding="utf-8")
    lines: list[str] = []

    def _run(_request, *, root: Path, notification_callback=None):
        assert root == tmp_path
        assert callable(notification_callback)
        notification_callback({"method": "ignored"})
        notification_callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "post",
                        "work_done": 1,
                        "work_total": 1,
                        "done": True,
                    },
                },
            }
        )
        return {"exit_code": 0}

    assert (
        delta_state_emit.main(
            run_command_direct_fn=_run,
            print_fn=lines.append,
            expected_state_paths=(expected_state_path,),
            root_path=tmp_path,
        )
        == 0
    )
    assert any("timeline" in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_triplets_timeout_heartbeat_and_nonzero_result_paths::delta_triplets.py::gabion.tooling.delta_triplets._run_step_callable::delta_triplets.py::gabion.tooling.delta_triplets.main
def test_delta_triplets_timeout_heartbeat_and_nonzero_result_paths() -> None:
    lines: list[str] = []
    monotonic_values = [0.0, 0.2, 0.2, 1.0]

    def _monotonic() -> float:
        if monotonic_values:
            return monotonic_values.pop(0)
        return 1.0

    step = delta_triplets.StepSpec(
        id="slow",
        label="slow",
        kind="gate",
        run=lambda: (time.sleep(1.1) or 0),
    )
    assert (
        delta_triplets._run_step_callable(
            name="trip",
            step=step,
            step_index=1,
            step_total=1,
            step_heartbeat_seconds=0.1,
            print_fn=lines.append,
            monotonic_fn=_monotonic,
        )
        == 0
    )
    assert any("step heartbeat 1/1: slow" in line for line in lines)

    lines.clear()
    assert (
        delta_triplets._run_triplet(
            "ok",
            [
                delta_triplets.StepSpec(
                    id="g1",
                    label="g1",
                    kind="gate",
                    run=lambda: 0,
                ),
                delta_triplets.StepSpec(
                    id="g2",
                    label="g2",
                    kind="gate",
                    run=lambda: 0,
                ),
            ],
            run_step_fn=lambda **_kwargs: 0,
            print_fn=lines.append,
            step_heartbeat_seconds=0.0,
        )
        == 0
    )
    assert any("ok triplet complete: exit=0" in line for line in lines)

    with env_scope({"GABION_TRIPLET_NONPOSITIVE": "0"}):
        assert delta_triplets._heartbeat_seconds("GABION_TRIPLET_NONPOSITIVE", 3.0) == 0.0

    lines.clear()
    assert (
        delta_triplets.main(
            triplets={"bad": (delta_triplets.StepSpec("s", "s", "gate", lambda: 0),)},
            run_triplet_fn=lambda _name, _steps: 1,
            print_fn=lines.append,
            pending_heartbeat_seconds=0.0,
        )
        == 1
    )
    assert any("complete failures=1 total=1" in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_triplets_non_emit_failure_continues_and_step_callable_without_heartbeat::delta_triplets.py::gabion.tooling.delta_triplets._run_step_callable::delta_triplets.py::gabion.tooling.delta_triplets._run_triplet
def test_delta_triplets_non_emit_failure_continues_and_step_callable_without_heartbeat() -> None:
    lines: list[str] = []
    assert (
        delta_triplets._run_step_callable(
            name="trip",
            step=delta_triplets.StepSpec(
                id="ok",
                label="ok",
                kind="gate",
                run=lambda: 0,
            ),
            step_index=1,
            step_total=1,
            step_heartbeat_seconds=0.0,
            print_fn=lines.append,
            monotonic_fn=lambda: 0.0,
        )
        == 0
    )
    assert lines == []

    executed: list[str] = []

    def _run_step(**kwargs) -> int:
        step = kwargs["step"]
        executed.append(step.label)
        return 2 if step.label == "gate_fail" else 0

    exit_code = delta_triplets._run_triplet(
        "trip",
        (
            delta_triplets.StepSpec("gate_fail", "gate_fail", "gate", lambda: 2),
            delta_triplets.StepSpec("after_fail", "after_fail", "gate", lambda: 0),
        ),
        run_step_fn=_run_step,
        print_fn=lines.append,
        monotonic_fn=lambda: 0.0,
        step_heartbeat_seconds=0.0,
    )
    assert exit_code == 2
    assert executed == ["gate_fail", "after_fail"]


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_triplets_step_callable_timeout_without_heartbeat_branch::delta_triplets.py::gabion.tooling.delta_triplets._run_step_callable
def test_delta_triplets_step_callable_timeout_without_heartbeat_branch() -> None:
    lines: list[str] = []

    def _slow_step() -> int:
        time.sleep(0.6)
        return 0

    assert (
        delta_triplets._run_step_callable(
            name="trip",
            step=delta_triplets.StepSpec(
                id="slow",
                label="slow",
                kind="gate",
                run=_slow_step,
            ),
            step_index=1,
            step_total=1,
            step_heartbeat_seconds=0.0,
            print_fn=lines.append,
            monotonic_fn=lambda: 0.0,
        )
        == 0
    )
    assert lines == []


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_state_emit_main_for_emitter_output_path_branch::delta_state_emit.py::gabion.tooling.delta_state_emit.main_for_emitter
def test_delta_state_emit_main_for_emitter_output_path_branch(tmp_path: Path) -> None:
    output_path = tmp_path / "custom_delta.json"

    def _run(_request, *, root: Path, notification_callback=None):
        _ = notification_callback
        assert root == tmp_path
        return {"exit_code": 1}

    assert (
        delta_state_emit.main_for_emitter(
            "obsolescence_delta_emit",
            run_command_direct_fn=_run,
            root_path=tmp_path,
            output_path=output_path,
            print_fn=lambda _line: None,
            monotonic_fn=time.monotonic,
        )
        == 1
    )


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_state_emit_main_for_emitter_without_output_path_branch::delta_state_emit.py::gabion.tooling.delta_state_emit.main_for_emitter
def test_delta_state_emit_main_for_emitter_without_output_path_branch(
    tmp_path: Path,
) -> None:
    def _run(_request, *, root: Path, notification_callback=None):
        _ = notification_callback
        assert root == tmp_path
        return {"exit_code": 1}

    assert (
        delta_state_emit.main_for_emitter(
            "obsolescence_delta_emit",
            run_command_direct_fn=_run,
            root_path=tmp_path,
            output_path=None,
            expected_outputs=None,
            print_fn=lambda _line: None,
            monotonic_fn=time.monotonic,
        )
        == 1
    )


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_triplets_emit_checkpoint_paths_are_partitioned::delta_triplets.py::gabion.tooling.delta_triplets._triplet_resume_checkpoint_path
def test_delta_triplets_emit_checkpoint_paths_are_partitioned() -> None:
    obsolescence_path = delta_triplets._triplet_resume_checkpoint_path("obsolescence")
    annotation_path = delta_triplets._triplet_resume_checkpoint_path("annotation-drift")
    ambiguity_path = delta_triplets._triplet_resume_checkpoint_path("ambiguity")
    assert obsolescence_path.name == "dataflow_resume_checkpoint_ci_obsolescence.json"
    assert annotation_path.name == "dataflow_resume_checkpoint_ci_annotation_drift.json"
    assert ambiguity_path.name == "dataflow_resume_checkpoint_ci_ambiguity.json"
    assert len({obsolescence_path, annotation_path, ambiguity_path}) == 3


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_triplets_emit_wrappers_pass_partitioned_resume_checkpoints::delta_triplets.py::gabion.tooling.delta_triplets._run_ambiguity_emit::delta_triplets.py::gabion.tooling.delta_triplets._run_annotation_drift_emit::delta_triplets.py::gabion.tooling.delta_triplets._run_obsolescence_emit
def test_delta_triplets_emit_wrappers_pass_partitioned_resume_checkpoints() -> None:
    captured: list[Path] = []

    def _capture(*, resume_checkpoint: Path) -> int:
        captured.append(resume_checkpoint)
        return 0

    assert delta_triplets._run_obsolescence_emit(run_emit=_capture) == 0
    assert delta_triplets._run_annotation_drift_emit(run_emit=_capture) == 0
    assert delta_triplets._run_ambiguity_emit(run_emit=_capture) == 0
    assert [path.name for path in captured] == [
        "dataflow_resume_checkpoint_ci_obsolescence.json",
        "dataflow_resume_checkpoint_ci_annotation_drift.json",
        "dataflow_resume_checkpoint_ci_ambiguity.json",
    ]


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_emit_modules_stream_phase_timeline_rows::delta_state_emit.py::gabion.tooling.delta_state_emit.obsolescence_main::delta_state_emit.py::gabion.tooling.delta_state_emit.annotation_drift_main::delta_state_emit.py::gabion.tooling.delta_state_emit.ambiguity_main
@pytest.mark.parametrize(
    "run_main",
    [
        delta_state_emit.obsolescence_main,
        delta_state_emit.annotation_drift_main,
        delta_state_emit.ambiguity_main,
    ],
)
def test_delta_emit_modules_stream_phase_timeline_rows(
    tmp_path: Path,
    run_main,
) -> None:
    delta_path = tmp_path / "delta.json"
    delta_path.write_text("{}\n", encoding="utf-8")
    lines: list[str] = []

    def _run(_request, *, root: Path, notification_callback=None) -> dict[str, object]:
        assert root == tmp_path
        assert callable(notification_callback)
        notification_callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "collection",
                        "event_seq": 1,
                        "phase_timeline_header": "| ts | phase |",
                        "phase_timeline_row": "| t0 | collection |",
                    },
                },
            }
        )
        notification_callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "post",
                        "event_seq": 2,
                        "phase_timeline_row": "| t1 | post |",
                    },
                },
            }
        )
        return {"exit_code": 0}

    exit_code = run_main(
        run_command_direct_fn=_run,
        root_path=tmp_path,
        delta_path=delta_path,
        print_fn=lines.append,
        monotonic_fn=time.monotonic,
    )
    assert exit_code == 0
    assert any("timeline:" in line for line in lines)
    assert any("| t0 | collection |" in line for line in lines)
    assert any("| t1 | post |" in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_emit_modules_supports_notification_callback_signature_edges::delta_state_emit.py::gabion.tooling.delta_state_emit._supports_notification_callback
def test_delta_emit_modules_supports_notification_callback_signature_edges() -> None:
    assert delta_state_emit._supports_notification_callback(object()) is True

    def _with_kwargs(*args, **kwargs):
        _ = args, kwargs
        return {"exit_code": 0}

    assert delta_state_emit._supports_notification_callback(_with_kwargs) is True


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_emit_modules_ignore_non_mapping_progress_and_dedupe_duplicates::delta_state_emit.py::gabion.tooling.delta_state_emit.obsolescence_main::delta_state_emit.py::gabion.tooling.delta_state_emit.annotation_drift_main::delta_state_emit.py::gabion.tooling.delta_state_emit.ambiguity_main
@pytest.mark.parametrize(
    "run_main",
    [
        delta_state_emit.obsolescence_main,
        delta_state_emit.annotation_drift_main,
        delta_state_emit.ambiguity_main,
    ],
)
def test_delta_emit_modules_ignore_non_mapping_progress_and_dedupe_duplicates(
    tmp_path: Path,
    run_main,
) -> None:
    delta_path = tmp_path / "delta.json"
    delta_path.write_text("{}\n", encoding="utf-8")
    lines: list[str] = []

    def _run(_request, *, root: Path, notification_callback=None) -> dict[str, object]:
        assert root == tmp_path
        assert callable(notification_callback)
        notification_callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": [],
                },
            }
        )
        progress = {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "phase": "collection",
                    "event_seq": 1,
                    "phase_timeline_header": "| ts | phase |",
                    "phase_timeline_row": "| t0 | collection |",
                },
            },
        }
        notification_callback(progress)
        notification_callback(progress)
        return {"exit_code": 0}

    exit_code = run_main(
        run_command_direct_fn=_run,
        root_path=tmp_path,
        delta_path=delta_path,
        print_fn=lines.append,
        monotonic_fn=time.monotonic,
    )
    assert exit_code == 0
    assert sum(1 for line in lines if line == "| t0 | collection |") == 1


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_shared_progress_timeline_helpers_cover_delta_state_wrappers::delta_state_emit.py::gabion.tooling.delta_state_emit._phase_timeline_header_columns::delta_state_emit.py::gabion.tooling.delta_state_emit._phase_timeline_header_block::progress_contract.py::gabion.commands.progress_contract.is_heartbeat_progress
def test_shared_progress_timeline_helpers_cover_delta_state_wrappers() -> None:
    assert (
        delta_state_emit._phase_timeline_header_columns()
        == progress_timeline.phase_timeline_header_columns()
    )
    assert (
        delta_state_emit._phase_timeline_header_block()
        == progress_timeline.phase_timeline_header_block()
    )
    assert progress_timeline.is_heartbeat_progress({"event_kind": "heartbeat"}) is True
    assert progress_timeline.is_heartbeat_progress({"event_kind": "progress"}) is False


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_progress_timeline_phase_progress_emit_due_edges::progress_contract.py::gabion.commands.progress_contract.phase_progress_emit_due
def test_progress_timeline_phase_progress_emit_due_edges() -> None:
    base = {"phase": "collection", "event_kind": "progress", "done": False}
    assert (
        progress_timeline.phase_progress_emit_due(
            phase_progress=base,
            timeline_header_emitted=False,
            last_emitted_phase=None,
            last_emitted_monotonic=None,
            now_monotonic=0.0,
        )
        is True
    )
    assert (
        progress_timeline.phase_progress_emit_due(
            phase_progress=base,
            timeline_header_emitted=True,
            last_emitted_phase="collection",
            last_emitted_monotonic=None,
            now_monotonic=0.1,
        )
        is True
    )
    assert (
        progress_timeline.phase_progress_emit_due(
            phase_progress=base,
            timeline_header_emitted=True,
            last_emitted_phase="collection",
            last_emitted_monotonic=1.0,
            now_monotonic=1.1,
        )
        is False
    )
    assert (
        progress_timeline.phase_progress_emit_due(
            phase_progress={**base, "done": True},
            timeline_header_emitted=True,
            last_emitted_phase="collection",
            last_emitted_monotonic=1.0,
            now_monotonic=1.1,
        )
        is True
    )
    assert (
        progress_timeline.phase_progress_emit_due(
            phase_progress={**base, "event_kind": "checkpoint"},
            timeline_header_emitted=True,
            last_emitted_phase="collection",
            last_emitted_monotonic=1.0,
            now_monotonic=1.1,
        )
        is True
    )
    assert (
        progress_timeline.phase_progress_emit_due(
            phase_progress={**base, "phase": "post"},
            timeline_header_emitted=True,
            last_emitted_phase="collection",
            last_emitted_monotonic=1.0,
            now_monotonic=1.1,
        )
        is True
    )


# gabion:evidence E:call_footprint::tests/test_delta_script_telemetry.py::test_delta_state_emit_main_throttles_same_phase_rows_but_forces_phase_transitions::delta_state_emit.py::gabion.tooling.delta_state_emit.main
def test_delta_state_emit_main_throttles_same_phase_rows_but_forces_phase_transitions(
    tmp_path: Path,
) -> None:
    expected_state_path = tmp_path / "state.json"
    expected_state_path.parent.mkdir(parents=True, exist_ok=True)
    expected_state_path.write_text("{}", encoding="utf-8")
    lines: list[str] = []
    monotonic_values = [0.0, 0.0, 0.1, 0.1, 0.2]

    def _monotonic() -> float:
        if monotonic_values:
            return monotonic_values.pop(0)
        return 0.2

    def _run(_request, *, root: Path, notification_callback=None):
        assert root == tmp_path
        assert callable(notification_callback)
        notification_callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "collection",
                        "event_seq": 1,
                        "phase_timeline_header": "| ts | phase |",
                        "phase_timeline_row": "| t0 | collection |",
                    },
                },
            }
        )
        notification_callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "collection",
                        "event_seq": 2,
                        "phase_timeline_row": "| t1 | collection |",
                    },
                },
            }
        )
        notification_callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "post",
                        "event_seq": 3,
                        "phase_timeline_row": "| t2 | post |",
                    },
                },
            }
        )
        return {"exit_code": 0}

    assert (
        delta_state_emit.main(
            run_command_direct_fn=_run,
            print_fn=lines.append,
            monotonic_fn=_monotonic,
            expected_state_paths=(expected_state_path,),
            root_path=tmp_path,
        )
        == 0
    )
    assert any(line == "| t0 | collection |" for line in lines)
    assert not any(line == "| t1 | collection |" for line in lines)
    assert any(line == "| t2 | post |" for line in lines)
