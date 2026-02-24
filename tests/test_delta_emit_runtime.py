from __future__ import annotations

import time
from pathlib import Path

from gabion.tooling import delta_emit_runtime
from gabion.commands import transport_policy


def _progress_notification(*, phase: str, event_seq: int, work_done: int) -> dict[str, object]:
    return {
        "method": "$/progress",
        "params": {
            "token": "gabion.dataflowAudit/progress-v1",
            "value": {
                "phase": phase,
                "event_kind": "progress",
                "event_seq": event_seq,
                "work_done": work_done,
                "work_total": 3,
            },
        },
    }


# gabion:evidence E:call_footprint::tests/test_delta_emit_runtime.py::test_emit_timeline_row_ignores_empty_phase::delta_emit_runtime.py::gabion.tooling.delta_emit_runtime._emit_timeline_row
def test_emit_timeline_row_ignores_empty_phase() -> None:
    state = delta_emit_runtime.TimelineEmitState()
    lines: list[str] = []
    delta_emit_runtime._emit_timeline_row(
        state=state,
        phase_progress={"phase": ""},
        signature=("phase",),
        script_name="delta-test",
        print_fn=lines.append,
        now_monotonic=1.0,
    )
    assert lines == []
    assert state.last_signature is None


# gabion:evidence E:call_footprint::tests/test_delta_emit_runtime.py::test_flush_pending_timeline_row_normalizes_signature_and_keys::delta_emit_runtime.py::gabion.tooling.delta_emit_runtime.flush_pending_timeline_row_if_due
def test_flush_pending_timeline_row_normalizes_signature_and_keys() -> None:
    state = delta_emit_runtime.TimelineEmitState(
        timeline_header_emitted=True,
        last_timeline_emit_monotonic=0.0,
        last_timeline_phase="collection",
        pending_phase_progress={
            "phase": "collection",
            "event_kind": "progress",
            "event_seq": 2,
            "work_done": 2,
            "work_total": 3,
            9: "coerced-key",
        },
        pending_signature=None,
    )
    lines: list[str] = []
    delta_emit_runtime.flush_pending_timeline_row_if_due(
        state=state,
        script_name="delta-test",
        print_fn=lines.append,
        monotonic_fn=lambda: 1.0,
        force=True,
    )
    assert lines[0] == "delta-test timeline:"
    assert "collection" in lines[1]
    assert state.pending_phase_progress is None
    assert state.pending_signature is None


# gabion:evidence E:call_footprint::tests/test_delta_emit_runtime.py::test_run_delta_emit_flush_thread_emits_pending_rows::delta_emit_runtime.py::gabion.tooling.delta_emit_runtime.run_delta_emit
def test_run_delta_emit_flush_thread_emits_pending_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "delta_output.json"
    output_path.write_text("{}", encoding="utf-8")
    run_spec = delta_emit_runtime.DeltaEmitRunSpec(
        script_name="delta-runtime",
        failure_label="delta-runtime",
        expected_outputs=(output_path,),
    )
    lines: list[str] = []

    def _runner(_request, *, root=None, notification_callback=None):
        _ = root
        assert callable(notification_callback)
        notification_callback(
            _progress_notification(phase="collection", event_seq=1, work_done=1)
        )
        notification_callback(
            _progress_notification(phase="collection", event_seq=2, work_done=2)
        )
        time.sleep(0.30)
        return {"exit_code": 0}

    exit_code = delta_emit_runtime.run_delta_emit(
        run_spec=run_spec,
        payload={"analysis_timeout_ticks": 10, "analysis_timeout_tick_ns": 1},
        run_command_direct_fn=_runner,
        root_path=tmp_path,
        print_fn=lines.append,
        min_interval_seconds=0.2,
    )

    assert exit_code == 0
    assert any(line == "delta-runtime timeline:" for line in lines)
    assert any("delta-runtime: complete exit=0" in line for line in lines)


def test_run_delta_emit_handles_run_command_branch(tmp_path: Path) -> None:
    output_path = tmp_path / "delta_output.json"
    output_path.write_text("{}", encoding="utf-8")
    run_spec = delta_emit_runtime.DeltaEmitRunSpec(
        script_name="delta-runtime-run-command",
        failure_label="delta-runtime-run-command",
        expected_outputs=(output_path,),
    )
    lines: list[str] = []

    def _stub_runner(_request, *, root=None, timeout_ticks=None, timeout_tick_ns=None, notification_callback=None):
        _ = root, timeout_ticks, timeout_tick_ns
        if callable(notification_callback):
            notification_callback(
                _progress_notification(phase="collection", event_seq=1, work_done=1)
            )
        return {"exit_code": 0}

    original_runtime_run = delta_emit_runtime.run_command
    original_runtime_direct = delta_emit_runtime.run_command_direct
    original_transport_run = transport_policy.run_command
    try:
        delta_emit_runtime.run_command = _stub_runner
        delta_emit_runtime.run_command_direct = _stub_runner
        transport_policy.run_command = _stub_runner
        exit_code = delta_emit_runtime.run_delta_emit(
            run_spec=run_spec,
            payload={"analysis_timeout_ticks": 10, "analysis_timeout_tick_ns": 1},
            run_command_fn=_stub_runner,
            run_command_direct_fn=_stub_runner,
            root_path=tmp_path,
            print_fn=lines.append,
        )
    finally:
        delta_emit_runtime.run_command = original_runtime_run
        delta_emit_runtime.run_command_direct = original_runtime_direct
        transport_policy.run_command = original_transport_run

    assert exit_code == 0
    assert any("delta-runtime-run-command: complete exit=0" in line for line in lines)


def test_run_delta_emit_handles_custom_runner_branch(tmp_path: Path) -> None:
    output_path = tmp_path / "delta_output.json"
    output_path.write_text("{}", encoding="utf-8")
    run_spec = delta_emit_runtime.DeltaEmitRunSpec(
        script_name="delta-runtime-custom",
        failure_label="delta-runtime-custom",
        expected_outputs=(output_path,),
    )
    lines: list[str] = []

    def _custom_runner(_request, *, root=None):
        _ = root
        return {"exit_code": 0}

    exit_code = delta_emit_runtime.run_delta_emit(
        run_spec=run_spec,
        payload={"analysis_timeout_ticks": 10, "analysis_timeout_tick_ns": 1},
        run_command_fn=_custom_runner,
        run_command_direct_fn=_custom_runner,
        root_path=tmp_path,
        print_fn=lines.append,
    )

    assert exit_code == 0
    assert any("delta-runtime-custom: complete exit=0" in line for line in lines)
