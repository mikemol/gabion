from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

from gabion.commands import transport_policy
from gabion.order_contract import ordered_or_sorted
from gabion.runtime import env_policy
from gabion.tooling import run_dataflow_stage
from tests.env_helpers import env_scope


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _base_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "report": tmp_path / "artifacts/audit_reports/dataflow_report.md",
        "timeout_json": tmp_path / "artifacts/audit_reports/timeout_progress.json",
        "timeout_md": tmp_path / "artifacts/audit_reports/timeout_progress.md",
        "deadline_json": tmp_path / "artifacts/out/deadline_profile.json",
        "deadline_md": tmp_path / "artifacts/out/deadline_profile.md",
        "resume": tmp_path / "artifacts/audit_reports/dataflow_resume_checkpoint_ci.json",
        "baseline": tmp_path / "baselines/dataflow_baseline.txt",
        "summary": tmp_path / "summary.md",
        "output": tmp_path / "github_output.txt",
    }


def _stage_paths(paths: dict[str, Path]) -> run_dataflow_stage.StagePaths:
    return run_dataflow_stage.StagePaths(
        report_path=paths["report"],
        timeout_progress_json_path=paths["timeout_json"],
        timeout_progress_md_path=paths["timeout_md"],
        deadline_profile_json_path=paths["deadline_json"],
        deadline_profile_md_path=paths["deadline_md"],
        obligation_trace_json_path=_obligation_trace_path(paths),
        resume_checkpoint_path=paths["resume"],
        baseline_path=paths["baseline"],
    )


def _obligation_trace_path(paths: dict[str, Path]) -> Path:
    return paths["deadline_json"].parent / "obligation_trace.json"


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_stage_ids_are_bounded_and_ordered::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._stage_ids
def test_stage_ids_are_bounded_and_ordered() -> None:
    assert run_dataflow_stage._stage_ids("run", 3) == ["run", "retry1", "retry2"]
    assert run_dataflow_stage._stage_ids("x", 2) == ["run", "retry1"]
    assert run_dataflow_stage._stage_ids("run", 6) == [
        "run",
        "retry1",
        "retry2",
        "retry3",
        "retry4",
        "retry5",
    ]
    assert run_dataflow_stage._stage_ids("run", 0) == []


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_parse_stage_strictness_profile_supports_named_and_positional::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._parse_stage_strictness_profile
def test_parse_stage_strictness_profile_supports_named_and_positional() -> None:
    assert run_dataflow_stage._parse_stage_strictness_profile("run=low,b=high,c=low") == {
        "run": "low",
    }
    assert run_dataflow_stage._parse_stage_strictness_profile("high,low") == {
        "run": "high",
        "retry1": "low",
    }
    assert run_dataflow_stage._parse_stage_strictness_profile("run=bad,b=high") == {}


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_check_command_includes_strictness_when_provided::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._check_command::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths
def test_check_command_includes_strictness_when_provided(tmp_path: Path) -> None:
    paths = _stage_paths(_base_paths(tmp_path))
    command = run_dataflow_stage._check_command(
        paths=paths,
        resume_on_timeout=1,
        strictness="low",
    )
    assert command[:6] == [sys.executable, "-m", "gabion", "check", "obsolescence", "delta"]
    assert "--baseline" in command
    assert str(run_dataflow_stage._OBSOLESCENCE_BASELINE_PATH) in command
    assert "--strictness" in command
    assert "low" in command


# gabion:evidence E:function_site::test_run_dataflow_stage.py::tests.test_run_dataflow_stage.test_check_command_includes_context_runtime_overrides
def test_check_command_includes_context_runtime_overrides(tmp_path: Path) -> None:
    paths = _stage_paths(_base_paths(tmp_path))
    with env_policy.lsp_timeout_override_scope(
        env_policy.LspTimeoutConfig(ticks=101, tick_ns=103)
    ):
        with transport_policy.transport_override_scope(
            transport_policy.TransportOverrideConfig(
                direct_requested=False,
                override_record_path="/tmp/override_record.json",
            )
        ):
            command = run_dataflow_stage._check_command(
                paths=paths,
                resume_on_timeout=1,
                strictness=None,
            )
    assert "--timeout" in command
    assert "10403ns" in command
    assert "--carrier" in command
    assert "lsp" in command
    assert "--carrier-override-record" in command
    assert "/tmp/override_record.json" in command


# gabion:evidence E:function_site::test_run_dataflow_stage.py::tests.test_run_dataflow_stage.test_check_command_uses_env_timeout_fallback_when_context_missing
def test_check_command_uses_env_timeout_fallback_when_context_missing(tmp_path: Path) -> None:
    paths = _stage_paths(_base_paths(tmp_path))
    with env_scope(
        {
            "GABION_LSP_TIMEOUT_TICKS": "31",
            "GABION_LSP_TIMEOUT_TICK_NS": "37",
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": None,
        }
    ):
        command = run_dataflow_stage._check_command(
            paths=paths,
            resume_on_timeout=1,
            strictness=None,
        )
    assert "--timeout" in command
    assert "1147ns" in command


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_stage_uses_progress_classification_fallback::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage.run_stage::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_json::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
def test_run_stage_uses_progress_classification_fallback(tmp_path: Path) -> None:
    paths = _base_paths(tmp_path)
    _write_text(paths["report"], "# report\n")
    _write_text(paths["timeout_md"], "timeout md\n")
    _write_text(paths["deadline_md"], "deadline md\n")
    _write_json(
        paths["timeout_json"],
        {"progress": {"classification": "timed_out_progress_resume"}},
    )
    _write_json(
        paths["deadline_json"],
        {
            "ticks_consumed": 10,
            "checks_total": 5,
            "ticks_per_ns": 0.01,
            "wall_total_elapsed_ns": 1_000_000_000,
        },
    )

    result = run_dataflow_stage.run_stage(
        stage_id="a",
        paths=_stage_paths(paths),
        resume_on_timeout=1,
        step_summary_path=paths["summary"],
        run_command_fn=lambda _cmd: 2,
    )

    assert result.analysis_state == "timed_out_progress_resume"
    assert result.is_timeout_resume is True
    assert result.terminal_status == "timeout_resume"
    assert "stage A" in paths["summary"].read_text()
    assert paths["report"].with_name("dataflow_report_stage_a.md").exists()
    assert paths["timeout_json"].with_name("timeout_progress_stage_a.json").exists()
    assert paths["deadline_json"].with_name("deadline_profile_stage_a.json").exists()


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_staged_retries_until_success::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage.run_staged::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_json::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
def test_run_staged_retries_until_success(tmp_path: Path) -> None:
    paths = _base_paths(tmp_path)
    _write_text(paths["timeout_md"], "timeout md\n")
    _write_text(paths["deadline_md"], "deadline md\n")
    calls = {"count": 0}

    def _run(_cmd: list[str] | tuple[str, ...]) -> int:
        calls["count"] += 1
        stage = "a" if calls["count"] == 1 else "b"
        _write_text(paths["report"], f"# report stage {stage}\n")
        _write_json(
            paths["timeout_json"],
            {"analysis_state": "timed_out_progress_resume" if stage == "a" else "done"},
        )
        _write_json(
            paths["deadline_json"],
            {
                "ticks_consumed": 10 * calls["count"],
                "checks_total": 5 * calls["count"],
                "ticks_per_ns": 0.01,
                "wall_total_elapsed_ns": 1_000_000_000 * calls["count"],
            },
        )
        return 2 if stage == "a" else 0

    results = run_dataflow_stage.run_staged(
        stage_ids=["a", "b", "c"],
        paths=_stage_paths(paths),
        resume_on_timeout=1,
        step_summary_path=paths["summary"],
        run_command_fn=_run,
        run_gate_fn=lambda _cmd: 0,
    )

    assert [result.stage_id for result in results] == ["a", "b"]
    assert results[-1].terminal_status == "success"
    assert paths["report"].with_name("dataflow_report_stage_a.md").exists()
    assert paths["report"].with_name("dataflow_report_stage_b.md").exists()
    assert not paths["report"].with_name("dataflow_report_stage_c.md").exists()


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_staged_skips_retry_when_wall_budget_reserved::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage.run_staged::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_json::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
def test_run_staged_skips_retry_when_wall_budget_reserved(tmp_path: Path) -> None:
    paths = _base_paths(tmp_path)
    _write_text(paths["timeout_md"], "timeout md\n")
    _write_text(paths["deadline_md"], "deadline md\n")
    calls = {"count": 0}

    def _run(_cmd: list[str] | tuple[str, ...]) -> int:
        calls["count"] += 1
        _write_text(paths["report"], "# report stage a\n")
        _write_json(paths["timeout_json"], {"analysis_state": "timed_out_progress_resume"})
        _write_json(
            paths["deadline_json"],
            {
                "ticks_consumed": 10,
                "checks_total": 5,
                "ticks_per_ns": 0.01,
                "wall_total_elapsed_ns": 1_000_000_000,
            },
        )
        return 2

    clock_samples = [0.0, 58.0]

    def _fake_monotonic() -> float:
        if clock_samples:
            return clock_samples.pop(0)
        return 58.0

    results = run_dataflow_stage.run_staged(
        stage_ids=["a", "b"],
        paths=_stage_paths(paths),
        resume_on_timeout=1,
        step_summary_path=paths["summary"],
        run_command_fn=_run,
        run_gate_fn=lambda _cmd: 0,
        max_wall_seconds=60,
        finalize_reserve_seconds=5,
        monotonic_fn=_fake_monotonic,
    )

    assert calls["count"] == 1
    assert [result.stage_id for result in results] == ["a"]
    assert "skipped due remaining wall budget" in paths["summary"].read_text()


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_staged_passes_stage_specific_strictness::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage.run_staged::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_json::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
def test_run_staged_passes_stage_specific_strictness(tmp_path: Path) -> None:
    paths = _base_paths(tmp_path)
    _write_text(paths["timeout_md"], "timeout md\n")
    _write_text(paths["deadline_md"], "deadline md\n")
    observed: list[tuple[str, ...]] = []

    def _run(cmd: list[str] | tuple[str, ...]) -> int:
        observed.append(tuple(cmd))
        _write_text(paths["report"], "# report\n")
        _write_json(paths["timeout_json"], {"analysis_state": "hard_failure"})
        _write_json(paths["deadline_json"], {"ticks_consumed": 1, "checks_total": 1})
        return 1

    run_dataflow_stage.run_staged(
        stage_ids=["a"],
        paths=_stage_paths(paths),
        resume_on_timeout=1,
        step_summary_path=paths["summary"],
        run_command_fn=_run,
        strictness_by_stage={"a": "low"},
    )
    assert observed
    assert "--strictness" in observed[0]
    strict_idx = observed[0].index("--strictness")
    assert observed[0][strict_idx + 1] == "low"


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_staged_stops_on_hard_failure::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage.run_staged::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_json::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
def test_run_staged_stops_on_hard_failure(tmp_path: Path) -> None:
    paths = _base_paths(tmp_path)
    _write_text(paths["report"], "# report\n")
    _write_text(paths["timeout_md"], "timeout md\n")
    _write_text(paths["deadline_md"], "deadline md\n")
    _write_json(paths["timeout_json"], {"analysis_state": "hard_failure"})
    _write_json(paths["deadline_json"], {"ticks_consumed": 1, "checks_total": 1})

    results = run_dataflow_stage.run_staged(
        stage_ids=["a", "b"],
        paths=_stage_paths(paths),
        resume_on_timeout=1,
        step_summary_path=paths["summary"],
        run_command_fn=lambda _cmd: 1,
    )

    assert [result.stage_id for result in results] == ["a"]
    assert results[-1].terminal_status == "hard_failure"
    assert not paths["report"].with_name("dataflow_report_stage_b.md").exists()


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_emit_stage_outputs_writes_terminal_and_stage_keys::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._emit_stage_outputs
def test_emit_stage_outputs_writes_terminal_and_stage_keys(tmp_path: Path) -> None:
    output_path = tmp_path / "github_output.txt"
    results = [
        run_dataflow_stage.StageResult(
            stage_id="a",
            exit_code=2,
            analysis_state="timed_out_progress_resume",
            is_timeout_resume=True,
            metrics_line="ticks=1 checks=1 ticks_per_ns=0.1 wall_s=1.000",
            obligation_rows=(),
            incompleteness_markers=(),
        ),
        run_dataflow_stage.StageResult(
            stage_id="b",
            exit_code=0,
            analysis_state="done",
            is_timeout_resume=False,
            metrics_line="ticks=2 checks=2 ticks_per_ns=0.2 wall_s=2.000",
            obligation_rows=(),
            incompleteness_markers=(),
        ),
    ]
    run_dataflow_stage._emit_stage_outputs(output_path, results)
    payload = output_path.read_text()
    assert "stage_a_exit=2" in payload
    assert "stage_b_exit=0" in payload
    assert "attempts_run=2" in payload
    assert "terminal_stage=B" in payload
    assert "terminal_status=success" in payload
    assert "analysis_state=done" in payload


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_staged_marks_success_as_failure_when_delta_gate_fails::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage.run_staged::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_json::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
def test_run_staged_marks_success_as_failure_when_delta_gate_fails(tmp_path: Path) -> None:
    paths = _base_paths(tmp_path)
    _write_text(paths["timeout_md"], "timeout md\n")
    _write_text(paths["deadline_md"], "deadline md\n")

    def _run(_cmd: list[str] | tuple[str, ...]) -> int:
        _write_text(paths["report"], "# report stage a\n")
        _write_json(paths["timeout_json"], {"analysis_state": "done"})
        _write_json(
            paths["deadline_json"],
            {
                "ticks_consumed": 10,
                "checks_total": 5,
                "ticks_per_ns": 0.01,
                "wall_total_elapsed_ns": 1_000_000_000,
            },
        )
        return 0

    gate_calls: list[str] = []

    def _run_gate(step_id: str) -> int:
        gate_calls.append(step_id)
        return 1 if step_id == "annotation_drift_orphaned_gate" else 0

    results = run_dataflow_stage.run_staged(
        stage_ids=["a", "b"],
        paths=_stage_paths(paths),
        resume_on_timeout=1,
        step_summary_path=paths["summary"],
        run_command_fn=_run,
        run_gate_fn=_run_gate,
    )

    assert [result.stage_id for result in results] == ["a"]
    assert results[-1].terminal_status == "hard_failure"
    assert results[-1].analysis_state == "delta_gate_failure"
    assert len(gate_calls) >= 1
    assert any(step_id == "annotation_drift_orphaned_gate" for step_id in gate_calls)

# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_obligation_trace_payload_covers_satisfied_unsatisfied_and_policy_skip::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._obligation_rows_from_timeout_payload::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._obligation_trace_payload
def test_obligation_trace_payload_covers_satisfied_unsatisfied_and_policy_skip() -> None:
    rows, markers = run_dataflow_stage._obligation_rows_from_timeout_payload(
        stage_id="a",
        analysis_state="timed_out_progress_resume",
        timeout_payload={
            "incremental_obligations": [
                {
                    "contract": "resume_contract",
                    "kind": "checkpoint_present_when_resumable",
                    "status": "SATISFIED",
                    "detail": "checkpoint.json",
                },
                {
                    "contract": "progress_report_contract",
                    "kind": "partial_report_emitted",
                    "status": "VIOLATION",
                    "detail": "partial report emission on timeout",
                },
                {
                    "contract": "incremental_projection_contract",
                    "kind": "section_projection_state",
                    "status": "OBLIGATION",
                    "detail": "policy",
                    "section_id": "components",
                },
            ]
        },
    )

    assert markers == ()
    assert ordered_or_sorted(
        (row["status"] for row in rows),
        source="test_obligation_trace_payload_covers_satisfied_unsatisfied_and_policy_skip.statuses",
    ) == [
        "satisfied",
        "skipped_by_policy",
        "unsatisfied",
    ]
    trace = run_dataflow_stage._obligation_trace_payload(
        [
            run_dataflow_stage.StageResult(
                stage_id="a",
                exit_code=2,
                analysis_state="timed_out_progress_resume",
                is_timeout_resume=True,
                metrics_line="ticks=n/a checks=n/a ticks_per_ns=n/a wall_s=n/a",
                obligation_rows=rows,
                incompleteness_markers=(),
            )
        ]
    )
    assert trace["summary"] == {
        "total": 3,
        "satisfied": 1,
        "unsatisfied": 1,
        "skipped_by_policy": 1,
    }
    assert trace["complete"] is False
    assert "timeout_or_partial_run" in trace["incompleteness_markers"]


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_timeout_stage_with_missing_incremental_obligations_marks_incomplete::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._obligation_rows_from_timeout_payload
def test_timeout_stage_with_missing_incremental_obligations_marks_incomplete() -> None:
    rows, markers = run_dataflow_stage._obligation_rows_from_timeout_payload(
        stage_id="a",
        analysis_state="timed_out_progress_resume",
        timeout_payload={"analysis_state": "timed_out_progress_resume"},
    )
    assert rows == ()
    assert markers == ("missing_incremental_obligations",)


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_debug_dump_state_transitions_track_active_stage::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._debug_dump_stage_end::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._debug_dump_stage_start
def test_debug_dump_state_transitions_track_active_stage() -> None:
    state = run_dataflow_stage.DebugDumpState(
        stage_ids=("a", "b"),
        started_wall_seconds=10.0,
    )
    command = ("/usr/bin/python", "-m", "gabion", "check")
    run_dataflow_stage._debug_dump_stage_start(
        state=state,
        stage_id="a",
        command=command,
        strictness="high",
        monotonic_fn=lambda: 15.0,
    )
    assert state.attempts_started == 1
    assert state.active_stage_id == "a"
    assert state.active_stage_started_wall_seconds == 15.0
    assert state.active_stage_strictness == "high"
    assert state.active_command == command
    result = run_dataflow_stage.StageResult(
        stage_id="a",
        exit_code=2,
        analysis_state="timed_out_progress_resume",
        is_timeout_resume=True,
        metrics_line="ticks=1 checks=1 ticks_per_ns=0.1 wall_s=1.000",
        obligation_rows=(),
        incompleteness_markers=(),
    )
    run_dataflow_stage._debug_dump_stage_end(state=state, result=result)
    assert state.attempts_completed == 1
    assert state.active_stage_id is None
    assert state.active_stage_started_wall_seconds is None
    assert state.active_stage_strictness is None
    assert state.active_command == ()
    assert state.last_analysis_state == "timed_out_progress_resume"
    assert state.last_terminal_status == "timeout_resume"


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_emit_debug_dump_writes_state_lines::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._emit_debug_dump::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._phase_timeline_markdown_path::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_json::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
def test_emit_debug_dump_writes_state_lines(tmp_path: Path, capsys) -> None:
    paths = _base_paths(tmp_path)
    stage_paths = _stage_paths(paths)
    _write_json(
        paths["timeout_json"],
        {
            "analysis_state": "timed_out_progress_resume",
            "progress": {
                "classification": "timed_out_progress_resume",
                "phase": "post",
                "work_done": 1,
                "work_total": 6,
                "completed_files": 283,
                "remaining_files": 0,
                "total_files": 283,
            },
        },
    )
    _write_json(
        paths["deadline_json"],
        {
            "ticks_consumed": 10,
            "checks_total": 5,
            "ticks_per_ns": 0.01,
            "wall_total_elapsed_ns": 1_000_000_000,
        },
    )
    _write_json(
        paths["resume"],
        {
            "completed_paths": [{"path": "sample.py"}],
            "analysis_index_resume": {
                "hydrated_paths_count": 1,
                "profiling_v1": {"counters": {"analysis_index.paths_parsed": 2}},
            },
        },
    )
    timeline_path = run_dataflow_stage._phase_timeline_markdown_path(paths["report"])
    timeline_jsonl_path = run_dataflow_stage._phase_timeline_jsonl_path(paths["report"])
    _write_text(
        timeline_path,
        "\n".join(
            (
                "# Dataflow Phase Timeline",
                "",
                "| ts_utc | event_seq | event_kind | phase | analysis_state | classification | progress_marker | primary | files | resume_checkpoint | stale_for_s | dimensions |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                "| 2026-02-20T00:00:00Z | 1 | heartbeat | post | analysis_post_in_progress |  | deadline_obligations:start | 0/1 post_tasks | 283/283 rem=0 |  | 12.0 | post_tasks=0/1 |",
            )
        )
        + "\n",
    )
    _write_text(
        timeline_jsonl_path,
        '{"event_seq":1,"event_kind":"heartbeat","phase":"post"}\n',
    )
    state = run_dataflow_stage.DebugDumpState(
        stage_ids=("a", "b", "c"),
        started_wall_seconds=0.0,
        attempts_started=1,
        attempts_completed=0,
        active_stage_id="a",
        active_stage_started_wall_seconds=5.0,
        active_stage_strictness="high",
        active_command=("/usr/bin/python", "-m", "gabion", "check"),
    )
    with run_dataflow_stage.deadline_scope_from_lsp_env():
        run_dataflow_stage._emit_debug_dump(
            reason="SIGUSR1",
            state=state,
            paths=stage_paths,
            step_summary_path=paths["summary"],
            monotonic_fn=lambda: 10.0,
        )
    captured = capsys.readouterr().out
    assert "debug dump: reason=SIGUSR1" in captured
    assert "active_stage=a" in captured
    assert "resume_checkpoint=present completed_paths=1 hydrated_paths=1" in captured
    assert "timeout_progress_state=timed_out_progress_resume" in captured
    assert "phase_timeline_rows=1" in captured
    assert "phase_timeline_jsonl_present=yes" in captured
    assert "phase_timeline_last_row=| 2026-02-20T00:00:00Z | 1 | heartbeat | post |" in captured
    assert "phase_timeline_last_stale_for_s=12.0" in captured
    assert "debug dump: reason=SIGUSR1" in paths["summary"].read_text()


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_reset_run_observability_artifacts_clears_stale_debug_inputs_only::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._phase_timeline_markdown_path::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._phase_timeline_jsonl_path::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._reset_run_observability_artifacts::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
def test_reset_run_observability_artifacts_clears_stale_debug_inputs_only(
    tmp_path: Path,
) -> None:
    paths = _base_paths(tmp_path)
    stage_paths = _stage_paths(paths)
    timeline_md_path = run_dataflow_stage._phase_timeline_markdown_path(paths["report"])
    timeline_jsonl_path = run_dataflow_stage._phase_timeline_jsonl_path(paths["report"])
    stale_paths = (
        paths["timeout_json"],
        paths["timeout_md"],
        paths["deadline_json"],
        paths["deadline_md"],
        _obligation_trace_path(paths),
        timeline_md_path,
        timeline_jsonl_path,
        paths["report"].parent / "dataflow_checkpoint_intro_timeline.md",
    )
    for stale_path in stale_paths:
        _write_text(stale_path, "stale\n")
    _write_text(paths["resume"], "{\"checkpoint\":\"keep\"}\n")

    run_dataflow_stage._reset_run_observability_artifacts(stage_paths)

    for stale_path in stale_paths:
        assert not stale_path.exists()
    assert paths["resume"].exists()


class _FakeSignalModule:
    SIGUSR1 = object()

    def __init__(self, previous_handler) -> None:
        self._handlers = {self.SIGUSR1: previous_handler}

    def getsignal(self, sig):
        return self._handlers.get(sig)

    def signal(self, sig, handler):
        previous = self._handlers.get(sig)
        self._handlers[sig] = handler
        return previous


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_install_signal_debug_dump_handler_registers_and_restores::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._install_signal_debug_dump_handler
def test_install_signal_debug_dump_handler_registers_and_restores() -> None:
    observed: list[str] = []

    def _previous_handler(_signum, _frame) -> None:
        return None

    signal_module = _FakeSignalModule(_previous_handler)
    restore = run_dataflow_stage._install_signal_debug_dump_handler(
        emit_dump_fn=observed.append,
        signal_module=signal_module,
    )

    active_handler = signal_module.getsignal(signal_module.SIGUSR1)
    assert callable(active_handler)
    assert active_handler is not _previous_handler
    active_handler(10, None)
    assert observed == ["SIGUSR1"]

    restore()
    assert signal_module.getsignal(signal_module.SIGUSR1) is _previous_handler


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_parse_args_and_run_subprocess_edges::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._parse_args::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._run_subprocess
def test_parse_args_and_run_subprocess_edges() -> None:
    with env_scope({"GABION_DATAFLOW_DEBUG_DUMP_INTERVAL_SECONDS": "5"}):
        args = run_dataflow_stage._parse_args([])
    assert args.debug_dump_interval_seconds == 5
    assert args.stage_id == "run"

    assert (
        run_dataflow_stage._run_subprocess(
            ["ignored"],
            popen_fn=lambda _command: (_ for _ in ()).throw(OSError("nope")),
        )
        == 127
    )

    heartbeats: list[str] = []

    class _FakeProcess:
        def __init__(self) -> None:
            self._polls = 0

        def poll(self):
            self._polls += 1
            if self._polls >= 3:
                return 0
            return None

    monotonic_samples = [0.0, 2.0, 4.0, 6.0]

    def _fake_monotonic() -> float:
        if monotonic_samples:
            return monotonic_samples.pop(0)
        return 10.0

    assert (
        run_dataflow_stage._run_subprocess(
            ["ignored"],
            heartbeat_interval_seconds=1,
            on_heartbeat=lambda: heartbeats.append("tick"),
            popen_fn=lambda _command: _FakeProcess(),
            monotonic_fn=_fake_monotonic,
            sleep_fn=lambda _seconds: None,
        )
        == 0
    )
    assert heartbeats


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_named_delta_gate_unknown_and_crash::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._run_named_delta_gate
def test_run_named_delta_gate_unknown_and_crash() -> None:
    assert run_dataflow_stage._run_named_delta_gate("missing-gate") == 2
    run_dataflow_stage._DELTA_GATE_REGISTRY["boom"] = lambda: (_ for _ in ()).throw(
        RuntimeError("boom")
    )
    try:
        assert run_dataflow_stage._run_named_delta_gate("boom") == 2
    finally:
        run_dataflow_stage._DELTA_GATE_REGISTRY.pop("boom", None)


@contextmanager
def _noop_scope():
    yield


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_main_supports_injected_orchestration::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage.main
def test_main_supports_injected_orchestration(tmp_path: Path) -> None:
    paths = _base_paths(tmp_path)
    restore_calls: list[str] = []
    reset_calls: list[str] = []
    append_calls: list[str] = []
    output_calls: list[str] = []
    subprocess_calls: list[tuple[str, ...]] = []

    def _install_handler(*, emit_dump_fn):
        _ = emit_dump_fn

        def _restore() -> None:
            restore_calls.append("restored")

        return _restore

    def _run_subprocess_fn(command, **_kwargs):
        subprocess_calls.append(tuple(command))
        return 0

    def _run_staged_fn(**kwargs):
        run_command_fn = kwargs["run_command_fn"]
        assert run_command_fn(["echo", "ok"]) == 0
        return [
            run_dataflow_stage.StageResult(
                stage_id="run",
                exit_code=0,
                analysis_state="done",
                is_timeout_resume=False,
                metrics_line="ticks=1 checks=1 ticks_per_ns=1 wall_s=1.0",
                obligation_rows=(),
                incompleteness_markers=(),
            )
        ]

    def _write_trace(path: Path, _results):
        assert path == _obligation_trace_path(paths)
        return {
            "summary": {
                "total": 0,
                "satisfied": 0,
                "unsatisfied": 0,
                "skipped_by_policy": 0,
            },
            "complete": True,
            "incompleteness_markers": [],
        }

    args = [
        "--stage-id",
        "run",
        "--max-attempts",
        "1",
        "--report",
        str(paths["report"]),
        "--resume-checkpoint",
        str(paths["resume"]),
        "--baseline",
        str(paths["baseline"]),
        "--timeout-progress-json",
        str(paths["timeout_json"]),
        "--timeout-progress-md",
        str(paths["timeout_md"]),
        "--deadline-profile-json",
        str(paths["deadline_json"]),
        "--deadline-profile-md",
        str(paths["deadline_md"]),
        "--obligation-trace-json",
        str(_obligation_trace_path(paths)),
        "--github-output",
        str(paths["output"]),
        "--step-summary",
        str(paths["summary"]),
        "--debug-dump-interval-seconds",
        "1",
    ]

    exit_code = run_dataflow_stage.main(
        args,
        run_staged_fn=_run_staged_fn,
        write_obligation_trace_fn=_write_trace,
        append_markdown_summary_fn=lambda _path, _payload: append_calls.append("md"),
        append_lines_fn=lambda _path, _lines: append_calls.append("lines"),
        emit_stage_outputs_fn=lambda _path, _results: output_calls.append("outputs"),
        reset_run_observability_artifacts_fn=lambda _paths: reset_calls.append("reset"),
        install_signal_debug_dump_handler_fn=_install_handler,
        run_subprocess_fn=_run_subprocess_fn,
        deadline_scope_factory=_noop_scope,
    )
    assert exit_code == 0
    assert reset_calls == ["reset"]
    assert output_calls == ["outputs"]
    assert restore_calls == ["restored"]
    assert subprocess_calls == [("echo", "ok")]


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_main_returns_error_when_no_stage_ids_requested::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage.main
def test_main_returns_error_when_no_stage_ids_requested(tmp_path: Path) -> None:
    output_path = tmp_path / "output.env"
    summary_path = tmp_path / "summary.md"
    exit_code = run_dataflow_stage.main(
        [
            "--max-attempts",
            "0",
            "--github-output",
            str(output_path),
            "--step-summary",
            str(summary_path),
        ]
    )
    assert exit_code == 2


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_dataflow_stage_helper_branch_coverage::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._load_json_object::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._emit_stage_outputs
def test_run_dataflow_stage_helper_branch_coverage(tmp_path: Path) -> None:
    # _load_json_object branches
    missing_payload = run_dataflow_stage._load_json_object(tmp_path / "missing.json")
    assert missing_payload == {}
    non_dict_path = tmp_path / "list.json"
    non_dict_path.write_text("[]", encoding="utf-8")
    assert run_dataflow_stage._load_json_object(non_dict_path) == {}
    invalid_utf_path = tmp_path / "invalid.json"
    invalid_utf_path.write_bytes(b"\xff")
    assert run_dataflow_stage._load_json_object(invalid_utf_path) == {}

    # _analysis_state fallback to none
    timeout_json = tmp_path / "timeout.json"
    timeout_json.write_text(json.dumps({"progress": {"classification": 1}}), encoding="utf-8")
    assert run_dataflow_stage._analysis_state(timeout_json) == "none"

    # _metrics_line + resume helpers
    assert (
        run_dataflow_stage._metrics_line(tmp_path / "missing_deadline.json")
        == "ticks=n/a checks=n/a ticks_per_ns=n/a wall_s=n/a"
    )
    resume_path = tmp_path / "resume.json"
    resume_path.write_text(json.dumps({"completed_paths": ["x"], "analysis_index_resume": []}), encoding="utf-8")
    assert "hydrated_paths=n/a" in run_dataflow_stage._resume_checkpoint_metrics_line(resume_path)

    assert run_dataflow_stage._normalize_obligation_status("OBLIGATION", "other") == "unsatisfied"

    rows, markers = run_dataflow_stage._obligation_rows_from_timeout_payload(
        stage_id="run",
        analysis_state="timed_out_progress_resume",
        timeout_payload={
            "incremental_obligations": [
                "skip",
                {"contract": "", "kind": "x"},
                {"contract": "c", "kind": ""},
                {"contract": "c", "kind": "k", "status": "SATISFIED"},
            ],
            "cleanup_truncated": True,
        },
    )
    assert len(rows) == 1
    assert markers == ("cleanup_truncated",)

    trace_path = tmp_path / "trace.json"
    trace_payload = run_dataflow_stage._write_obligation_trace(
        trace_path,
        [
            run_dataflow_stage.StageResult(
                stage_id="run",
                exit_code=0,
                analysis_state="done",
                is_timeout_resume=False,
                metrics_line="m",
                obligation_rows=rows,
                incompleteness_markers=(),
            )
        ],
    )
    assert trace_path.exists()
    assert isinstance(trace_payload, dict)
    assert run_dataflow_stage._obligation_trace_summary_lines({"summary": []}) == []

    summary_md = tmp_path / "summary.md"
    summary_md.write_text("# summary\n", encoding="utf-8")
    run_dataflow_stage._append_markdown_summary(summary_md, trace_payload)
    assert "Obligation trace summary" in summary_md.read_text(encoding="utf-8")
    run_dataflow_stage._append_markdown_summary(tmp_path / "missing_summary.md", trace_payload)
    run_dataflow_stage._append_lines(None, ["ignored"])

    timeline = tmp_path / "timeline.md"
    assert run_dataflow_stage._markdown_timeline_row_count(timeline) == 0
    timeline_dir = tmp_path / "timeline_dir"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    assert run_dataflow_stage._markdown_timeline_row_count(timeline_dir) == 0
    assert run_dataflow_stage._markdown_timeline_last_row(timeline) == ""
    assert run_dataflow_stage._markdown_timeline_last_row(timeline_dir) == ""
    timeline.write_text("no-table\n", encoding="utf-8")
    assert run_dataflow_stage._markdown_timeline_last_row(timeline) == ""
    assert run_dataflow_stage._phase_timeline_stale_for_s_from_row("bad") == ""
    assert run_dataflow_stage._phase_timeline_stale_for_s_from_row("| only | two |") == ""
    assert run_dataflow_stage._command_preview(("x" * 400,), max_chars=10).endswith("...")

    assert (
        run_dataflow_stage._timeout_progress_metrics_line(tmp_path / "missing_timeout.json")
        == "timeout_progress=missing"
    )
    timeout_json.write_text(json.dumps({"analysis_state": "x", "progress": []}), encoding="utf-8")
    assert "classification=n/a phase=n/a" in run_dataflow_stage._timeout_progress_metrics_line(timeout_json)

    # _unlink_if_exists OSError branch via directory unlink.
    run_dataflow_stage._unlink_if_exists(tmp_path / "not_here")
    run_dataflow_stage._unlink_if_exists(timeline_dir)
    assert timeline_dir.exists()

    class _NoSignal:
        pass

    restore = run_dataflow_stage._install_signal_debug_dump_handler(
        emit_dump_fn=lambda _reason: None,
        signal_module=_NoSignal(),
    )
    restore()

    with env_scope({"TEST_INT_VALUE": "bad"}):
        assert run_dataflow_stage._env_int("TEST_INT_VALUE", 7) == 7

    assert run_dataflow_stage._parse_stage_strictness_profile("run=low,garbage") == {
        "run": "low"
    }

    output_env = tmp_path / "output.env"
    run_dataflow_stage._emit_stage_outputs(output_env, [])
    assert not output_env.exists()


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_staged_invokes_callbacks::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage.run_staged
def test_run_staged_invokes_callbacks(tmp_path: Path) -> None:
    paths = _base_paths(tmp_path)
    _write_text(paths["report"], "# report\n")
    _write_text(paths["timeout_md"], "timeout md\n")
    _write_text(paths["deadline_md"], "deadline md\n")
    _write_json(paths["timeout_json"], {"analysis_state": "done"})
    _write_json(paths["deadline_json"], {"ticks_consumed": 1, "checks_total": 1})

    starts: list[tuple[str, tuple[str, ...], str | None]] = []
    ends: list[str] = []

    def _on_start(stage_id: str, command: list[str] | tuple[str, ...], strictness: str | None) -> None:
        starts.append((stage_id, tuple(command), strictness))

    def _on_end(result: run_dataflow_stage.StageResult) -> None:
        ends.append(result.stage_id)

    results = run_dataflow_stage.run_staged(
        stage_ids=["run"],
        paths=_stage_paths(paths),
        resume_on_timeout=0,
        step_summary_path=paths["summary"],
        run_command_fn=lambda _cmd: 0,
        run_gate_fn=lambda _step_id: 0,
        on_stage_start=_on_start,
        on_stage_end=_on_end,
    )
    assert [result.stage_id for result in results] == ["run"]
    assert starts and starts[0][0] == "run"
    assert ends == ["run"]


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_main_uses_env_defaults_and_exercises_debug_callbacks::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage.main
def test_main_uses_env_defaults_and_exercises_debug_callbacks(tmp_path: Path) -> None:
    paths = _base_paths(tmp_path)
    output_env_file = tmp_path / "env_output.txt"
    summary_env_file = tmp_path / "env_summary.txt"
    traces: list[str] = []

    def _install_handler(*, emit_dump_fn):
        emit_dump_fn("unit-test")

        def _restore() -> None:
            traces.append("restored")

        return _restore

    def _run_staged_fn(**kwargs):
        on_stage_start = kwargs["on_stage_start"]
        on_stage_end = kwargs["on_stage_end"]
        stage_result = run_dataflow_stage.StageResult(
            stage_id="run",
            exit_code=0,
            analysis_state="done",
            is_timeout_resume=False,
            metrics_line="ticks=1 checks=1 ticks_per_ns=1 wall_s=1.0",
            obligation_rows=(),
            incompleteness_markers=(),
        )
        on_stage_start("run", ["echo", "ok"], None)
        on_stage_end(stage_result)
        return [stage_result]

    with env_scope(
        {
            "GITHUB_OUTPUT": str(output_env_file),
            "GITHUB_STEP_SUMMARY": str(summary_env_file),
        }
    ):
        exit_code = run_dataflow_stage.main(
            [
                "--stage-id",
                "run",
                "--max-attempts",
                "1",
                "--report",
                str(paths["report"]),
                "--resume-checkpoint",
                str(paths["resume"]),
                "--baseline",
                str(paths["baseline"]),
                "--timeout-progress-json",
                str(paths["timeout_json"]),
                "--timeout-progress-md",
                str(paths["timeout_md"]),
                "--deadline-profile-json",
                str(paths["deadline_json"]),
                "--deadline-profile-md",
                str(paths["deadline_md"]),
                "--obligation-trace-json",
                str(_obligation_trace_path(paths)),
            ],
            run_staged_fn=_run_staged_fn,
            write_obligation_trace_fn=lambda _path, _results: {
                "summary": {
                    "total": 0,
                    "satisfied": 0,
                    "unsatisfied": 0,
                    "skipped_by_policy": 0,
                },
                "complete": True,
                "incompleteness_markers": [],
            },
            append_markdown_summary_fn=lambda _path, _payload: None,
            append_lines_fn=lambda path, lines: run_dataflow_stage._append_lines(path, lines),
            emit_stage_outputs_fn=lambda path, results: run_dataflow_stage._emit_stage_outputs(path, results),
            reset_run_observability_artifacts_fn=lambda _paths: None,
            install_signal_debug_dump_handler_fn=_install_handler,
            run_subprocess_fn=lambda _command, **_kwargs: 0,
            deadline_scope_factory=_noop_scope,
        )
    assert exit_code == 0
    assert "restored" in traces
    assert output_env_file.exists()


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_dataflow_stage_additional_branch_edges_for_signal_subprocess_and_env_outputs::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._install_signal_debug_dump_handler::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._parse_stage_strictness_profile::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._run_subprocess::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage.main
def test_run_dataflow_stage_additional_branch_edges_for_signal_subprocess_and_env_outputs(
    tmp_path: Path,
) -> None:
    signal_module = _FakeSignalModule(previous_handler=None)
    restore = run_dataflow_stage._install_signal_debug_dump_handler(
        emit_dump_fn=lambda _reason: None,
        signal_module=signal_module,
    )
    restore()
    assert signal_module.getsignal(signal_module.SIGUSR1) is not None

    assert run_dataflow_stage._parse_stage_strictness_profile("low,bad") == {"run": "low"}
    timeout_json = tmp_path / "timeout_state.json"
    timeout_json.write_text(
        json.dumps({"progress": {"classification": "timed_out_progress_resume"}}),
        encoding="utf-8",
    )
    assert (
        run_dataflow_stage._analysis_state(timeout_json)
        == "timed_out_progress_resume"
    )

    class _FakeProcess:
        def __init__(self) -> None:
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return 0 if self.poll_count >= 2 else None

    assert (
        run_dataflow_stage._run_subprocess(
            ["ignored"],
            heartbeat_interval_seconds=1,
            on_heartbeat="not-callable",  # type: ignore[arg-type]
            popen_fn=lambda _command: _FakeProcess(),
            monotonic_fn=lambda: 0.0,
            sleep_fn=lambda _seconds: None,
        )
        == 0
    )

    heartbeat_calls: list[str] = []
    monotonic_values = [0.0, 0.1, 0.2]

    def _monotonic() -> float:
        if monotonic_values:
            return monotonic_values.pop(0)
        return 0.2

    assert (
        run_dataflow_stage._run_subprocess(
            ["ignored"],
            heartbeat_interval_seconds=1,
            on_heartbeat=lambda: heartbeat_calls.append("hb"),
            popen_fn=lambda _command: _FakeProcess(),
            monotonic_fn=_monotonic,
            sleep_fn=lambda _seconds: None,
        )
        == 0
    )
    assert heartbeat_calls == []

    paths = _base_paths(tmp_path)
    output_env_file = tmp_path / "output.env"
    summary_env_file = tmp_path / "summary.md"
    observed_append_paths: list[Path | None] = []

    def _run_staged_fn(**_kwargs):
        return [
            run_dataflow_stage.StageResult(
                stage_id="run",
                exit_code=0,
                analysis_state="done",
                is_timeout_resume=False,
                metrics_line="m",
                obligation_rows=(),
                incompleteness_markers=(),
            )
        ]

    with env_scope(
        {
            "GITHUB_OUTPUT": str(output_env_file),
            "GITHUB_STEP_SUMMARY": str(summary_env_file),
        }
    ):
        assert (
            run_dataflow_stage.main(
                [
                    "--stage-id",
                    "run",
                    "--max-attempts",
                    "1",
                    "--report",
                    str(paths["report"]),
                    "--resume-checkpoint",
                    str(paths["resume"]),
                    "--baseline",
                    str(paths["baseline"]),
                    "--timeout-progress-json",
                    str(paths["timeout_json"]),
                    "--timeout-progress-md",
                    str(paths["timeout_md"]),
                    "--deadline-profile-json",
                    str(paths["deadline_json"]),
                    "--deadline-profile-md",
                    str(paths["deadline_md"]),
                    "--obligation-trace-json",
                    str(_obligation_trace_path(paths)),
                ],
                run_staged_fn=_run_staged_fn,
                write_obligation_trace_fn=lambda _path, _results: {
                    "summary": {
                        "total": 0,
                        "satisfied": 0,
                        "unsatisfied": 0,
                        "skipped_by_policy": 0,
                    },
                    "complete": True,
                    "incompleteness_markers": [],
                },
                append_markdown_summary_fn=lambda _path, _payload: None,
                append_lines_fn=lambda path, lines: (
                    observed_append_paths.append(path),
                    run_dataflow_stage._append_lines(path, lines),
                )[-1],
                emit_stage_outputs_fn=lambda path, results: run_dataflow_stage._emit_stage_outputs(path, results),
                reset_run_observability_artifacts_fn=lambda _paths: None,
                install_signal_debug_dump_handler_fn=lambda **_kwargs: (lambda: None),
                run_subprocess_fn=lambda _command, **_kwargs: 0,
                deadline_scope_factory=_noop_scope,
            )
            == 0
        )
    assert summary_env_file in observed_append_paths
    assert output_env_file.exists()
    assert "analysis_state=done" in output_env_file.read_text(encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_dataflow_stage_remaining_branch_edges_for_progress_resume_and_main_env_fallbacks::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._analysis_state::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._resume_checkpoint_metrics_line::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._emit_debug_dump::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage._parse_stage_strictness_profile::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage.run_staged::run_dataflow_stage.py::gabion.tooling.run_dataflow_stage.main
def test_run_dataflow_stage_remaining_branch_edges_for_progress_resume_and_main_env_fallbacks(
    tmp_path: Path,
    capsys,
) -> None:
    timeout_json = tmp_path / "timeout_non_mapping_progress.json"
    timeout_json.write_text(json.dumps({"progress": []}), encoding="utf-8")
    assert run_dataflow_stage._analysis_state(timeout_json) == "none"

    resume_path = tmp_path / "resume_metrics.json"
    resume_path.write_text(
        json.dumps(
            {
                "completed_paths": ["a.py"],
                "analysis_index_resume": {
                    "hydrated_paths_count": 1,
                    "profiling_v1": [],
                },
            }
        ),
        encoding="utf-8",
    )
    assert (
        "paths_parsed_after_resume=n/a"
        in run_dataflow_stage._resume_checkpoint_metrics_line(resume_path)
    )
    resume_path.write_text(
        json.dumps(
            {
                "completed_paths": ["a.py"],
                "analysis_index_resume": {
                    "hydrated_paths_count": 1,
                    "profiling_v1": {"counters": []},
                },
            }
        ),
        encoding="utf-8",
    )
    assert (
        "paths_parsed_after_resume=n/a"
        in run_dataflow_stage._resume_checkpoint_metrics_line(resume_path)
    )

    paths = _base_paths(tmp_path)
    _write_text(paths["report"], "# report\n")
    _write_text(paths["timeout_md"], "timeout md\n")
    _write_text(paths["deadline_md"], "deadline md\n")
    _write_json(paths["timeout_json"], {"analysis_state": "done"})
    _write_json(paths["deadline_json"], {"ticks_consumed": 1, "checks_total": 1})
    _write_json(
        paths["resume"],
        {"completed_paths": [], "analysis_index_resume": {"hydrated_paths_count": 0}},
    )
    timeline_md = run_dataflow_stage._phase_timeline_markdown_path(paths["report"])
    _write_text(
        timeline_md,
        "| ts_utc | idx | kind | phase |\n"
        "| --- | --- | --- | --- |\n"
        "| 2026-02-20T00:00:00Z | 1 | heartbeat | post |\n",
    )
    timeline_jsonl = run_dataflow_stage._phase_timeline_jsonl_path(paths["report"])
    _write_text(timeline_jsonl, "{}\n")
    run_dataflow_stage._emit_debug_dump(
        reason="unit",
        state=run_dataflow_stage.DebugDumpState(
            stage_ids=("run",),
            started_wall_seconds=0.0,
        ),
        paths=_stage_paths(paths),
        step_summary_path=None,
        monotonic_fn=lambda: 1.0,
    )
    captured = capsys.readouterr().out
    assert "phase_timeline_last_row=" in captured
    assert "phase_timeline_last_stale_for_s=" not in captured

    assert run_dataflow_stage._parse_stage_strictness_profile("bad") == {}

    assert (
        run_dataflow_stage.run_staged(
            stage_ids=[],
            paths=_stage_paths(paths),
            resume_on_timeout=1,
            step_summary_path=paths["summary"],
            run_command_fn=lambda _command: 0,
        )
        == []
    )

    run_calls = {"count": 0}

    def _run_timeout_then_fail(_command: list[str] | tuple[str, ...]) -> int:
        run_calls["count"] += 1
        _write_text(paths["report"], "# report\n")
        _write_json(
            paths["timeout_json"],
            {
                "analysis_state": (
                    "timed_out_progress_resume"
                    if run_calls["count"] == 1
                    else "hard_failure"
                )
            },
        )
        _write_json(paths["deadline_json"], {"ticks_consumed": 1, "checks_total": 1})
        return 2

    monotonic_values = [0.0, 1.0]

    def _monotonic() -> float:
        if monotonic_values:
            return monotonic_values.pop(0)
        return 1.0

    results = run_dataflow_stage.run_staged(
        stage_ids=["run", "run"],
        paths=_stage_paths(paths),
        resume_on_timeout=1,
        step_summary_path=paths["summary"],
        run_command_fn=_run_timeout_then_fail,
        run_gate_fn=lambda _step: 0,
        max_wall_seconds=60,
        finalize_reserve_seconds=5,
        monotonic_fn=_monotonic,
    )
    assert [result.analysis_state for result in results] == [
        "timed_out_progress_resume",
        "hard_failure",
    ]

    output_paths: list[Path | None] = []
    summary_paths: list[Path | None] = []

    def _run_staged_fn(**_kwargs):
        return [
            run_dataflow_stage.StageResult(
                stage_id="run",
                exit_code=0,
                analysis_state="done",
                is_timeout_resume=False,
                metrics_line="m",
                obligation_rows=(),
                incompleteness_markers=(),
            )
        ]

    with env_scope({"GITHUB_OUTPUT": "", "GITHUB_STEP_SUMMARY": ""}):
        assert (
            run_dataflow_stage.main(
                [
                    "--stage-id",
                    "run",
                    "--max-attempts",
                    "1",
                    "--report",
                    str(paths["report"]),
                    "--resume-checkpoint",
                    str(paths["resume"]),
                    "--baseline",
                    str(paths["baseline"]),
                    "--timeout-progress-json",
                    str(paths["timeout_json"]),
                    "--timeout-progress-md",
                    str(paths["timeout_md"]),
                    "--deadline-profile-json",
                    str(paths["deadline_json"]),
                    "--deadline-profile-md",
                    str(paths["deadline_md"]),
                    "--obligation-trace-json",
                    str(_obligation_trace_path(paths)),
                ],
                run_staged_fn=_run_staged_fn,
                write_obligation_trace_fn=lambda _path, _results: {
                    "summary": {
                        "total": 0,
                        "satisfied": 0,
                        "unsatisfied": 0,
                        "skipped_by_policy": 0,
                    },
                    "complete": True,
                    "incompleteness_markers": [],
                },
                append_markdown_summary_fn=lambda _path, _payload: None,
                append_lines_fn=lambda path, _lines: summary_paths.append(path),
                emit_stage_outputs_fn=lambda path, _results: output_paths.append(path),
                reset_run_observability_artifacts_fn=lambda _paths: None,
                install_signal_debug_dump_handler_fn=lambda **_kwargs: (lambda: None),
                run_subprocess_fn=lambda _command, **_kwargs: 0,
                deadline_scope_factory=_noop_scope,
            )
            == 0
        )
    assert output_paths == [None]
    assert summary_paths == [None]
