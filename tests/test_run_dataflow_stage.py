from __future__ import annotations

import json
from pathlib import Path

from scripts import run_dataflow_stage


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


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_stage_ids_are_bounded_and_ordered::run_dataflow_stage.py::scripts.run_dataflow_stage._stage_ids
def test_stage_ids_are_bounded_and_ordered() -> None:
    assert run_dataflow_stage._stage_ids("a", 3) == ["a", "b", "c"]
    assert run_dataflow_stage._stage_ids("b", 3) == ["b", "c"]
    assert run_dataflow_stage._stage_ids("x", 2) == ["a", "b"]
    assert run_dataflow_stage._stage_ids("a", 0) == []


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_parse_stage_strictness_profile_supports_named_and_positional::run_dataflow_stage.py::scripts.run_dataflow_stage._parse_stage_strictness_profile
def test_parse_stage_strictness_profile_supports_named_and_positional() -> None:
    assert run_dataflow_stage._parse_stage_strictness_profile("a=low,b=high,c=low") == {
        "a": "low",
        "b": "high",
        "c": "low",
    }
    assert run_dataflow_stage._parse_stage_strictness_profile("low,high") == {
        "a": "low",
        "b": "high",
    }
    assert run_dataflow_stage._parse_stage_strictness_profile("a=bad,b=high") == {
        "b": "high"
    }


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_check_command_includes_strictness_when_provided::run_dataflow_stage.py::scripts.run_dataflow_stage._check_command::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths
def test_check_command_includes_strictness_when_provided(tmp_path: Path) -> None:
    paths = _stage_paths(_base_paths(tmp_path))
    command = run_dataflow_stage._check_command(
        paths=paths,
        resume_on_timeout=1,
        strictness="low",
    )
    assert "--strictness" in command
    assert "low" in command


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_stage_uses_progress_classification_fallback::run_dataflow_stage.py::scripts.run_dataflow_stage.run_stage::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_json::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
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


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_staged_retries_until_success::run_dataflow_stage.py::scripts.run_dataflow_stage.run_staged::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_json::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
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


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_staged_skips_retry_when_wall_budget_reserved::run_dataflow_stage.py::scripts.run_dataflow_stage.run_staged::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_json::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
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


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_staged_passes_stage_specific_strictness::run_dataflow_stage.py::scripts.run_dataflow_stage.run_staged::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_json::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
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


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_staged_stops_on_hard_failure::run_dataflow_stage.py::scripts.run_dataflow_stage.run_staged::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_json::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
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


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_emit_stage_outputs_writes_terminal_and_stage_keys::run_dataflow_stage.py::scripts.run_dataflow_stage._emit_stage_outputs
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


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_run_staged_marks_success_as_failure_when_delta_gate_fails::run_dataflow_stage.py::scripts.run_dataflow_stage.run_staged::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._base_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._stage_paths::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_json::test_run_dataflow_stage.py::tests.test_run_dataflow_stage._write_text
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

    gate_calls: list[tuple[str, ...]] = []

    def _run_gate(cmd: list[str] | tuple[str, ...]) -> int:
        gate_calls.append(tuple(cmd))
        return 1 if "annotation_drift_orphaned_gate.py" in cmd[-1] else 0

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
    assert any("annotation_drift_orphaned_gate.py" in cmd[-1] for cmd in gate_calls)

# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_obligation_trace_payload_covers_satisfied_unsatisfied_and_policy_skip::run_dataflow_stage.py::scripts.run_dataflow_stage._obligation_rows_from_timeout_payload::run_dataflow_stage.py::scripts.run_dataflow_stage._obligation_trace_payload
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
    assert sorted(row["status"] for row in rows) == [
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


# gabion:evidence E:call_footprint::tests/test_run_dataflow_stage.py::test_timeout_stage_with_missing_incremental_obligations_marks_incomplete::run_dataflow_stage.py::scripts.run_dataflow_stage._obligation_rows_from_timeout_payload
def test_timeout_stage_with_missing_incremental_obligations_marks_incomplete() -> None:
    rows, markers = run_dataflow_stage._obligation_rows_from_timeout_payload(
        stage_id="a",
        analysis_state="timed_out_progress_resume",
        timeout_payload={"analysis_state": "timed_out_progress_resume"},
    )
    assert rows == ()
    assert markers == ("missing_incremental_obligations",)
