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
        resume_checkpoint_path=paths["resume"],
        baseline_path=paths["baseline"],
    )


def test_stage_ids_are_bounded_and_ordered() -> None:
    assert run_dataflow_stage._stage_ids("a", 3) == ["a", "b", "c"]
    assert run_dataflow_stage._stage_ids("b", 3) == ["b", "c"]
    assert run_dataflow_stage._stage_ids("x", 2) == ["a", "b"]
    assert run_dataflow_stage._stage_ids("a", 0) == []


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


def test_emit_stage_outputs_writes_terminal_and_stage_keys(tmp_path: Path) -> None:
    output_path = tmp_path / "github_output.txt"
    results = [
        run_dataflow_stage.StageResult(
            stage_id="a",
            exit_code=2,
            analysis_state="timed_out_progress_resume",
            is_timeout_resume=True,
            metrics_line="ticks=1 checks=1 ticks_per_ns=0.1 wall_s=1.000",
        ),
        run_dataflow_stage.StageResult(
            stage_id="b",
            exit_code=0,
            analysis_state="done",
            is_timeout_resume=False,
            metrics_line="ticks=2 checks=2 ticks_per_ns=0.2 wall_s=2.000",
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
