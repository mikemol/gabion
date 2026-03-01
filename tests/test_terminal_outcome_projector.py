from __future__ import annotations

from pathlib import Path

import pytest

from gabion.exceptions import NeverThrown
from gabion.tooling import terminal_outcome_projector


# gabion:evidence E:call_footprint::tests/test_terminal_outcome_projector.py::test_project_terminal_outcome_normalizes_status_matrix::terminal_outcome_projector.py::gabion.tooling.terminal_outcome_projector.project_terminal_outcome
def test_project_terminal_outcome_normalizes_status_matrix() -> None:
    cases = [
        (
            terminal_outcome_projector.TerminalOutcomeInput(
                terminal_exit=0,
                terminal_state="succeeded",
                terminal_stage="run",
                terminal_status="unknown",
                attempts_run=2,
            ),
            "success",
        ),
        (
            terminal_outcome_projector.TerminalOutcomeInput(
                terminal_exit=1,
                terminal_state="timed_out_progress_resume",
                terminal_stage="retry1",
                terminal_status="unknown",
                attempts_run=3,
            ),
            "timeout_resume",
        ),
        (
            terminal_outcome_projector.TerminalOutcomeInput(
                terminal_exit=2,
                terminal_state="failed",
                terminal_stage="retry2",
                terminal_status="unknown",
                attempts_run=1,
            ),
            "hard_failure",
        ),
        (
            terminal_outcome_projector.TerminalOutcomeInput(
                terminal_exit=0,
                terminal_state="succeeded",
                terminal_stage="run",
                terminal_status="success",
                attempts_run=1,
            ),
            "success",
        ),
    ]

    for payload, expected_status in cases:
        projected = terminal_outcome_projector.project_terminal_outcome(payload)
        assert projected.terminal_status == expected_status
        assert projected.terminal_stage == payload.terminal_stage.upper()


# gabion:evidence E:call_footprint::tests/test_terminal_outcome_projector.py::test_project_terminal_outcome_invalid_status_raises::terminal_outcome_projector.py::gabion.tooling.terminal_outcome_projector.project_terminal_outcome
def test_project_terminal_outcome_invalid_status_raises() -> None:
    with pytest.raises(NeverThrown):
        terminal_outcome_projector.project_terminal_outcome(
            terminal_outcome_projector.TerminalOutcomeInput(
                terminal_exit=1,
                terminal_state="failed",
                terminal_stage="run",
                terminal_status="bogus",
                attempts_run=1,
            )
        )


# gabion:evidence E:call_footprint::tests/test_terminal_outcome_projector.py::test_terminal_outcome_from_stage_results_and_roundtrip::terminal_outcome_projector.py::gabion.tooling.terminal_outcome_projector.project_terminal_outcome
def test_terminal_outcome_from_stage_results_and_roundtrip(tmp_path: Path) -> None:
    assert terminal_outcome_projector.terminal_outcome_from_stage_results([]) is None

    outcome = terminal_outcome_projector.terminal_outcome_from_stage_results(
        [
            {"stage_id": "run", "exit_code": 124, "analysis_state": "timed_out_progress_resume"},
        ]
    )
    assert outcome is not None
    assert outcome.terminal_status == "timeout_resume"

    path = tmp_path / "terminal.json"
    terminal_outcome_projector.write_terminal_outcome_artifact(path, outcome)
    loaded = terminal_outcome_projector.read_terminal_outcome_artifact(path)
    assert loaded == outcome


# gabion:evidence E:call_footprint::tests/test_terminal_outcome_projector.py::test_terminal_outcome_read_non_mapping_returns_none::terminal_outcome_projector.py::gabion.tooling.terminal_outcome_projector.project_terminal_outcome
def test_terminal_outcome_read_non_mapping_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "terminal.json"
    path.write_text("[]\n", encoding="utf-8")
    assert terminal_outcome_projector.read_terminal_outcome_artifact(path) is None


# gabion:evidence E:call_footprint::tests/test_terminal_outcome_projector.py::test_terminal_outcome_read_missing_path_returns_none::terminal_outcome_projector.py::gabion.tooling.terminal_outcome_projector.project_terminal_outcome
def test_terminal_outcome_read_missing_path_returns_none(tmp_path: Path) -> None:
    assert (
        terminal_outcome_projector.read_terminal_outcome_artifact(
            tmp_path / "missing-terminal.json"
        )
        is None
    )


# gabion:evidence E:call_footprint::tests/test_terminal_outcome_projector.py::test_terminal_outcome_render_line::terminal_outcome_projector.py::gabion.tooling.terminal_outcome_projector.project_terminal_outcome
def test_terminal_outcome_render_line() -> None:
    outcome = terminal_outcome_projector.project_terminal_outcome(
        terminal_outcome_projector.TerminalOutcomeInput(
            terminal_exit=0,
            terminal_state="succeeded",
            terminal_stage="run",
            terminal_status="success",
            attempts_run=1,
        )
    )
    line = terminal_outcome_projector.render_terminal_outcome_line(outcome)
    assert "terminal_stage=RUN" in line
    assert "status=success" in line
    output_lines = outcome.to_output_lines(stage_metrics="")
    assert not any(line.startswith("stage_metrics=") for line in output_lines)
