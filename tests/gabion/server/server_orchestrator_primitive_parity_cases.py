from __future__ import annotations

from pathlib import Path

from gabion import server
from gabion.server_core import command_orchestrator_primitives


def test_report_path_resolution_parity(tmp_path: Path) -> None:
    inputs = [None, "", "-", "/dev/stdout", "report.md", "/tmp/report.md"]
    for report_path in inputs:
        assert server._resolve_report_output_path(
            root=tmp_path,
            report_path=report_path,
        ) == command_orchestrator_primitives._resolve_report_output_path(
            root=tmp_path,
            report_path=report_path,
        )


def test_report_section_journal_path_resolution_parity(tmp_path: Path) -> None:
    for report_path in ("dataflow_report.md", "subdir/custom.md", None):
        assert server._resolve_report_section_journal_path(
            root=tmp_path,
            report_path=report_path,
        ) == command_orchestrator_primitives._resolve_report_section_journal_path(
            root=tmp_path,
            report_path=report_path,
        )


def test_normalize_dataflow_response_parity() -> None:
    payload = {
        "lint_lines": ["a.py:1:2: W sample", 1],
        "analysis_surfaces": ["decision", "taint", 7],
        "disabled_surfaces": ["exception", 9],
        "payload_capabilities": {"report_projection": 1, "resume": True},
    }
    assert server._normalize_dataflow_response(payload) == (
        command_orchestrator_primitives._serialize_dataflow_response(
            command_orchestrator_primitives._normalize_dataflow_response(payload)
        )
    )


def test_analysis_timeout_budget_parity() -> None:
    payload = {
        "analysis_timeout_ms": 500,
        "analysis_timeout_grace_ms": 50,
    }
    assert server._analysis_timeout_budget_ns(payload) == (
        command_orchestrator_primitives._analysis_timeout_budget_ns(payload)
    )


def test_flush_decision_helpers_parity() -> None:
    checkpoint_kwargs = {
        "intro_changed": False,
        "remaining_files": 2,
        "semantic_substantive_progress": True,
        "now_ns": 5_000_000_000,
        "last_flush_ns": 3_000_000_000,
    }
    assert server._collection_checkpoint_flush_due(**checkpoint_kwargs) == (
        command_orchestrator_primitives._collection_checkpoint_flush_due(**checkpoint_kwargs)
    )

    report_kwargs = {
        "completed_files": 9,
        "remaining_files": 1,
        "now_ns": 20_000_000_000,
        "last_flush_ns": 5_000_000_000,
        "last_flush_completed": 0,
    }
    assert server._collection_report_flush_due(**report_kwargs) == (
        command_orchestrator_primitives._collection_report_flush_due(**report_kwargs)
    )

    assert server._projection_phase_flush_due(
        phase="post",
        now_ns=0,
        last_flush_ns=10_000_000_000,
    ) == command_orchestrator_primitives._projection_phase_flush_due(
        phase="post",
        now_ns=0,
        last_flush_ns=10_000_000_000,
    )


def test_progress_heartbeat_seconds_parity() -> None:
    payload = {"progress_heartbeat_seconds": "4"}
    assert server._progress_heartbeat_seconds(payload) == (
        command_orchestrator_primitives._progress_heartbeat_seconds(payload)
    )
