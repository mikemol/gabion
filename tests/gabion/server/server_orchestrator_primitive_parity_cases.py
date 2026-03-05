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
        command_orchestrator_primitives._normalize_dataflow_response(payload)
    )


def test_analysis_timeout_budget_parity() -> None:
    payload = {
        "analysis_timeout_ms": 500,
        "analysis_timeout_grace_ms": 50,
    }
    assert server._analysis_timeout_budget_ns(payload) == (
        command_orchestrator_primitives._analysis_timeout_budget_ns(payload)
    )
