from __future__ import annotations

import time
from pathlib import Path

import pytest

from gabion import server
from gabion.analysis.aspf import Forest
from gabion.analysis.dataflow_audit import AnalysisResult
from gabion.exceptions import NeverThrown
from gabion.execution_plan import ExecutionPlan
from gabion.server_core import command_orchestrator as orchestrator


def _empty_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        type_suggestions=[],
        type_ambiguities=[],
        type_callsite_evidence=[],
        constant_smells=[],
        unused_arg_smells=[],
        forest=Forest(),
    )


def _timeout_context(
    *,
    tmp_path: Path,
    deps: server.ExecuteCommandDeps,
) -> orchestrator._TimeoutCleanupContext:
    return orchestrator._TimeoutCleanupContext(
        timeout_hard_deadline_ns=time.monotonic_ns() + 1_000_000_000,
        cleanup_grace_ns=100_000_000,
        timeout_total_ns=1_000_000_000,
        analysis_window_ns=900_000_000,
        analysis_resume_checkpoint_path=tmp_path / "resume.json",
        analysis_resume_input_manifest_digest="digest",
        last_collection_resume_payload=None,
        execute_deps=deps,
        analysis_resume_input_witness=None,
        emit_checkpoint_intro_timeline=False,
        checkpoint_intro_timeline_path=tmp_path / "timeline.md",
        analysis_resume_total_files=5,
        analysis_resume_checkpoint_status="checkpoint_seeded",
        analysis_resume_reused_files=0,
        profile_enabled=False,
        latest_collection_progress={},
        semantic_progress_cumulative=None,
        report_output_path=tmp_path / "report.md",
        projection_rows=[],
        report_phase_checkpoint_path=tmp_path / "phase.json",
        report_section_journal_path=tmp_path / "sections.json",
        report_section_witness_digest=None,
        phase_checkpoint_state={},
        enable_phase_projection_checkpoints=False,
        forest=Forest(),
        analysis_resume_intro_payload=None,
        runtime_root=tmp_path,
        initial_paths_count_value=1,
        execution_plan=ExecutionPlan(),
        ensure_report_sections_cache_fn=None,
        emit_lsp_progress_fn=lambda **_kwargs: None,
    )


def test_emit_annotation_drift_outputs_emit_only_path_skips_delta_block(
    tmp_path: Path,
) -> None:
    orchestrator._bind_server_symbols()
    state_path = tmp_path / "drift_state.json"
    state_path.write_text('{"summary":{"changed_tests":1}}', encoding="utf-8")
    response: dict[str, object] = {}
    orchestrator._emit_annotation_drift_outputs(
        response=response,
        root=str(tmp_path),
        paths=[],
        test_annotation_drift_state_path=state_path,
        emit_test_annotation_drift=True,
        emit_test_annotation_drift_delta=False,
        write_test_annotation_drift_baseline=False,
    )
    assert response["test_annotation_drift_summary"] == {"changed_tests": 1}
    assert "test_annotation_drift_baseline_path" not in response


def test_emit_primary_outputs_synthesis_report_without_plan_path(
    tmp_path: Path,
) -> None:
    orchestrator._bind_server_symbols()
    response: dict[str, object] = {}
    artifacts = orchestrator._emit_primary_outputs(
        response=response,
        context=orchestrator._PrimaryOutputContext(
            analysis=_empty_analysis_result(),
            root=str(tmp_path),
            paths=[],
            payload={},
            config=orchestrator.AuditConfig(project_root=tmp_path),
            synthesis_plan_path=None,
            synthesis_report=True,
            synthesis_protocols_path=None,
            synthesis_protocols_kind=None,
            synthesis_max_tier=3,
            synthesis_min_bundle_size=2,
            synthesis_allow_singletons=False,
            refactor_plan=False,
            refactor_plan_json=None,
            decision_snapshot_path=None,
            structure_tree_path=None,
            structure_metrics_path=None,
        ),
    )
    assert artifacts.synthesis_plan is not None
    assert "synthesis_plan" in response


def test_finalize_report_refactor_enabled_without_payload_keeps_report_stable(
    tmp_path: Path,
) -> None:
    orchestrator._bind_server_symbols()
    outcome = orchestrator._finalize_report_and_violations(
        context=orchestrator._ReportFinalizationContext(
            analysis=_empty_analysis_result(),
            root=str(tmp_path),
            max_components=1,
            report_path=True,
            report_output_path=None,
            projection_rows=[],
            report_section_journal_path=tmp_path / "sections.json",
            report_section_witness_digest=None,
            report_phase_checkpoint_path=tmp_path / "phase.json",
            analysis_resume_checkpoint_path=None,
            analysis_resume_reused_files=0,
            type_audit_report=False,
            baseline_path=None,
            baseline_write=False,
            decision_snapshot_path=None,
            structure_tree_path=None,
            structure_metrics_path=None,
            structure_metrics_payload=None,
            synthesis_plan=None,
            synthesis_plan_path=None,
            synthesis_report=False,
            synthesis_protocols_path=None,
            refactor_plan=True,
            refactor_plan_json=None,
            refactor_plan_payload=None,
        ),
        phase_checkpoint_state={},
    )
    assert isinstance(outcome.report, str)


def test_load_timeout_resume_progress_uses_manifest_resume_pair(
    tmp_path: Path,
) -> None:
    orchestrator._bind_server_symbols()
    resume_payload = {
        "completed_paths": ["a.py", "b.py"],
        "in_progress_scan_by_path": {"c.py": {"phase": "collection"}},
        "semantic_progress": {"substantive_progress": True},
    }
    deps = server._default_execute_command_deps().with_overrides(
        load_analysis_resume_checkpoint_manifest_fn=lambda **_kwargs: (
            {"witness_digest": "digest-1"},
            resume_payload,
        )
    )
    context = _timeout_context(tmp_path=tmp_path, deps=deps)
    progress_payload: dict[str, object] = {"classification": "timed_out_no_progress"}
    loaded = orchestrator._load_timeout_resume_progress(
        context=context,
        progress_payload=progress_payload,
        timeout_collection_resume_payload=None,
        mark_cleanup_timeout_fn=lambda _step: None,
    )
    assert loaded == resume_payload
    assert progress_payload["classification"] == "timed_out_progress_resume"
    assert progress_payload["resume_supported"] is True
    resume = progress_payload.get("resume")
    assert isinstance(resume, dict)
    assert resume["resume_token"]["witness_digest"] == "digest-1"


def test_load_timeout_resume_progress_manifest_loader_none_keeps_previous_payload(
    tmp_path: Path,
) -> None:
    orchestrator._bind_server_symbols()
    deps = server._default_execute_command_deps().with_overrides(
        load_analysis_resume_checkpoint_manifest_fn=lambda **_kwargs: None
    )
    context = _timeout_context(tmp_path=tmp_path, deps=deps)
    progress_payload: dict[str, object] = {"classification": "timed_out_no_progress"}
    loaded = orchestrator._load_timeout_resume_progress(
        context=context,
        progress_payload=progress_payload,
        timeout_collection_resume_payload=None,
        mark_cleanup_timeout_fn=lambda _step: None,
    )
    assert loaded is None
    assert progress_payload["classification"] == "timed_out_no_progress"


def test_finalize_report_without_report_path_applies_baseline(tmp_path: Path) -> None:
    orchestrator._bind_server_symbols()
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("sample violation\n", encoding="utf-8")
    context = orchestrator._ReportFinalizationContext(
        analysis=_empty_analysis_result(),
        root=str(tmp_path),
        max_components=1,
        report_path=False,
        report_output_path=None,
        projection_rows=[],
        report_section_journal_path=tmp_path / "sections.json",
        report_section_witness_digest=None,
        report_phase_checkpoint_path=tmp_path / "phase.json",
        analysis_resume_checkpoint_path=None,
        analysis_resume_reused_files=0,
        type_audit_report=False,
        baseline_path=baseline_path,
        baseline_write=False,
        decision_snapshot_path=None,
        structure_tree_path=None,
        structure_metrics_path=None,
        structure_metrics_payload=None,
        synthesis_plan=None,
        synthesis_plan_path=None,
        synthesis_report=False,
        synthesis_protocols_path=None,
        refactor_plan=False,
        refactor_plan_json=None,
        refactor_plan_payload=None,
    )
    outcome = orchestrator._finalize_report_and_violations(
        context=context,
        phase_checkpoint_state={},
    )
    assert outcome.report is None
    assert outcome.violations == []


def test_notification_runtime_rejects_non_callable_sender() -> None:
    orchestrator._bind_server_symbols()
    with pytest.raises(NeverThrown):
        orchestrator._notification_runtime("not-callable")


def test_render_timeout_partial_report_handles_non_callable_cache_loader(
    tmp_path: Path,
) -> None:
    orchestrator._bind_server_symbols()
    deps = server._default_execute_command_deps()
    projection_rows = deps.report_projection_spec_rows_fn()
    context = orchestrator._TimeoutCleanupContext(
        timeout_hard_deadline_ns=time.monotonic_ns() + 1_000_000_000,
        cleanup_grace_ns=100_000_000,
        timeout_total_ns=1_000_000_000,
        analysis_window_ns=900_000_000,
        analysis_resume_checkpoint_path=None,
        analysis_resume_input_manifest_digest=None,
        last_collection_resume_payload=None,
        execute_deps=deps,
        analysis_resume_input_witness=None,
        emit_checkpoint_intro_timeline=False,
        checkpoint_intro_timeline_path=tmp_path / "timeline.md",
        analysis_resume_total_files=0,
        analysis_resume_checkpoint_status=None,
        analysis_resume_reused_files=0,
        profile_enabled=False,
        latest_collection_progress={},
        semantic_progress_cumulative=None,
        report_output_path=tmp_path / "report.md",
        projection_rows=[projection_rows[0]],
        report_phase_checkpoint_path=tmp_path / "phase.json",
        report_section_journal_path=tmp_path / "sections.json",
        report_section_witness_digest=None,
        phase_checkpoint_state={},
        enable_phase_projection_checkpoints=False,
        forest=Forest(),
        analysis_resume_intro_payload=None,
        runtime_root=tmp_path,
        initial_paths_count_value=1,
        execution_plan=ExecutionPlan(),
        ensure_report_sections_cache_fn=None,
        emit_lsp_progress_fn=None,
    )
    outcome = orchestrator._render_timeout_partial_report(
        context=context,
        analysis_state="timed_out_no_progress",
        progress_payload={"classification": "timed_out_no_progress"},
        timeout_collection_resume_payload=None,
        phase_checkpoint_state={},
        mark_cleanup_timeout_fn=None,
    )
    assert outcome.partial_report_written is True
    assert "intro" in outcome.resolved_sections
