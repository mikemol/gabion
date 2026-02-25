from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

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
    last_collection_resume_payload: dict[str, object] | None = None,
) -> orchestrator._TimeoutCleanupContext:
    return orchestrator._TimeoutCleanupContext(
        timeout_hard_deadline_ns=time.monotonic_ns() + 1_000_000_000,
        cleanup_grace_ns=100_000_000,
        timeout_total_ns=1_000_000_000,
        analysis_window_ns=900_000_000,
        analysis_resume_checkpoint_path=tmp_path / "resume.json",
        analysis_resume_input_manifest_digest="digest",
        last_collection_resume_payload=last_collection_resume_payload,
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


def _analysis_context(
    *,
    tmp_path: Path,
    deps: server.ExecuteCommandDeps,
    source_path: Path,
    emit_checkpoint_intro_timeline: bool,
    emitted_events: list[dict[str, object]],
) -> orchestrator._AnalysisExecutionContext:
    return orchestrator._AnalysisExecutionContext(
        execute_deps=deps,
        runtime_state=orchestrator.CommandRuntimeState(latest_collection_progress={}),
        forest=Forest(),
        paths=[source_path],
        no_recursive=False,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=100,
        report_path=False,
        include_coherence=False,
        include_rewrite_plans=False,
        include_exception_obligations=False,
        include_handledness_witnesses=False,
        include_never_invariants=False,
        include_wl_refinement=False,
        include_decisions=False,
        lint=False,
        include_ambiguities=False,
        config=orchestrator.AuditConfig(project_root=tmp_path),
        needs_analysis=True,
        file_paths_for_run=[source_path],
        analysis_resume_intro_payload=None,
        analysis_resume_reused_files=0,
        analysis_resume_total_files=1,
        analysis_resume_checkpoint_path=tmp_path / "resume.json",
        analysis_resume_checkpoint_status="checkpoint_seeded",
        analysis_resume_input_manifest_digest="digest",
        analysis_resume_input_witness=None,
        analysis_resume_intro_timeline_header=None,
        analysis_resume_intro_timeline_row=None,
        checkpoint_intro_timeline_path=tmp_path / "timeline.md",
        emit_checkpoint_intro_timeline=emit_checkpoint_intro_timeline,
        enable_phase_projection_checkpoints=False,
        report_output_path=None,
        projection_rows=[],
        report_section_journal_path=tmp_path / "sections.json",
        report_section_witness_digest=None,
        report_phase_checkpoint_path=tmp_path / "phase.json",
        phase_checkpoint_state={},
        profile_enabled=False,
        emit_phase_progress_events=False,
        fingerprint_deadness_json=None,
        emit_lsp_progress_fn=lambda **kwargs: emitted_events.append(
            {str(key): kwargs[key] for key in kwargs}
        ),
        ensure_report_sections_cache_fn=lambda: ({}, None),
        clear_report_sections_cache_reason_fn=lambda: None,
        check_deadline_fn=lambda: None,
        profiling_stage_ns={"server.analysis_call": 0, "server.projection_emit": 0},
        profiling_counters={
            "server.collection_resume_persist_calls": 0,
            "server.projection_emit_calls": 0,
        },
    )


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._emit_annotation_drift_outputs
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


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._emit_primary_outputs
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


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._finalize_report_and_violations E:decision_surface/direct::command_orchestrator.py::gabion.server_core.command_orchestrator._finalize_report_and_violations::stale_f2f5df7d0b69_366adc93
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


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._load_timeout_resume_progress E:decision_surface/direct::command_orchestrator.py::gabion.server_core.command_orchestrator._load_timeout_resume_progress::stale_a34f1f47eb2e
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


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._load_timeout_resume_progress E:decision_surface/direct::command_orchestrator.py::gabion.server_core.command_orchestrator._load_timeout_resume_progress::stale_c63e5782a009_20498f3c
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


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._finalize_report_and_violations E:decision_surface/direct::command_orchestrator.py::gabion.server_core.command_orchestrator._finalize_report_and_violations::stale_951f0c40d59e
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


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._notification_runtime
def test_notification_runtime_rejects_non_callable_sender() -> None:
    orchestrator._bind_server_symbols()
    with pytest.raises(NeverThrown):
        orchestrator._notification_runtime("not-callable")


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._render_timeout_partial_report
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


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._prepare_analysis_resume_state
def test_prepare_analysis_resume_state_skips_intro_timeline_when_disabled(
    tmp_path: Path,
) -> None:
    orchestrator._bind_server_symbols()
    source_path = tmp_path / "module.py"
    source_path.write_text("def f() -> None:\n    return None\n", encoding="utf-8")
    checkpoint_writes: list[dict[str, object]] = []
    deps = server._default_execute_command_deps().with_overrides(
        load_analysis_resume_checkpoint_manifest_fn=lambda **_kwargs: None,
        write_analysis_resume_checkpoint_fn=lambda **kwargs: checkpoint_writes.append(
            {str(key): kwargs[key] for key in kwargs}
        ),
    )
    state = orchestrator._AnalysisResumePreparationState(
        analysis_resume_checkpoint_path=None,
        analysis_resume_input_witness=None,
        analysis_resume_input_manifest_digest=None,
        analysis_resume_total_files=0,
        analysis_resume_reused_files=0,
        analysis_resume_checkpoint_status=None,
        analysis_resume_checkpoint_compatibility_status=None,
        analysis_resume_intro_payload=None,
        analysis_resume_intro_timeline_header=None,
        analysis_resume_intro_timeline_row=None,
        report_section_witness_digest=None,
        phase_checkpoint_state={},
        semantic_progress_cumulative=None,
        last_collection_resume_payload=None,
    )
    runtime_state = orchestrator.CommandRuntimeState(latest_collection_progress={})
    _file_paths_for_run, collection_resume_payload = orchestrator._prepare_analysis_resume_state(
        execute_deps=deps,
        needs_analysis=True,
        paths=[source_path],
        root=str(tmp_path),
        payload={"resume_checkpoint": str(tmp_path / "resume.json")},
        no_recursive=False,
        report_path=False,
        include_wl_refinement=False,
        config=orchestrator.AuditConfig(project_root=tmp_path),
        explicit_resume_checkpoint=False,
        emit_checkpoint_intro_timeline=False,
        checkpoint_intro_timeline_path=tmp_path / "timeline.md",
        report_output_path=None,
        report_phase_checkpoint_path=tmp_path / "phase.json",
        state=state,
        runtime_state=runtime_state,
    )
    assert collection_resume_payload is not None
    assert checkpoint_writes
    assert state.analysis_resume_checkpoint_status == "checkpoint_seeded"
    assert state.analysis_resume_intro_timeline_header is None
    assert state.analysis_resume_intro_timeline_row is None


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._run_analysis_with_progress
def test_run_analysis_with_progress_skips_checkpoint_serialized_event_when_timeline_disabled(
    tmp_path: Path,
) -> None:
    orchestrator._bind_server_symbols()
    source_path = tmp_path / "module.py"
    source_path.write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    emitted_events: list[dict[str, object]] = []
    checkpoint_writes: list[dict[str, object]] = []
    deps = server._default_execute_command_deps().with_overrides(
        analyze_paths_fn=lambda *_args, **_kwargs: _empty_analysis_result(),
        collection_semantic_progress_fn=lambda **_kwargs: {
            "substantive_progress": False,
            "current_witness_digest": "digest-1",
        },
        collection_checkpoint_flush_due_fn=lambda **_kwargs: True,
        write_analysis_resume_checkpoint_fn=lambda **kwargs: checkpoint_writes.append(
            {str(key): kwargs[key] for key in kwargs}
        ),
    )
    context = _analysis_context(
        tmp_path=tmp_path,
        deps=deps,
        source_path=source_path,
        emit_checkpoint_intro_timeline=False,
        emitted_events=emitted_events,
    )
    state = orchestrator._AnalysisExecutionMutableState(
        last_collection_resume_payload=None,
        semantic_progress_cumulative=None,
        latest_collection_progress={},
    )
    outcome = orchestrator._run_analysis_with_progress(
        context=context,
        state=state,
        collection_resume_payload=None,
    )
    assert checkpoint_writes
    assert outcome.latest_collection_progress["total_files"] == 1
    assert not any(
        event.get("analysis_state") == "analysis_collection_checkpoint_serialized"
        for event in emitted_events
    )


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._persist_timeout_resume_checkpoint
def test_persist_timeout_resume_checkpoint_skips_checkpoint_event_when_timeline_disabled(
    tmp_path: Path,
) -> None:
    orchestrator._bind_server_symbols()
    checkpoint_writes: list[dict[str, object]] = []
    emitted_events: list[dict[str, object]] = []
    deps = server._default_execute_command_deps().with_overrides(
        write_analysis_resume_checkpoint_fn=lambda **kwargs: checkpoint_writes.append(
            {str(key): kwargs[key] for key in kwargs}
        ),
    )
    context = _timeout_context(
        tmp_path=tmp_path,
        deps=deps,
        last_collection_resume_payload={
            "completed_paths": ["module.py"],
            "in_progress_scan_by_path": {},
        },
    )
    persisted_payload = orchestrator._persist_timeout_resume_checkpoint(
        context=context,
        timeout_collection_resume_payload=None,
        mark_cleanup_timeout_fn=lambda _step: None,
        emit_lsp_progress_fn=lambda **kwargs: emitted_events.append(
            {str(key): kwargs[key] for key in kwargs}
        ),
    )
    assert isinstance(persisted_payload, dict)
    assert checkpoint_writes
    assert emitted_events == []


def test_emit_primary_outputs_writes_synthesis_protocols_to_response_for_stdout(
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
            synthesis_protocols_path="-",
            synthesis_protocols_kind="dataclass",
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
    assert isinstance(response.get("synthesis_protocols"), str)


@pytest.mark.parametrize(
    ("domain", "action", "baseline", "state_in"),
    [
        ("obsolescence", "state", None, "state.json"),
        ("annotation-drift", "delta", "drift-baseline.json", "state.json"),
        ("ambiguity", "baseline-write", "ambiguity-baseline.json", "state.json"),
    ],
)
def test_parse_execution_payload_options_aux_operation_domain_routing(
    tmp_path: Path,
    domain: str,
    action: str,
    baseline: str | None,
    state_in: str,
) -> None:
    orchestrator._bind_server_symbols()
    payload: dict[str, object] = {
        "strictness": "high",
        "aux_operation": {
            "domain": domain,
            "action": action,
            "baseline_path": baseline,
            "state_in": state_in,
        },
    }
    options = orchestrator._parse_execution_payload_options(
        payload=payload,
        root=tmp_path,
    )
    if domain == "obsolescence":
        assert options.emit_test_obsolescence_state is True
        assert options.test_obsolescence_state_path == state_in
    elif domain == "annotation-drift":
        assert options.emit_test_annotation_drift_delta is True
        assert options.annotation_drift_baseline_path_override is not None
        assert options.test_annotation_drift_state_path == state_in
    else:
        assert options.write_ambiguity_baseline is True
        assert options.ambiguity_baseline_path_override is not None
        assert options.ambiguity_state_path == state_in


def test_parse_execution_payload_options_aux_operation_invalid_paths_raise() -> None:
    orchestrator._bind_server_symbols()
    with pytest.raises(NeverThrown):
        orchestrator._parse_execution_payload_options(
            payload={
                "strictness": "high",
                "aux_operation": {"domain": "obsolescence", "action": "delta"},
            },
            root=Path("."),
        )
    with pytest.raises(NeverThrown):
        orchestrator._parse_execution_payload_options(
            payload={
                "strictness": "high",
                "aux_operation": {"domain": "invalid", "action": "state"},
            },
            root=Path("."),
        )


def test_emit_test_obsolescence_outputs_ignores_non_mapping_active_summary(
    tmp_path: Path,
) -> None:
    orchestrator._bind_server_symbols()
    state_path = tmp_path / "state.json"
    state_path.write_text("{}\n", encoding="utf-8")
    original = orchestrator.test_obsolescence_state.load_state
    try:
        orchestrator.test_obsolescence_state.load_state = lambda _path: SimpleNamespace(
            candidates=[],
            baseline=SimpleNamespace(
                summary={},
                active={"summary": ["not-a-mapping"]},
            ),
            baseline_payload={"summary": {}, "active": {"summary": ["not-a-mapping"]}},
        )

        response: dict[str, object] = {}
        orchestrator._emit_test_obsolescence_outputs(
            response=response,
            root=str(tmp_path),
            emit_test_obsolescence=True,
            emit_test_obsolescence_state=False,
            test_obsolescence_state_path=state_path,
            emit_test_obsolescence_delta=False,
            write_test_obsolescence_baseline=False,
        )
    finally:
        orchestrator.test_obsolescence_state.load_state = original
    assert response["test_obsolescence_active_summary"] == {}
