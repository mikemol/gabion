from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from gabion import server
from gabion.analysis.aspf import Forest
from gabion.analysis.dataflow_audit import AnalysisResult
from gabion.exceptions import NeverThrown
from gabion.server_core import command_orchestrator as orchestrator


def _bind() -> None:
    orchestrator._bind_server_symbols()


class _Plan:
    def as_json_dict(self) -> dict[str, object]:
        return {}


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


def _analysis_context(
    *,
    tmp_path: Path,
    payload: dict[str, object],
) -> orchestrator._AnalysisExecutionContext:
    _bind()
    return orchestrator._AnalysisExecutionContext(
        execute_deps=server._default_execute_command_deps(),
        aspf_trace_state=None,
        runtime_state=orchestrator.CommandRuntimeState(latest_collection_progress={}),
        forest=Forest(),
        paths=[tmp_path / "sample.py"],
        no_recursive=False,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=50,
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
        needs_analysis=False,
        file_paths_for_run=[tmp_path / "sample.py"],
        analysis_resume_intro_payload=None,
        analysis_resume_reused_files=0,
        analysis_resume_total_files=0,
        analysis_resume_state_path=None,
        analysis_resume_state_status=None,
        analysis_resume_input_manifest_digest=None,
        analysis_resume_input_witness=None,
        analysis_resume_intro_timeline_header=None,
        analysis_resume_intro_timeline_row=None,
        phase_timeline_path=tmp_path / "timeline.md",
        emit_phase_timeline=False,
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
        emit_lsp_progress_fn=lambda **_kwargs: None,
        ensure_report_sections_cache_fn=lambda: ({}, None),
        clear_report_sections_cache_reason_fn=lambda: None,
        check_deadline_fn=lambda: None,
        profiling_stage_ns={"server.analysis_call": 0, "server.projection_emit": 0},
        profiling_counters={
            "server.collection_resume_persist_calls": 0,
            "server.projection_emit_calls": 0,
        },
        payload=payload,
    )


def test_normalize_duration_timeout_clock_ticks_without_duration_returns_total() -> None:
    _bind()
    total = orchestrator._normalize_duration_timeout_clock_ticks(
        timeout=orchestrator._TimeoutIngressCarrier(
            has_tick_timeout=False,
            has_duration_timeout=False,
        ),
        total_ticks=9,
    )
    assert total == 9


def test_auxiliary_mode_from_payload_legacy_baseline_write_branch() -> None:
    _bind()
    mode = orchestrator._auxiliary_mode_from_payload(
        payload={"write_test_obsolescence_baseline": True},
        mode_key="obsolescence_mode",
        state_key="test_obsolescence_state",
        emit_state_key="emit_test_obsolescence_state",
        emit_delta_key="emit_test_obsolescence_delta",
        write_baseline_key="write_test_obsolescence_baseline",
        emit_report_key="emit_test_obsolescence",
        domain="obsolescence",
        allow_report=True,
    )
    assert mode.kind == "baseline-write"


def test_auxiliary_mode_from_payload_rejects_invalid_mode_kind() -> None:
    _bind()
    with pytest.raises(NeverThrown):
        orchestrator._auxiliary_mode_from_payload(
            payload={"obsolescence_mode": {"kind": "not-valid"}},
            mode_key="obsolescence_mode",
            state_key="test_obsolescence_state",
            emit_state_key="emit_test_obsolescence_state",
            emit_delta_key="emit_test_obsolescence_delta",
            write_baseline_key="write_test_obsolescence_baseline",
            emit_report_key="emit_test_obsolescence",
            domain="obsolescence",
            allow_report=True,
        )


def test_select_auxiliary_mode_selection_rejects_invalid_aux_operation() -> None:
    _bind()
    with pytest.raises(NeverThrown):
        orchestrator._select_auxiliary_mode_selection(
            payload={},
            aux_operation=orchestrator._AuxOperationIngressCarrier(
                domain="obsolescence",
                action="not-valid",
                state_in=None,
                baseline_path=None,
            ),
        )
    with pytest.raises(NeverThrown):
        orchestrator._select_auxiliary_mode_selection(
            payload={},
            aux_operation=orchestrator._AuxOperationIngressCarrier(
                domain="not-a-domain",
                action="state",
                state_in=None,
                baseline_path=None,
            ),
        )


def test_select_auxiliary_mode_selection_ambiguity_domain_branch() -> None:
    _bind()
    selection = orchestrator._select_auxiliary_mode_selection(
        payload={},
        aux_operation=orchestrator._AuxOperationIngressCarrier(
            domain="ambiguity",
            action="state",
            state_in="state.json",
            baseline_path=None,
        ),
    )
    assert selection.ambiguity.kind == "state"
    assert selection.obsolescence.kind == "off"
    assert selection.annotation_drift.kind == "off"


def test_select_auxiliary_mode_selection_taint_lifecycle_domain_branch() -> None:
    _bind()
    selection = orchestrator._select_auxiliary_mode_selection(
        payload={},
        aux_operation=orchestrator._AuxOperationIngressCarrier(
            domain="taint",
            action="lifecycle",
            state_in="taint_state.json",
            baseline_path=None,
        ),
    )
    assert selection.taint.kind == "lifecycle"
    assert selection.taint.state_path == "taint_state.json"
    assert selection.ambiguity.kind == "off"


def test_execute_analysis_phase_applies_runtime_payload_overrides_without_analysis(
    tmp_path: Path,
) -> None:
    _bind()
    context = _analysis_context(
        tmp_path=tmp_path,
        payload={
            "proof_mode": "on",
            "order_policy": "sort",
            "order_telemetry": True,
            "order_enforce_canonical_allowlist": True,
            "order_deadline_probe": False,
            "derivation_cache_max_entries": "2",
            "projection_registry_gas_limit": "3",
        },
    )
    outcome = orchestrator._run_analysis_with_progress(
        context=context,
        state=orchestrator._AnalysisExecutionMutableState(
            last_collection_resume_payload=None,
            semantic_progress_cumulative=None,
            latest_collection_progress={},
        ),
        collection_resume_payload=None,
    )
    assert isinstance(outcome.analysis, AnalysisResult)


def test_run_analysis_with_progress_skips_resume_seed_when_resume_payload_present(
    tmp_path: Path,
) -> None:
    _bind()
    deps = server._default_execute_command_deps().with_overrides(
        analyze_paths_fn=lambda *_args, **_kwargs: _empty_analysis_result(),
        build_analysis_collection_resume_seed_fn=(
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError("resume seed should be skipped"))
        ),
    )
    context = dataclasses.replace(
        _analysis_context(tmp_path=tmp_path, payload={}),
        execute_deps=deps,
        needs_analysis=True,
    )
    outcome = orchestrator._run_analysis_with_progress(
        context=context,
        state=orchestrator._AnalysisExecutionMutableState(
            last_collection_resume_payload=None,
            semantic_progress_cumulative=None,
            latest_collection_progress={},
        ),
        collection_resume_payload={
            "completed_paths": [],
            "in_progress_scan_by_path": {},
            "semantic_progress": {},
        },
    )
    assert isinstance(outcome.analysis, AnalysisResult)


def test_build_success_response_emits_analysis_resume_block_when_resume_source_present(
    tmp_path: Path,
) -> None:
    _bind()
    options = orchestrator._parse_execution_payload_options(
        payload={},
        root=tmp_path,
    )
    context = orchestrator._SuccessResponseContext(
        execute_deps=server._default_execute_command_deps(),
        aspf_trace_state=None,
        analysis=_empty_analysis_result(),
        root=str(tmp_path),
        paths=[],
        payload={},
        config=orchestrator.AuditConfig(project_root=tmp_path),
        options=options,
        name_filter_bundle=orchestrator.DataflowNameFilterBundle(
            exclude_dirs=set(),
            ignore_params=set(),
            decision_ignore_params=set(),
            transparent_decorators=set(),
        ),
        report_path=False,
        report_output_path=None,
        report_section_journal_path=tmp_path / "sections.json",
        report_section_witness_digest=None,
        report_phase_checkpoint_path=tmp_path / "phase.json",
        projection_rows=[],
        analysis_resume_state_path=None,
        analysis_resume_source="resume_manifest",
        analysis_resume_state_status="checkpoint_seeded",
        analysis_resume_state_compatibility_status="compatible",
        analysis_resume_manifest_digest="digest",
        analysis_resume_reused_files=1,
        analysis_resume_total_files=3,
        profiling_stage_ns={"server.analysis_call": 0, "server.projection_emit": 0},
        profiling_counters={
            "server.collection_resume_persist_calls": 0,
            "server.projection_emit_calls": 0,
        },
        phase_checkpoint_state={},
        execution_plan=_Plan(),
        last_collection_resume_payload=None,
        semantic_progress_cumulative=None,
        latest_collection_progress={},
        emit_lsp_progress_fn=lambda **_kwargs: None,
        dataflow_capabilities=orchestrator._DataflowCapabilityAnnotations(
            selected_adapter="python:default",
            supported_analysis_surfaces=[],
            disabled_surface_reasons={},
        ),
    )
    outcome = orchestrator._build_success_response(context=context)
    resume_payload = outcome.response.get("analysis_resume")
    assert isinstance(resume_payload, dict)
    assert resume_payload["source"] == "resume_manifest"
    assert resume_payload["cache_verdict"] in {"seeded", "warm"}
