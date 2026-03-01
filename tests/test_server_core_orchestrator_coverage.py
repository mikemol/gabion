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


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._select_auxiliary_mode_selection
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


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._taint_marker_row_from_ambiguity_witness
def test_taint_marker_row_from_ambiguity_witness_edge_inputs() -> None:
    _bind()
    assert orchestrator._taint_marker_row_from_ambiguity_witness("bad") is None
    row = orchestrator._taint_marker_row_from_ambiguity_witness(
        {
            "kind": "k",
            "candidate_count": "bad-count",
            "site": "not-a-mapping",
        }
    )
    assert isinstance(row, dict)
    assert row["reason"] == "ambiguity witness kind=k candidates=0"
    assert "span" not in row["site"]


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._taint_marker_row_from_type_ambiguity
def test_taint_marker_row_from_type_ambiguity_edge_inputs() -> None:
    _bind()
    assert orchestrator._taint_marker_row_from_type_ambiguity("") is None
    path, function = orchestrator._type_ambiguity_site("only-function downstream types conflict: x")
    assert path == ""
    assert function == "only-function"


def _progress_transition_state(
    *,
    done: int,
    marker: str,
    event_kind: orchestrator._ProgressEventKind = "progress",
    analysis_state: str = "analysis_post_in_progress",
    total: int = 6,
) -> orchestrator._ProgressTransitionState:
    _bind()
    return orchestrator._normalize_progress_transition_state(
        phase="post",
        analysis_state=analysis_state,
        event_kind=event_kind,
        primary_unit="post_tasks",
        primary_done=done,
        primary_total=total,
        progress_marker=marker,
    )


# gabion:evidence E:call_footprint::tests/test_server_core_orchestrator_coverage.py::test_progress_transition_validator_allows_table_driven_valid_sequences::command_orchestrator.py::gabion.server_core.command_orchestrator.validate_progress_transition_contract
@pytest.mark.parametrize(
    ("previous_kwargs", "current_kwargs", "expected_reason", "expected_event_kind"),
    [
        (
            {"done": 5, "marker": "fingerprint:normalize"},
            {"done": 5, "marker": "fingerprint:warnings"},
            "parent_held",
            "progress",
        ),
        (
            {"done": 5, "marker": "fingerprint:rewrite_plans"},
            {"done": 6, "marker": "fingerprint:done"},
            "parent_advanced",
            "progress",
        ),
        (
            {"done": 6, "marker": "fingerprint:done"},
            {"done": 6, "marker": "complete"},
            "terminal_transition",
            "terminal",
        ),
    ],
)
def test_progress_transition_validator_allows_table_driven_valid_sequences(
    previous_kwargs: dict[str, object],
    current_kwargs: dict[str, object],
    expected_reason: str,
    expected_event_kind: str,
) -> None:
    previous = _progress_transition_state(**previous_kwargs)
    current = _progress_transition_state(**current_kwargs)
    decision = orchestrator._validate_progress_transition_or_never(
        previous=previous,
        current=current,
    )
    assert decision.valid is True
    assert decision.suppress_emit is False
    assert decision.reason == expected_reason
    assert decision.effective_event_kind == expected_event_kind


# gabion:evidence E:call_footprint::tests/test_server_core_orchestrator_coverage.py::test_progress_transition_validator_rejects_table_driven_invalid_sequences::command_orchestrator.py::gabion.server_core.command_orchestrator.validate_progress_transition_contract
@pytest.mark.parametrize(
    ("previous_kwargs", "current_kwargs"),
    [
        (
            {"done": 5, "marker": "fingerprint:normalize"},
            {"done": 6, "marker": "fingerprint:normalize"},
        ),
        (
            {"done": 5, "marker": "fingerprint:normalize"},
            {"done": 6, "marker": "fingerprint:warnings"},
        ),
        (
            {"done": 5, "marker": "fingerprint:normalize"},
            {"done": 5, "marker": "fingerprint:warnings", "analysis_state": "analysis_edge_in_progress"},
        ),
        (
            {"done": 5, "marker": "fingerprint:done"},
            {"done": 6, "marker": "complete"},
        ),
        (
            {"done": 6, "marker": "complete", "event_kind": "terminal"},
            {"done": 6, "marker": "fingerprint:done", "event_kind": "heartbeat"},
        ),
    ],
)
def test_progress_transition_validator_rejects_table_driven_invalid_sequences(
    previous_kwargs: dict[str, object],
    current_kwargs: dict[str, object],
) -> None:
    previous = _progress_transition_state(**previous_kwargs)
    current = _progress_transition_state(**current_kwargs)
    with pytest.raises(NeverThrown):
        orchestrator._validate_progress_transition_or_never(
            previous=previous,
            current=current,
        )


# gabion:evidence E:call_footprint::tests/test_server_core_orchestrator_coverage.py::test_progress_transition_validator_rejects_complete_marker_before_parent_completion::command_orchestrator.py::gabion.server_core.command_orchestrator.validate_progress_transition_contract
def test_progress_transition_validator_rejects_complete_marker_before_parent_completion() -> None:
    with pytest.raises(NeverThrown):
        orchestrator._validate_progress_transition_or_never(
            previous=None,
            current=_progress_transition_state(done=5, marker="complete"),
        )


# gabion:evidence E:call_footprint::tests/test_server_core_orchestrator_coverage.py::test_progress_transition_validator_normalizes_terminal_and_suppresses_replay::command_orchestrator.py::gabion.server_core.command_orchestrator.validate_progress_transition_contract
def test_progress_transition_validator_normalizes_terminal_and_suppresses_replay() -> None:
    previous = _progress_transition_state(done=6, marker="fingerprint:done")
    terminal = _progress_transition_state(done=6, marker="complete")
    first_decision = orchestrator._validate_progress_transition_or_never(
        previous=previous,
        current=terminal,
    )
    assert first_decision.effective_event_kind == "terminal"
    assert first_decision.reason == "terminal_transition"
    replay_decision = orchestrator._validate_progress_transition_or_never(
        previous=terminal,
        current=terminal,
    )
    assert replay_decision.suppress_emit is True
    assert replay_decision.reason == "terminal_replay_suppressed"

    heartbeat_terminal = _progress_transition_state(
        done=6,
        marker="complete",
        event_kind="heartbeat",
    )
    heartbeat_decision = orchestrator._validate_progress_transition_or_never(
        previous=terminal,
        current=heartbeat_terminal,
    )
    assert heartbeat_decision.reason == "terminal_keepalive"
    assert heartbeat_decision.suppress_emit is False


# gabion:evidence E:call_footprint::tests/test_server_core_orchestrator_coverage.py::test_create_progress_emitter_emits_non_complete_terminal_without_terminal_latch::command_orchestrator.py::gabion.server_core.command_orchestrator._create_progress_emitter
def test_create_progress_emitter_emits_non_complete_terminal_without_terminal_latch(
    tmp_path: Path,
) -> None:
    notifications: list[tuple[str, object]] = []

    def _send_notification(method: str, params: object) -> None:
        notifications.append((method, params))

    emitter = orchestrator._create_progress_emitter(
        notification_runtime=orchestrator._NotificationRuntime(
            send_notification_fn=_send_notification,
            emit_phase_progress_events=False,
        ),
        phase_timeline_markdown_path=tmp_path / "timeline.md",
        phase_timeline_jsonl_path=tmp_path / "timeline.jsonl",
        progress_heartbeat_seconds=1.0,
        profiling_stage_ns={},
        profiling_counters={},
    )
    assert callable(emitter.emit)
    emitter.emit(
        phase="post",
        collection_progress={
            "completed_files": 0,
            "in_progress_files": 0,
            "remaining_files": 1,
            "total_files": 1,
        },
        semantic_progress=None,
        work_done=5,
        work_total=6,
        include_timing=False,
        done=False,
        analysis_state="analysis_post_in_progress",
        classification="active",
        phase_progress_v2={
            "primary_unit": "post_tasks",
            "primary_done": 5,
            "primary_total": 6,
            "dimensions": {"post_tasks": {"done": 5, "total": 6}},
        },
        progress_marker="fingerprint:done",
        event_kind="terminal",
    )
    emitter.stop()

    assert len(notifications) == 1
    _, params = notifications[0]
    assert isinstance(params, dict)
    value = params.get("value")
    assert isinstance(value, dict)
    assert value.get("event_kind") == "terminal"
    transition_v2 = value.get("progress_transition_v2")
    assert isinstance(transition_v2, dict)
    assert transition_v2.get("format_version") == 2
    transition = value.get("progress_transition_v1")
    assert isinstance(transition, dict)
    assert transition.get("terminal_complete") is False
    assert transition.get("reason") == "initial_transition"


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


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._emit_taint_outputs
def test_emit_taint_outputs_rejects_missing_or_invalid_state_payload(
    tmp_path: Path,
) -> None:
    _bind()
    with pytest.raises(NeverThrown):
        orchestrator._emit_taint_outputs(
            response={},
            analysis=_empty_analysis_result(),
            root=str(tmp_path),
            taint_state_path="missing-state.json",
            emit_taint_delta=False,
            emit_taint_state=False,
            write_taint_baseline=False,
            emit_taint_lifecycle=False,
            taint_profile_name="observe",
            taint_boundary_registry_payload=None,
        )

    bad_state = tmp_path / "bad_state.json"
    bad_state.write_text("[]", encoding="utf-8")
    with pytest.raises(NeverThrown):
        orchestrator._emit_taint_outputs(
            response={},
            analysis=_empty_analysis_result(),
            root=str(tmp_path),
            taint_state_path=str(bad_state),
            emit_taint_delta=False,
            emit_taint_state=False,
            write_taint_baseline=False,
            emit_taint_lifecycle=False,
            taint_profile_name="observe",
            taint_boundary_registry_payload=None,
        )


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
