from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from gabion import server
from gabion.analysis.aspf.aspf import Forest
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.dataflow.engine.dataflow_contracts import AnalysisResult
from gabion.analysis.foundation.identity_shadow_runtime import (
    IntegerAnchorDecode,
    build_identity_shadow_runtime,
)
from gabion.analysis.foundation.identity_shadow_session import IdentityShadowSession
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


# gabion:behavior primary=desired
def test_execute_command_total_starts_and_stops_identity_shadow_session(
    tmp_path: Path,
) -> None:
    _bind()
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "def callee(x):\n"
        "    return x\n"
        "\n"
        "def caller(a, b):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )
    mirror_events: list[str] = []

    class _Mirror:
        def start(self) -> None:
            mirror_events.append("start")

        def stop(self) -> None:
            mirror_events.append("stop")

    def _build_session(
        *,
        registry: object,
        root: Path,
    ) -> IdentityShadowSession:
        _ = root
        assert isinstance(registry, PrimeRegistry)
        mirror_events.append("build")
        runtime = build_identity_shadow_runtime(
            run_id="run:coverage:session",
            registry=registry,
        )
        return IdentityShadowSession(runtime=runtime, registry_mirror=_Mirror())

    class _Workspace:
        def __init__(self, root_path: str) -> None:
            self.root_path = root_path

    class _DummyNotifyingServer:
        def __init__(self, root_path: str) -> None:
            self.workspace = _Workspace(root_path)
            self.notifications: list[tuple[str, dict[str, object]]] = []

        def send_notification(self, method: str, params: dict[str, object]) -> None:
            self.notifications.append((method, params))

    ls = _DummyNotifyingServer(str(tmp_path))
    payload = {
        "root": str(tmp_path),
        "paths": [str(module_path)],
        "report": "-",
        "analysis_timeout_ticks": 50_000,
        "analysis_timeout_tick_ns": 1_000_000,
    }
    deps = server._default_execute_command_deps().with_overrides(
        analyze_paths_fn=lambda *_args, **_kwargs: _empty_analysis_result(),
    )
    response = orchestrator.execute_command_total(
        ls,
        payload,
        deps=deps,
        build_identity_shadow_session_fn=_build_session,
    )

    assert response.canonical.analysis_state == "succeeded"
    assert mirror_events[0] == "build"
    assert mirror_events[1] == "start"
    assert mirror_events[-1] == "stop"
    assert mirror_events.count("start") == 1
    assert mirror_events.count("stop") == 1

    by_token: dict[str, dict[str, object]] = {}
    for method, params in ls.notifications:
        if method != orchestrator._LSP_PROGRESS_NOTIFICATION_METHOD:
            continue
        token = params.get("token")
        value = params.get("value")
        if isinstance(token, str) and isinstance(value, dict):
            by_token[token] = value
    assert orchestrator._LSP_PROGRESS_TOKEN_V2 in by_token
    assert (
        by_token[orchestrator._LSP_PROGRESS_TOKEN_V2].get("schema")
        == "gabion/canonical_progress_event_v2"
    )


# gabion:behavior primary=desired
def test_execute_command_total_uses_protocol_notify_when_send_notification_missing(
    tmp_path: Path,
) -> None:
    _bind()
    module_path = tmp_path / "sample.py"
    module_path.write_text("def sample(x):\n    return x\n")

    class _Workspace:
        def __init__(self, root_path: str) -> None:
            self.root_path = root_path

    notifications: list[tuple[str, dict[str, object]]] = []

    class _Protocol:
        def notify(self, method: str, params: dict[str, object]) -> None:
            notifications.append((method, params))

    class _DummyProtocolServer:
        def __init__(self, root_path: str) -> None:
            self.workspace = _Workspace(root_path)
            self.protocol = _Protocol()

    ls = _DummyProtocolServer(str(tmp_path))
    payload = {
        "root": str(tmp_path),
        "paths": [str(module_path)],
        "report": "-",
        "analysis_timeout_ticks": 50_000,
        "analysis_timeout_tick_ns": 1_000_000,
    }
    deps = server._default_execute_command_deps().with_overrides(
        analyze_paths_fn=lambda *_args, **_kwargs: _empty_analysis_result(),
    )

    response = orchestrator.execute_command_total(
        ls,
        payload,
        deps=deps,
    )

    assert response.canonical.analysis_state == "succeeded"
    assert any(
        method == orchestrator._LSP_PROGRESS_NOTIFICATION_METHOD
        for method, _params in notifications
    )


def _analysis_context(
    *,
    tmp_path: Path,
    payload: dict[str, object],
) -> orchestrator._AnalysisExecutionContext:
    _bind()
    return orchestrator._AnalysisExecutionContext(
        trace_runtime_context=orchestrator._TraceRuntimeContext(
            execute_deps=server._default_execute_command_deps(),
            aspf_trace_state=None,
        ),
        runtime_state=orchestrator.CommandRuntimeState(latest_collection_progress={}),
        forest=Forest(),
        paths=[tmp_path / "sample.py"],
        no_recursive=False,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=50,
        report_request_state=orchestrator.ReportRequestState(
            report_path=False,
            runtime_state=orchestrator.ReportRuntimeState(
                projection_state=orchestrator.ReportProjectionState(
                    output_path=None,
                    section_journal_path=tmp_path / "sections.json",
                    phase_checkpoint_path=tmp_path / "phase.json",
                    projection_rows=(),
                ),
                checkpoint_state=orchestrator.ReportCheckpointState(),
            ),
        ),
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
        analysis_resume_state=orchestrator.AnalysisResumeState(),
        phase_timeline_path=tmp_path / "timeline.md",
        emit_phase_timeline=False,
        enable_phase_projection_checkpoints=False,
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


# gabion:behavior primary=verboten facets=timeout
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


# gabion:behavior primary=allowed_unwanted facets=legacy
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


# gabion:behavior primary=allowed_unwanted facets=legacy
@pytest.mark.parametrize(
    ("payload", "expected_kind"),
    [
        ({"emit_test_obsolescence": True}, "report"),
        ({"emit_test_obsolescence_state": True}, "state"),
        ({"emit_test_obsolescence_delta": True}, "delta"),
        ({"write_test_obsolescence_baseline": True}, "baseline-write"),
        ({}, "off"),
    ],
)
def test_auxiliary_mode_from_payload_legacy_kind_resolution_order(
    payload: dict[str, object],
    expected_kind: str,
) -> None:
    _bind()
    mode = orchestrator._auxiliary_mode_from_payload(
        payload=payload,
        mode_key="obsolescence_mode",
        state_key="test_obsolescence_state",
        emit_state_key="emit_test_obsolescence_state",
        emit_delta_key="emit_test_obsolescence_delta",
        write_baseline_key="write_test_obsolescence_baseline",
        emit_report_key="emit_test_obsolescence",
        domain="obsolescence",
        allow_report=True,
    )
    assert mode.kind == expected_kind


# gabion:behavior primary=allowed_unwanted facets=legacy
def test_auxiliary_mode_from_payload_legacy_kind_resolution_conflict_still_rejected() -> None:
    _bind()
    with pytest.raises(NeverThrown):
        orchestrator._auxiliary_mode_from_payload(
            payload={
                "emit_test_obsolescence": True,
                "emit_test_obsolescence_state": True,
            },
            mode_key="obsolescence_mode",
            state_key="test_obsolescence_state",
            emit_state_key="emit_test_obsolescence_state",
            emit_delta_key="emit_test_obsolescence_delta",
            write_baseline_key="write_test_obsolescence_baseline",
            emit_report_key="emit_test_obsolescence",
            domain="obsolescence",
            allow_report=True,
        )


# gabion:behavior primary=verboten facets=invalid
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


# gabion:behavior primary=verboten facets=invalid
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


# gabion:behavior primary=desired
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
# gabion:behavior primary=desired
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
# gabion:behavior primary=verboten facets=edge
def test_taint_marker_row_from_ambiguity_witness_edge_inputs() -> None:
    _bind()
    with pytest.raises(NeverThrown):
        orchestrator._taint_marker_row_from_ambiguity_witness("bad")
    with pytest.raises(NeverThrown):
        orchestrator._taint_marker_row_from_ambiguity_witness(
            {
                "kind": "k",
                "candidate_count": "bad-count",
                "site": "not-a-mapping",
            }
        )
    row = orchestrator._taint_marker_row_from_ambiguity_witness(
        {
            "kind": "k",
            "candidate_count": 2,
            "site": {"path": "a.py", "function": "fn"},
        }
    )
    assert row["reason"] == "ambiguity witness kind=k candidates=2"


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._taint_marker_row_from_type_ambiguity
# gabion:behavior primary=verboten facets=edge
def test_taint_marker_row_from_type_ambiguity_edge_inputs() -> None:
    _bind()
    with pytest.raises(NeverThrown):
        orchestrator._taint_marker_row_from_type_ambiguity("")
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
# gabion:behavior primary=desired
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
# gabion:behavior primary=verboten facets=invalid
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
# gabion:behavior primary=desired
def test_progress_transition_validator_rejects_complete_marker_before_parent_completion() -> None:
    with pytest.raises(NeverThrown):
        orchestrator._validate_progress_transition_or_never(
            previous=None,
            current=_progress_transition_state(done=5, marker="complete"),
        )


# gabion:evidence E:call_footprint::tests/test_server_core_orchestrator_coverage.py::test_progress_transition_validator_normalizes_terminal_and_suppresses_replay::command_orchestrator.py::gabion.server_core.command_orchestrator.validate_progress_transition_contract
# gabion:behavior primary=desired
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
# gabion:behavior primary=desired
def test_create_progress_emitter_emits_non_complete_terminal_without_terminal_latch(
    tmp_path: Path,
) -> None:
    notifications: list[tuple[str, object]] = []
    identity_shadow_runtime = build_identity_shadow_runtime(
        run_id="run:coverage",
        registry=PrimeRegistry(),
    )

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
        identity_shadow_runtime=identity_shadow_runtime,
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
    by_token: dict[str, dict[str, object]] = {}
    for _method, params in notifications:
        assert isinstance(params, dict)
        token = params.get("token")
        value = params.get("value")
        if isinstance(token, str) and isinstance(value, dict):
            by_token[token] = value
    assert orchestrator._LSP_PROGRESS_TOKEN_V2 in by_token

    v2_value = by_token[orchestrator._LSP_PROGRESS_TOKEN_V2]
    assert v2_value.get("schema") == "gabion/canonical_progress_event_v2"
    assert v2_value.get("adaptation_kind") == "valid"
    event = v2_value.get("event")
    assert isinstance(event, dict)
    payload = event.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("event_kind") == "terminal"
    transition_v2 = payload.get("progress_transition_v2")
    assert isinstance(transition_v2, dict)
    assert transition_v2.get("format_version") == 2
    transition = payload.get("progress_transition_v2")
    assert isinstance(transition, dict)
    assert transition.get("terminal_complete") is False
    assert transition.get("reason") == "initial_transition"
    assert v2_value.get("adaptation_error") == ""
    assert v2_value.get("rejected_progress_payload_v2") is None
    assert isinstance(v2_value.get("identity_allocation_delta_v1"), list)


# gabion:behavior primary=allowed_unwanted facets=fallback
def test_create_progress_emitter_emits_rejected_canonical_v2_with_v1_fallback(
    tmp_path: Path,
) -> None:
    notifications: list[tuple[str, object]] = []

    class _RejectingIntegerCarrier:
        def encode_anchor_tokens(
            self,
            *,
            namespace: str,
            key: str,
            value: int,
        ) -> tuple[str, ...]:
            _ = (namespace, key, value)
            return ()

        def decode_anchor_tokens(
            self,
            *,
            namespace: str,
            key: str,
            tokens: tuple[str, ...],
        ) -> IntegerAnchorDecode:
            _ = (namespace, key, tokens)
            return IntegerAnchorDecode(is_present=False)

    identity_shadow_runtime = build_identity_shadow_runtime(
        run_id="run:coverage:rejected",
        registry=PrimeRegistry(),
        integer_carrier=_RejectingIntegerCarrier(),
    )

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
        identity_shadow_runtime=identity_shadow_runtime,
    )
    emitter.emit(
        phase="post",
        collection_progress={
            "completed_files": 0,
            "in_progress_files": 0,
            "remaining_files": 1,
            "total_files": 1,
        },
        semantic_progress=None,
        work_done=1,
        work_total=2,
        include_timing=False,
        done=False,
        analysis_state="analysis_post_in_progress",
        classification="active",
        progress_marker="fingerprint:normalize",
        event_kind="progress",
    )
    emitter.stop()

    by_token: dict[str, dict[str, object]] = {}
    for _method, params in notifications:
        assert isinstance(params, dict)
        token = params.get("token")
        value = params.get("value")
        if isinstance(token, str) and isinstance(value, dict):
            by_token[token] = value
    assert orchestrator._LSP_PROGRESS_TOKEN_V2 in by_token

    v2_value = by_token[orchestrator._LSP_PROGRESS_TOKEN_V2]
    assert v2_value.get("schema") == "gabion/canonical_progress_event_v2"
    assert v2_value.get("adaptation_kind") == "rejected"
    assert v2_value.get("event") is None
    assert isinstance(v2_value.get("adaptation_error"), str)
    fallback_payload = v2_value.get("rejected_progress_payload_v2")
    assert isinstance(fallback_payload, dict)
    assert fallback_payload.get("schema") == "gabion/dataflow_progress_v1"


# gabion:behavior primary=desired
def test_execute_analysis_phase_applies_runtime_payload_overrides_without_analysis(
    tmp_path: Path,
) -> None:
    _bind()
    context = _analysis_context(
        tmp_path=tmp_path,
        payload={
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
            collection_progress_runtime_state=orchestrator.CollectionProgressRuntimeState(),
        ),
        collection_resume_payload=None,
    )
    assert isinstance(outcome.analysis, AnalysisResult)


# gabion:evidence E:function_site::command_orchestrator.py::gabion.server_core.command_orchestrator._emit_taint_outputs
# gabion:behavior primary=verboten facets=invalid,missing
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


# gabion:behavior primary=desired
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
        trace_runtime_context=orchestrator._TraceRuntimeContext(
            execute_deps=deps,
            aspf_trace_state=None,
        ),
        needs_analysis=True,
    )
    outcome = orchestrator._run_analysis_with_progress(
        context=context,
        state=orchestrator._AnalysisExecutionMutableState(
            collection_progress_runtime_state=orchestrator.CollectionProgressRuntimeState(),
        ),
        collection_resume_payload={
            "completed_paths": [],
            "in_progress_scan_by_path": {},
            "semantic_progress": {},
        },
    )
    assert isinstance(outcome.analysis, AnalysisResult)


# gabion:behavior primary=desired
def test_build_success_response_emits_analysis_resume_block_when_resume_source_present(
    tmp_path: Path,
) -> None:
    _bind()
    options = orchestrator._parse_execution_payload_options(
        payload={},
        root=tmp_path,
    )
    context = orchestrator._SuccessResponseContext(
        continuation_runtime_context=orchestrator._ContinuationRuntimeContext(
            trace_runtime_context=orchestrator._TraceRuntimeContext(
                execute_deps=server._default_execute_command_deps(),
                aspf_trace_state=None,
            ),
            continuation_state=orchestrator.AnalysisContinuationState(
                resume_state=orchestrator.AnalysisResumeState(
                    projection_state=orchestrator.AnalysisResumeProjectionState(
                        runtime_state=orchestrator.AnalysisResumeRuntimeState(
                            state_path=None,
                            state_status="checkpoint_seeded",
                            reused_files=1,
                            total_files=3,
                        ),
                        source="resume_manifest",
                        compatibility_status="compatible",
                    ),
                    support_state=orchestrator.AnalysisResumeSupportState(
                        input_state=orchestrator.AnalysisResumeInputState(
                            manifest_digest="digest"
                        )
                    ),
                ),
                collection_progress_runtime_state=orchestrator.CollectionProgressRuntimeState(),
            ),
        ),
        report_analysis_state=orchestrator.ReportAnalysisState(
            analysis=_empty_analysis_result(),
            root=str(tmp_path),
            request_state=orchestrator.ReportRequestState(
                report_path=False,
                runtime_state=orchestrator.ReportRuntimeState(
                    projection_state=orchestrator.ReportProjectionState(
                        output_path=None,
                        section_journal_path=tmp_path / "sections.json",
                        phase_checkpoint_path=tmp_path / "phase.json",
                        projection_rows=(),
                    ),
                    checkpoint_state=orchestrator.ReportCheckpointState(),
                ),
            ),
        ),
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
        profiling_stage_ns={"server.analysis_call": 0, "server.projection_emit": 0},
        profiling_counters={
            "server.collection_resume_persist_calls": 0,
            "server.projection_emit_calls": 0,
        },
        execution_plan=_Plan(),
        emit_lsp_progress_fn=lambda **_kwargs: None,
        dataflow_capabilities=orchestrator._DataflowCapabilityAnnotations(
            selected_adapter="python:default",
            supported_analysis_surfaces=[],
            disabled_surface_reasons={},
        ),
    )
    outcome = orchestrator._build_success_response(context=context)
    resume_payload = outcome.response.payload.get("analysis_resume")
    assert isinstance(resume_payload, dict)
    assert resume_payload["source"] == "resume_manifest"
    assert resume_payload["cache_verdict"] in {"seeded", "warm"}


# gabion:behavior primary=desired
def test_execute_command_total_uses_analysis_stage_module(
    tmp_path: Path,
) -> None:
    _bind()
    module_path = tmp_path / "sample.py"
    module_path.write_text("def f(x):\n    return x\n")

    class _Workspace:
        def __init__(self, root_path: str) -> None:
            self.root_path = root_path

    class _DummyServer:
        def __init__(self, root_path: str) -> None:
            self.workspace = _Workspace(root_path)

    called = {"analysis_stage": False}

    def _wrapped_run_analysis_stage(**kwargs: object):
        called["analysis_stage"] = True
        return orchestrator.run_analysis_stage(**kwargs)

    deps = server._default_execute_command_deps().with_overrides(
        analyze_paths_fn=lambda *_args, **_kwargs: _empty_analysis_result(),
    )
    response = orchestrator.execute_command_total(
        _DummyServer(str(tmp_path)),
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "report": "-",
            "analysis_timeout_ticks": 5_000,
            "analysis_timeout_tick_ns": 1_000_000,
        },
        deps=deps,
        run_analysis_stage_fn=_wrapped_run_analysis_stage,
    )

    assert called["analysis_stage"] is True
    assert response.canonical.analysis_state == "succeeded"
