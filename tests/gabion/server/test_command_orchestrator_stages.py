from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from gabion.analysis.foundation.timeout_context import TimeoutExceeded
from gabion.server_core import command_orchestrator as orchestrator


# gabion:behavior primary=desired
def test_stage_finalize_success_projects_resume_compatibility() -> None:
    captured: dict[str, object] = {}

    def _fake_build_success_response(*, context: object) -> orchestrator._SuccessResponseOutcome:
        captured["context"] = context
        return orchestrator._SuccessResponseOutcome(
            response={"ok": True},
            report_request_state=orchestrator.ReportRequestState(),
        )

    stage = orchestrator._ExecuteCommandFinalizeSuccessStage(
        execute_deps=cast(orchestrator.CommandEffects, SimpleNamespace()),
        aspf_trace_state=None,
        report_analysis_state=orchestrator.ReportAnalysisState(
            analysis=cast(orchestrator.AnalysisResult, SimpleNamespace()),
            root=".",
            request_state=orchestrator.ReportRequestState(
                report_path=None,
                runtime_state=orchestrator.ReportRuntimeState(
                    projection_state=orchestrator.ReportProjectionState(
                        output_path=None,
                        section_journal_path=Path("report_journal.json"),
                        phase_checkpoint_path=None,
                        projection_rows=(),
                    ),
                    checkpoint_state=orchestrator.ReportCheckpointState(),
                ),
            ),
        ),
        paths=[],
        payload={},
        config=cast(orchestrator.AuditConfig, SimpleNamespace()),
        options=cast(orchestrator._ExecutionPayloadOptions, SimpleNamespace()),
        name_filter_bundle=cast(orchestrator.DataflowNameFilterBundle, SimpleNamespace()),
        continuation_state=orchestrator.AnalysisContinuationState(
            resume_state=orchestrator.AnalysisResumeState(
                projection_state=orchestrator.AnalysisResumeProjectionState(
                    compatibility_status="compatible",
                )
            ),
            collection_progress_runtime_state=orchestrator.CollectionProgressRuntimeState(),
        ),
        profiling_stage_ns={},
        profiling_counters={},
        execution_plan=cast(orchestrator.ExecutionPlan, SimpleNamespace()),
        emit_lsp_progress_fn=lambda **_kwargs: None,
        dataflow_capabilities=orchestrator._DataflowCapabilityAnnotations(
            selected_adapter="python",
            supported_analysis_surfaces=[],
            disabled_surface_reasons={},
        ),
        identity_shadow_runtime=None,
    )

    outcome = orchestrator._stage_finalize_success(
        stage=stage,
        build_success_response_fn=_fake_build_success_response,
    )

    assert outcome.response == {"ok": True}
    context = cast(orchestrator._SuccessResponseContext, captured["context"])
    assert (
        context.continuation_state.resume_state.projection_state.compatibility_status
        == "compatible"
    )


# gabion:behavior primary=verboten facets=timeout
def test_stage_finalize_timeout_delegates_cleanup() -> None:
    sentinel = {"timeout": True}

    def _fake_timeout_cleanup(*, exc: TimeoutExceeded, context: object) -> dict:
        assert isinstance(exc, TimeoutExceeded)
        assert context is timeout_context
        return sentinel

    timeout_context = cast(orchestrator._TimeoutCleanupContext, SimpleNamespace())
    outcome = orchestrator._stage_finalize_timeout(
        exc=TimeoutExceeded("timed out"),
        context=timeout_context,
        cleanup_handler_fn=_fake_timeout_cleanup,
    )

    assert outcome is sentinel

# gabion:behavior primary=verboten facets=timeout
def test_stage_execute_analysis_propagates_timeout() -> None:
    def _raise_timeout(*, context: object, state: object, collection_resume_payload: object) -> object:
        raise TimeoutExceeded("timed out")

    try:
        orchestrator._stage_execute_analysis(
            context=cast(orchestrator._AnalysisExecutionContext, SimpleNamespace()),
            state=orchestrator._AnalysisExecutionMutableState(
                collection_progress_runtime_state=orchestrator.CollectionProgressRuntimeState(),
            ),
            collection_resume_payload=None,
            run_analysis_with_progress_fn=_raise_timeout,
        )
    except TimeoutExceeded:
        return
    raise AssertionError("expected TimeoutExceeded")
