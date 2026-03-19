from __future__ import annotations

from gabion.server_core.stage_contracts import (
    AnalysisContextContract,
    AnalysisRunner,
    AnalysisStateContract,
    JSONObject,
    StageAnalysisResult,
)


def run_analysis_stage(
    *,
    context: AnalysisContextContract,
    state: AnalysisStateContract,
    collection_resume_payload: JSONObject | None,
    run_analysis_with_progress: AnalysisRunner,
) -> StageAnalysisResult:
    outcome = run_analysis_with_progress(
        context=context,
        state=state,
        collection_resume_payload=collection_resume_payload,
    )
    return StageAnalysisResult(
        analysis_outcome=outcome,
        collection_progress_runtime_state=outcome.collection_progress_runtime_state,
    )
