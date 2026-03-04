from __future__ import annotations

from gabion.server_core.stage_contracts import AnalysisRunner, JSONObject, StageAnalysisResult


def run_analysis_stage(
    *,
    context: object,
    state: object,
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
        semantic_progress_cumulative=outcome.semantic_progress_cumulative,
        latest_collection_progress=outcome.latest_collection_progress,
        last_collection_resume_payload=outcome.last_collection_resume_payload,
    )
