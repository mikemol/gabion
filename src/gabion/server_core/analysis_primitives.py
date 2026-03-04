from __future__ import annotations

from dataclasses import dataclass

from gabion.server_core import command_orchestrator_primitives as legacy


@dataclass(frozen=True)
class AnalysisPrimitives:
    analysis_index_resume_hydrated_count = staticmethod(legacy._analysis_index_resume_hydrated_count)
    analysis_index_resume_signature = staticmethod(legacy._analysis_index_resume_signature)
    analysis_resume_cache_verdict = staticmethod(legacy._analysis_resume_cache_verdict)
    analysis_resume_progress = staticmethod(legacy._analysis_resume_progress)
    groups_by_path_from_collection_resume = staticmethod(legacy._groups_by_path_from_collection_resume)
    latest_report_phase = staticmethod(legacy._latest_report_phase)
    normalize_dataflow_response = staticmethod(legacy._normalize_dataflow_response)
    report_witness_digest = staticmethod(legacy._report_witness_digest)
    render_incremental_report = staticmethod(legacy._render_incremental_report)


def default_analysis_primitives() -> AnalysisPrimitives:
    return AnalysisPrimitives()
