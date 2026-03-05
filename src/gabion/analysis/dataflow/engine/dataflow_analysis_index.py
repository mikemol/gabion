# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Analysis-index ownership boundary during runtime retirement."""


from typing import Any

from gabion.analysis.dataflow.engine.dataflow_analysis_index_owner import (
    _PROGRESS_EMIT_MIN_INTERVAL_SECONDS as _indexed_progress_emit_min_interval_seconds,
    _PhaseWorkProgress as _indexed_phase_work_progress_type,
    _analyze_file_internal as _indexed_analyze_file_internal,
    _build_analysis_collection_resume_payload as _indexed_build_analysis_collection_resume_payload,
    _build_analysis_index as _indexed_build_analysis_index,
    _build_call_graph as _indexed_build_call_graph,
    _iter_monotonic_paths_owner as _indexed_iter_monotonic_paths,
    _load_analysis_collection_resume_payload as _indexed_load_analysis_collection_resume_payload,
    _phase_work_progress_owner as _indexed_phase_work_progress,
    _profiling_v1_payload_owner as _indexed_profiling_v1_payload,
)
from gabion.analysis.dataflow.engine.dataflow_resume_serialization import (
    _analysis_collection_resume_path_key as _resume_analysis_collection_resume_path_key,
)

AnalysisIndex = Any
_PROGRESS_EMIT_MIN_INTERVAL_SECONDS = _indexed_progress_emit_min_interval_seconds

_PhaseWorkProgress = _indexed_phase_work_progress_type


_analysis_collection_resume_path_key = _resume_analysis_collection_resume_path_key


_iter_monotonic_paths = _indexed_iter_monotonic_paths


_phase_work_progress = _indexed_phase_work_progress


_profiling_v1_payload = _indexed_profiling_v1_payload


_analyze_file_internal = _indexed_analyze_file_internal

_build_analysis_collection_resume_payload = _indexed_build_analysis_collection_resume_payload

_build_analysis_index = _indexed_build_analysis_index

_build_call_graph = _indexed_build_call_graph

_load_analysis_collection_resume_payload = _indexed_load_analysis_collection_resume_payload


__all__ = [
    "AnalysisIndex",
    "_PROGRESS_EMIT_MIN_INTERVAL_SECONDS",
    "_build_call_graph",
    "_analysis_collection_resume_path_key",
    "_analyze_file_internal",
    "_build_analysis_collection_resume_payload",
    "_build_analysis_index",
    "_iter_monotonic_paths",
    "_load_analysis_collection_resume_payload",
    "_phase_work_progress",
    "_profiling_v1_payload",
]
