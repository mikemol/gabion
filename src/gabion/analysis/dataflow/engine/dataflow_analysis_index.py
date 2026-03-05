# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Analysis-index ownership boundary during runtime retirement."""


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.dataflow.engine.dataflow_analysis_index_owner import (
    _analyze_file_internal as _indexed_analyze_file_internal,
    _build_analysis_collection_resume_payload as _indexed_build_analysis_collection_resume_payload,
    _build_analysis_index as _indexed_build_analysis_index,
    _build_call_graph as _indexed_build_call_graph,
    _load_analysis_collection_resume_payload as _indexed_load_analysis_collection_resume_payload,
)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never

AnalysisIndex = Any
_PROGRESS_EMIT_MIN_INTERVAL_SECONDS = 1.0
_ANALYSIS_PROFILING_FORMAT_VERSION = 1


@dataclass(frozen=True)
class _PhaseWorkProgress:
    work_done: int
    work_total: int


def _analysis_collection_resume_path_key(path: Path) -> str:
    return str(path)


def _iter_monotonic_paths(
    paths: Iterable[Path],
    *,
    source: str,
) -> list[Path]:
    ordered: list[Path] = []
    previous_path_key = ""
    has_previous_path_key = False
    for path in paths:
        check_deadline()
        path_key = _analysis_collection_resume_path_key(path)
        if has_previous_path_key and previous_path_key > path_key:
            never(
                "path order regression",
                source=source,
                previous_path=previous_path_key,
                current_path=path_key,
            )
        previous_path_key = path_key
        has_previous_path_key = True
        ordered.append(path)
    return ordered


def _phase_work_progress(*, work_done: int, work_total: int) -> _PhaseWorkProgress:
    check_deadline()
    normalized_total = max(int(work_total), 0)
    normalized_done = max(int(work_done), 0)
    if normalized_total:
        normalized_done = min(normalized_done, normalized_total)
    return _PhaseWorkProgress(work_done=normalized_done, work_total=normalized_total)


def _profiling_v1_payload(*, stage_ns: Mapping[str, int], counters: Mapping[str, int]) -> JSONObject:
    return {
        "format_version": _ANALYSIS_PROFILING_FORMAT_VERSION,
        "stage_ns": {str(key): int(stage_ns[key]) for key in stage_ns},
        "counters": {str(key): int(counters[key]) for key in counters},
    }


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
