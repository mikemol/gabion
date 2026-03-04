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


def _analyze_file_internal(path, *, recursive, config, resume_state, on_progress, on_profile):
    return _indexed_analyze_file_internal(
        path,
        recursive=recursive,
        config=config,
        resume_state=resume_state,
        on_progress=on_progress,
        on_profile=on_profile,
    )


def _build_analysis_collection_resume_payload(
    *,
    groups_by_path,
    param_spans_by_path,
    bundle_sites_by_path,
    invariant_propositions,
    completed_paths,
    in_progress_scan_by_path,
    analysis_index_resume,
    file_stage_timings_v1_by_path,
):
    return _indexed_build_analysis_collection_resume_payload(
        groups_by_path=groups_by_path,
        param_spans_by_path=param_spans_by_path,
        bundle_sites_by_path=bundle_sites_by_path,
        invariant_propositions=invariant_propositions,
        completed_paths=completed_paths,
        in_progress_scan_by_path=in_progress_scan_by_path,
        analysis_index_resume=analysis_index_resume,
        file_stage_timings_v1_by_path=file_stage_timings_v1_by_path,
    )


def _build_analysis_index(
    paths,
    *,
    project_root,
    ignore_params,
    strictness,
    external_filter,
    transparent_decorators,
    parse_failure_witnesses,
    resume_payload=None,
    on_progress=None,
    forest_spec_id=None,
    fingerprint_seed_revision=None,
    decision_ignore_params=None,
    decision_require_tiers=False,
):
    return _indexed_build_analysis_index(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        resume_payload=resume_payload,
        on_progress=on_progress,
        forest_spec_id=forest_spec_id,
        fingerprint_seed_revision=fingerprint_seed_revision,
        decision_ignore_params=decision_ignore_params,
        decision_require_tiers=decision_require_tiers,
    )


def _build_call_graph(
    paths,
    *,
    project_root,
    ignore_params,
    strictness,
    external_filter,
    transparent_decorators=None,
    parse_failure_witnesses,
    analysis_index=None,
):
    return _indexed_build_call_graph(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
    )


def _load_analysis_collection_resume_payload(
    *,
    payload,
    file_paths,
    include_invariant_propositions,
):
    return _indexed_load_analysis_collection_resume_payload(
        payload=payload,
        file_paths=file_paths,
        include_invariant_propositions=include_invariant_propositions,
    )


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
