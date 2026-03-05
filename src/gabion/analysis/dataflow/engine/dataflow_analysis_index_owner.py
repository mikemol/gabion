# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Analysis-index compatibility owner during WS-5 migration."""

from gabion.analysis.dataflow.engine import dataflow_analysis_index as _analysis_index
from gabion.analysis.dataflow.engine.dataflow_analysis_index import (
    OptionalDecorators,
    OptionalParseFailures,
    OptionalProjectRoot,
    _EMPTY_CACHE_SEMANTIC_CONTEXT,
    _FILE_SCAN_PROGRESS_EMIT_INTERVAL,
    _IndexedPassContext,
    _IndexedPassSpec,
    _PROGRESS_EMIT_MIN_INTERVAL_SECONDS,
    _accumulate_function_index_for_tree,
    _analysis_index_ctor,
    _analysis_index_module_trees,
    _analysis_index_resolved_call_edges,
    _analysis_index_resolved_call_edges_by_caller,
    _analysis_index_stage_cache,
    _analysis_index_transitive_callers,
    _analyze_file_internal,
    _build_analysis_collection_resume_payload,
    _build_analysis_index,
    _build_call_graph,
    _build_function_index,
    _build_single_module_artifact,
    _build_symbol_table,
    _function_index_acc_ctor,
    _iter_monotonic_paths,
    _iter_resolved_edge_param_events,
    _load_analysis_collection_resume_payload,
    _parse_stage_cache_key,
    _phase_work_progress,
    _profiling_v1_payload,
    _progress_emit_min_interval_seconds,
    _reduce_resolved_call_edges,
    _run_indexed_pass,
    _sorted_text,
    _stage_cache_key_aliases,
    analyze_file,
)

# Temporary boundary adapter retained for external import compatibility.
_BOUNDARY_ADAPTER_LIFECYCLE: dict[str, object] = {
    "actor": "codex",
    "rationale": "WS-5 hard-cut completed; retain compatibility alias while external importers migrate",
    "scope": "dataflow_analysis_index_owner.alias_surface",
    "start": "2026-03-05",
    "expiry": "WS-5 compatibility-shim retirement",
    "rollback_condition": "no external consumers require owner path aliases",
    "evidence_links": ["docs/ws5_decomposition_ledger.md"],
}

__all__ = list(getattr(_analysis_index, "__all__", ()))
