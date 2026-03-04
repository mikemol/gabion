# gabion:boundary_normalization_module
from __future__ import annotations

"""Analysis-index owner compatibility surface during WS-5 migration."""

import importlib

_RUNTIME_MODULE = "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan"
_runtime = importlib.import_module(_RUNTIME_MODULE)

AnalysisIndex = _runtime.AnalysisIndex
_CacheIdentity = _runtime._CacheIdentity
_CacheSemanticContext = _runtime._CacheSemanticContext
_ResolvedCallEdge = _runtime._ResolvedCallEdge
_ResumeCacheIdentityPair = _runtime._ResumeCacheIdentityPair
_StageCacheIdentitySpec = _runtime._StageCacheIdentitySpec
_StageCacheSpec = _runtime._StageCacheSpec
_analysis_index_module_trees = _runtime._analysis_index_module_trees
_analysis_index_resolved_call_edges = _runtime._analysis_index_resolved_call_edges
_analysis_index_resolved_call_edges_by_caller = _runtime._analysis_index_resolved_call_edges_by_caller
_analysis_index_stage_cache = _runtime._analysis_index_stage_cache
_analysis_index_transitive_callers = _runtime._analysis_index_transitive_callers
_analyze_file_internal = _runtime._analyze_file_internal
_build_analysis_collection_resume_payload = _runtime._build_analysis_collection_resume_payload
_build_analysis_index = _runtime._build_analysis_index
_build_call_graph = _runtime._build_call_graph
_load_analysis_collection_resume_payload = _runtime._load_analysis_collection_resume_payload
_reduce_resolved_call_edges = _runtime._reduce_resolved_call_edges
_run_indexed_pass = _runtime._run_indexed_pass


__all__ = [
    "AnalysisIndex",
    "_CacheIdentity",
    "_CacheSemanticContext",
    "_ResolvedCallEdge",
    "_ResumeCacheIdentityPair",
    "_StageCacheIdentitySpec",
    "_StageCacheSpec",
    "_analysis_index_module_trees",
    "_analysis_index_resolved_call_edges",
    "_analysis_index_resolved_call_edges_by_caller",
    "_analysis_index_stage_cache",
    "_analysis_index_transitive_callers",
    "_analyze_file_internal",
    "_build_analysis_collection_resume_payload",
    "_build_analysis_index",
    "_build_call_graph",
    "_load_analysis_collection_resume_payload",
    "_reduce_resolved_call_edges",
    "_run_indexed_pass",
]
