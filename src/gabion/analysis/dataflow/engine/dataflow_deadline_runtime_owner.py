# gabion:boundary_normalization_module
from __future__ import annotations

"""Deadline-runtime compatibility owner during WS-5 migration."""

import importlib

from gabion.analysis.dataflow.engine.dataflow_analysis_index import _build_analysis_index
from gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms import (
    _collect_recursive_functions,
    _collect_recursive_nodes,
    _reachable_from_roots,
)

_RUNTIME_MODULE = "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan"
_runtime = importlib.import_module(_RUNTIME_MODULE)

_CalleeResolutionOutcome = _runtime._CalleeResolutionOutcome
_DeadlineFunctionCollector = _runtime._DeadlineFunctionCollector
_DeadlineFunctionFacts = _runtime._DeadlineFunctionFacts
_DeadlineLoopFacts = _runtime._DeadlineLoopFacts
_collect_call_edges = _runtime._collect_call_edges
_collect_call_nodes_by_path = _runtime._collect_call_nodes_by_path
_collect_deadline_function_facts = _runtime._collect_deadline_function_facts
_collect_deadline_local_info = _runtime._collect_deadline_local_info
_normalize_snapshot_path = _runtime._normalize_snapshot_path
_resolve_callee_outcome = _runtime._resolve_callee_outcome


__all__ = [
    "_CalleeResolutionOutcome",
    "_DeadlineFunctionCollector",
    "_DeadlineFunctionFacts",
    "_DeadlineLoopFacts",
    "_build_analysis_index",
    "_collect_call_edges",
    "_collect_call_nodes_by_path",
    "_collect_deadline_function_facts",
    "_collect_deadline_local_info",
    "_collect_recursive_functions",
    "_collect_recursive_nodes",
    "_normalize_snapshot_path",
    "_reachable_from_roots",
    "_resolve_callee_outcome",
]
