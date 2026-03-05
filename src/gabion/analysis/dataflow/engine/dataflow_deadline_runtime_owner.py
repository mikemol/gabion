# gabion:boundary_normalization_module
from __future__ import annotations

"""Deadline-runtime compatibility owner during WS-5 migration."""

from gabion.analysis.dataflow.engine.dataflow_analysis_index import _build_analysis_index
from gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms import (
    _collect_recursive_functions,
    _collect_recursive_nodes,
    _reachable_from_roots,
)
from gabion.analysis.dataflow.engine.dataflow_facade import (
    _CalleeResolutionOutcome,
    _DeadlineFunctionCollector,
    _DeadlineFunctionFacts,
    _DeadlineLoopFacts,
    _collect_call_edges,
    _collect_call_nodes_by_path,
    _collect_deadline_function_facts,
    _collect_deadline_local_info,
    _normalize_snapshot_path,
    _resolve_callee_outcome,
)


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
