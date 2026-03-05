# gabion:boundary_normalization_module
from __future__ import annotations

"""Deadline-runtime compatibility owner during WS-5 migration."""

from gabion.analysis.dataflow.engine.dataflow_analysis_index import _build_analysis_index
from gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms import (
    _collect_recursive_functions,
    _collect_recursive_nodes,
    _reachable_from_roots,
)
from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import _is_test_path
from gabion.analysis.dataflow.engine.dataflow_deadline_contracts import (
    _CalleeResolutionOutcome,
    _DeadlineFunctionFacts,
    _DeadlineLoopFacts,
)
from gabion.analysis.dataflow.engine.dataflow_resume_paths import (
    normalize_snapshot_path as _normalize_snapshot_path,
)
from gabion.analysis.dataflow.engine.dataflow_facade import (
    _DeadlineFunctionCollector,
    _collect_call_nodes_by_path,
    _collect_deadline_function_facts,
    _collect_deadline_local_info,
    _resolve_callee_outcome,
)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.indexed_scan.calls.call_edges import (
    CollectCallEdgesDeps as _CollectCallEdgesDeps,
    collect_call_edges as _collect_call_edges_impl,
)


def _collect_call_edges(
    *,
    by_name,
    by_qual,
    symbol_table,
    project_root,
    class_index,
):
    return _collect_call_edges_impl(
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=symbol_table,
        project_root=project_root,
        class_index=class_index,
        resolve_callee_outcome_fn=_resolve_callee_outcome,
        deps=_CollectCallEdgesDeps(
            check_deadline_fn=check_deadline,
            is_test_path_fn=_is_test_path,
        ),
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
