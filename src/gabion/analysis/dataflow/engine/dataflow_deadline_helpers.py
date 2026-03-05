# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Deadline helper owner surface for obligations extraction."""

import re
from pathlib import Path

from gabion.analysis.aspf.aspf import Forest, NodeId
from gabion.analysis.dataflow.engine.dataflow_analysis_index_owner import (
    _build_analysis_index,
)
from gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms import (
    _collect_recursive_functions,
    _collect_recursive_nodes,
    _reachable_from_roots,
)
from gabion.analysis.dataflow.engine.dataflow_contracts import CallArgs, FunctionInfo, OptionalSpan4
from gabion.analysis.dataflow.engine.dataflow_deadline_contracts import (
    _CalleeResolutionOutcome,
    _DeadlineFunctionFacts,
    _DeadlineLoopFacts,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_runtime_owner import (
    _DeadlineFunctionCollector,
    _collect_call_edges,
    _collect_call_nodes_by_path,
    _collect_deadline_function_facts,
    _collect_deadline_local_info,
    _is_dynamic_dispatch_callee_key,
    _materialize_call_candidates,
    _resolve_callee,
    _resolve_callee_outcome,
)
from gabion.analysis.dataflow.engine.dataflow_resume_paths import (
    normalize_snapshot_path as _normalize_snapshot_path,
)
from gabion.analysis.indexed_scan.deadline.deadline_runtime import (
    DeadlineArgInfo as _DeadlineArgInfo,
    bind_call_args as _bind_call_args,
    caller_param_bindings_for_call as _caller_param_bindings_for_call,
    classify_deadline_expr as _classify_deadline_expr,
    collect_call_edges_from_forest as _collect_call_edges_from_forest,
    collect_call_resolution_obligation_details_from_forest as _collect_call_resolution_obligation_details_from_forest,
    collect_call_resolution_obligations_from_forest as _collect_call_resolution_obligations_from_forest,
    deadline_arg_info_map as _deadline_arg_info_map,
    deadline_loop_forwarded_params as _deadline_loop_forwarded_params,
    function_suite_id as _function_suite_id,
    function_suite_key as _function_suite_key,
    is_deadline_origin_call as _is_deadline_origin_call,
)
from gabion.analysis.indexed_scan.deadline.deadline_fallback import (
    fallback_deadline_arg_info as _fallback_deadline_arg_info)
from gabion.analysis.foundation.json_types import JSONObject

_DEADLINE_HELPER_QUALS = {
    "gabion.analysis.timeout_context.check_deadline",
    "gabion.analysis.timeout_context.deadline_loop_iter",
    "gabion.analysis.timeout_context.set_deadline",
    "gabion.analysis.timeout_context.reset_deadline",
    "gabion.analysis.timeout_context.get_deadline",
    "gabion.analysis.timeout_context.deadline_scope",
}
_DEADLINE_EXEMPT_PREFIXES = ("gabion.analysis.timeout_context.",)


def _is_deadline_annot(annot: object) -> bool:
    if not annot:
        return False
    return bool(re.search(r"\bDeadline\b", str(annot)))


def _is_deadline_param(name: str, annot: object) -> bool:
    if _is_deadline_annot(annot):
        return True
    if annot is None and name.lower() == "deadline":
        return True
    return False


__all__ = [
    "_CalleeResolutionOutcome",
    "_DEADLINE_EXEMPT_PREFIXES",
    "_DEADLINE_HELPER_QUALS",
    "_DeadlineArgInfo",
    "_DeadlineFunctionCollector",
    "_DeadlineFunctionFacts",
    "_build_analysis_index",
    "_bind_call_args",
    "_caller_param_bindings_for_call",
    "_classify_deadline_expr",
    "_collect_call_edges",
    "_collect_call_edges_from_forest",
    "_collect_call_nodes_by_path",
    "_collect_call_resolution_obligation_details_from_forest",
    "_collect_call_resolution_obligations_from_forest",
    "_collect_deadline_function_facts",
    "_collect_deadline_local_info",
    "_collect_recursive_functions",
    "_collect_recursive_nodes",
    "_deadline_arg_info_map",
    "_deadline_loop_forwarded_params",
    "_fallback_deadline_arg_info",
    "_function_suite_id",
    "_function_suite_key",
    "_is_deadline_origin_call",
    "_is_deadline_param",
    "_is_dynamic_dispatch_callee_key",
    "_materialize_call_candidates",
    "_normalize_snapshot_path",
    "_reachable_from_roots",
    "_resolve_callee",
    "_resolve_callee_outcome",
]
