# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Deadline helper owner surface for obligations extraction."""

import re
from pathlib import Path

from gabion.analysis.aspf.aspf import Forest, NodeId
from gabion.analysis.dataflow.engine.dataflow_contracts import CallArgs, FunctionInfo, OptionalSpan4
from gabion.analysis.dataflow.engine.dataflow_deadline_runtime_owner import (
    _CalleeResolutionOutcome,
    _DeadlineFunctionCollector,
    _DeadlineFunctionFacts,
    _DeadlineLoopFacts,
    _build_analysis_index as _indexed_build_analysis_index,
    _collect_call_edges as _collect_call_edges,
    _collect_call_nodes_by_path as _indexed_collect_call_nodes_by_path,
    _collect_deadline_function_facts as _indexed_collect_deadline_function_facts,
    _collect_deadline_local_info,
    _collect_recursive_functions,
    _collect_recursive_nodes as _indexed_collect_recursive_nodes,
    _normalize_snapshot_path,
    _reachable_from_roots as _indexed_reachable_from_roots,
    _resolve_callee_outcome as _indexed_resolve_callee_outcome,
)
from gabion.analysis.indexed_scan.deadline.deadline_runtime import (
    DeadlineArgInfo as _DeadlineArgInfo, caller_param_bindings_for_call as _indexed_caller_param_bindings_for_call, classify_deadline_expr as _classify_deadline_expr, collect_call_edges_from_forest as _indexed_collect_call_edges_from_forest, collect_call_resolution_obligation_details_from_forest as _indexed_collect_call_resolution_obligation_details_from_forest, collect_call_resolution_obligations_from_forest as _indexed_collect_call_resolution_obligations_from_forest, deadline_arg_info_map as _indexed_deadline_arg_info_map, deadline_loop_forwarded_params as _indexed_deadline_loop_forwarded_params, function_suite_id as _function_suite_id, function_suite_key as _function_suite_key, is_deadline_origin_call as _is_deadline_origin_call, materialize_call_candidates as _indexed_materialize_call_candidates)
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


def _build_analysis_index(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators,
    parse_failure_witnesses: list[JSONObject],
):
    return _indexed_build_analysis_index(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
    )


def _collect_call_edges_from_forest(
    forest: Forest,
    *,
    by_name: dict[str, list[FunctionInfo]],
) -> dict[NodeId, set[NodeId]]:
    return _indexed_collect_call_edges_from_forest(forest, by_name=by_name)


def _collect_call_nodes_by_path(
    paths: list[Path],
    *,
    trees=None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index=None,
):
    return _indexed_collect_call_nodes_by_path(
        paths,
        trees=trees,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
    )


def _collect_call_resolution_obligations_from_forest(
    forest: Forest,
) -> list[tuple[NodeId, NodeId, tuple[int, int, int, int], str]]:
    return _indexed_collect_call_resolution_obligations_from_forest(forest)


def _collect_call_resolution_obligation_details_from_forest(
    forest: Forest,
) -> list[tuple[NodeId, NodeId, tuple[int, int, int, int], str, str]]:
    return _indexed_collect_call_resolution_obligation_details_from_forest(forest)


def _collect_deadline_function_facts(
    paths: list[Path],
    *,
    project_root=None,
    ignore_params: set[str],
    parse_failure_witnesses: list[JSONObject],
    trees=None,
    analysis_index=None,
    stage_cache_fn=None,
) -> dict[str, _DeadlineFunctionFacts]:
    return _indexed_collect_deadline_function_facts(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        parse_failure_witnesses=parse_failure_witnesses,
        trees=trees,
        analysis_index=analysis_index,
        stage_cache_fn=stage_cache_fn,
    )


def _collect_recursive_nodes(edges) -> set[object]:
    return _indexed_collect_recursive_nodes(edges)


def _caller_param_bindings_for_call(
    call: CallArgs,
    callee: FunctionInfo,
    *,
    strictness: str,
) -> dict[str, set[str]]:
    return _indexed_caller_param_bindings_for_call(call, callee, strictness=strictness)


def _deadline_arg_info_map(
    call: CallArgs,
    callee: FunctionInfo,
    *,
    call_node,
    alias_to_param,
    origin_vars,
    strictness: str,
) -> dict[str, _DeadlineArgInfo]:
    return _indexed_deadline_arg_info_map(
        call,
        callee,
        call_node=call_node,
        alias_to_param=alias_to_param,
        origin_vars=origin_vars,
        strictness=strictness,
    )


def _deadline_loop_forwarded_params(
    *,
    qual: str,
    loop_fact: _DeadlineLoopFacts,
    deadline_params,
    call_infos,
) -> set[str]:
    return _indexed_deadline_loop_forwarded_params(
        qual=qual,
        loop_fact=loop_fact,
        deadline_params=deadline_params,
        call_infos=call_infos,
    )


def _materialize_call_candidates(
    *,
    forest: Forest,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table,
    project_root,
    class_index,
    resolve_callee_outcome_fn=None,
) -> None:
    if resolve_callee_outcome_fn is None:
        resolve_callee_outcome_fn = _indexed_resolve_callee_outcome
    _indexed_materialize_call_candidates(
        forest=forest,
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=symbol_table,
        project_root=project_root,
        class_index=class_index,
        resolve_callee_outcome_fn=resolve_callee_outcome_fn,
        normalize_snapshot_path_fn=_normalize_snapshot_path,
    )


def _reachable_from_roots(edges, roots):
    return _indexed_reachable_from_roots(edges, roots)


def _resolve_callee_outcome(
    callee_key: str,
    caller: FunctionInfo,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    *,
    symbol_table=None,
    project_root=None,
    class_index=None,
    call=None,
    ambiguity_sink=None,
    local_lambda_bindings=None,
):
    return _indexed_resolve_callee_outcome(
        callee_key,
        caller,
        by_name,
        by_qual,
        symbol_table=symbol_table,
        project_root=project_root,
        class_index=class_index,
        call=call,
        local_lambda_bindings=local_lambda_bindings,
    )


__all__ = [
    "_CalleeResolutionOutcome",
    "_DEADLINE_EXEMPT_PREFIXES",
    "_DEADLINE_HELPER_QUALS",
    "_DeadlineArgInfo",
    "_DeadlineFunctionCollector",
    "_DeadlineFunctionFacts",
    "_build_analysis_index",
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
    "_materialize_call_candidates",
    "_normalize_snapshot_path",
    "_reachable_from_roots",
    "_resolve_callee_outcome",
]
