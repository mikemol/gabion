# gabion:boundary_normalization_module
from __future__ import annotations

"""Deadline-runtime compatibility owner during WS-5 migration."""

from functools import partial
from pathlib import Path
import re

from gabion.analysis.dataflow.engine.dataflow_analysis_index import _build_analysis_index
from gabion.analysis.dataflow.engine.dataflow_analysis_index_owner import (
    _EMPTY_CACHE_SEMANTIC_CONTEXT,
    _analysis_index_stage_cache,
    _parse_stage_cache_key,
    _sorted_text,
)
from gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms import (
    _collect_recursive_functions,
    _collect_recursive_nodes,
    _reachable_from_roots,
)
from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import (
    _is_test_path,
    _module_name,
    _target_names,
)
from gabion.analysis.dataflow.engine.dataflow_callee_resolution import (
    CalleeResolutionContext as _CalleeResolutionContextCore,
    collect_callee_resolution_effects as _collect_callee_resolution_effects_impl,
    resolve_callee_with_effects as _resolve_callee_with_effects_impl,
)
from gabion.analysis.dataflow.engine.dataflow_callee_resolution_support import _callee_key
from gabion.analysis.dataflow.engine.dataflow_contracts import CallArgs, FunctionInfo
from gabion.analysis.dataflow.engine.dataflow_deadline_collector import (
    make_deadline_function_collector,
)
from gabion.analysis.dataflow.engine.dataflow_post_phase_analyses import (
    _StageCacheSpec as _StageCacheSpec_owner,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_contracts import (
    _CalleeResolutionOutcome,
    _DeadlineFunctionFacts,
    _DeadlineLocalInfo,
    _DeadlineLoopFacts,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_helpers import (
    _collect_functions,
    _enclosing_scopes,
    _node_span,
    _param_names,
)
from gabion.analysis.dataflow.engine.dataflow_resume_paths import (
    normalize_snapshot_path as _normalize_snapshot_path,
)
from gabion.analysis.dataflow.io.dataflow_parse_helpers import (
    _ParseModuleFailure,
    _ParseModuleStage,
    _ParseModuleSuccess,
    _parse_module_tree as _parse_module_tree_outcome,
)
from gabion.analysis.core.visitors import ParentAnnotator
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.indexed_scan.calls.call_edges import (
    CollectCallEdgesDeps as _CollectCallEdgesDeps,
    collect_call_edges as _collect_call_edges_impl,
)
from gabion.analysis.indexed_scan.calls.callee_outcome_runtime import (
    CalleeOutcomeDeps as _CalleeOutcomeDeps,
    ResolveCalleeDeps as _ResolveCalleeDeps,
    resolve_callee as _resolve_callee_impl,
    resolve_callee_outcome as _resolve_callee_outcome_impl,
)
from gabion.analysis.indexed_scan.calls.call_nodes_by_path import (
    CallNodesForTreeDeps as _CallNodesForTreeDeps,
    CollectCallNodesByPathDeps as _CollectCallNodesByPathDeps,
    call_nodes_for_tree as _call_nodes_for_tree_impl,
    collect_call_nodes_by_path as _collect_call_nodes_by_path_impl,
)
from gabion.analysis.indexed_scan.deadline.deadline_local_info import (
    CollectDeadlineLocalInfoDeps as _CollectDeadlineLocalInfoDeps,
    collect_deadline_local_info as _collect_deadline_local_info_impl,
)
from gabion.analysis.indexed_scan.deadline.deadline_function_facts import (
    CollectDeadlineFunctionFactsDeps as _CollectDeadlineFunctionFactsDeps,
    collect_deadline_function_facts as _collect_deadline_function_facts_impl,
)
from gabion.analysis.indexed_scan.deadline.deadline_runtime import (
    DeadlineArgInfo as _DeadlineArgInfoRuntime,
    FunctionSuiteKey as _FunctionSuiteKeyRuntime,
    FunctionSuiteLookupOutcome as _FunctionSuiteLookupOutcomeRuntime,
    FunctionSuiteLookupStatus as _FunctionSuiteLookupStatusRuntime,
    bind_call_args as _bind_call_args_impl,
    call_candidate_target_site as _call_candidate_target_site_impl,
    caller_param_bindings_for_call as _caller_param_bindings_for_call_impl,
    classify_deadline_expr as _classify_deadline_expr_impl,
    collect_call_edges_from_forest as _collect_call_edges_from_forest_impl,
    collect_call_resolution_obligation_details_from_forest as _collect_call_resolution_obligation_details_from_forest_impl,
    collect_call_resolution_obligations_from_forest as _collect_call_resolution_obligations_from_forest_impl,
    deadline_arg_info_map as _deadline_arg_info_map_impl,
    deadline_loop_forwarded_params as _deadline_loop_forwarded_params_impl,
    fallback_deadline_arg_info as _fallback_deadline_arg_info_runtime_impl,
    function_suite_id as _function_suite_id_impl,
    function_suite_key as _function_suite_key_impl,
    is_deadline_origin_call as _is_deadline_origin_call,
    materialize_call_candidates as _materialize_call_candidates_impl,
    node_to_function_suite_id as _node_to_function_suite_id_impl,
    node_to_function_suite_lookup_outcome as _node_to_function_suite_lookup_outcome_impl,
    obligation_candidate_suite_ids as _obligation_candidate_suite_ids_impl,
    suite_caller_function_id as _suite_caller_function_id_impl,
)
from gabion.order_contract import sort_once


_StageCacheSpec = _StageCacheSpec_owner


_DeadlineFunctionCollector = make_deadline_function_collector(
    node_span_fn=_node_span,
    check_deadline_fn=check_deadline,
    deadline_loop_facts_ctor=_DeadlineLoopFacts,
)
_DeadlineArgInfo = _DeadlineArgInfoRuntime
_FunctionSuiteKey = _FunctionSuiteKeyRuntime
_FunctionSuiteLookupStatus = _FunctionSuiteLookupStatusRuntime
_FunctionSuiteLookupOutcome = _FunctionSuiteLookupOutcomeRuntime


def _is_dynamic_dispatch_callee_key(callee_key: str) -> bool:
    check_deadline()
    text = callee_key.strip()
    if not text:
        return False
    if text.startswith("getattr("):
        return True
    if "." not in text:
        return False
    base, _, _ = text.partition(".")
    base = base.strip()
    if not base or base in {"self", "cls"}:
        return False
    if any(token in base for token in ("(", "[", "{")):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", base) is None:
        return True
    return False


def _dedupe_resolution_candidates(
    candidates,
) -> tuple[FunctionInfo, ...]:
    deduped: dict[str, FunctionInfo] = {}
    for candidate in candidates:
        check_deadline()
        if _is_test_path(candidate.path):
            continue
        deduped[candidate.qual] = candidate
    return tuple(
        sort_once(
            deduped.values(),
            key=lambda info: info.qual,
            source="gabion.analysis.dataflow.engine.dataflow_deadline_runtime_owner._dedupe_resolution_candidates",
        )
    )


_COLLECT_CALL_EDGES_DEPS = _CollectCallEdgesDeps(
    check_deadline_fn=check_deadline,
    is_test_path_fn=_is_test_path,
)

_collect_call_edges_with_static_deps = partial(
    _collect_call_edges_impl,
    deps=_COLLECT_CALL_EDGES_DEPS,
)


def _collect_call_edges(
    *,
    by_name,
    by_qual,
    symbol_table,
    project_root,
    class_index,
    resolve_callee_outcome_fn=None,
):
    resolver = (
        _resolve_callee_outcome
        if resolve_callee_outcome_fn is None
        else resolve_callee_outcome_fn
    )
    return _collect_call_edges_with_static_deps(
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=symbol_table,
        project_root=project_root,
        class_index=class_index,
        resolve_callee_outcome_fn=resolver,
    )


_RESOLVE_CALLEE_DEPS = _ResolveCalleeDeps(
    check_deadline_fn=check_deadline,
    callee_resolution_context_core_ctor=_CalleeResolutionContextCore,
    resolve_callee_with_effects_fn=_resolve_callee_with_effects_impl,
    collect_callee_resolution_effects_fn=_collect_callee_resolution_effects_impl,
    module_name_fn=_module_name,
)

_resolve_callee = partial(
    _resolve_callee_impl,
    deps=_RESOLVE_CALLEE_DEPS,
)

_CALLEE_OUTCOME_DEPS = _CalleeOutcomeDeps(
    check_deadline_fn=check_deadline,
    callee_resolution_context_core_ctor=_CalleeResolutionContextCore,
    resolve_callee_with_effects_fn=_resolve_callee_with_effects_impl,
    collect_callee_resolution_effects_fn=_collect_callee_resolution_effects_impl,
    module_name_fn=_module_name,
    dedupe_resolution_candidates_fn=_dedupe_resolution_candidates,
    callee_key_fn=_callee_key,
    is_dynamic_dispatch_callee_key_fn=_is_dynamic_dispatch_callee_key,
    outcome_ctor=_CalleeResolutionOutcome,
    default_resolve_callee_fn=_resolve_callee,
)

_resolve_callee_outcome = partial(
    _resolve_callee_outcome_impl,
    deps=_CALLEE_OUTCOME_DEPS,
)


_COLLECT_DEADLINE_LOCAL_INFO_DEPS = _CollectDeadlineLocalInfoDeps(
    check_deadline_fn=check_deadline,
    is_deadline_origin_call_fn=_is_deadline_origin_call,
    target_names_fn=_target_names,
    deadline_local_info_ctor=_DeadlineLocalInfo,
)

_collect_deadline_local_info = partial(
    _collect_deadline_local_info_impl,
    deps=_COLLECT_DEADLINE_LOCAL_INFO_DEPS,
)


_CALL_NODES_FOR_TREE_DEPS = _CallNodesForTreeDeps(
    check_deadline_fn=check_deadline,
    node_span_fn=_node_span,
)

_call_nodes_for_tree = partial(
    _call_nodes_for_tree_impl,
    deps=_CALL_NODES_FOR_TREE_DEPS,
)


def _parse_module_tree_or_none(
    path,
    *,
    stage,
    parse_failure_witnesses,
):
    outcome = _parse_module_tree_outcome(
        path,
        stage=stage,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    match outcome:
        case _ParseModuleSuccess(kind="parsed", tree=tree):
            return tree
        case _ParseModuleFailure(kind="parse_failure"):
            return None


_COLLECT_CALL_NODES_BY_PATH_DEPS = _CollectCallNodesByPathDeps(
    check_deadline_fn=check_deadline,
    analysis_index_stage_cache_fn=_analysis_index_stage_cache,
    stage_cache_spec_ctor=_StageCacheSpec,
    parse_module_stage_call_nodes=_ParseModuleStage.CALL_NODES,
    parse_stage_cache_key_fn=_parse_stage_cache_key,
    empty_cache_semantic_context=_EMPTY_CACHE_SEMANTIC_CONTEXT,
    call_nodes_for_tree_fn=_call_nodes_for_tree,
    parse_module_tree_fn=_parse_module_tree_or_none,
)

_collect_call_nodes_by_path = partial(
    _collect_call_nodes_by_path_impl,
    deps=_COLLECT_CALL_NODES_BY_PATH_DEPS,
)


def _deadline_function_facts_for_tree(
    path: Path,
    tree,
    *,
    project_root,
    ignore_params,
):
    check_deadline()
    parents = ParentAnnotator()
    parents.visit(tree)
    module = _module_name(path, project_root)
    facts = {}
    for fn in _collect_functions(tree):
        check_deadline()
        scopes = _enclosing_scopes(fn, parents.parents)
        qual_parts = [module] if module else []
        if scopes:
            qual_parts.extend(scopes)
        qual_parts.append(fn.name)
        qual = ".".join(qual_parts)
        params = set(_param_names(fn, ignore_params))
        collector = _DeadlineFunctionCollector(fn, params)
        collector.visit(fn)
        local_info = _collect_deadline_local_info(collector.assignments, params)
        facts[qual] = _DeadlineFunctionFacts(
            path=path,
            qual=qual,
            span=_node_span(fn),
            loop=collector.loop,
            check_params=set(collector.check_params),
            ambient_check=collector.ambient_check,
            loop_sites=list(collector.loop_sites),
            local_info=local_info,
        )
    return facts


_COLLECT_DEADLINE_FUNCTION_FACTS_DEPS = _CollectDeadlineFunctionFactsDeps(
    check_deadline_fn=check_deadline,
    analysis_index_stage_cache_fn=_analysis_index_stage_cache,
    stage_cache_spec_ctor=_StageCacheSpec,
    parse_stage_cache_key_fn=_parse_stage_cache_key,
    deadline_function_facts_stage=_ParseModuleStage.DEADLINE_FUNCTION_FACTS,
    empty_cache_semantic_context=_EMPTY_CACHE_SEMANTIC_CONTEXT,
    sorted_text_fn=_sorted_text,
    deadline_function_facts_for_tree_fn=_deadline_function_facts_for_tree,
    parse_module_tree_fn=_parse_module_tree_or_none,
)

_collect_deadline_function_facts = partial(
    _collect_deadline_function_facts_impl,
    deps=_COLLECT_DEADLINE_FUNCTION_FACTS_DEPS,
)

_bind_call_args = _bind_call_args_impl

_caller_param_bindings_for_call = _caller_param_bindings_for_call_impl

_classify_deadline_expr = _classify_deadline_expr_impl

_fallback_deadline_arg_info = _fallback_deadline_arg_info_runtime_impl

_deadline_arg_info_map = _deadline_arg_info_map_impl

_deadline_loop_forwarded_params = _deadline_loop_forwarded_params_impl

_function_suite_key = _function_suite_key_impl

_function_suite_id = _function_suite_id_impl

_node_to_function_suite_lookup_outcome = _node_to_function_suite_lookup_outcome_impl

_suite_caller_function_id = _suite_caller_function_id_impl

_node_to_function_suite_id = _node_to_function_suite_id_impl

_obligation_candidate_suite_ids = _obligation_candidate_suite_ids_impl

_collect_call_edges_from_forest = _collect_call_edges_from_forest_impl

_collect_call_resolution_obligations_from_forest = (
    _collect_call_resolution_obligations_from_forest_impl
)

_collect_call_resolution_obligation_details_from_forest = (
    _collect_call_resolution_obligation_details_from_forest_impl
)

_call_candidate_target_site = _call_candidate_target_site_impl

_materialize_call_candidates_with_static_deps = partial(
    _materialize_call_candidates_impl,
    normalize_snapshot_path_fn=_normalize_snapshot_path,
)


def _materialize_call_candidates(
    *,
    forest,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table,
    project_root,
    class_index: dict[str, object],
    resolve_callee_outcome_fn=None,
) -> None:
    resolver = (
        _resolve_callee_outcome
        if resolve_callee_outcome_fn is None
        else resolve_callee_outcome_fn
    )
    _materialize_call_candidates_with_static_deps(
        forest=forest,
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=symbol_table,
        project_root=project_root,
        class_index=class_index,
        resolve_callee_outcome_fn=resolver,
    )


__all__ = [
    "_CalleeResolutionOutcome",
    "_DeadlineArgInfo",
    "_DeadlineFunctionCollector",
    "_DeadlineFunctionFacts",
    "_DeadlineLoopFacts",
    "_FunctionSuiteKey",
    "_FunctionSuiteLookupOutcome",
    "_FunctionSuiteLookupStatus",
    "_bind_call_args",
    "_build_analysis_index",
    "_call_candidate_target_site",
    "_call_nodes_for_tree",
    "_collect_call_edges_from_forest",
    "_collect_call_resolution_obligation_details_from_forest",
    "_collect_call_resolution_obligations_from_forest",
    "_caller_param_bindings_for_call",
    "_classify_deadline_expr",
    "_collect_call_edges",
    "_collect_call_nodes_by_path",
    "_collect_deadline_function_facts",
    "_collect_deadline_local_info",
    "_collect_recursive_functions",
    "_collect_recursive_nodes",
    "_deadline_arg_info_map",
    "_deadline_function_facts_for_tree",
    "_deadline_loop_forwarded_params",
    "_fallback_deadline_arg_info",
    "_function_suite_id",
    "_function_suite_key",
    "_materialize_call_candidates",
    "_node_to_function_suite_id",
    "_node_to_function_suite_lookup_outcome",
    "_normalize_snapshot_path",
    "_obligation_candidate_suite_ids",
    "_reachable_from_roots",
    "_resolve_callee",
    "_resolve_callee_outcome",
    "_suite_caller_function_id",
]
