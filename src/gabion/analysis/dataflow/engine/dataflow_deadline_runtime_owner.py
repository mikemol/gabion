# gabion:boundary_normalization_module
from __future__ import annotations

"""Deadline-runtime compatibility owner during WS-5 migration."""

from dataclasses import dataclass
from pathlib import Path

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
from gabion.analysis.dataflow.engine.dataflow_facade import (
    _DeadlineFunctionCollector,
    _resolve_callee_outcome,
)
from gabion.analysis.foundation.json_types import JSONObject
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
    is_deadline_origin_call as _is_deadline_origin_call,
)


@dataclass(frozen=True)
class _StageCacheSpec:
    stage: object
    cache_key: object
    build: object


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


def _collect_deadline_local_info(
    assignments,
    params,
):
    return _collect_deadline_local_info_impl(
        assignments,
        params,
        deps=_CollectDeadlineLocalInfoDeps(
            check_deadline_fn=check_deadline,
            is_deadline_origin_call_fn=_is_deadline_origin_call,
            target_names_fn=_target_names,
            deadline_local_info_ctor=_DeadlineLocalInfo,
        ),
    )


def _call_nodes_for_tree(tree):
    return _call_nodes_for_tree_impl(
        tree,
        deps=_CallNodesForTreeDeps(
            check_deadline_fn=check_deadline,
            node_span_fn=_node_span,
        ),
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


def _collect_call_nodes_by_path(
    paths,
    *,
    trees=None,
    parse_failure_witnesses,
    analysis_index=None,
):
    return _collect_call_nodes_by_path_impl(
        paths,
        trees=trees,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        deps=_CollectCallNodesByPathDeps(
            check_deadline_fn=check_deadline,
            analysis_index_stage_cache_fn=_analysis_index_stage_cache,
            stage_cache_spec_ctor=_StageCacheSpec,
            parse_module_stage_call_nodes=_ParseModuleStage.CALL_NODES,
            parse_stage_cache_key_fn=_parse_stage_cache_key,
            empty_cache_semantic_context=_EMPTY_CACHE_SEMANTIC_CONTEXT,
            call_nodes_for_tree_fn=_call_nodes_for_tree,
            parse_module_tree_fn=_parse_module_tree_or_none,
        ),
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


def _collect_deadline_function_facts(
    paths,
    *,
    project_root=None,
    ignore_params,
    parse_failure_witnesses: list[JSONObject],
    trees=None,
    analysis_index=None,
    stage_cache_fn=None,
):
    return _collect_deadline_function_facts_impl(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        parse_failure_witnesses=parse_failure_witnesses,
        trees=trees,
        analysis_index=analysis_index,
        stage_cache_fn=stage_cache_fn,
        deps=_CollectDeadlineFunctionFactsDeps(
            check_deadline_fn=check_deadline,
            analysis_index_stage_cache_fn=_analysis_index_stage_cache,
            stage_cache_spec_ctor=_StageCacheSpec,
            parse_stage_cache_key_fn=_parse_stage_cache_key,
            deadline_function_facts_stage=_ParseModuleStage.DEADLINE_FUNCTION_FACTS,
            empty_cache_semantic_context=_EMPTY_CACHE_SEMANTIC_CONTEXT,
            sorted_text_fn=_sorted_text,
            deadline_function_facts_for_tree_fn=_deadline_function_facts_for_tree,
            parse_module_tree_fn=_parse_module_tree_or_none,
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
