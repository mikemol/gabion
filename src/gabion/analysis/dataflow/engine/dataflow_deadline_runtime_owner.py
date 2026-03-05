# gabion:boundary_normalization_module
from __future__ import annotations

"""Deadline-runtime compatibility owner during WS-5 migration."""

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import cast

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
    _resolve_callee,
    _target_names,
)
from gabion.analysis.dataflow.engine.dataflow_callee_resolution import (
    CalleeResolutionContext as _CalleeResolutionContextCore,
    collect_callee_resolution_effects as _collect_callee_resolution_effects_impl,
    resolve_callee_with_effects as _resolve_callee_with_effects_impl,
)
from gabion.analysis.dataflow.engine.dataflow_callee_resolution_support import _callee_key
from gabion.analysis.dataflow.engine.dataflow_contracts import FunctionInfo
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
from gabion.analysis.indexed_scan.calls.callee_outcome_runtime import (
    CalleeOutcomeDeps as _CalleeOutcomeDeps,
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
    is_deadline_origin_call as _is_deadline_origin_call,
)
from gabion.order_contract import sort_once


@dataclass(frozen=True)
class _StageCacheSpec:
    stage: object
    cache_key: object
    build: object


_DEADLINE_CHECK_METHODS = {"check", "expired"}


class _DeadlineFunctionCollector(ast.NodeVisitor):
    def __init__(self, root: ast.AST, params: set[str]) -> None:
        self._root = root
        self._params = params
        self.loop = False
        self.check_params: set[str] = set()
        self.ambient_check = False
        self.loop_sites: list[_DeadlineLoopFacts] = []
        self._loop_stack: list[_DeadlineLoopFacts] = []
        self.assignments: list[tuple[list[ast.AST], object, object]] = []

    def _mark_param_check(self, name: str) -> None:
        if self._loop_stack:
            self._loop_stack[-1].check_params.add(name)
        else:
            self.check_params.add(name)

    def _mark_ambient_check(self) -> None:
        if self._loop_stack:
            self._loop_stack[-1].ambient_check = True
        else:
            self.ambient_check = True

    def _record_call_span(self, node: ast.AST) -> None:
        if self._loop_stack:
            span = _node_span(node)
            if span is not None:
                self._loop_stack[-1].call_spans.add(span)

    def _iter_marks_ambient(self, expr: ast.AST) -> bool:
        if type(expr) is ast.Call:
            func = cast(ast.Call, expr).func
            if type(func) is ast.Name:
                return cast(ast.Name, func).id == "deadline_loop_iter"
            if type(func) is ast.Attribute:
                return cast(ast.Attribute, func).attr == "deadline_loop_iter"
        return False

    def _visit_loop_body(
        self,
        node: ast.AST,
        kind: str,
        *,
        ambient_check: bool = False,
    ) -> None:
        self.loop = True
        loop_fact = _DeadlineLoopFacts(
            span=_node_span(node),
            kind=kind,
            depth=len(self._loop_stack) + 1,
            ambient_check=ambient_check,
        )
        self._loop_stack.append(loop_fact)
        for stmt in getattr(node, "body", []):
            check_deadline()
            self.visit(stmt)
        self._loop_stack.pop()
        self.loop_sites.append(loop_fact)
        for stmt in getattr(node, "orelse", []):
            check_deadline()
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is not self._root:
            return
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is not self._root:
            return
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node
        return

    def visit_For(self, node: ast.For) -> None:
        self.loop = True
        ambient_check = self._iter_marks_ambient(node.iter)
        self.visit(node.target)
        self.visit(node.iter)
        self._visit_loop_body(node, "for", ambient_check=ambient_check)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.loop = True
        ambient_check = self._iter_marks_ambient(node.iter)
        self.visit(node.target)
        self.visit(node.iter)
        self._visit_loop_body(node, "async_for", ambient_check=ambient_check)

    def visit_While(self, node: ast.While) -> None:
        self.loop = True
        self.visit(node.test)
        self._visit_loop_body(node, "while")

    def visit_Call(self, node: ast.Call) -> None:
        self._record_call_span(node)
        func = node.func
        func_type = type(func)
        if func_type is ast.Attribute:
            attribute_func = cast(ast.Attribute, func)
            if attribute_func.attr == "deadline_loop_iter":
                self._mark_ambient_check()
            if (
                attribute_func.attr in _DEADLINE_CHECK_METHODS
                and type(attribute_func.value) is ast.Name
                and cast(ast.Name, attribute_func.value).id in self._params
            ):
                self._mark_param_check(cast(ast.Name, attribute_func.value).id)
            if attribute_func.attr == "check_deadline" and node.args:
                first = node.args[0]
                if type(first) is ast.Name and cast(ast.Name, first).id in self._params:
                    self._mark_param_check(cast(ast.Name, first).id)
            if attribute_func.attr in {"check_deadline", "require_deadline"} and not node.args:
                self._mark_ambient_check()
        elif func_type is ast.Name:
            name_func = cast(ast.Name, func)
            if name_func.id == "deadline_loop_iter":
                self._mark_ambient_check()
            if name_func.id == "check_deadline" and node.args:
                first = node.args[0]
                if type(first) is ast.Name and cast(ast.Name, first).id in self._params:
                    self._mark_param_check(cast(ast.Name, first).id)
            if name_func.id in {"check_deadline", "require_deadline"} and not node.args:
                self._mark_ambient_check()
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.assignments.append((node.targets, node.value, _node_span(node)))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.assignments.append(([node.target], node.value, _node_span(node)))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.assignments.append(([node.target], node.value, _node_span(node)))
        self.generic_visit(node)


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
    local_lambda_bindings=None,
    resolve_callee_fn=_resolve_callee,
) -> _CalleeResolutionOutcome:
    return _resolve_callee_outcome_impl(
        callee_key,
        caller,
        by_name,
        by_qual,
        symbol_table=symbol_table,
        project_root=project_root,
        class_index=class_index,
        call=call,
        local_lambda_bindings=local_lambda_bindings,
        resolve_callee_fn=resolve_callee_fn,
        deps=_CalleeOutcomeDeps(
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
