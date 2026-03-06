# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Post-phase analysis owner module for WS-5 decomposition."""

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence, cast

from gabion.analysis.dataflow.engine.dataflow_contracts import (
    CallArgs,
    ClassInfo,
    FunctionInfo,
    InvariantProposition,
    SymbolTable,
)
from gabion.analysis.dataflow.engine.dataflow_analysis_index import (
    _EMPTY_CACHE_SEMANTIC_CONTEXT,
    _IndexedPassContext,
    _IndexedPassSpec,
    _build_call_graph,
    _analysis_index_stage_cache,
    _analysis_index_resolved_call_edges,
    _analysis_index_resolved_call_edges_by_caller,
    _iter_resolved_edge_param_events,
    _parse_stage_cache_key,
    _reduce_resolved_call_edges,
    _run_indexed_pass,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_helpers import (
    _collect_functions,
    _enclosing_scopes,
    _is_test_path,
    _module_name,
    _node_span,
    _param_names,
    _param_annotations,
)
from gabion.analysis.dataflow.io.dataflow_parse_helpers import (
    _ParseModuleStage,
    _ParseModuleSuccess,
    _forbid_adhoc_bundle_discovery,
    _parse_module_tree,
    _parse_module_tree_or_none,
)
from gabion.analysis.dataflow.engine.dataflow_lint_helpers import (
    _constant_smells_from_details as _constant_smells_from_details_impl,
    _deadness_witnesses_from_constant_details as _deadness_witnesses_from_constant_details_impl,
    _expand_type_hint as _expand_type_hint_impl,
    _split_top_level as _split_top_level_impl,
)
from gabion.analysis.dataflow.engine.dataflow_exception_obligations import (
    _builtin_exception_class as _exc_builtin_exception_class,
    exception_handler_compatibility as _exc_exception_handler_compatibility,
    exception_param_names as _exc_exception_param_names,
    exception_type_name as _exc_exception_type_name,
    handler_label as _exc_handler_label,
    handler_type_names as _exc_handler_type_names,
    node_in_try_body as _exc_node_in_try_body,
)
from gabion.analysis.dataflow.engine.dataflow_resume_paths import (
    normalize_snapshot_path as _normalize_snapshot_path_impl,
)
from gabion.analysis.dataflow.engine.dataflow_resume_serialization import (
    _deserialize_invariants_for_resume,
    _invariant_confidence,
    _invariant_digest,
    _normalize_invariant_proposition,
)
from gabion.analysis.foundation.json_types import JSONObject, JSONValue
from gabion.analysis.foundation.resume_codec import (
    int_tuple4_or_none,
    mapping_or_none,
    sequence_or_none,
)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.core.visitors import ParentAnnotator
from gabion.analysis.dataflow.io.dataflow_reporting_helpers import _format_span_fields
from gabion.analysis.indexed_scan.calls.callsite_evidence import (
    CallsiteEvidenceDeps as _CallsiteEvidenceDeps,
    callsite_evidence_for_bundle as _callsite_evidence_for_bundle_impl,
)
from gabion.analysis.indexed_scan.calls.callee_resolution_helpers import (
    decorator_name as _decorator_name,
)
from gabion.analysis.indexed_scan.ast.expression_eval import (
    branch_reachability_under_env as _branch_reachability_under_env_impl,
    eval_bool_expr as _eval_bool_expr_impl,
    eval_value_expr as _eval_value_expr_impl,
    is_reachability_false as _is_reachability_false_impl,
    is_reachability_true as _is_reachability_true_impl,
)
from gabion.analysis.indexed_scan.deadline.deadline_runtime import (
    caller_param_bindings_for_call as _caller_param_bindings_for_call_impl,
)
from gabion.analysis.indexed_scan.scanners.flow.type_flow import (
    TypeFlowInferDeps as _TypeFlowInferDeps,
    infer_type_flow as _infer_type_flow_impl,
)
from gabion.analysis.indexed_scan.scanners.flow.unused_arg_flow import (
    analyze_unused_arg_flow_indexed as _analyze_unused_arg_flow_indexed_impl,
)
from gabion.analysis.indexed_scan.scanners.flow.constant_flow_details import (
    CollectConstantFlowDetailsDeps as _CollectConstantFlowDetailsDeps,
    collect_constant_flow_details as _collect_constant_flow_details_impl,
)
from gabion.analysis.indexed_scan.index.analysis_index_stage_cache import (
    AnalysisIndexStageCacheFn,
)
from gabion.analysis.indexed_scan.obligations.exception_obligations import (
    collect_exception_obligations as _collect_exception_obligations_impl,
    dead_env_map as _dead_env_map_impl,
)
from gabion.analysis.indexed_scan.obligations.handledness import (
    collect_handledness_witnesses as _collect_handledness_witnesses_impl,
)
from gabion.analysis.indexed_scan.obligations.invariant_propositions import (
    CollectInvariantPropositionsDeps as _CollectInvariantPropositionsDeps,
    collect_invariant_propositions as _collect_invariant_propositions_impl,
)
from gabion.analysis.indexed_scan.obligations.never_invariants import (
    collect_never_invariants as _collect_never_invariants_impl,
    keyword_links_literal as _keyword_links_literal_impl,
    keyword_string_literal as _keyword_string_literal_impl,
    never_reason as _never_reason_impl,
)
from gabion.analysis.indexed_scan.obligations.decision_surface_runtime import (
    DecisionSurfaceAnalyzeDeps as _DecisionSurfaceAnalyzeDeps,
    analyze_decision_surface_indexed as _analyze_decision_surface_indexed_impl,
)
from gabion.analysis.indexed_scan.scanners.config_fields import (
    CollectConfigBundlesDeps as _CollectConfigBundlesDeps,
    IterConfigFieldsDeps as _IterConfigFieldsDeps,
    collect_config_bundles as _collect_config_bundles_impl,
    iter_config_fields as _iter_config_fields_impl,
)
from gabion.analysis.indexed_scan.scanners.knob_param_names import (
    ComputeKnobParamNamesDeps as _ComputeKnobParamNamesDeps,
    KnobFlowFoldAccumulator as _KnobFlowFoldAccumulator,
    compute_knob_param_names as _compute_knob_param_names_impl,
)
from gabion.analysis.indexed_scan.scanners.materialization.dataclass_registry import (
    CollectDataclassRegistryDeps as _CollectDataclassRegistryDeps,
    DataclassRegistryForTreeDeps as _DataclassRegistryForTreeDeps,
    collect_dataclass_registry as _collect_dataclass_registry_impl,
    dataclass_registry_for_tree as _dataclass_registry_for_tree_impl,
)
from gabion.analysis.indexed_scan.scanners.materialization.property_hook_manifest import (
    PropertyHookCallableIndexDeps as _PropertyHookCallableIndexDeps,
    PropertyHookManifestDeps as _PropertyHookManifestDeps,
    build_property_hook_callable_index as _build_property_hook_callable_index_impl,
    generate_property_hook_manifest as _generate_property_hook_manifest_impl,
)
from gabion.analysis.dataflow.engine.dataflow_bundle_iteration import (
    iter_dataclass_call_bundle_effects as _iter_dataclass_call_bundle_effects_impl,
)
from gabion.analysis.semantics.semantic_primitives import (
    AnalysisPassPrerequisites,
    DecisionPredicateEvidence,
    ParameterId,
    SpanIdentity,
)
from gabion.invariants import never, require_not_none
from gabion.order_contract import OrderPolicy, sort_once

# Temporary boundary adapters for unmoved post-phase owners.
_BOUNDARY_ADAPTER_LIFECYCLE: dict[str, object] = {
    "actor": "codex",
    "rationale": "WS-5 staged post-phase hard-cut; keep import surface stable while migrating in slices",
    "scope": "dataflow_post_phase_analyses.runtime_delegates",
    "start": "2026-03-04",
    "expiry": "WS-5-D completion",
    "rollback_condition": "post-phase owner extraction complete in canonical module",
    "evidence_links": ["docs/ws5_decomposition_ledger.md"],
}

_LITERAL_EVAL_ERROR_TYPES = (
    SyntaxError,
    ValueError,
    TypeError,
    MemoryError,
    RecursionError,
)

_NONE_TYPES = {"None", "NoneType", "type(None)"}


def _parse_module_source(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _simple_store_name(target: ast.AST):
    if type(target) is ast.Name:
        return cast(ast.Name, target).id
    return None


def _decorator_matches(name: str, allowlist: set[str]) -> bool:
    if name in allowlist:
        return True
    if "." in name and name.split(".")[-1] in allowlist:
        return True
    return False


def _decorator_name_local(node: ast.AST):
    return _decorator_name(node, check_deadline_fn=check_deadline)


def _is_marker_call(call: ast.Call, aliases: set[str]) -> bool:
    name = _decorator_name_local(call.func)
    if not name:
        return False
    return _decorator_matches(name, aliases)


def _is_never_marker_raise(
    function: str,
    exception_name,
    never_exceptions: set[str],
) -> bool:
    if not exception_name or not never_exceptions:
        return False
    if not _decorator_matches(exception_name, never_exceptions):
        return False
    return function == "never" or function.endswith(".never")


def _function_key(scope, name: str) -> str:
    parts = list(scope)
    parts.append(name)
    return ".".join(parts)


def _invariant_term(expr: ast.AST, params: set[str]):
    expr_type = type(expr)
    if expr_type is ast.Name:
        name_expr = cast(ast.Name, expr)
        return next(iter(params.intersection({name_expr.id})), None)
    if expr_type is ast.Call:
        call_expr = cast(ast.Call, expr)
        if type(call_expr.func) is ast.Name:
            func_name = cast(ast.Name, call_expr.func)
            if func_name.id == "len" and len(call_expr.args) == 1:
                arg = call_expr.args[0]
                if type(arg) is ast.Name:
                    arg_id = cast(ast.Name, arg).id
                    return next((f"{entry}.length" for entry in params.intersection({arg_id})), None)
    return None


def _extract_invariant_from_expr(
    expr: ast.AST,
    params: set[str],
    *,
    scope: str,
    source: str = "assert",
) -> object:
    if type(expr) is not ast.Compare:
        return None
    compare_expr = cast(ast.Compare, expr)
    if len(compare_expr.ops) != 1 or len(compare_expr.comparators) != 1:
        return None
    if type(compare_expr.ops[0]) is not ast.Eq:
        return None
    left = _invariant_term(compare_expr.left, params)
    right = _invariant_term(compare_expr.comparators[0], params)
    if left is not None and right is not None:
        return InvariantProposition(
            form="Equal",
            terms=(left, right),
            scope=scope,
            source=source,
        )
    return None


class _InvariantCollector(ast.NodeVisitor):
    # dataflow-bundle: params, scope
    def __init__(self, params: set[str], scope: str) -> None:
        self._params = params
        self._scope = scope
        self.propositions: list[InvariantProposition] = []
        self._seen: set[tuple[str, tuple[str, ...], str]] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Assert(self, node: ast.Assert) -> None:
        prop = _extract_invariant_from_expr(
            node.test,
            self._params,
            scope=self._scope,
        )
        if prop is not None:
            normalized = _normalize_invariant_proposition(
                prop,
                default_scope=self._scope,
                default_source="assert",
            )
            key = (normalized.form, normalized.terms, normalized.scope or "")
            if key not in self._seen:
                self._seen.add(key)
                self.propositions.append(normalized)
        self.generic_visit(node)


def _scope_path(path: Path, root) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def _enclosing_function_node(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
):
    check_deadline()
    current = parents.get(node)
    while current is not None:
        check_deadline()
        current_type = type(current)
        if current_type is ast.FunctionDef or current_type is ast.AsyncFunctionDef:
            return cast(ast.FunctionDef | ast.AsyncFunctionDef, current)
        current = parents.get(current)
    return None


def _node_in_block(node: ast.AST, block: list[ast.stmt]) -> bool:
    check_deadline()
    for stmt in block:
        check_deadline()
        if node is stmt:
            return True
        for child in ast.walk(stmt):
            check_deadline()
            if node is child:
                return True
    return False


def _names_in_expr(expr: ast.AST) -> set[str]:
    check_deadline()
    names: set[str] = set()
    for node in ast.walk(expr):
        check_deadline()
        if type(node) is ast.Name:
            names.add(cast(ast.Name, node).id)
    return names


def _eval_value_expr(expr: ast.AST, env: dict[str, JSONValue]):
    return _eval_value_expr_impl(
        expr,
        env,
        check_deadline_fn=check_deadline,
    )


def _eval_bool_expr(expr: ast.AST, env: dict[str, JSONValue]):
    return _eval_bool_expr_impl(
        expr,
        env,
        check_deadline_fn=check_deadline,
    )


def _branch_reachability_under_env(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    env: dict[str, JSONValue],
):
    return _branch_reachability_under_env_impl(
        node,
        parents,
        env,
        check_deadline_fn=check_deadline,
        node_in_block_fn=_node_in_block,
    )


def _is_reachability_false(reachability) -> bool:
    return _is_reachability_false_impl(reachability)


def _is_reachability_true(reachability) -> bool:
    return _is_reachability_true_impl(reachability)


def _dead_env_map(
    deadness_witnesses,
) -> dict[tuple[str, str], dict[str, tuple[JSONValue, JSONObject]]]:
    return _dead_env_map_impl(
        deadness_witnesses,
        check_deadline_fn=check_deadline,
        sequence_or_none_fn=sequence_or_none,
        mapping_or_none_fn=mapping_or_none,
        literal_eval_error_types=_LITERAL_EVAL_ERROR_TYPES,
    )


def _exception_param_names(expr, params: set[str]) -> list[str]:
    return _exc_exception_param_names(expr, params, check_deadline=check_deadline)


def _exception_type_name(expr):
    return _exc_exception_type_name(expr, decorator_name=_decorator_name_local)


def _annotation_exception_candidates(annotation) -> tuple[str, ...]:
    check_deadline()
    if not annotation:
        return ()
    try:
        expr = ast.parse(annotation, mode="eval").body
    except SyntaxError:
        return ()
    candidates: set[str] = set()
    for node in ast.walk(expr):
        check_deadline()
        node_type = type(node)
        if node_type is ast.Name:
            node_name = cast(ast.Name, node)
            cls = _exc_builtin_exception_class(node_name.id)
            if cls is not None:
                candidates.add(node_name.id)
        elif node_type is ast.Attribute:
            node_attr = cast(ast.Attribute, node)
            cls = _exc_builtin_exception_class(node_attr.attr)
            if cls is not None:
                candidates.add(node_attr.attr)
    return tuple(
        sort_once(
            candidates,
            source="_annotation_exception_candidates.candidates",
            policy=OrderPolicy.SORT,
        )
    )


def _refine_exception_name_from_annotations(
    expr,
    *,
    param_annotations: object,
):
    check_deadline()
    direct_name = _exception_type_name(expr)
    if type(expr) is not ast.Name:
        return direct_name, None, ()
    annotations = cast(dict[str, str | None], param_annotations)
    annotation = annotations.get(cast(ast.Name, expr).id)
    candidates = _annotation_exception_candidates(annotation)
    if not candidates:
        return direct_name, None, ()
    if len(candidates) == 1:
        return candidates[0], "PARAM_ANNOTATION", candidates
    return direct_name, "PARAM_ANNOTATION_AMBIGUOUS", candidates


def _handler_type_names(handler_type) -> tuple[str, ...]:
    return _exc_handler_type_names(
        handler_type,
        decorator_name=_decorator_name_local,
        check_deadline=check_deadline,
    )


def _exception_handler_compatibility(
    exception_name,
    handler_type,
) -> str:
    return _exc_exception_handler_compatibility(
        exception_name,
        handler_type,
        decorator_name=_decorator_name_local,
        check_deadline=check_deadline,
    )


def _exception_path_id(
    *,
    path: str,
    function: str,
    source_kind: str,
    lineno: int,
    col: int,
    kind: str,
) -> str:
    return f"{path}:{function}:{source_kind}:{lineno}:{col}:{kind}"


def _handler_label(handler: ast.ExceptHandler) -> str:
    return _exc_handler_label(handler)


def _node_in_try_body(node: ast.AST, try_node: ast.Try) -> bool:
    return _exc_node_in_try_body(node, try_node, check_deadline=check_deadline)


def _find_handling_try(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
):
    check_deadline()
    current = parents.get(node)
    try_ancestors: list[ast.Try] = []
    while current is not None:
        check_deadline()
        if type(current) is ast.Try:
            try_ancestors.append(cast(ast.Try, current))
        current = parents.get(current)
    return next(
        (try_node for try_node in try_ancestors if _node_in_try_body(node, try_node)),
        None,
    )


def _keyword_string_literal(call: ast.Call, key: str) -> str:
    return _keyword_string_literal_impl(
        call,
        key,
        check_deadline_fn=check_deadline,
    )


def _keyword_links_literal(call: ast.Call) -> list[JSONObject]:
    return _keyword_links_literal_impl(
        call,
        check_deadline_fn=check_deadline,
        sort_once_fn=sort_once,
    )


def _never_reason(call: ast.Call):
    return _never_reason_impl(call, check_deadline_fn=check_deadline)


def _normalize_snapshot_path(path: Path, root) -> str:
    return _normalize_snapshot_path_impl(path, root)


def _type_from_const_repr(value: str):
    try:
        literal = ast.literal_eval(value)
    except _LITERAL_EVAL_ERROR_TYPES:
        return None
    literal_type = type(literal)
    if literal is None:
        return "None"
    if literal_type is bool:
        return "bool"
    if literal_type is int:
        return "int"
    if literal_type is float:
        return "float"
    if literal_type is complex:
        return "complex"
    if literal_type is str:
        return "str"
    if literal_type is bytes:
        return "bytes"
    if literal_type is list:
        return "list"
    if literal_type is tuple:
        return "tuple"
    if literal_type is set:
        return "set"
    if literal_type is dict:
        return "dict"
    return None


def _is_broad_type(annot) -> bool:
    if annot is None:
        return True
    base = annot.replace("typing.", "")
    return base in {"Any", "object"}


def _split_top_level(value: str, sep: str) -> list[str]:
    return _split_top_level_impl(value, sep)


def _expand_type_hint(hint: str) -> set[str]:
    return _expand_type_hint_impl(hint)


def _combine_type_hints(types: set[str]) -> tuple[str, bool]:
    check_deadline()
    normalized_sets = []
    for hint in types:
        check_deadline()
        expanded = _expand_type_hint(hint)
        normalized_sets.append(
            tuple(
                sort_once(
                    (t for t in expanded if t not in _NONE_TYPES),
                    source="gabion.analysis.dataflow_post_phase_analyses._combine_type_hints.site_1",
                )
            )
        )
    unique_normalized = {norm for norm in normalized_sets if norm}
    expanded: set[str] = set()
    for hint in types:
        check_deadline()
        expanded.update(_expand_type_hint(hint))
    none_types = {t for t in expanded if t in _NONE_TYPES}
    expanded -= none_types
    if not expanded:
        return "Any", bool(types)
    sorted_types = sort_once(
        expanded,
        source="gabion.analysis.dataflow_post_phase_analyses._combine_type_hints.site_2",
    )
    if len(sorted_types) == 1:
        base = sorted_types[0]
        if none_types:
            conflicted = len(unique_normalized) > 1
            return f"Optional[{base}]", conflicted
        return base, len(unique_normalized) > 1
    union = f"Union[{', '.join(sorted_types)}]"
    if none_types:
        return f"Optional[{union}]", len(unique_normalized) > 1
    return union, len(unique_normalized) > 1


def _format_call_site(caller: FunctionInfo, call: CallArgs) -> str:
    """Render a stable, human-friendly call site identifier."""
    caller_name = _function_key(caller.scope, caller.name)
    span = call.span
    if span is None:
        return f"{caller.path.name}:{caller_name}"
    line, col, _, _ = span
    return f"{caller.path.name}:{line + 1}:{col + 1}:{caller_name}"


def _format_type_flow_site(
    *,
    caller: FunctionInfo,
    call: CallArgs,
    callee: FunctionInfo,
    caller_param: str,
    callee_param: str,
    annot: str,
    project_root,
) -> str:
    """Format a stable, machine-actionable callsite for type-flow evidence."""
    caller_name = _function_key(caller.scope, caller.name)
    caller_path = _normalize_snapshot_path(caller.path, project_root)
    if call.span is None:
        loc = f"{caller_path}:{caller_name}"
    else:
        line, col, _, _ = call.span
        loc = f"{caller_path}:{line + 1}:{col + 1}"
    return (
        f"{loc}: {caller_name}.{caller_param} -> {callee.qual}.{callee_param} expects {annot}"
    )


def _callsite_evidence_for_bundle(
    calls: list[CallArgs],
    bundle: set[str],
    *,
    limit: int = 12,
) -> list[JSONObject]:
    return cast(
        list[JSONObject],
        _callsite_evidence_for_bundle_impl(
            calls,
            bundle,
            limit=limit,
            deps=_CallsiteEvidenceDeps(
                check_deadline_fn=check_deadline,
                sort_once_fn=sort_once,
                require_not_none_fn=require_not_none,
                span_identity_from_tuple_fn=SpanIdentity.from_tuple,
            ),
        ),
    )


def generate_property_hook_manifest(
    invariants: Sequence[InvariantProposition],
    *,
    min_confidence: float = 0.7,
    emit_hypothesis_templates: bool = False,
) -> JSONObject:
    return _generate_property_hook_manifest_impl(
        invariants,
        min_confidence=min_confidence,
        emit_hypothesis_templates=emit_hypothesis_templates,
        deps=_PropertyHookManifestDeps(
            check_deadline_fn=check_deadline,
            sort_once_fn=sort_once,
            invariant_confidence_fn=_invariant_confidence,
            normalize_invariant_proposition_fn=_normalize_invariant_proposition,
            invariant_digest_fn=_invariant_digest,
        ),
    )


def _build_property_hook_callable_index(hooks: Sequence[JSONValue]) -> list[JSONObject]:
    return _build_property_hook_callable_index_impl(
        hooks,
        deps=_PropertyHookCallableIndexDeps(
            check_deadline_fn=check_deadline,
            sort_once_fn=sort_once,
        ),
    )


def analyze_type_flow_repo_with_map(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    parse_failure_witnesses=None,
    analysis_index=None,
):
    check_deadline()
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="type_flow_with_map",
            run=lambda context: _infer_type_flow(
                context.paths,
                project_root=context.project_root,
                ignore_params=context.ignore_params,
                strictness=context.strictness,
                external_filter=context.external_filter,
                transparent_decorators=context.transparent_decorators,
                parse_failure_witnesses=context.parse_failure_witnesses,
                analysis_index=context.analysis_index,
            )[:3],
        ),
    )


def analyze_type_flow_repo(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    parse_failure_witnesses=None,
    analysis_index=None,
) -> tuple[list[str], list[str]]:
    _inferred, suggestions, ambiguities = analyze_type_flow_repo_with_map(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
    )
    return suggestions, ambiguities


def analyze_type_flow_repo_with_evidence(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    max_sites_per_param: int = 3,
    parse_failure_witnesses=None,
    analysis_index=None,
) -> tuple[list[str], list[str], list[str]]:
    check_deadline()
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="type_flow_with_evidence",
            run=lambda context: _infer_type_flow(
                context.paths,
                project_root=context.project_root,
                ignore_params=context.ignore_params,
                strictness=context.strictness,
                external_filter=context.external_filter,
                transparent_decorators=context.transparent_decorators,
                max_sites_per_param=max_sites_per_param,
                parse_failure_witnesses=context.parse_failure_witnesses,
                analysis_index=context.analysis_index,
            )[1:],
        ),
    )


def analyze_constant_flow_repo(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    parse_failure_witnesses=None,
    analysis_index=None,
) -> list[str]:
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="constant_flow",
            run=lambda context: _constant_smells_from_details_impl(
                _collect_constant_flow_details(
                    context.paths,
                    project_root=context.project_root,
                    ignore_params=context.ignore_params,
                    strictness=context.strictness,
                    external_filter=context.external_filter,
                    transparent_decorators=context.transparent_decorators,
                    parse_failure_witnesses=context.parse_failure_witnesses,
                    analysis_index=context.analysis_index,
                )
            ),
        ),
    )


def analyze_deadness_flow_repo(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    parse_failure_witnesses=None,
    analysis_index=None,
) -> list[JSONObject]:
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="deadness_flow",
            run=lambda context: _deadness_witnesses_from_constant_details_impl(
                _collect_constant_flow_details(
                    context.paths,
                    project_root=context.project_root,
                    ignore_params=context.ignore_params,
                    strictness=context.strictness,
                    external_filter=context.external_filter,
                    transparent_decorators=context.transparent_decorators,
                    parse_failure_witnesses=context.parse_failure_witnesses,
                    analysis_index=context.analysis_index,
                ),
                project_root=context.project_root,
            ),
        ),
    )


def analyze_unused_arg_flow_repo(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    parse_failure_witnesses=None,
    analysis_index=None,
) -> list[str]:
    check_deadline()
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="unused_arg_flow",
            run=_analyze_unused_arg_flow_indexed,
        ),
    )


def _infer_type_flow(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    max_sites_per_param: int = 3,
    parse_failure_witnesses: list[JSONObject],
    analysis_index=None,
):
    return _infer_type_flow_impl(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        max_sites_per_param=max_sites_per_param,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        deps=_TypeFlowInferDeps(
            check_deadline_fn=check_deadline,
            analysis_pass_prerequisites_ctor=AnalysisPassPrerequisites,
            require_not_none_fn=require_not_none,
            analysis_index_resolved_call_edges_by_caller_fn=_analysis_index_resolved_call_edges_by_caller,
            caller_param_bindings_for_call_fn=_caller_param_bindings_for_call_impl,
            function_key_fn=_function_key,
            normalize_snapshot_path_fn=_normalize_snapshot_path,
            is_test_path_fn=_is_test_path,
            is_broad_type_fn=_is_broad_type,
            sort_once_fn=sort_once,
        ),
    )


def _analyze_unused_arg_flow_indexed(
    context: _IndexedPassContext,
) -> list[str]:
    return _analyze_unused_arg_flow_indexed_impl(
        context,
        analysis_index_resolved_call_edges_fn=_analysis_index_resolved_call_edges,
        check_deadline_fn=check_deadline,
        sort_once_fn=sort_once,
    )


def _collect_constant_flow_details(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index=None,
    iter_resolved_edge_param_events_fn: Callable[
        ...,
        Iterable[tuple[object, str, object, object]],
    ] = _iter_resolved_edge_param_events,
    reduce_resolved_call_edges_fn: Callable[..., object] = _reduce_resolved_call_edges,
):
    return cast(
        list[object],
        _collect_constant_flow_details_impl(
            paths,
            project_root=project_root,
            ignore_params=ignore_params,
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=analysis_index,
            iter_resolved_edge_param_events_fn=iter_resolved_edge_param_events_fn,
            reduce_resolved_call_edges_fn=reduce_resolved_call_edges_fn,
            deps=_CollectConstantFlowDetailsDeps(
                check_deadline_fn=check_deadline,
                require_not_none_fn=require_not_none,
                resolved_edge_reducer_spec_ctor=_ResolvedEdgeReducerSpec,
                constant_flow_fold_accumulator_ctor=_ConstantFlowFoldAccumulator,
                format_call_site_fn=_format_call_site,
                function_key_fn=_function_key,
                sort_once_fn=sort_once,
                constant_flow_detail_ctor=ConstantFlowDetail,
            ),
        ),
    )


@dataclass
class _ConstantFlowFoldAccumulator:
    const_values: dict[tuple[str, str], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    non_const: dict[tuple[str, str], bool] = field(
        default_factory=lambda: defaultdict(bool)
    )
    call_counts: dict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    call_sites: dict[tuple[str, str], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )


@dataclass(frozen=True)
class _ResolvedEdgeReducerSpec:
    reducer_id: str
    init: Callable[[], object]
    fold: Callable[[object, object], None]
    finish: Callable[[object], object]


@dataclass(frozen=True)
class ConstantFlowDetail:
    path: Path
    qual: str
    name: str
    param: str
    value: str
    count: int
    sites: tuple[str, ...] = ()


@dataclass(frozen=True)
class _StageCacheSpec:
    stage: _ParseModuleStage
    cache_key: object
    build: Callable[[ast.Module, Path], object]


@dataclass
class _AnalysisIndexLite:
    by_name: dict[str, list[FunctionInfo]]
    by_qual: dict[str, FunctionInfo]
    symbol_table: SymbolTable
    class_index: dict[str, ClassInfo]
    transitive_callers: object = None
    resolved_call_edges: object = None
    resolved_transparent_call_edges: object = None
    resolved_transparent_edges_by_caller: object = None


def _dataclass_registry_for_tree(
    path: Path,
    tree: ast.AST,
    *,
    project_root=None,
) -> dict[str, list[str]]:
    return cast(
        dict[str, list[str]],
        _dataclass_registry_for_tree_impl(
            path,
            tree,
            project_root=project_root,
            deps=_DataclassRegistryForTreeDeps(
                check_deadline_fn=check_deadline,
                module_name_fn=_module_name,
                simple_store_name_fn=_simple_store_name,
                decorator_text_fn=lambda node: _decorator_name_local(node) or "",
            ),
        ),
    )


def _collect_exception_obligations(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    handledness_witnesses=None,
    deadness_witnesses=None,
    never_exceptions=None,
):
    return cast(
        list[JSONObject],
        _collect_exception_obligations_impl(
            paths,
            project_root=project_root,
            ignore_params=ignore_params,
            handledness_witnesses=handledness_witnesses,
            deadness_witnesses=deadness_witnesses,
            never_exceptions=never_exceptions,
            check_deadline_fn=check_deadline,
            parent_annotator_factory=ParentAnnotator,
            collect_functions_fn=_collect_functions,
            param_names_fn=_param_names,
            normalize_snapshot_path_fn=_normalize_snapshot_path,
            enclosing_function_node_fn=_enclosing_function_node,
            enclosing_scopes_fn=_enclosing_scopes,
            function_key_fn=_function_key,
            exception_type_name_fn=_exception_type_name,
            decorator_matches_fn=_decorator_matches,
            is_never_marker_raise_fn=_is_never_marker_raise,
            exception_param_names_fn=_exception_param_names,
            exception_path_id_fn=_exception_path_id,
            sequence_or_none_fn=sequence_or_none,
            branch_reachability_under_env_fn=_branch_reachability_under_env,
            is_reachability_false_fn=_is_reachability_false,
            is_reachability_true_fn=_is_reachability_true,
            names_in_expr_fn=_names_in_expr,
            sort_once_fn=sort_once,
            order_policy_sort=OrderPolicy.SORT,
            order_policy_enforce=OrderPolicy.ENFORCE,
            mapping_or_none_fn=mapping_or_none,
            literal_eval_error_types=_LITERAL_EVAL_ERROR_TYPES,
        ),
    )


def _collect_handledness_witnesses(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
) -> list[JSONObject]:
    return cast(
        list[JSONObject],
        _collect_handledness_witnesses_impl(
            paths,
            project_root=project_root,
            ignore_params=ignore_params,
            check_deadline_fn=check_deadline,
            parent_annotator_factory=ParentAnnotator,
            collect_functions_fn=_collect_functions,
            param_names_fn=_param_names,
            param_annotations_fn=_param_annotations,
            normalize_snapshot_path_fn=_normalize_snapshot_path,
            find_handling_try_fn=_find_handling_try,
            enclosing_function_node_fn=_enclosing_function_node,
            enclosing_scopes_fn=_enclosing_scopes,
            function_key_fn=_function_key,
            refine_exception_name_from_annotations_fn=_refine_exception_name_from_annotations,
            exception_param_names_fn=_exception_param_names,
            exception_path_id_fn=_exception_path_id,
            exception_handler_compatibility_fn=_exception_handler_compatibility,
            handler_label_fn=_handler_label,
            handler_type_names_fn=_handler_type_names,
        ),
    )


def _collect_never_invariants(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    forest,
    marker_aliases: Sequence[str] = (),
    deadness_witnesses=None,
) -> list[JSONObject]:
    return cast(
        list[JSONObject],
        _collect_never_invariants_impl(
            paths,
            project_root=project_root,
            ignore_params=ignore_params,
            forest=forest,
            marker_aliases=marker_aliases,
            deadness_witnesses=deadness_witnesses,
            check_deadline_fn=check_deadline,
            parent_annotator_factory=ParentAnnotator,
            collect_functions_fn=_collect_functions,
            param_names_fn=_param_names,
            normalize_snapshot_path_fn=_normalize_snapshot_path,
            enclosing_function_node_fn=_enclosing_function_node,
            enclosing_scopes_fn=_enclosing_scopes,
            function_key_fn=_function_key,
            exception_param_names_fn=_exception_param_names,
            node_span_fn=_node_span,
            dead_env_map_fn=_dead_env_map,
            branch_reachability_under_env_fn=_branch_reachability_under_env,
            is_reachability_false_fn=_is_reachability_false,
            is_reachability_true_fn=_is_reachability_true,
            names_in_expr_fn=_names_in_expr,
            sort_once_fn=sort_once,
            order_policy_sort=OrderPolicy.SORT,
            order_policy_enforce=OrderPolicy.ENFORCE,
            is_marker_call_fn=_is_marker_call,
            decorator_name_fn=_decorator_name_local,
            require_not_none_fn=require_not_none,
        ),
    )


def _collect_invariant_propositions(
    path: Path,
    *,
    ignore_params: set[str],
    project_root,
    emitters: Iterable[Callable[[ast.FunctionDef], Iterable[InvariantProposition]]] = (),
) -> list[InvariantProposition]:
    return cast(
        list[InvariantProposition],
        _collect_invariant_propositions_impl(
            path,
            ignore_params=ignore_params,
            project_root=project_root,
            emitters=cast(Iterable[Callable[[object], Iterable[object]]], emitters),
            deps=_CollectInvariantPropositionsDeps(
                check_deadline_fn=check_deadline,
                parse_module_source_fn=_parse_module_source,
                collect_functions_fn=cast(Callable[[object], Iterable[object]], _collect_functions),
                param_names_fn=_param_names,
                scope_path_fn=_scope_path,
                invariant_collector_ctor=cast(Callable[..., object], _InvariantCollector),
                invariant_proposition_type=InvariantProposition,
                normalize_invariant_proposition_fn=_normalize_invariant_proposition,
            ),
        ),
    )


def _param_annotations_by_path(
    paths: list[Path],
    *,
    ignore_params: set[str],
    parse_failure_witnesses: list[JSONObject],
) -> dict[Path, dict[str, object]]:
    check_deadline()
    annotations: dict[Path, dict[str, object]] = {}
    for path in paths:
        check_deadline()
        parse_outcome = _parse_module_tree(
            path,
            stage=_ParseModuleStage.PARAM_ANNOTATIONS,
            parse_failure_witnesses=parse_failure_witnesses,
        )
        if type(parse_outcome) is not _ParseModuleSuccess:
            continue
        tree = parse_outcome.tree
        parent = ParentAnnotator()
        parent.visit(tree)
        parents = parent.parents
        by_fn: dict[str, object] = {}
        for fn in _collect_functions(tree):
            check_deadline()
            scopes = _enclosing_scopes(fn, parents)
            fn_key = _function_key(scopes, fn.name)
            by_fn[fn_key] = _param_annotations(fn, ignore_params)
        annotations[path] = by_fn
    return annotations


def analyze_decision_surfaces_repo(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    decision_tiers=None,
    require_tiers: bool = False,
    forest=None,
    parse_failure_witnesses=None,
    analysis_index=None,
) -> tuple[list[str], list[str], list[str]]:
    from gabion.analysis.dataflow.engine.decision_surface_analyzer import (
        DecisionSurfaceAnalyzerInput,
        analyze_decision_surfaces,
    )
    from gabion.analysis.dataflow.engine.scan_kernel import (
        ScanKernelDeps,
        ScanKernelRequest,
    )

    check_deadline()
    analyzer_output = analyze_decision_surfaces(
        data=DecisionSurfaceAnalyzerInput(
            kernel_request=ScanKernelRequest(
                paths=paths,
                project_root=project_root,
                ignore_params=ignore_params,
                strictness=strictness,
                external_filter=external_filter,
                transparent_decorators=transparent_decorators,
                parse_failure_witnesses=parse_failure_witnesses,
                analysis_index=analysis_index,
            ),
            decision_tiers=decision_tiers,
            require_tiers=require_tiers,
            forest=forest,
        ),
        deps=ScanKernelDeps(run_indexed_pass_fn=_run_indexed_pass),
        runner=lambda context: _analyze_decision_surfaces_indexed(
            context,
            decision_tiers=decision_tiers,
            require_tiers=require_tiers,
            forest=forest,
        ),
    )
    return (
        analyzer_output.surfaces,
        analyzer_output.warnings,
        analyzer_output.lint_lines,
    )


def analyze_value_encoded_decisions_repo(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    decision_tiers=None,
    require_tiers: bool = False,
    forest=None,
    parse_failure_witnesses=None,
    analysis_index=None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    from gabion.analysis.dataflow.engine.decision_surface_analyzer import (
        DecisionSurfaceAnalyzerInput,
        analyze_value_encoded_decisions,
    )
    from gabion.analysis.dataflow.engine.scan_kernel import (
        ScanKernelDeps,
        ScanKernelRequest,
    )

    check_deadline()
    analyzer_output = analyze_value_encoded_decisions(
        data=DecisionSurfaceAnalyzerInput(
            kernel_request=ScanKernelRequest(
                paths=paths,
                project_root=project_root,
                ignore_params=ignore_params,
                strictness=strictness,
                external_filter=external_filter,
                transparent_decorators=transparent_decorators,
                parse_failure_witnesses=parse_failure_witnesses,
                analysis_index=analysis_index,
            ),
            decision_tiers=decision_tiers,
            require_tiers=require_tiers,
            forest=forest,
        ),
        deps=ScanKernelDeps(run_indexed_pass_fn=_run_indexed_pass),
        runner=lambda context: _analyze_value_encoded_decisions_indexed(
            context,
            decision_tiers=decision_tiers,
            require_tiers=require_tiers,
            forest=forest,
        ),
    )
    return (
        analyzer_output.surfaces,
        analyzer_output.warnings,
        analyzer_output.rewrites,
        analyzer_output.lint_lines,
    )


def run_scan_domain_orchestrator(
    *,
    paths: list[Path],
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    forest,
    transparent_decorators=None,
    decision_tiers=None,
    require_tiers: bool = False,
    parse_failure_witnesses=None,
    analysis_index=None,
) -> dict[str, list[str]]:
    decision_surfaces, decision_warnings, decision_lint = analyze_decision_surfaces_repo(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        decision_tiers=decision_tiers,
        require_tiers=require_tiers,
        forest=forest,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
    )
    value_surfaces, value_warnings, value_rewrites, value_lint = (
        analyze_value_encoded_decisions_repo(
            paths,
            project_root=project_root,
            ignore_params=ignore_params,
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            decision_tiers=decision_tiers,
            require_tiers=require_tiers,
            forest=forest,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=analysis_index,
        )
    )
    return {
        "decision_surfaces": decision_surfaces,
        "decision_warnings": decision_warnings,
        "decision_lint": decision_lint,
        "value_surfaces": value_surfaces,
        "value_warnings": value_warnings,
        "value_rewrites": value_rewrites,
        "value_lint": value_lint,
    }


def _span_line_col(span):
    from gabion.analysis.dataflow.engine.dataflow_lint_helpers import _span_line_col as _impl

    return _impl(span)


def _lint_line(path: str, line: int, col: int, code: str, message: str) -> str:
    return f"{path}:{line}:{col}: {code} {message}".strip()


def _decision_param_lint_line(
    info: "FunctionInfo",
    param: str,
    *,
    project_root,
    code: str,
    message: str,
):
    from gabion.analysis.dataflow.engine.dataflow_lint_helpers import _decision_param_lint_line as _impl

    return _impl(
        info,
        param,
        project_root=project_root,
        code=code,
        message=message,
    )


def _decision_tier_for(
    info: "FunctionInfo",
    param: str,
    *,
    tier_map: dict[str, int],
    project_root,
):
    check_deadline()
    if not tier_map:
        return None
    span = info.param_spans.get(param)
    if span is not None:
        path = _normalize_snapshot_path(info.path, project_root)
        line, col, _, _ = span
        location = f"{path}:{line + 1}:{col + 1}"
        for key in (location, f"{location}:{param}"):
            check_deadline()
            if key in tier_map:
                return tier_map[key]
    for key in (f"{info.qual}:{param}", f"{info.qual}.{param}", param):
        check_deadline()
        if key in tier_map:
            return tier_map[key]
    return None


@dataclass(frozen=True)
class _DecisionSurfaceSpec:
    pass_id: str
    alt_kind: str
    surface_label: str
    params: Callable[[FunctionInfo], set[str]]
    descriptor: Callable[[FunctionInfo, str], str]
    alt_evidence: Callable[[str, str], JSONObject]
    surface_lint_code: str
    surface_lint_message: Callable[[str, str, str], str]
    emit_surface_lint: Callable[[int, object], bool]
    tier_lint_code: str
    tier_missing_message: Callable[[str, str], str]
    tier_internal_message: Callable[[str, int, str, str], str]
    rewrite_line: object = None


def _decision_reason_summary(info: FunctionInfo, params: Iterable[str]) -> str:
    labels: set[str] = set()
    for param in params:
        check_deadline()
        labels.update(info.decision_surface_reasons.get(param, set()))
    if not labels:
        return "heuristic"
    return ", ".join(
        sort_once(labels, source="_decision_reason_summary.labels"),
    )


def _decision_predicate_evidence(
    info: FunctionInfo,
    param: str,
) -> DecisionPredicateEvidence:
    reasons = tuple(
        sort_once(
            info.decision_surface_reasons.get(param, set()),
            source="_decision_predicate_evidence.reasons",
        )
    )
    span = info.param_spans.get(param)
    return DecisionPredicateEvidence(
        parameter=ParameterId.from_raw(param),
        reasons=reasons,
        spans=(SpanIdentity.from_tuple(span),) if span is not None else (),
    )


def _boundary_tier_obligation(caller_count: int) -> str:
    if caller_count > 0:
        return "tier-2:decision-bundle-elevation"
    return "tier-3:decision-table-boundary"


def _decision_surface_alt_evidence(
    *,
    spec: _DecisionSurfaceSpec,
    boundary: str,
    descriptor: str,
    params: Iterable[str],
    caller_count: int,
    reason_summary: str,
) -> JSONObject:
    base_evidence = dict(spec.alt_evidence(boundary, descriptor))
    payload: JSONObject = {
        "boundary": base_evidence.get("boundary", boundary),
        "classification_descriptor": descriptor,
        "classification_reason": reason_summary,
        "decision_params": sort_once(
            set(params),
            source="_decision_surface_alt_evidence.params",
        ),
    }
    if "meta" in base_evidence:
        payload["meta"] = base_evidence["meta"]
    for key in sort_once(
        (str(k) for k in base_evidence if str(k) not in {"boundary", "meta"}),
        source="_decision_surface_alt_evidence.base_evidence",
    ):
        payload[key] = base_evidence[key]
    payload["tier_obligation"] = _boundary_tier_obligation(caller_count)
    payload["tier_pathway"] = "internal" if caller_count > 0 else "boundary"
    return payload


def _suite_site_label(*, forest: object, suite_id: object) -> str:
    suite_node = forest.nodes.get(suite_id)
    if suite_node is None:
        never("suite site missing during label projection", suite_id=str(suite_id))  # pragma: no cover - invariant sink
    path = str(suite_node.meta.get("path", "") or "")
    qual = str(suite_node.meta.get("qual", "") or "")
    suite_kind = str(suite_node.meta.get("suite_kind", "") or "")
    span = int_tuple4_or_none(suite_node.meta.get("span"))
    if not path or not qual or not suite_kind or span is None:
        never(  # pragma: no cover - invariant sink
            "suite site label projection missing identity",
            path=path,
            qual=qual,
            suite_kind=suite_kind,
            span=suite_node.meta.get("span"),
        )
    span_text = _format_span_fields(*span)
    if span_text:
        return f"{path}:{qual}[{suite_kind}]@{span_text}"
    return f"{path}:{qual}[{suite_kind}]"


_DIRECT_DECISION_SURFACE_SPEC = _DecisionSurfaceSpec(
    pass_id="decision_surfaces",
    alt_kind="DecisionSurface",
    surface_label="decision surface params",
    params=lambda info: info.decision_params,
    descriptor=lambda info, boundary: (
        f"{boundary}; reason={_decision_reason_summary(info, info.decision_params)}"
    ),
    alt_evidence=lambda boundary, _descriptor: {
        "meta": boundary,
        "boundary": boundary,
    },
    surface_lint_code="GABION_DECISION_SURFACE",
    surface_lint_message=lambda param, boundary, _descriptor: (
        f"decision surface param '{param}' ({boundary})"
    ),
    emit_surface_lint=lambda caller_count, tier: caller_count == 0 and tier is None,
    tier_lint_code="GABION_DECISION_TIER",
    tier_missing_message=lambda param, _descriptor: (
        f"decision param '{param}' missing decision tier metadata"
    ),
    tier_internal_message=lambda param, tier, boundary, _descriptor: (
        f"tier-{tier} decision param '{param}' used below boundary ({boundary})"
    ),
)


_VALUE_DECISION_SURFACE_SPEC = _DecisionSurfaceSpec(
    pass_id="value_encoded_decisions",
    alt_kind="ValueDecisionSurface",
    surface_label="value-encoded decision params",
    params=lambda info: info.value_decision_params,
    descriptor=lambda info, _boundary: ", ".join(
        sort_once(
            info.value_decision_reasons,
            source="_VALUE_DECISION_SURFACE_SPEC.descriptor",
        )
    )
    or "heuristic",
    alt_evidence=lambda boundary, descriptor: {
        "meta": descriptor,
        "boundary": boundary,
        "reasons": descriptor,
    },
    surface_lint_code="GABION_VALUE_DECISION_SURFACE",
    surface_lint_message=lambda param, boundary, descriptor: (
        f"value-encoded decision param '{param}' ({boundary}; {descriptor})"
    ),
    emit_surface_lint=lambda _caller_count, tier: tier is None,
    tier_lint_code="GABION_VALUE_DECISION_TIER",
    tier_missing_message=lambda param, descriptor: (
        f"value-encoded decision param '{param}' missing decision tier metadata ({descriptor})"
    ),
    tier_internal_message=lambda param, tier, boundary, descriptor: (
        f"tier-{tier} value-encoded decision param '{param}' used below boundary ({boundary}; {descriptor})"
    ),
    rewrite_line=lambda info, params, descriptor: (
        f"{info.path.name}:{info.qual} consider rebranching value-encoded decision params: "
        + ", ".join(params)
        + f" ({descriptor})"
    ),
)


def _analyze_decision_surface_indexed(
    context: _IndexedPassContext,
    *,
    spec: _DecisionSurfaceSpec,
    decision_tiers,
    require_tiers: bool,
    forest: object,
) -> tuple[list[str], list[str], list[str], list[str]]:
    return _analyze_decision_surface_indexed_impl(
        context,
        spec=spec,
        decision_tiers=decision_tiers,
        require_tiers=require_tiers,
        forest=forest,
        deps=_DecisionSurfaceAnalyzeDeps(
            build_call_graph_fn=_build_call_graph,
            check_deadline_fn=check_deadline,
            is_test_path_fn=_is_test_path,
            sort_once_fn=sort_once,
            decision_reason_summary_fn=_decision_reason_summary,
            decision_surface_alt_evidence_fn=_decision_surface_alt_evidence,
            suite_site_label_fn=_suite_site_label,
            decision_tier_for_fn=_decision_tier_for,
            decision_param_lint_line_fn=_decision_param_lint_line,
        ),
    )


def _analyze_decision_surfaces_indexed(
    context: _IndexedPassContext,
    *,
    decision_tiers,
    require_tiers: bool,
    forest: object,
    run_fn: Callable[
        ...,
        tuple[list[str], list[str], list[str], list[str]],
    ] = _analyze_decision_surface_indexed,
) -> tuple[list[str], list[str], list[str]]:
    surfaces, warnings, rewrites, lint_lines = run_fn(
        context,
        spec=_DIRECT_DECISION_SURFACE_SPEC,
        decision_tiers=decision_tiers,
        require_tiers=require_tiers,
        forest=forest,
    )
    if rewrites:
        never(
            "decision_surfaces rewrites must be empty",
            pass_id=_DIRECT_DECISION_SURFACE_SPEC.pass_id,
        )
    return surfaces, warnings, lint_lines


def _analyze_value_encoded_decisions_indexed(
    context: _IndexedPassContext,
    *,
    decision_tiers,
    require_tiers: bool,
    forest: object,
) -> tuple[list[str], list[str], list[str], list[str]]:
    return _analyze_decision_surface_indexed(
        context,
        spec=_VALUE_DECISION_SURFACE_SPEC,
        decision_tiers=decision_tiers,
        require_tiers=require_tiers,
        forest=forest,
    )


def _compute_knob_param_names(
    *,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table: SymbolTable,
    project_root,
    class_index: dict[str, ClassInfo],
    strictness: str,
    analysis_index=None,
) -> set[str]:
    return cast(
        set[str],
        _compute_knob_param_names_impl(
            by_name=by_name,
            by_qual=by_qual,
            symbol_table=symbol_table,
            project_root=project_root,
            class_index=class_index,
            strictness=strictness,
            analysis_index=analysis_index,
            deps=_ComputeKnobParamNamesDeps(
                check_deadline_fn=check_deadline,
                analysis_index_ctor=_AnalysisIndexLite,
                iter_resolved_edge_param_events_fn=_iter_resolved_edge_param_events,
                reduce_resolved_call_edges_fn=_reduce_resolved_call_edges,
                resolved_edge_reducer_spec_ctor=_ResolvedEdgeReducerSpec,
                knob_flow_fold_acc_ctor=_KnobFlowFoldAccumulator,
            ),
        ),
    )


def _collect_config_bundles(
    paths: list[Path],
    *,
    parse_failure_witnesses: list[JSONObject],
    analysis_index=None,
) -> dict[Path, dict[str, set[str]]]:
    return cast(
        dict[Path, dict[str, set[str]]],
        _collect_config_bundles_impl(
            paths,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=analysis_index,
            deps=_CollectConfigBundlesDeps(
                check_deadline_fn=check_deadline,
                forbid_adhoc_bundle_discovery_fn=_forbid_adhoc_bundle_discovery,
                analysis_index_stage_cache_fn=_analysis_index_stage_cache,
                stage_cache_spec_ctor=_StageCacheSpec,
                parse_module_stage_config_fields=_ParseModuleStage.CONFIG_FIELDS,
                parse_stage_cache_key_fn=_parse_stage_cache_key,
                empty_cache_semantic_context=_EMPTY_CACHE_SEMANTIC_CONTEXT,
                iter_config_fields_fn=_iter_config_fields,
            ),
        ),
    )


def _iter_config_fields(
    path: Path,
    *,
    tree=None,
    parse_failure_witnesses: list[JSONObject],
) -> dict[str, set[str]]:
    return cast(
        dict[str, set[str]],
        _iter_config_fields_impl(
            path,
            tree=tree,
            parse_failure_witnesses=parse_failure_witnesses,
            deps=_IterConfigFieldsDeps(
                check_deadline_fn=check_deadline,
                parse_module_tree_fn=_parse_module_tree_or_none,
                parse_module_stage_config_fields=_ParseModuleStage.CONFIG_FIELDS,
                simple_store_name_fn=_simple_store_name,
            ),
        ),
    )


def _collect_dataclass_registry(
    paths: list[Path],
    *,
    project_root,
    parse_failure_witnesses: list[JSONObject],
    analysis_index=None,
    stage_cache_fn: AnalysisIndexStageCacheFn[object] = _analysis_index_stage_cache,
) -> dict[str, list[str]]:
    return cast(
        dict[str, list[str]],
        _collect_dataclass_registry_impl(
            paths,
            project_root=project_root,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=analysis_index,
            stage_cache_fn=stage_cache_fn,
            deps=_CollectDataclassRegistryDeps(
                check_deadline_fn=check_deadline,
                stage_cache_spec_ctor=_StageCacheSpec,
                parse_module_stage_dataclass_registry=_ParseModuleStage.DATACLASS_REGISTRY,
                parse_stage_cache_key_fn=_parse_stage_cache_key,
                empty_cache_semantic_context=_EMPTY_CACHE_SEMANTIC_CONTEXT,
                dataclass_registry_for_tree_fn=_dataclass_registry_for_tree,
                parse_module_tree_fn=_parse_module_tree_or_none,
            ),
        ),
    )


def _iter_dataclass_call_bundles(
    path: Path,
    *,
    project_root=None,
    symbol_table=None,
    dataclass_registry=None,
    parse_failure_witnesses: list[JSONObject],
) -> set[tuple[str, ...]]:
    check_deadline()
    outcome = _iter_dataclass_call_bundle_effects_impl(
        path,
        project_root=project_root,
        symbol_table=symbol_table,
        dataclass_registry=dataclass_registry,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    parse_failure_witnesses.extend(outcome.witness_effects)
    return set(outcome.bundles)


__all__ = [
    "_StageCacheSpec",
    "_annotation_exception_candidates",
    "_branch_reachability_under_env",
    "_build_property_hook_callable_index",
    "_callsite_evidence_for_bundle",
    "_collect_config_bundles",
    "_collect_constant_flow_details",
    "_collect_dataclass_registry",
    "_dead_env_map",
    "_enclosing_function_node",
    "_eval_bool_expr",
    "_eval_value_expr",
    "_exception_param_names",
    "_exception_type_name",
    "_handler_label",
    "_handler_type_names",
    "_lint_line",
    "_keyword_links_literal",
    "_keyword_string_literal",
    "_names_in_expr",
    "_node_in_block",
    "_refine_exception_name_from_annotations",
    "_collect_exception_obligations",
    "_collect_handledness_witnesses",
    "_collect_invariant_propositions",
    "_collect_never_invariants",
    "_combine_type_hints",
    "_infer_type_flow",
    "_compute_knob_param_names",
    "_deserialize_invariants_for_resume",
    "_expand_type_hint",
    "_format_call_site",
    "_format_type_flow_site",
    "_iter_config_fields",
    "_iter_dataclass_call_bundles",
    "_param_annotations_by_path",
    "_parse_module_source",
    "_span_line_col",
    "_split_top_level",
    "_type_from_const_repr",
    "analyze_constant_flow_repo",
    "analyze_deadness_flow_repo",
    "analyze_decision_surfaces_repo",
    "analyze_type_flow_repo_with_evidence",
    "analyze_type_flow_repo_with_map",
    "analyze_unused_arg_flow_repo",
    "analyze_value_encoded_decisions_repo",
    "generate_property_hook_manifest",
    "_decision_param_lint_line",
    "_decision_tier_for",
]
