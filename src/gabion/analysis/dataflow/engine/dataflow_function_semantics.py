# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Function-semantics owner surface for indexed dataflow analysis."""

import ast
from collections.abc import Mapping
from typing import cast

from gabion.analysis.core.visitors import UseVisitor
from gabion.analysis.dataflow.engine.dataflow_contracts import CallArgs, ParamUse
from gabion.analysis.dataflow.engine.dataflow_function_index_helpers import (
    _enclosing_class,
    _enclosing_scopes,
    _param_names,
)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.indexed_scan.scanners.key_aliases import (
    normalize_key_expr as _normalize_key_expr_impl,
)

_AST_UNPARSE_ERROR_TYPES = (
    AttributeError,
    TypeError,
    ValueError,
    RecursionError,
)

_LITERAL_EVAL_ERROR_TYPES = (
    SyntaxError,
    ValueError,
    TypeError,
    MemoryError,
    RecursionError,
)

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


class _ReturnAliasCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.returns: list[object] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        del node
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        del node
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node
        return

    def visit_Return(self, node: ast.Return) -> None:
        self.returns.append(node.value)


def _function_key(scope: list[str], name: str) -> str:
    if not scope:
        return name
    return ".".join([*scope, name])


def _callee_name(call: ast.Call) -> str:
    try:
        return ast.unparse(call.func)
    except _AST_UNPARSE_ERROR_TYPES:
        return "<call>"


def _normalize_callee(name: str, class_name = None) -> str:
    if not class_name:
        return name
    if name.startswith("self.") or name.startswith("cls."):
        parts = name.split(".")
        if len(parts) == 2:
            return f"{class_name}.{parts[1]}"
    return name


def _call_context(node: ast.AST, parents: dict[ast.AST, ast.AST]):
    check_deadline()
    child = node
    parent = parents.get(child)
    while parent is not None:
        check_deadline()
        if type(parent) is ast.Call:
            call_parent = cast(ast.Call, parent)
            if child in call_parent.args:
                return call_parent, True
            for kw in call_parent.keywords:
                check_deadline()
                if child is kw or child is kw.value:
                    return call_parent, True
            return call_parent, False
        child = parent
        parent = parents.get(child)
    return None, False


def _return_aliases(
    fn: ast.AST,
    ignore_params = None,
):
    check_deadline()
    params = _param_names(cast(FunctionNode, fn), ignore_params or set())
    if not params:
        return None
    param_set = set(params)
    collector = _ReturnAliasCollector()
    for stmt in cast(FunctionNode, fn).body:
        check_deadline()
        collector.visit(stmt)
    if not collector.returns:
        return None
    alias = None

    def _alias_from_expr(expr = None):
        check_deadline()
        if expr is not None:
            expr_type = type(expr)
            if expr_type is ast.Name:
                name_node = cast(ast.Name, expr)
                if name_node.id in param_set:
                    return [name_node.id]
            if expr_type in {ast.Tuple, ast.List}:
                sequence_node = cast(ast.Tuple | ast.List, expr)
                names: list[str] = []
                for elt in sequence_node.elts:
                    check_deadline()
                    if type(elt) is ast.Name and cast(ast.Name, elt).id in param_set:
                        names.append(cast(ast.Name, elt).id)
                    else:
                        return None
                return names
        return None

    for expr in collector.returns:
        check_deadline()
        candidate = _alias_from_expr(expr)
        if candidate is not None:
            if alias is None:
                alias = candidate
                continue
            if alias != candidate:
                return None
            continue
        return None
    return alias


def _collect_return_aliases(
    funcs: list[FunctionNode],
    parents: dict[ast.AST, ast.AST],
    *,
    ignore_params: set[str],
) -> dict[str, tuple[list[str], list[str]]]:
    check_deadline()
    aliases: dict[str, tuple[list[str], list[str]]] = {}
    conflicts: set[str] = set()
    for fn in funcs:
        check_deadline()
        alias = _return_aliases(fn, ignore_params)
        if not alias:
            continue
        params = _param_names(fn, ignore_params)
        class_name = _enclosing_class(fn, parents)
        scopes = _enclosing_scopes(fn, parents)
        keys = {fn.name}
        if class_name:
            keys.add(f"{class_name}.{fn.name}")
        if scopes:
            keys.add(_function_key(scopes, fn.name))
        info = (params, alias)
        for key in keys:
            check_deadline()
            if key in conflicts:
                continue
            if key in aliases:
                aliases.pop(key, None)
                conflicts.add(key)
                continue
            aliases[key] = info
    return aliases


def _const_repr(node: ast.AST):
    node_type = type(node)
    if node_type is ast.Constant:
        return repr(cast(ast.Constant, node).value)
    if node_type is ast.UnaryOp:
        unary_node = cast(ast.UnaryOp, node)
        if type(unary_node.op) in {ast.USub, ast.UAdd} and type(unary_node.operand) is ast.Constant:
            try:
                return ast.unparse(unary_node)
            except _AST_UNPARSE_ERROR_TYPES:
                return None
    if node_type is ast.Attribute:
        attribute_node = cast(ast.Attribute, node)
        if attribute_node.attr.isupper():
            try:
                return ast.unparse(attribute_node)
            except _AST_UNPARSE_ERROR_TYPES:
                return None
        return None
    return None


def _normalize_key_expr(
    node: ast.AST,
    *,
    const_bindings: Mapping[str, ast.AST],
):
    return _normalize_key_expr_impl(
        node,
        const_bindings=const_bindings,
        check_deadline_fn=check_deadline,
        literal_eval_error_types=_LITERAL_EVAL_ERROR_TYPES,
    )


def _analyze_function(
    fn: FunctionNode,
    parents: dict[ast.AST, ast.AST],
    *,
    is_test: bool,
    ignore_params = None,
    strictness: str = "high",
    class_name = None,
    return_aliases = None,
) -> tuple[dict[str, ParamUse], list[CallArgs]]:
    params = _param_names(fn, ignore_params or set())
    use_map = {p: ParamUse(set(), False, {p}) for p in params}
    alias_to_param: dict[str, str] = {p: p for p in params}
    call_args: list[CallArgs] = []

    visitor = UseVisitor(
        parents=parents,
        use_map=use_map,
        call_args=call_args,
        alias_to_param=alias_to_param,
        is_test=is_test,
        strictness=strictness,
        const_repr=_const_repr,
        callee_name=lambda call: _normalize_callee(_callee_name(call), class_name),
        call_args_factory=CallArgs,
        call_context=_call_context,
        return_aliases=return_aliases,
        normalize_key_expr=_normalize_key_expr,
    )
    visitor.visit(fn)
    return use_map, call_args


__all__ = [
    "_analyze_function",
    "_call_context",
    "_collect_return_aliases",
    "_const_repr",
    "_normalize_key_expr",
    "_return_aliases",
]
