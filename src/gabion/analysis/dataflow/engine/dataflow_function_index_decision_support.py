# gabion:boundary_normalization_module gabion:decision_protocol_module
from __future__ import annotations

"""Canonical decision/decorator helpers for function-index accumulation."""

import ast
from collections import defaultdict
from typing import cast

from gabion.analysis.dataflow.engine.dataflow_function_index_helpers import _param_names
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.indexed_scan.calls.callee_resolution_helpers import (
    decorator_name as _decorator_name_impl,
)
from gabion.analysis.indexed_scan.scanners.flow.value_encoded_decision_params import (
    ValueEncodedDecisionParamsDeps as _ValueEncodedDecisionParamsDeps,
    value_encoded_decision_params as _value_encoded_decision_params_impl,
)

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
OptionalIgnoredParams = set[str] | None


def _decorator_name(node: ast.AST):
    return _decorator_name_impl(node, check_deadline_fn=check_deadline)


def _decorator_matches(name: str, allowlist: set[str]) -> bool:
    if name in allowlist:
        return True
    if "." in name and name.split(".")[-1] in allowlist:
        return True
    return False


def _decorators_transparent(
    fn: FunctionNode,
    transparent_decorators,
) -> bool:
    check_deadline()
    if not fn.decorator_list:
        return True
    if not transparent_decorators:
        return True
    for deco in fn.decorator_list:
        check_deadline()
        name = _decorator_name(deco)
        if not name:
            return False
        if not _decorator_matches(name, transparent_decorators):
            return False
    return True


def _decision_root_name(node: ast.AST):
    check_deadline()
    current = node
    while True:
        check_deadline()
        current_type = type(current)
        if current_type is ast.Attribute:
            current = cast(ast.Attribute, current).value
        elif current_type is ast.Subscript:
            current = cast(ast.Subscript, current).value
        else:
            break
    if type(current) is ast.Name:
        return cast(ast.Name, current).id
    return None


def is_decision_surface(node: ast.AST) -> bool:
    node_type = type(node)
    return (
        node_type is ast.If
        or node_type is ast.While
        or node_type is ast.Assert
        or node_type is ast.IfExp
        or node_type is ast.Match
        or node_type is ast.comprehension
    )


def _decision_surface_form_entries(
    fn: ast.AST,
) -> list[tuple[str, ast.AST]]:
    check_deadline()
    entries: list[tuple[str, ast.AST]] = []
    for node in ast.walk(fn):
        check_deadline()
        if not is_decision_surface(node):
            continue
        node_type = type(node)
        if node_type is ast.If:
            entries.append(("if", cast(ast.If, node).test))
            continue
        if node_type is ast.While:
            entries.append(("while", cast(ast.While, node).test))
            continue
        if node_type is ast.Assert:
            entries.append(("assert", cast(ast.Assert, node).test))
            continue
        if node_type is ast.IfExp:
            entries.append(("ifexp", cast(ast.IfExp, node).test))
            continue
        if node_type is ast.Match:
            match_node = cast(ast.Match, node)
            entries.append(("match_subject", match_node.subject))
            for case in match_node.cases:
                check_deadline()
                if case.guard is not None:
                    entries.append(("match_guard", case.guard))
            continue
        for guard in cast(ast.comprehension, node).ifs:
            check_deadline()
            entries.append(("comprehension_guard", guard))
    return entries


def _mark_param_roots(expr: ast.AST, params: set[str], out: set[str]) -> None:
    check_deadline()
    for node in ast.walk(expr):
        check_deadline()
        node_type = type(node)
        if node_type is ast.Name and cast(ast.Name, node).id in params:
            out.add(cast(ast.Name, node).id)
            continue
        if node_type is ast.Attribute or node_type is ast.Subscript:
            root = _decision_root_name(node)
            if root in params:
                out.add(root)


def _collect_param_roots(expr: ast.AST, params: set[str]) -> set[str]:
    found: set[str] = set()
    _mark_param_roots(expr, params, found)
    return found


def _contains_boolish(expr: ast.AST) -> bool:
    check_deadline()
    for node in ast.walk(expr):
        check_deadline()
        node_type = type(node)
        if node_type is ast.Compare or node_type is ast.BoolOp:
            return True
        if node_type is ast.UnaryOp and type(cast(ast.UnaryOp, node).op) is ast.Not:
            return True
    return False


def _decision_surface_reason_map(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> dict[str, set[str]]:
    check_deadline()
    params = set(_param_names(fn, ignore_params))
    if not params:
        return {}
    reason_map: dict[str, set[str]] = defaultdict(set)
    for reason, expr in _decision_surface_form_entries(fn):
        check_deadline()
        found = _collect_param_roots(expr, params)
        for param in found:
            check_deadline()
            reason_map[param].add(reason)
    return reason_map


def _decision_surface_params(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> set[str]:
    check_deadline()
    reason_map = _decision_surface_reason_map(fn, ignore_params)
    return set(reason_map)


def _value_encoded_decision_params(
    fn: ast.AST,
    ignore_params=None,
) -> tuple[set[str], set[str]]:
    return _value_encoded_decision_params_impl(
        fn,
        ignore_params,
        deps=_ValueEncodedDecisionParamsDeps(
            check_deadline_fn=check_deadline,
            param_names_fn=_param_names,
            mark_param_roots_fn=_mark_param_roots,
            contains_boolish_fn=_contains_boolish,
            collect_param_roots_fn=_collect_param_roots,
        ),
    )


__all__ = [
    "_collect_param_roots",
    "_contains_boolish",
    "_decorator_name",
    "_decision_surface_form_entries",
    "_decision_surface_params",
    "_decision_surface_reason_map",
    "_decorators_transparent",
    "_mark_param_roots",
    "_value_encoded_decision_params",
    "is_decision_surface",
]
