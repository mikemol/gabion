"""Exception-obligation helpers extracted from ``dataflow_audit``."""

from __future__ import annotations

import ast
from collections.abc import Callable

from gabion.order_contract import ordered_or_sorted

_AST_UNPARSE_ERROR_TYPES = (
    AttributeError,
    TypeError,
    ValueError,
    RecursionError,
)


def exception_param_names(
    expr: ast.AST | None,
    params: set[str],
    *,
    check_deadline: Callable[[], None],
) -> list[str]:
    check_deadline()
    if expr is None:
        return []
    names: set[str] = set()
    for node in ast.walk(expr):
        check_deadline()
        if isinstance(node, ast.Name) and node.id in params:
            names.add(node.id)
    return ordered_or_sorted(names, source="_exception_param_names.names")


def exception_type_name(
    expr: ast.AST | None,
    *,
    decorator_name: Callable[[ast.AST], str | None],
) -> str | None:
    if expr is None:
        return None
    if isinstance(expr, ast.Call):
        return decorator_name(expr.func)
    return decorator_name(expr)


def handler_is_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name):
        return handler.type.id in {"Exception", "BaseException"}
    if isinstance(handler.type, ast.Attribute):
        return handler.type.attr in {"Exception", "BaseException"}
    return False


def handler_label(handler: ast.ExceptHandler) -> str:
    if handler.type is None:
        return "except:"
    try:
        return f"except {ast.unparse(handler.type)}"
    except _AST_UNPARSE_ERROR_TYPES:
        return "except <unknown>"


def node_in_try_body(
    node: ast.AST,
    try_node: ast.Try,
    *,
    check_deadline: Callable[[], None],
) -> bool:
    check_deadline()
    for stmt in try_node.body:
        check_deadline()
        if node is stmt:
            return True
        for child in ast.walk(stmt):
            check_deadline()
            if node is child:
                return True
    return False
