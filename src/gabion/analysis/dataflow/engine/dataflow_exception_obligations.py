# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_exception_obligations
"""Exception-obligation helpers extracted from ``legacy_dataflow_monolith``."""

from __future__ import annotations

import ast
import builtins
from collections.abc import Callable, Mapping
from enum import StrEnum
from functools import singledispatch

from gabion.order_contract import sort_once

_AST_UNPARSE_ERROR_TYPES = (
    AttributeError,
    TypeError,
    ValueError,
    RecursionError,
)

class _EvalDecision(StrEnum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@singledispatch
def _exception_target_expr(expr: ast.AST) -> ast.AST:
    return expr


@_exception_target_expr.register(ast.Call)
def _exception_target_expr_for_call(expr: ast.Call) -> ast.AST:
    return expr.func


def exception_param_names(
    expr: ast.AST,
    params: set[str],
    *,
    check_deadline: Callable[[], None],
) -> list[str]:
    check_deadline()
    return sort_once(
        {
            node.id
            for node in ast.walk(expr)
            if isinstance(node, ast.Name) and node.id in params
        },
        source="_exception_param_names.names",
    )


def exception_type_name(
    expr: ast.AST,
    *,
    decorator_name: Callable[[ast.AST], object],
) -> object:
    return decorator_name(_exception_target_expr(expr))


@singledispatch
def _handler_type_is_broad(expr: ast.AST) -> bool:
    return False


@_handler_type_is_broad.register(ast.Name)
def _handler_name_is_broad(expr: ast.Name) -> bool:
    return expr.id in {"Exception", "BaseException"}


@_handler_type_is_broad.register(ast.Attribute)
def _handler_attribute_is_broad(expr: ast.Attribute) -> bool:
    return expr.attr in {"Exception", "BaseException"}


def handler_is_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    return _handler_type_is_broad(handler.type)


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


@singledispatch
def _handler_type_names_from_expr(
    handler_expr: ast.AST,
    *,
    decorator_name: Callable[[ast.AST], object],
    check_deadline: Callable[[], None],
) -> tuple[str, ...]:
    name = decorator_name(handler_expr)
    return (str(name),) if name else ()


@_handler_type_names_from_expr.register(ast.Tuple)
def _handler_type_names_from_tuple(
    handler_expr: ast.Tuple,
    *,
    decorator_name: Callable[[ast.AST], object],
    check_deadline: Callable[[], None],
) -> tuple[str, ...]:
    names: list[str] = []
    for elt in handler_expr.elts:
        check_deadline()
        name = decorator_name(elt)
        if name:
            names.append(str(name))
    return tuple(names)


def handler_type_names(
    handler_type: ast.AST | None,
    *,
    decorator_name: Callable[[ast.AST], object],
    check_deadline: Callable[[], None],
) -> tuple[str, ...]:
    check_deadline()
    if handler_type is None:
        return ()
    return _handler_type_names_from_expr(
        handler_type,
        decorator_name=decorator_name,
        check_deadline=check_deadline,
    )


def exception_handler_compatibility(
    exception_name: str | None,
    handler_type: ast.AST | None,
    *,
    decorator_name: Callable[[ast.AST], object],
    check_deadline: Callable[[], None],
) -> str:
    check_deadline()
    if handler_type is None:
        return "compatible"
    handler_names = handler_type_names(
        handler_type,
        decorator_name=decorator_name,
        check_deadline=check_deadline,
    )
    if not handler_names:
        return "unknown"
    exception_cls = (
        _builtin_exception_class(exception_name)
        if exception_name is not None
        else None
    )
    if exception_cls is None:
        return "unknown"
    any_unknown = False
    for handler_name in handler_names:
        check_deadline()
        handler_cls = _builtin_exception_class(handler_name)
        if handler_cls is None:
            any_unknown = True
            continue
        if issubclass(exception_cls, handler_cls):
            return "compatible"
    return "unknown" if any_unknown else "incompatible"


def _builtin_exception_class(name: str) -> object:
    value = getattr(builtins, name, None)
    try:
        if issubclass(value, BaseException):
            return value
    except TypeError:
        return None
    return None
