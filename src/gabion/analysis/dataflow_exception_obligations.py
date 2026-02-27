# gabion:boundary_normalization_module
# gabion:decision_protocol_module
"""Exception-obligation helpers extracted from ``dataflow_audit``."""

from __future__ import annotations

import ast
import builtins
from collections.abc import Callable

from gabion.order_contract import sort_once

_AST_UNPARSE_ERROR_TYPES = (
    AttributeError,
    TypeError,
    ValueError,
    RecursionError,
)


def exception_param_names(
    expr: object,
    params: set[str],
    *,
    check_deadline: Callable[[], None],
) -> list[str]:
    check_deadline()
    names: set[str] = set()
    match expr:
        case ast.AST() as expression:
            for node in ast.walk(expression):
                check_deadline()
                match node:
                    case ast.Name(id=name_text) if name_text in params:
                        names.add(name_text)
                    case _:
                        pass
        case _:
            pass
    return sort_once(names, source="_exception_param_names.names")


def exception_type_name(
    expr: object,
    *,
    decorator_name: Callable[[ast.AST], object],
) -> object:
    match expr:
        case ast.Call(func=call_target):
            return decorator_name(call_target)
        case ast.AST() as expression:
            return decorator_name(expression)
        case _:
            return None


def handler_is_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    match handler.type:
        case ast.Name(id=name_text):
            return name_text in {"Exception", "BaseException"}
        case ast.Attribute(attr=attr_text):
            return attr_text in {"Exception", "BaseException"}
        case _:
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


def handler_type_names(
    handler_type: object,
    *,
    decorator_name: Callable[[ast.AST], object],
    check_deadline: Callable[[], None],
) -> tuple[str, ...]:
    check_deadline()
    match handler_type:
        case ast.Tuple(elts=elements):
            names: list[str] = []
            for elt in elements:
                check_deadline()
                name = decorator_name(elt)
                if name:
                    names.append(str(name))
            return tuple(names)
        case ast.AST() as handler_expr:
            name = decorator_name(handler_expr)
            return (str(name),) if name else ()
        case _:
            return ()


def exception_handler_compatibility(
    exception_name: object,
    handler_type: object,
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
    match exception_name:
        case str() as exception_name_text:
            exception_cls = _builtin_exception_class(exception_name_text)
        case _:
            exception_cls = None
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
    match value:
        case type() as value_type:
            if issubclass(value_type, BaseException):
                return value_type
            return None
        case _:
            return None
