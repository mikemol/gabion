from __future__ import annotations

import ast

from gabion.analysis.dataflow_exception_obligations import (
    exception_param_names,
    exception_type_name,
    handler_is_broad,
    handler_label,
    node_in_try_body,
)


def _check_deadline() -> None:
    return None


def _decorator_name(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def test_exception_obligation_module_edges() -> None:
    assert exception_param_names(None, {"a"}, check_deadline=_check_deadline) == []
    expr = ast.parse("a + b").body[0].value
    assert exception_param_names(expr, {"a"}, check_deadline=_check_deadline) == ["a"]

    assert exception_type_name(None, decorator_name=_decorator_name) is None
    assert exception_type_name(ast.parse("ValueError").body[0].value, decorator_name=_decorator_name) == "ValueError"
    assert exception_type_name(ast.parse("ValueError()") .body[0].value, decorator_name=_decorator_name) == "ValueError"

    handler_any = ast.ExceptHandler(type=None, name=None, body=[])
    assert handler_is_broad(handler_any) is True
    assert handler_label(handler_any) == "except:"

    handler_attr = ast.ExceptHandler(
        type=ast.Attribute(
            value=ast.Name(id="builtins", ctx=ast.Load()),
            attr="Exception",
            ctx=ast.Load(),
        ),
        name=None,
        body=[],
    )
    assert handler_is_broad(handler_attr) is True

    handler_weird = ast.ExceptHandler(type=object(), name=None, body=[])
    assert handler_is_broad(handler_weird) is False
    assert handler_label(handler_weird) == "except <unknown>"

    tree = ast.parse("try:\n    foo(1)\nexcept Exception:\n    pass\n")
    try_node = tree.body[0]
    assert isinstance(try_node, ast.Try)
    call_node = try_node.body[0].value
    assert node_in_try_body(call_node, try_node, check_deadline=_check_deadline) is True
    other_call = ast.parse("bar()").body[0].value
    assert node_in_try_body(other_call, try_node, check_deadline=_check_deadline) is False
