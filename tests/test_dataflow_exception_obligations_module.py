from __future__ import annotations

import ast

from gabion.analysis.dataflow_exception_obligations import (
    exception_handler_compatibility,
    exception_param_names,
    exception_type_name,
    handler_is_broad,
    handler_label,
    handler_type_names,
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


# gabion:evidence E:call_footprint::tests/test_dataflow_exception_obligations_module.py::test_exception_obligation_module_edges::dataflow_exception_obligations.py::gabion.analysis.dataflow_exception_obligations.exception_param_names::dataflow_exception_obligations.py::gabion.analysis.dataflow_exception_obligations.exception_type_name::dataflow_exception_obligations.py::gabion.analysis.dataflow_exception_obligations.handler_is_broad::dataflow_exception_obligations.py::gabion.analysis.dataflow_exception_obligations.handler_label::dataflow_exception_obligations.py::gabion.analysis.dataflow_exception_obligations.node_in_try_body
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


# gabion:evidence E:call_footprint::tests/test_dataflow_exception_obligations_module.py::test_exception_handler_compatibility_edges::dataflow_exception_obligations.py::gabion.analysis.dataflow_exception_obligations.exception_handler_compatibility::dataflow_exception_obligations.py::gabion.analysis.dataflow_exception_obligations.handler_type_names
def test_exception_handler_compatibility_edges() -> None:
    broad = ast.ExceptHandler(type=None, name=None, body=[])
    assert (
        exception_handler_compatibility(
            "ValueError",
            broad.type,
            decorator_name=_decorator_name,
            check_deadline=_check_deadline,
        )
        == "compatible"
    )

    typed = ast.ExceptHandler(type=ast.parse("ValueError").body[0].value, name=None, body=[])
    assert handler_type_names(typed.type, decorator_name=_decorator_name, check_deadline=_check_deadline) == ("ValueError",)
    assert (
        exception_handler_compatibility(
            "RuntimeError",
            typed.type,
            decorator_name=_decorator_name,
            check_deadline=_check_deadline,
        )
        == "incompatible"
    )

    tuple_handler = ast.ExceptHandler(
        type=ast.Tuple(
            elts=[ast.Name(id="ValueError", ctx=ast.Load()), ast.Name(id="TypeError", ctx=ast.Load())],
            ctx=ast.Load(),
        ),
        name=None,
        body=[],
    )
    assert handler_type_names(tuple_handler.type, decorator_name=_decorator_name, check_deadline=_check_deadline) == ("ValueError", "TypeError")
    assert (
        exception_handler_compatibility(
            "RuntimeError",
            tuple_handler.type,
            decorator_name=_decorator_name,
            check_deadline=_check_deadline,
        )
        == "incompatible"
    )
    assert (
        exception_handler_compatibility(
            "UnicodeError",
            tuple_handler.type,
            decorator_name=_decorator_name,
            check_deadline=_check_deadline,
        )
        == "compatible"
    )

    unresolved = ast.ExceptHandler(type=ast.parse("pkg.CustomError").body[0].value, name=None, body=[])
    assert (
        exception_handler_compatibility(
            "ValueError",
            unresolved.type,
            decorator_name=_decorator_name,
            check_deadline=_check_deadline,
        )
        == "unknown"
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_exception_obligations_module.py::test_exception_handler_compatibility_additional_edges::dataflow_exception_obligations.py::gabion.analysis.dataflow_exception_obligations.exception_handler_compatibility::dataflow_exception_obligations.py::gabion.analysis.dataflow_exception_obligations.handler_is_broad::dataflow_exception_obligations.py::gabion.analysis.dataflow_exception_obligations.handler_type_names
def test_exception_handler_compatibility_additional_edges() -> None:
    named_broad = ast.ExceptHandler(
        type=ast.Name(id="Exception", ctx=ast.Load()),
        name=None,
        body=[],
    )
    assert handler_is_broad(named_broad) is True
    assert handler_type_names(
        None,
        decorator_name=_decorator_name,
        check_deadline=_check_deadline,
    ) == ()
    assert handler_type_names(
        ast.Tuple(
            elts=[
                ast.Name(id="ValueError", ctx=ast.Load()),
                ast.Constant(value=42),
            ],
            ctx=ast.Load(),
        ),
        decorator_name=_decorator_name,
        check_deadline=_check_deadline,
    ) == ("ValueError",)
    assert (
        exception_handler_compatibility(
            "ValueError",
            ast.parse("42").body[0].value,
            decorator_name=_decorator_name,
            check_deadline=_check_deadline,
        )
        == "unknown"
    )
    assert (
        exception_handler_compatibility(
            None,
            ast.parse("ValueError").body[0].value,
            decorator_name=_decorator_name,
            check_deadline=_check_deadline,
        )
        == "unknown"
    )
    assert (
        exception_handler_compatibility(
            "DefinitelyNotBuiltinError",
            ast.parse("ValueError").body[0].value,
            decorator_name=_decorator_name,
            check_deadline=_check_deadline,
        )
        == "unknown"
    )
    assert (
        exception_handler_compatibility(
            "ValueError",
            ast.parse("int").body[0].value,
            decorator_name=_decorator_name,
            check_deadline=_check_deadline,
        )
        == "unknown"
    )
