from __future__ import annotations

import ast
from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import CallArgs, ParamUse, _call_context, _callee_name, _const_repr
    from gabion.analysis.visitors import ParentAnnotator, UseVisitor

    return CallArgs, ParamUse, _call_context, _callee_name, _const_repr, ParentAnnotator, UseVisitor


def _call_args_factory(**kwargs):
    CallArgs, *_ = _load()
    return CallArgs(**kwargs)


def _make_visitor(
    tree: ast.AST,
    strictness: str,
    *,
    return_aliases: dict[str, tuple[list[str], list[str]]] | None = None,
):
    CallArgs, ParamUse, _call_context, _callee_name, _const_repr, ParentAnnotator, UseVisitor = _load()
    annotator = ParentAnnotator()
    annotator.visit(tree)
    use_map = {
        "a": ParamUse(direct_forward=set(), non_forward=False, current_aliases={"a"}),
        "b": ParamUse(direct_forward=set(), non_forward=False, current_aliases={"b"}),
        "args": ParamUse(direct_forward=set(), non_forward=False, current_aliases={"args"}),
        "kwargs": ParamUse(direct_forward=set(), non_forward=False, current_aliases={"kwargs"}),
    }
    alias_to_param = {name: name for name in use_map}
    call_args: list[CallArgs] = []
    return_aliases = return_aliases or {
        "ret": (["a", "b"], ["a", "b"]),
        "ret1": (["a"], ["a"]),
    }
    visitor = UseVisitor(
        parents=annotator.parents,
        use_map=use_map,
        call_args=call_args,
        alias_to_param=alias_to_param,
        is_test=False,
        strictness=strictness,
        const_repr=_const_repr,
        callee_name=_callee_name,
        call_args_factory=_call_args_factory,
        call_context=_call_context,
        return_aliases=return_aliases,
    )
    return visitor, use_map, alias_to_param, call_args


def test_usevisitor_star_forwarding_low_strictness() -> None:
    tree = ast.parse(
        "def f(a, b, *args, **kwargs):\n"
        "    g(a, b, *args, **kwargs, k=kwargs)\n"
    )
    visitor, use_map, _, call_args = _make_visitor(tree, strictness="low")
    visitor.visit(tree)
    assert call_args
    assert ("args[*]", "arg[*]") in use_map["args"].direct_forward
    assert ("kwargs[*]", "kw[*]") in use_map["kwargs"].direct_forward


def test_usevisitor_span_adjusts_zero_width_call() -> None:
    tree = ast.parse("def f(a, b, *args, **kwargs):\n    g(a)\n")
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    # Force a degenerate end position to exercise the span widening branch.
    call.end_lineno = call.lineno
    call.end_col_offset = call.col_offset

    visitor, _, _, call_args = _make_visitor(tree, strictness="high")
    visitor.visit(tree)
    assert call_args
    assert call_args[0].span == (
        call.lineno - 1,
        call.col_offset,
        call.lineno - 1,
        call.col_offset + 1,
    )


def test_usevisitor_alias_binding_and_non_forward() -> None:
    tree = ast.parse(
        "def ret(a, b):\n"
        "    return a, b\n"
        "def ret1(a):\n"
        "    return a\n"
        "def f(a, b, *args, **kwargs):\n"
        "    x, y = ret(a, b)\n"
        "    z: int = ret1(a)\n"
        "    (u, v) = (a, b)\n"
        "    obj = a\n"
        "    obj.attr = a\n"
        "    record = a\n"
        "    record['key'] = b\n"
        "    val = obj.attr\n"
        "    val2 = record['key']\n"
        "    nested = (a.b).c\n"
        "    a = 1\n"
        "    a += 2\n"
        "    g(*args, **kwargs)\n"
        "    return a\n"
    )
    visitor, use_map, alias_to_param, call_args = _make_visitor(tree, strictness="high")
    visitor.visit(tree)
    assert call_args
    assert alias_to_param.get("x") == "a"
    assert alias_to_param.get("y") == "b"
    assert alias_to_param.get("z") == "a"
    assert alias_to_param.get("u") == "a"
    assert alias_to_param.get("v") == "b"
    assert use_map["a"].non_forward is True
    assert use_map["b"].non_forward is True
    assert use_map["args"].non_forward is True
    assert use_map["kwargs"].non_forward is True


def test_return_alias_binding_tuple_and_rejects_mismatch() -> None:
    tree = ast.parse("def f(a, b):\n    return a, b\n")
    visitor, _, alias_to_param, _ = _make_visitor(tree, strictness="high")
    targets = [ast.Tuple(elts=[ast.Name(id="x", ctx=ast.Store()), ast.Name(id="y", ctx=ast.Store())], ctx=ast.Store())]
    assert visitor._bind_return_alias(targets, ["a", "b"]) is True
    assert alias_to_param["x"] == "a"
    assert alias_to_param["y"] == "b"
    assert visitor._bind_return_alias(targets, ["a"]) is False


def test_alias_from_call_rejects_starred() -> None:
    tree = ast.parse(
        "def ret(a, b):\n"
        "    return a, b\n"
        "def f(a, b, *args):\n"
        "    return ret(*args)\n"
    )
    visitor, _, _, _ = _make_visitor(
        tree,
        strictness="high",
        return_aliases={"ret": (["a", "b"], ["a", "b"])},
    )
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    assert visitor._alias_from_call(call) is None


def test_attribute_and_subscript_forwarding() -> None:
    tree = ast.parse(
        "def f(a):\n"
        "    obj.attr = a\n"
        "    record['key'] = a\n"
        "    g(obj.attr)\n"
        "    h(record['key'])\n"
        "    i(record[0])\n"
        "    j(a.b.c)\n"
    )
    visitor, use_map, _, _ = _make_visitor(tree, strictness="high")
    visitor.visit(tree)
    assert ("g", "arg[0]") in use_map["a"].direct_forward
    assert ("h", "arg[0]") in use_map["a"].direct_forward
    assert use_map["a"].non_forward is True


def test_bind_sequence_mismatch_marks_non_forward() -> None:
    tree = ast.parse("def f(a, b):\n    pass\n")
    visitor, use_map, _, _ = _make_visitor(tree, strictness="high")
    target = ast.Tuple(elts=[ast.Name(id="x", ctx=ast.Store())], ctx=ast.Store())
    rhs = ast.Tuple(
        elts=[ast.Name(id="a", ctx=ast.Load()), ast.Name(id="b", ctx=ast.Load())],
        ctx=ast.Load(),
    )
    assert visitor._bind_sequence(target, rhs) is False
    assert use_map["a"].non_forward is False
    assert use_map["b"].non_forward is False


def test_import_visitor_basic_and_relative() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import SymbolTable
    from gabion.analysis.visitors import ImportVisitor

    table = SymbolTable()
    visitor = ImportVisitor("pkg.mod", table)
    visitor.visit(ast.parse("import os\nimport pkg.util as util\n"))
    assert table.imports[("pkg.mod", "os")] == "os"
    assert table.imports[("pkg.mod", "util")] == "pkg.util"

    visitor.visit(ast.parse("from .sub import Thing\n"))
    assert table.imports[("pkg.mod", "Thing")] == "pkg.sub.Thing"

    visitor.visit(ast.parse("from .. import Foo\n"))
    assert table.imports[("pkg.mod", "Foo")] == "Foo"

    visitor.visit(ast.parse("from ..star import *\n"))
    assert "star" in table.star_imports.get("pkg.mod", set())

    before = dict(table.imports)
    deep = ImportVisitor("a.b", table)
    deep.visit(ast.parse("from ..... import Nope\n"))
    assert dict(table.imports) == before
