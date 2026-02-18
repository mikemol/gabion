from __future__ import annotations

import ast
from pathlib import Path

import pytest

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis.dataflow_audit import (
        CallArgs,
        ParamUse,
        _callee_name,
        _call_context,
        _const_repr,
        _normalize_key_expr,
    )
    from gabion.analysis.visitors import ImportVisitor, ParentAnnotator, UseVisitor

    return (
        CallArgs,
        ParamUse,
        _callee_name,
        _call_context,
        _const_repr,
        _normalize_key_expr,
        ImportVisitor,
        ParentAnnotator,
        UseVisitor,
    )

def _make_use_visitor(
    code: str,
    params: list[str],
    *,
    strictness: str = "high",
    return_aliases: dict[str, tuple[list[str], list[str]]] | None = None,
):
    (
        CallArgs,
        ParamUse,
        _callee_name,
        _call_context,
        _const_repr,
        _normalize_key_expr,
        _,
        ParentAnnotator,
        UseVisitor,
    ) = _load()
    tree = ast.parse(code)
    annotator = ParentAnnotator()
    annotator.visit(tree)
    alias_to_param = {name: name for name in params}
    use_map = {
        name: ParamUse(direct_forward=set(), non_forward=False, current_aliases={name})
        for name in params
    }
    call_args: list[CallArgs] = []
    visitor = UseVisitor(
        parents=annotator.parents,
        use_map=use_map,
        call_args=call_args,
        alias_to_param=alias_to_param,
        is_test=False,
        strictness=strictness,
        const_repr=_const_repr,
        callee_name=_callee_name,
        call_args_factory=CallArgs,
        call_context=_call_context,
        return_aliases=return_aliases,
        normalize_key_expr=_normalize_key_expr,
    )
    return tree, visitor, use_map, call_args

# equivalent_witness — E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_alias_from_call_branches() -> None:
    code = "def f(a, b):\n    return a\n"
    tree, visitor, _, _ = _make_use_visitor(
        code,
        ["a", "b"],
        return_aliases={
            "identity": (["x"], ["x"]),
            "noop": (["x"], []),
        },
    )
    call = ast.parse("noop(a)").body[0].value
    assert visitor._alias_from_call(call) is None
    call = ast.parse("identity(*args)").body[0].value
    assert visitor._alias_from_call(call) is None
    call = ast.parse("identity(a, b)").body[0].value
    assert visitor._alias_from_call(call) is None
    call = ast.parse("identity(x=1)").body[0].value
    assert visitor._alias_from_call(call) is None
    call = ast.parse("identity(a)").body[0].value
    assert visitor._alias_from_call(call) == ["a"]
    visitor.visit(tree)

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_bind_sequence_and_return_alias_assignment() -> None:
    code = (
        "def f(a, b):\n"
        "    alias = identity(a)\n"
        "    extra: int = identity(a)\n"
        "    (x, y) = (a, b)\n"
    )
    tree, visitor, use_map, _ = _make_use_visitor(
        code,
        ["a", "b"],
        return_aliases={"identity": (["x"], ["x"])},
    )
    target = ast.parse("(x, y)").body[0].value
    rhs = ast.parse("(a,)").body[0].value
    assert visitor._bind_sequence(target, rhs) is False
    visitor.visit(tree)
    assert "alias" in visitor.alias_to_param
    assert use_map["a"].current_aliases

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_star_args_strictness() -> None:
    code = "def f(a, b):\n    foo(*a)\n    foo(**b)\n"
    tree, visitor, use_map, _ = _make_use_visitor(code, ["a", "b"], strictness="high")
    visitor.visit(tree)
    assert use_map["a"].non_forward is True
    assert use_map["b"].non_forward is True
    tree, visitor, use_map, _ = _make_use_visitor(code, ["a", "b"], strictness="low")
    visitor.visit(tree)
    assert ("args[*]", "arg[*]") in use_map["a"].direct_forward
    assert ("kwargs[*]", "kw[*]") in use_map["b"].direct_forward

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_attribute_and_subscript_aliases() -> None:
    code = (
        "def f(a, b):\n"
        "    alias = a\n"
        "    alias.attr = a\n"
        "    data = {}\n"
        "    data['k'] = b\n"
        "    foo(alias.attr)\n"
        "    foo(alias.other)\n"
        "    foo(alias['k'])\n"
        "    foo(alias[0])\n"
        "    foo(data['k'])\n"
        "    foo(alias.inner.attr)\n"
    )
    tree, visitor, use_map, _ = _make_use_visitor(code, ["a", "b"], strictness="low")
    visitor.visit(tree)
    assert any(slot == "arg[0]" for _, slot in use_map["a"].direct_forward)
    assert any(slot == "arg[0]" for _, slot in use_map["b"].direct_forward)
    assert use_map["a"].non_forward is True

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_check_write_clears_attr_and_key_aliases() -> None:
    tree, visitor, use_map, _ = _make_use_visitor("def f(a):\n    pass\n", ["a"])
    visitor._attr_alias_to_param[("obj", "field")] = "a"
    visitor._key_alias_to_param[("data", "k")] = "a"
    visitor._check_write(ast.Name(id="obj", ctx=ast.Store()))
    visitor._check_write(ast.Name(id="data", ctx=ast.Store()))
    assert ("obj", "field") not in visitor._attr_alias_to_param
    assert ("data", "k") not in visitor._key_alias_to_param
    assert use_map["a"].non_forward is True

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_bind_sequence_edge_cases() -> None:
    tree, visitor, _, _ = _make_use_visitor("def f(a, b):\n    pass\n", ["a", "b"])
    assert visitor._bind_sequence(
        ast.Name(id="x", ctx=ast.Store()),
        ast.Name(id="a", ctx=ast.Load()),
    ) is False
    target = ast.Tuple(
        elts=[ast.Name(id="x", ctx=ast.Store())],
        ctx=ast.Store(),
    )
    assert visitor._bind_sequence(target, ast.Name(id="a", ctx=ast.Load())) is False
    nested_target = ast.Tuple(
        elts=[
            ast.Tuple(
                elts=[ast.Name(id="x", ctx=ast.Store())],
                ctx=ast.Store(),
            )
        ],
        ctx=ast.Store(),
    )
    nested_rhs = ast.Tuple(
        elts=[
            ast.Tuple(
                elts=[ast.Name(id="a", ctx=ast.Load()), ast.Name(id="b", ctx=ast.Load())],
                ctx=ast.Load(),
            )
        ],
        ctx=ast.Load(),
    )
    assert visitor._bind_sequence(nested_target, nested_rhs) is True

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_starred_list_literal_records_forward_without_site() -> None:
    code = "def f(a):\n    return [*a]\n"
    tree, visitor, use_map, _ = _make_use_visitor(code, ["a"], strictness="low")
    visitor.visit(tree)
    assert ("args[*]", "arg[*]") in use_map["a"].direct_forward
    assert use_map["a"].forward_sites == {}

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_record_forward_skips_call_without_span() -> None:
    tree, visitor, use_map, _ = _make_use_visitor("def f(a):\n    pass\n", ["a"], strictness="low")
    call = ast.Call(func=ast.Name(id="g", ctx=ast.Load()), args=[], keywords=[])
    visitor._record_forward("a", "g", "arg[0]", call)
    assert ("g", "arg[0]") in use_map["a"].direct_forward
    assert use_map["a"].forward_sites == {}
    mismatch_target = ast.Tuple(
        elts=[ast.Name(id="x", ctx=ast.Store())],
        ctx=ast.Store(),
    )
    mismatch_rhs = ast.Tuple(
        elts=[ast.Constant(value=1)],
        ctx=ast.Load(),
    )
    assert visitor._bind_sequence(mismatch_target, mismatch_rhs) is True

@pytest.mark.parametrize(
    ("call_expr", "expected_alias"),
    [
        ("identity(**kwargs)", None),
        ("identity(extra=a)", None),
        ("identity(x=1)", None),
        ("identity(1)", None),
        ("identity(x=a)", ["a"]),
    ],
)
# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_alias_from_call_keyword_and_kw_aliases(
    call_expr: str,
    expected_alias: list[str] | None,
) -> None:
    tree, visitor, _, _ = _make_use_visitor(
        "def f(a):\n    return a\n",
        ["a"],
        return_aliases={"identity": (["x"], ["x"])},
    )
    call = ast.parse(call_expr).body[0].value
    assert visitor._alias_from_call(call) == expected_alias
    visitor.visit(tree)

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_bind_return_alias_rejects_invalid_targets() -> None:
    tree, visitor, _, _ = _make_use_visitor("def f(a):\n    return a\n", ["a"])
    targets = [
        ast.Name(id="x", ctx=ast.Store()),
        ast.Name(id="y", ctx=ast.Store()),
    ]
    assert visitor._bind_return_alias(targets, ["a"]) is False
    assert visitor._bind_return_alias([ast.Name(id="x", ctx=ast.Store())], ["a", "b"]) is False
    tuple_target = ast.Tuple(
        elts=[ast.Name(id="x", ctx=ast.Store()), ast.Constant(value=1)],
        ctx=ast.Store(),
    )
    assert visitor._bind_return_alias([tuple_target], ["a", "b"]) is False
    assert visitor._bind_return_alias([ast.Constant(value=1)], ["a"]) is False
    visitor.visit(tree)

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_annassign_edges() -> None:
    tree, visitor, use_map, _ = _make_use_visitor(
        "def f(a):\n    x: int = a\n",
        ["a"],
        return_aliases={"identity": (["x"], ["x"])},
    )
    visitor.visit(tree)
    assert "x" in visitor.alias_to_param
    assert use_map["a"].current_aliases
    visitor.visit_AnnAssign(
        ast.AnnAssign(
            target=ast.Name(id="y", ctx=ast.Store()),
            annotation=ast.Name(id="int", ctx=ast.Load()),
            value=None,
            simple=1,
        )
    )
    visitor.visit_AnnAssign(
        ast.AnnAssign(
            target=ast.Attribute(
                value=ast.Name(id="obj", ctx=ast.Load()),
                attr="attr",
                ctx=ast.Store(),
            ),
            annotation=ast.Name(id="int", ctx=ast.Load()),
            value=ast.Constant(value=1),
            simple=0,
        )
    )

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_visit_name_attribute_subscript_edges() -> None:
    tree, visitor, use_map, _ = _make_use_visitor("def f(a):\n    pass\n", ["a"])
    visitor.visit_Name(ast.Name(id="a", ctx=ast.Store()))
    assert visitor._root_name(ast.Constant(value=1)) is None
    visitor._attr_alias_to_param[("obj", "field")] = "a"
    visitor.alias_to_param["obj"] = "a"
    visitor._suspend_non_forward.add("a")
    visitor.visit_Attribute(
        ast.Attribute(
            value=ast.Name(id="obj", ctx=ast.Load()),
            attr="field",
            ctx=ast.Load(),
        )
    )
    visitor._suspend_non_forward.clear()
    visitor.visit_Attribute(
        ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="a", ctx=ast.Load()),
                attr="b",
                ctx=ast.Load(),
            ),
            attr="c",
            ctx=ast.Load(),
        )
    )
    assert use_map["a"].non_forward is True
    visitor.visit_Attribute(
        ast.Attribute(
            value=ast.Name(id="a", ctx=ast.Load()),
            attr="b",
            ctx=ast.Store(),
        )
    )
    visitor.visit_Subscript(
        ast.Subscript(
            value=ast.Name(id="a", ctx=ast.Load()),
            slice=ast.Name(id="k", ctx=ast.Load()),
            ctx=ast.Load(),
        )
    )
    visitor.visit_Subscript(
        ast.Subscript(
            value=ast.Name(id="a", ctx=ast.Load()),
            slice=ast.Constant(value="missing"),
            ctx=ast.Load(),
        )
    )
    visitor._key_alias_to_param[("a", ("literal", "str", "k"))] = "a"
    visitor._suspend_non_forward.add("a")
    visitor.visit_Subscript(
        ast.Subscript(
            value=ast.Name(id="a", ctx=ast.Load()),
            slice=ast.Constant(value="k"),
            ctx=ast.Load(),
        )
    )
    visitor._suspend_non_forward.clear()
    visitor.visit_Subscript(
        ast.Subscript(
            value=ast.Name(id="a", ctx=ast.Load()),
            slice=ast.Constant(value="k"),
            ctx=ast.Store(),
        )
    )


def test_record_unknown_key_without_span_marks_carrier_only() -> None:
    _, visitor, use_map, _ = _make_use_visitor("def f(a):\n    pass\n", ["a"])
    visitor._record_unknown_key("a", ast.Name(id="dynamic", ctx=ast.Load()))
    assert use_map["a"].unknown_key_carrier is True
    assert use_map["a"].unknown_key_sites == set()

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_collect_alias_sources_default() -> None:
    tree, visitor, _, _ = _make_use_visitor("def f(a):\n    pass\n", ["a"])
    assert visitor._collect_alias_sources(ast.Constant(value=1)) == set()
    visitor.visit(tree)

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_attr_and_subscript_slot_fallbacks_with_aliases() -> None:
    code = (
        "def f(a, data, obj):\n"
        "    obj.attr = a\n"
        "    data['k'] = a\n"
        "    foo(**obj.attr)\n"
        "    bar(**data['k'])\n"
    )
    tree, visitor, use_map, _ = _make_use_visitor(code, ["a", "data", "obj"])
    visitor.visit(tree)
    assert ("foo", "arg[?]") in use_map["a"].direct_forward
    assert ("bar", "arg[?]") in use_map["a"].direct_forward

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_subscript_non_name_root_sets_non_forward() -> None:
    code = "def f(a):\n    foo(a.b['k'])\n"
    tree, visitor, use_map, _ = _make_use_visitor(code, ["a"])
    visitor.visit(tree)
    assert use_map["a"].non_forward is True

# gabion:evidence E:call_cluster::test_visitors_edges.py::tests.test_visitors_edges._make_use_visitor
def test_subscript_and_attribute_slot_fallbacks() -> None:
    code = (
        "def f(a, data, obj):\n"
        "    foo(**data['k'])\n"
        "    bar(**obj.attr)\n"
    )
    tree, visitor, use_map, _ = _make_use_visitor(code, ["a", "data", "obj"])
    visitor.visit(tree)
    assert ("foo", "arg[?]") not in use_map["a"].direct_forward
    assert ("bar", "arg[?]") not in use_map["a"].direct_forward

def test_check_write_missing_use_map_entries() -> None:
    tree, visitor, use_map, _ = _make_use_visitor("def f(a):\n    pass\n", ["a"])
    visitor.alias_to_param["ghost"] = "missing"
    visitor._attr_alias_to_param[("obj", "field")] = "missing"
    visitor._key_alias_to_param[("data", "k")] = "missing"
    visitor._check_write(ast.Name(id="ghost", ctx=ast.Store()))
    visitor._check_write(ast.Name(id="obj", ctx=ast.Store()))
    visitor._check_write(ast.Name(id="data", ctx=ast.Store()))
    assert "missing" not in use_map
    visitor.visit(tree)

def test_bind_sequence_nested_mismatch_and_missing_use_map() -> None:
    tree, visitor, _, _ = _make_use_visitor("def f(a):\n    pass\n", ["a"])
    visitor.alias_to_param["rhs"] = "missing"
    lhs_nested = ast.Tuple(
        elts=[
            ast.Tuple(
                elts=[ast.Name(id="x", ctx=ast.Store())],
                ctx=ast.Store(),
            )
        ],
        ctx=ast.Store(),
    )
    rhs_nested = ast.Tuple(
        elts=[
            ast.Tuple(
                elts=[
                    ast.Name(id="rhs", ctx=ast.Load()),
                    ast.Name(id="other", ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            )
        ],
        ctx=ast.Load(),
    )
    assert visitor._bind_sequence(lhs_nested, rhs_nested) is True

    lhs = ast.Tuple(elts=[ast.Name(id="x", ctx=ast.Store())], ctx=ast.Store())
    rhs = ast.Tuple(elts=[ast.Name(id="rhs", ctx=ast.Load())], ctx=ast.Load())
    assert visitor._bind_sequence(lhs, rhs) is True
    visitor.visit(tree)

def test_bind_sequence_nested_match_branch_and_mark_non_forward_suspended() -> None:
    tree, visitor, use_map, _ = _make_use_visitor("def f(a):\n    pass\n", ["a"])
    lhs_nested = ast.Tuple(
        elts=[
            ast.Tuple(
                elts=[ast.Name(id="x", ctx=ast.Store())],
                ctx=ast.Store(),
            )
        ],
        ctx=ast.Store(),
    )
    rhs_nested = ast.Tuple(
        elts=[
            ast.Tuple(
                elts=[ast.Name(id="a", ctx=ast.Load())],
                ctx=ast.Load(),
            )
        ],
        ctx=ast.Load(),
    )
    assert visitor._bind_sequence(lhs_nested, rhs_nested) is True
    assert "x" in visitor.alias_to_param
    visitor._suspend_non_forward.add("a")
    assert visitor._mark_non_forward("a") is False
    assert use_map["a"].non_forward is False
    visitor.visit(tree)

def test_bind_return_alias_and_annassign_missing_use_map_entries() -> None:
    _, visitor, _, _ = _make_use_visitor(
        "def f(a):\n    x: int = identity(a)\n",
        ["a"],
        return_aliases={"identity": (["x"], ["x"])},
    )
    assert visitor._bind_return_alias([ast.Name(id="x", ctx=ast.Store())], ["missing"]) is True
    tuple_target = ast.Tuple(
        elts=[ast.Name(id="x", ctx=ast.Store()), ast.Name(id="y", ctx=ast.Store())],
        ctx=ast.Store(),
    )
    assert visitor._bind_return_alias([tuple_target], ["missing", "missing2"]) is True
    visitor.alias_to_param["a"] = "missing"
    annassign = ast.parse("x: int = identity(a)").body[0]
    assert isinstance(annassign, ast.AnnAssign)
    with pytest.raises(KeyError):
        visitor.visit_AnnAssign(annassign)

def test_attribute_and_subscript_non_forward_when_not_suspended() -> None:
    tree, visitor, use_map, _ = _make_use_visitor("def f(a):\n    pass\n", ["a"])
    visitor.visit_Attribute(
        ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="a", ctx=ast.Load()),
                attr="b",
                ctx=ast.Load(),
            ),
            attr="c",
            ctx=ast.Load(),
        )
    )
    visitor.visit_Attribute(
        ast.Attribute(
            value=ast.Name(id="a", ctx=ast.Load()),
            attr="unknown",
            ctx=ast.Load(),
        )
    )
    visitor.visit_Subscript(
        ast.Subscript(
            value=ast.Attribute(
                value=ast.Name(id="a", ctx=ast.Load()),
                attr="b",
                ctx=ast.Load(),
            ),
            slice=ast.Constant(value="k"),
            ctx=ast.Load(),
        )
    )
    visitor.visit_Subscript(
        ast.Subscript(
            value=ast.Name(id="a", ctx=ast.Load()),
            slice=ast.Name(id="idx", ctx=ast.Load()),
            ctx=ast.Load(),
        )
    )
    visitor.visit_Subscript(
        ast.Subscript(
            value=ast.Name(id="a", ctx=ast.Load()),
            slice=ast.Constant(value="k"),
            ctx=ast.Load(),
        )
    )
    assert use_map["a"].non_forward is True
    visitor.visit(tree)

def test_subscript_positional_slot_detection() -> None:
    code = (
        "def f(a):\n"
        "    data = {}\n"
        "    data['k'] = a\n"
        "    sink(data['k'])\n"
    )
    tree, visitor, use_map, _ = _make_use_visitor(code, ["a"], strictness="low")
    visitor.visit(tree)
    assert ("sink", "arg[0]") in use_map["a"].direct_forward


def test_subscript_name_bound_key_tracks_forward_and_unknown_key_state() -> None:
    code = (
        "def f(a):\n"
        "    k = 'k'\n"
        "    data = {}\n"
        "    data[k] = a\n"
        "    sink(data[k])\n"
        "    sink(data[get_key()])\n"
    )
    tree, visitor, use_map, _ = _make_use_visitor(code, ["a"], strictness="low")
    visitor.visit(tree)
    assert ("sink", "arg[0]") in use_map["a"].direct_forward
    assert use_map["a"].unknown_key_carrier is True
