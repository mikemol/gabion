from __future__ import annotations

from pathlib import Path
import ast

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import dataflow_audit as da

    return da

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._eval_bool_expr::env,expr E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._eval_value_expr::env,expr E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._node_in_block::node
def test_deadness_helper_evaluators_cover_edges(tmp_path: Path) -> None:
    da = _load()

    # _node_in_block: exercise subtree scanning path (not direct stmt identity).
    tree = ast.parse("if x:\n    raise ValueError(y)\n")
    if_node = tree.body[0]
    assert isinstance(if_node, ast.If)
    raise_stmt = if_node.body[0]
    assert isinstance(raise_stmt, ast.Raise)
    name_y = raise_stmt.exc.args[0]
    assert isinstance(name_y, ast.Name)
    assert da._node_in_block(name_y, if_node.body) is True
    other = ast.parse("z").body[0].value
    assert da._node_in_block(other, if_node.body) is False

    # _names_in_expr
    names = da._names_in_expr(ast.parse("a and b").body[0].value)
    assert names == {"a", "b"}

    # _eval_value_expr
    value_outcome = da._eval_value_expr(ast.parse("1").body[0].value, {})
    assert value_outcome.is_unknown() is False
    assert value_outcome.value == 1
    assert da._eval_value_expr(ast.parse("x").body[0].value, {}).is_unknown() is True
    neg_outcome = da._eval_value_expr(ast.parse("-1").body[0].value, {})
    assert neg_outcome.is_unknown() is False
    assert neg_outcome.value == -1
    plus_outcome = da._eval_value_expr(ast.parse("+2").body[0].value, {})
    assert plus_outcome.is_unknown() is False
    assert plus_outcome.value == 2
    assert da._eval_value_expr(ast.parse("b'hi'").body[0].value, {}).is_unknown() is True
    assert da._eval_value_expr(ast.parse("foo()").body[0].value, {}).is_unknown() is True

    # _eval_bool_expr: constants + unknown names.
    bool_outcome = da._eval_bool_expr(ast.parse("0").body[0].value, {})
    assert bool_outcome.is_unknown() is False
    assert bool_outcome.as_bool() is False
    assert da._eval_bool_expr(ast.parse("x").body[0].value, {}).is_unknown() is True

    # not <unknown> => unknown; not 0 => True
    assert da._eval_bool_expr(ast.parse("not x").body[0].value, {}).is_unknown() is True
    not_zero_outcome = da._eval_bool_expr(ast.parse("not 0").body[0].value, {})
    assert not_zero_outcome.is_unknown() is False
    assert not_zero_outcome.as_bool() is True

    # BoolOp AND: false dominates even if earlier terms are unknown.
    and_false = da._eval_bool_expr(ast.parse("x and 0").body[0].value, {})
    assert and_false.is_unknown() is False
    assert and_false.as_bool() is False
    assert da._eval_bool_expr(ast.parse("x and 1").body[0].value, {}).is_unknown() is True

    # BoolOp OR: true dominates even if later terms are unknown.
    or_true = da._eval_bool_expr(ast.parse("1 or x").body[0].value, {})
    assert or_true.is_unknown() is False
    assert or_true.as_bool() is True
    assert da._eval_bool_expr(ast.parse("x or 0").body[0].value, {}).is_unknown() is True
    or_false = da._eval_bool_expr(ast.parse("0 or 0").body[0].value, {})
    assert or_false.is_unknown() is False
    assert or_false.as_bool() is False

    # Compare: unknown side => unknown; cover comparison ops.
    assert da._eval_bool_expr(ast.parse("x == 1").body[0].value, {}).is_unknown() is True
    assert da._eval_bool_expr(ast.parse("1 == 1").body[0].value, {}).as_bool() is True
    assert da._eval_bool_expr(ast.parse("1 != 2").body[0].value, {}).as_bool() is True
    assert da._eval_bool_expr(ast.parse("1 < 2").body[0].value, {}).as_bool() is True
    assert da._eval_bool_expr(ast.parse("1 <= 1").body[0].value, {}).as_bool() is True
    assert da._eval_bool_expr(ast.parse("2 > 1").body[0].value, {}).as_bool() is True
    assert da._eval_bool_expr(ast.parse("2 >= 2").body[0].value, {}).as_bool() is True
    assert da._eval_bool_expr(ast.parse("foo()").body[0].value, {}).is_unknown() is True

    # _branch_reachability_under_env: no constraints => None
    assert da._branch_reachability_under_env(ast.parse("1").body[0].value, {}, {}) is None

    # Reachability for body + orelse branches + unknown env.
    mod = ast.parse(
        "def f(x):\n"
        "    if x:\n"
        "        raise ValueError()\n"
    )
    parent = da.ParentAnnotator()
    parent.visit(mod)
    if_stmt = mod.body[0].body[0]
    raise_body = if_stmt.body[0]
    assert da._branch_reachability_under_env(raise_body, parent.parents, {"x": 1}) is True
    assert da._branch_reachability_under_env(raise_body, parent.parents, {"x": 0}) is False
    assert da._branch_reachability_under_env(raise_body, parent.parents, {}) is None

    mod = ast.parse(
        "def g(x):\n"
        "    if x:\n"
        "        return 0\n"
        "    else:\n"
        "        raise ValueError()\n"
    )
    parent = da.ParentAnnotator()
    parent.visit(mod)
    if_stmt = mod.body[0].body[0]
    raise_else = if_stmt.orelse[0]
    assert da._branch_reachability_under_env(raise_else, parent.parents, {"x": 0}) is True
    assert da._branch_reachability_under_env(raise_else, parent.parents, {"x": 1}) is False

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorator_matches::allowlist,name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._dead_env_map::deadness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_never_marker_raise::exception_name,never_exceptions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_type_name::expr E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_param_names::expr,params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::stale_72be7680c751
def test_exception_obligation_deadness_parsing_skips_invalid_entries(tmp_path: Path) -> None:
    da = _load()
    obligations = da._collect_exception_obligations(
        [],
        project_root=tmp_path,
        ignore_params=set(),
        deadness_witnesses=[
            {"path": "x.py", "function": "f", "bundle": "not-a-list"},
            {"path": "x.py", "function": "f", "bundle": []},
            {"path": "x.py", "function": "f", "bundle": ["a"], "environment": "nope"},
            {"path": "x.py", "function": "f", "bundle": ["a"], "environment": {"a": 1}},
            {"path": "x.py", "function": "f", "bundle": ["a"], "environment": {"a": "NOPE("}},
        ],
    )
    assert obligations == []

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorator_matches::allowlist,name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._dead_env_map::deadness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_never_marker_raise::exception_name,never_exceptions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_type_name::expr E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_param_names::expr,params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::strictness
def test_exception_obligations_deadness_selection_skips_unknown_names(tmp_path: Path) -> None:
    da = _load()
    module = tmp_path / "mod.py"
    module.write_text(
        "def callee(flag):\n"
        "    if a and flag != 0:\n"
        "        raise RuntimeError('boom')\n"
        "    return 0\n"
        "\n"
        "def caller():\n"
        "    return callee(0)\n"
    )
    deadness = da.analyze_deadness_flow_repo(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    obligations = da._collect_exception_obligations(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        deadness_witnesses=deadness,
    )
    assert any(entry.get("status") == "DEAD" for entry in obligations)

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decorator_matches::allowlist,name E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._dead_env_map::deadness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_never_marker_raise::exception_name,never_exceptions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_type_name::expr E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_param_names::expr,params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::stale_5b7a7d4bac9d
def test_exception_obligations_skip_never_marker_raise(tmp_path: Path) -> None:
    da = _load()
    module = tmp_path / "mod.py"
    module.write_text(
        "from gabion.exceptions import NeverThrown\n"
        "\n"
        "def never():\n"
        "    raise NeverThrown('boom')\n"
    )
    obligations = da._collect_exception_obligations(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        never_exceptions={"NeverThrown"},
    )
    assert obligations == []
