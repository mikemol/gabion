from __future__ import annotations

from pathlib import Path
import ast
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


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
    assert da._eval_value_expr(ast.parse("1").body[0].value, {}) == 1
    assert da._eval_value_expr(ast.parse("x").body[0].value, {}) is None
    assert da._eval_value_expr(ast.parse("-1").body[0].value, {}) == -1
    assert da._eval_value_expr(ast.parse("+2").body[0].value, {}) == 2
    assert da._eval_value_expr(ast.parse("foo()").body[0].value, {}) is None

    # _eval_bool_expr: constants + unknown names.
    assert da._eval_bool_expr(ast.parse("0").body[0].value, {}) is False
    assert da._eval_bool_expr(ast.parse("x").body[0].value, {}) is None

    # not <unknown> => unknown; not 0 => True
    assert da._eval_bool_expr(ast.parse("not x").body[0].value, {}) is None
    assert da._eval_bool_expr(ast.parse("not 0").body[0].value, {}) is True

    # BoolOp AND: false dominates even if earlier terms are unknown.
    assert da._eval_bool_expr(ast.parse("x and 0").body[0].value, {}) is False
    assert da._eval_bool_expr(ast.parse("x and 1").body[0].value, {}) is None

    # BoolOp OR: true dominates even if later terms are unknown.
    assert da._eval_bool_expr(ast.parse("1 or x").body[0].value, {}) is True
    assert da._eval_bool_expr(ast.parse("x or 0").body[0].value, {}) is None
    assert da._eval_bool_expr(ast.parse("0 or 0").body[0].value, {}) is False

    # Compare: unknown side => unknown; cover comparison ops.
    assert da._eval_bool_expr(ast.parse("x == 1").body[0].value, {}) is None
    assert da._eval_bool_expr(ast.parse("1 == 1").body[0].value, {}) is True
    assert da._eval_bool_expr(ast.parse("1 != 2").body[0].value, {}) is True
    assert da._eval_bool_expr(ast.parse("1 < 2").body[0].value, {}) is True
    assert da._eval_bool_expr(ast.parse("1 <= 1").body[0].value, {}) is True
    assert da._eval_bool_expr(ast.parse("2 > 1").body[0].value, {}) is True
    assert da._eval_bool_expr(ast.parse("2 >= 2").body[0].value, {}) is True
    assert da._eval_bool_expr(ast.parse("foo()").body[0].value, {}) is None

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
