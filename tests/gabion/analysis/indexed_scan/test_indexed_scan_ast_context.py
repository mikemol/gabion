from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from gabion.analysis.indexed_scan.ast.ast_context import (
    PathAstContextBuildStatus, ancestor_if_names, build_path_ast_context, enclosing_function_context)


@dataclass
class _ParentAnnotator:
    parents: dict[ast.AST, ast.AST]

    def __init__(self) -> None:
        self.parents = {}

    def visit(self, tree: ast.AST) -> None:
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                self.parents[child] = parent


def _collect_functions(tree: ast.AST) -> list[ast.AST]:
    return [node for node in ast.walk(tree) if type(node) in {ast.FunctionDef, ast.AsyncFunctionDef}]


def _param_names(fn: ast.AST, _ignore_params: set[str]) -> list[str]:
    assert isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef)
    return [arg.arg for arg in fn.args.args]


def _param_annotations(fn: ast.AST, _ignore_params: set[str]) -> dict[str, object]:
    assert isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef)
    out: dict[str, object] = {}
    for arg in fn.args.args:
        out[arg.arg] = ast.unparse(arg.annotation) if arg.annotation is not None else None
    return out


def _normalize_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _check_deadline() -> None:
    return None


def _enclosing_function_node(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST | None:
    current = parents.get(node)
    while current is not None:
        if type(current) in {ast.FunctionDef, ast.AsyncFunctionDef}:
            return current
        current = parents.get(current)
    return None


def _enclosing_scopes(_fn: ast.AST, _parents: dict[ast.AST, ast.AST]) -> list[str]:
    return ["scope"]


def _function_key(scopes: list[str], name: str) -> str:
    return ".".join([*scopes, name])


def _names_in_expr(expr: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(expr) if type(node) is ast.Name}


#
# gabion:evidence E:function_site::indexed_scan/ast_context.py::gabion.analysis.indexed_scan.ast_context.build_path_ast_context E:function_site::indexed_scan/ast_context.py::gabion.analysis.indexed_scan.ast_context.enclosing_function_context
def test_build_path_ast_context_and_enclosing_context(tmp_path: Path) -> None:
    module = tmp_path / "mod.py"
    module.write_text("def f(x: int):\n    if x:\n        return x\n")

    context = build_path_ast_context(
        module,
        project_root=tmp_path,
        ignore_params=set(),
        check_deadline_fn=_check_deadline,
        parent_annotator_factory=_ParentAnnotator,
        collect_functions_fn=_collect_functions,
        param_names_fn=_param_names,
        normalize_snapshot_path_fn=_normalize_path,
        param_annotations_fn=_param_annotations,
    )
    assert context.status is PathAstContextBuildStatus.PARSED
    context_payload = context.contexts[0]
    assert context_payload.path_value == "mod.py"
    assert len(context_payload.params_by_fn) == 1

    fn = next(iter(context_payload.params_by_fn))
    if_stmt = next(node for node in ast.walk(context_payload.tree) if type(node) is ast.If)
    function, params, annotations = enclosing_function_context(
        if_stmt,
        parents=context_payload.parents,
        params_by_fn=context_payload.params_by_fn,
        param_annotations_by_fn=context_payload.param_annotations_by_fn,
        enclosing_function_node_fn=_enclosing_function_node,
        enclosing_scopes_fn=_enclosing_scopes,
        function_key_fn=_function_key,
    )
    assert function == "scope.f"
    assert params == {"x"}
    assert annotations == {"x": "int"}

    module_expr = ast.parse("x").body[0].value
    module_fn, module_params, module_annots = enclosing_function_context(
        module_expr,
        parents={},
        params_by_fn={fn: {"x"}},
        param_annotations_by_fn={fn: {"x": "int"}},
        enclosing_function_node_fn=_enclosing_function_node,
        enclosing_scopes_fn=_enclosing_scopes,
        function_key_fn=_function_key,
    )
    assert module_fn == "<module>"
    assert module_params == set()
    assert module_annots == {}


#
# gabion:evidence E:function_site::indexed_scan/ast_context.py::gabion.analysis.indexed_scan.ast_context.build_path_ast_context
def test_build_path_ast_context_syntax_error_returns_none(tmp_path: Path) -> None:
    module = tmp_path / "bad.py"
    module.write_text("def oops(:\n")
    context = build_path_ast_context(
        module,
        project_root=tmp_path,
        ignore_params=set(),
        check_deadline_fn=_check_deadline,
        parent_annotator_factory=_ParentAnnotator,
        collect_functions_fn=_collect_functions,
        param_names_fn=_param_names,
        normalize_snapshot_path_fn=_normalize_path,
        param_annotations_fn=_param_annotations,
    )
    assert context.status is PathAstContextBuildStatus.SYNTAX_ERROR
    assert context.contexts == ()


#
# gabion:evidence E:function_site::indexed_scan/ast_context.py::gabion.analysis.indexed_scan.ast_context.ancestor_if_names
def test_ancestor_if_names_collects_nested_conditions() -> None:
    tree = ast.parse(
        "def f(a, b):\n"
        "    if a:\n"
        "        if b:\n"
        "            return 1\n"
    )
    parent = _ParentAnnotator()
    parent.visit(tree)
    return_node = next(node for node in ast.walk(tree) if type(node) is ast.Return)
    names = ancestor_if_names(
        return_node,
        parents=parent.parents,
        names_in_expr_fn=_names_in_expr,
        check_deadline_fn=_check_deadline,
    )
    assert names == {"a", "b"}
