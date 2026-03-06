# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from gabion.analysis.foundation.json_types import JSONValue


@dataclass(frozen=True)
class PathAstContext:
    tree: ast.AST
    parents: dict[ast.AST, ast.AST]
    path_value: str
    params_by_fn: dict[ast.AST, set[str]]
    param_annotations_by_fn: dict[ast.AST, dict[str, JSONValue]]


class PathAstContextBuildStatus(StrEnum):
    PARSED = "parsed"
    SYNTAX_ERROR = "syntax_error"


@dataclass(frozen=True)
class PathAstContextBuildResult:
    status: PathAstContextBuildStatus
    contexts: tuple[PathAstContext, ...]


def build_path_ast_context(
    path: Path,
    *,
    project_root,
    ignore_params: set[str],
    check_deadline_fn: Callable[[], None],
    parent_annotator_factory: Callable[[], object],
    collect_functions_fn: Callable[[ast.AST], list[ast.AST]],
    param_names_fn: Callable[..., list[str]],
    normalize_snapshot_path_fn: Callable[..., str],
    param_annotations_fn: Callable[..., dict[str, JSONValue]],
) -> PathAstContextBuildResult:
    check_deadline_fn()
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return PathAstContextBuildResult(
            status=PathAstContextBuildStatus.SYNTAX_ERROR,
            contexts=(),
        )

    parent_annotator = parent_annotator_factory()
    parent_annotator.visit(tree)
    parents = parent_annotator.parents

    params_by_fn: dict[ast.AST, set[str]] = {}
    param_annotations_by_fn: dict[ast.AST, dict[str, JSONValue]] = {}
    for fn in collect_functions_fn(tree):
        check_deadline_fn()
        params_by_fn[fn] = set(param_names_fn(fn, ignore_params))
        param_annotations_by_fn[fn] = param_annotations_fn(fn, ignore_params)

    return PathAstContextBuildResult(
        status=PathAstContextBuildStatus.PARSED,
        contexts=(
            PathAstContext(
                tree=tree,
                parents=parents,
                path_value=normalize_snapshot_path_fn(path, project_root),
                params_by_fn=params_by_fn,
                param_annotations_by_fn=param_annotations_by_fn,
            ),
        ),
    )


def enclosing_function_context(
    node: ast.AST,
    *,
    parents: dict[ast.AST, ast.AST],
    params_by_fn: dict[ast.AST, set[str]],
    param_annotations_by_fn: dict[ast.AST, dict[str, JSONValue]],
    enclosing_function_node_fn: Callable[..., object],
    enclosing_scopes_fn: Callable[..., list[str]],
    function_key_fn: Callable[..., str],
) -> tuple[str, set[str], dict[str, JSONValue]]:
    fn_node = enclosing_function_node_fn(node, parents)
    if fn_node is None:
        return "<module>", set(), {}
    scopes = enclosing_scopes_fn(fn_node, parents)
    return (
        function_key_fn(scopes, fn_node.name),
        params_by_fn.get(fn_node, set()),
        param_annotations_by_fn.get(fn_node, {}),
    )


def ancestor_if_names(
    node: ast.AST,
    *,
    parents: dict[ast.AST, ast.AST],
    names_in_expr_fn: Callable[..., set[str]],
    check_deadline_fn: Callable[[], None],
) -> set[str]:
    names: set[str] = set()
    current = parents.get(node)
    while current is not None:
        check_deadline_fn()
        if type(current) is ast.If:
            names.update(names_in_expr_fn(current.test))
        current = parents.get(current)
    return names


__all__ = [
    "PathAstContextBuildResult",
    "PathAstContextBuildStatus",
    "PathAstContext",
    "ancestor_if_names",
    "build_path_ast_context",
    "enclosing_function_context",
]
