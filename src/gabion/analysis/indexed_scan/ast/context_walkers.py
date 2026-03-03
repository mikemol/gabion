# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from gabion.analysis.foundation.json_types import JSONValue

from gabion.analysis.indexed_scan.ast.ast_context import PathAstContext, PathAstContextBuildStatus, build_path_ast_context


def empty_param_annotations(
    _fn: ast.AST,
    _ignore_params: set[str],
) -> dict[str, JSONValue]:
    return {}


def iter_parsed_path_contexts(
    paths: Iterable[Path],
    *,
    project_root,
    ignore_params: set[str],
    check_deadline_fn: Callable[[], None],
    parent_annotator_factory: Callable[[], object],
    collect_functions_fn: Callable[[ast.AST], list[ast.AST]],
    param_names_fn: Callable[..., list[str]],
    normalize_snapshot_path_fn: Callable[..., str],
    param_annotations_fn: Callable[..., dict[str, JSONValue]],
) -> Iterator[PathAstContext]:
    for path in paths:
        check_deadline_fn()
        result = build_path_ast_context(
            path,
            project_root=project_root,
            ignore_params=ignore_params,
            check_deadline_fn=check_deadline_fn,
            parent_annotator_factory=parent_annotator_factory,
            collect_functions_fn=collect_functions_fn,
            param_names_fn=param_names_fn,
            normalize_snapshot_path_fn=normalize_snapshot_path_fn,
            param_annotations_fn=param_annotations_fn,
        )
        if result.status is not PathAstContextBuildStatus.PARSED:
            continue
        for context in result.contexts:
            check_deadline_fn()
            yield context


def iter_nodes_of_types(
    tree: ast.AST,
    node_types: tuple[type[ast.AST], ...],
    *,
    check_deadline_fn: Callable[[], None],
) -> Iterator[ast.AST]:
    for node in ast.walk(tree):
        check_deadline_fn()
        if type(node) in node_types:
            yield node


__all__ = [
    "empty_param_annotations",
    "iter_nodes_of_types",
    "iter_parsed_path_contexts",
]
