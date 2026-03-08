from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from gabion.analysis.foundation.json_types import JSONObject


@dataclass(frozen=True)
class MaterializeStructuredSuiteSitesForTreeDeps:
    check_deadline_fn: Callable[[], None]
    parent_annotator_factory: object
    module_name_fn: Callable[..., str]
    collect_functions_fn: Callable[[ast.Module], Sequence[object]]
    enclosing_scopes_fn: Callable[..., Sequence[str]]
    node_span_fn: Callable[[ast.AST], object]
    materialize_statement_suite_contains_fn: Callable[..., None]


def materialize_structured_suite_sites_for_tree(
    *,
    forest,
    path: Path,
    tree: ast.Module,
    project_root,
    deps: MaterializeStructuredSuiteSitesForTreeDeps,
) -> None:
    deps.check_deadline_fn()
    parent_annotator = deps.parent_annotator_factory()
    parent_annotator.visit(tree)
    parent_map = parent_annotator.parents
    module = deps.module_name_fn(path, project_root)
    path_name = path.name
    for fn in deps.collect_functions_fn(tree):
        deps.check_deadline_fn()
        scopes = deps.enclosing_scopes_fn(fn, parent_map)
        qual_parts = [module] if module else []
        if scopes:
            qual_parts.extend(scopes)
        qual_parts.append(fn.name)
        qual = ".".join(qual_parts)
        function_span = deps.node_span_fn(fn)
        function_suite = forest.add_suite_site(
            path_name,
            qual,
            "function",
            span=function_span,
        )
        parent_suite = function_suite
        if function_span is not None:
            parent_suite = forest.add_suite_site(
                path_name,
                qual,
                "function_body",
                span=function_span,
                parent=function_suite,
            )
        deps.materialize_statement_suite_contains_fn(
            forest=forest,
            path_name=path_name,
            qual=qual,
            statements=fn.body,
            parent_suite=parent_suite,
        )


@dataclass(frozen=True)
class MaterializeStructuredSuiteSitesDeps:
    check_deadline_fn: Callable[[], None]
    iter_monotonic_paths_fn: Callable[..., list[Path]]
    analysis_index_module_trees_fn: Callable[..., dict[Path, object]]
    parse_module_tree_fn: Callable[..., object]
    parse_module_stage_suite_containment: object
    materialize_structured_suite_sites_for_tree_fn: Callable[..., None]


def materialize_structured_suite_sites(
    *,
    forest,
    file_paths: list[Path],
    project_root,
    parse_failure_witnesses: list[JSONObject],
    analysis_index=None,
    deps: MaterializeStructuredSuiteSitesDeps,
) -> None:
    deps.check_deadline_fn()
    ordered_file_paths = deps.iter_monotonic_paths_fn(
        file_paths,
        source="_materialize_structured_suite_sites.file_paths",
    )
    if analysis_index is not None:
        trees = deps.analysis_index_module_trees_fn(
            analysis_index,
            ordered_file_paths,
            stage=deps.parse_module_stage_suite_containment,
            parse_failure_witnesses=parse_failure_witnesses,
        )
    else:
        trees = {}
        for path in ordered_file_paths:
            deps.check_deadline_fn()
            tree = deps.parse_module_tree_fn(
                path,
                stage=deps.parse_module_stage_suite_containment,
                parse_failure_witnesses=parse_failure_witnesses,
            )
            trees[path] = tree
    for path in deps.iter_monotonic_paths_fn(
        trees,
        source="_materialize_structured_suite_sites.trees",
    ):
        deps.check_deadline_fn()
        tree = trees[path]
        if tree is not None:
            deps.materialize_structured_suite_sites_for_tree_fn(
                forest=forest,
                path=path,
                tree=tree,
                project_root=project_root,
            )
