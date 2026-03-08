from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from gabion.analysis.foundation.json_types import ParseFailureWitnesses
from gabion.analysis.indexed_scan.index.analysis_index_stage_cache import (
    AnalysisIndexStageCacheFn,
)


@dataclass(frozen=True)
class CollectDataclassRegistryDeps:
    check_deadline_fn: Callable[[], None]
    stage_cache_spec_ctor: Callable[..., object]
    parse_module_stage_dataclass_registry: object
    parse_stage_cache_key_fn: Callable[..., object]
    empty_cache_semantic_context: object
    dataclass_registry_for_tree_fn: Callable[..., dict[str, list[str]]]
    parse_module_tree_fn: Callable[..., object]


@dataclass(frozen=True)
class DataclassRegistryForTreeDeps:
    check_deadline_fn: Callable[[], None]
    module_name_fn: Callable[..., str]
    simple_store_name_fn: Callable[..., object]
    decorator_text_fn: Callable[[ast.AST], str]


def collect_dataclass_registry(
    paths: list[Path],
    *,
    project_root,
    parse_failure_witnesses: ParseFailureWitnesses,
    analysis_index = None,
    stage_cache_fn: AnalysisIndexStageCacheFn[object],
    deps: CollectDataclassRegistryDeps,
) -> dict[str, list[str]]:
    deps.check_deadline_fn()
    registry: dict[str, list[str]] = {}
    if analysis_index is not None:
        registry_by_path = stage_cache_fn(
            analysis_index,
            paths,
            spec=deps.stage_cache_spec_ctor(
                stage=deps.parse_module_stage_dataclass_registry,
                cache_key=deps.parse_stage_cache_key_fn(
                    stage=deps.parse_module_stage_dataclass_registry,
                    cache_context=deps.empty_cache_semantic_context,
                    config_subset={
                        "project_root": str(project_root) if project_root is not None else "",
                    },
                    detail="dataclass_registry",
                ),
                build=lambda tree, path: deps.dataclass_registry_for_tree_fn(
                    path,
                    tree,
                    project_root=project_root,
                ),
            ),
            parse_failure_witnesses=parse_failure_witnesses,
        )
        for entries in registry_by_path.values():
            deps.check_deadline_fn()
            if entries is not None:
                registry.update(entries)
        return registry
    for path in paths:
        deps.check_deadline_fn()
        tree = deps.parse_module_tree_fn(
            path,
            stage=deps.parse_module_stage_dataclass_registry,
            parse_failure_witnesses=parse_failure_witnesses,
        )
        if tree is not None:
            registry.update(
                deps.dataclass_registry_for_tree_fn(
                    path,
                    tree,
                    project_root=project_root,
                )
            )
    return registry


def dataclass_registry_for_tree(
    path: Path,
    tree: ast.AST,
    *,
    project_root = None,
    deps: DataclassRegistryForTreeDeps,
) -> dict[str, list[str]]:
    deps.check_deadline_fn()
    registry: dict[str, list[str]] = {}
    module = deps.module_name_fn(path, project_root)
    for node in ast.walk(tree):
        deps.check_deadline_fn()
        if type(node) is not ast.ClassDef:
            continue
        class_node = cast(ast.ClassDef, node)
        decorators = {deps.decorator_text_fn(dec) for dec in class_node.decorator_list}
        if not any("dataclass" in dec for dec in decorators):
            continue
        fields: list[str] = []
        for stmt in class_node.body:
            deps.check_deadline_fn()
            stmt_type = type(stmt)
            if stmt_type is ast.AnnAssign:
                ann_stmt = cast(ast.AnnAssign, stmt)
                raw_name = deps.simple_store_name_fn(ann_stmt.target)
                if type(raw_name) is str:
                    fields.append(raw_name)
            elif stmt_type is ast.Assign:
                assign_stmt = cast(ast.Assign, stmt)
                for target in assign_stmt.targets:
                    deps.check_deadline_fn()
                    raw_name = deps.simple_store_name_fn(target)
                    if type(raw_name) is str:
                        fields.append(raw_name)
        if not fields:
            continue
        if module:
            registry[f"{module}.{class_node.name}"] = fields
        else:
            registry[class_node.name] = fields
    return registry
