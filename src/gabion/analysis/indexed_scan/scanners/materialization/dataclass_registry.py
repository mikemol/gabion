# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataclass_registry
from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import singledispatch
from itertools import chain
from pathlib import Path
from typing import TypeGuard

from gabion.analysis.foundation.json_types import ParseFailureWitnesses
from gabion.analysis.indexed_scan.index.analysis_index_stage_cache import (
    AnalysisIndexStageCacheFn,
)
from gabion.invariants import never


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


def _iter_dataclass_field_targets(
    class_node: ast.ClassDef,
) -> Iterator[ast.Name]:
    for stmt in class_node.body:
        yield from _dataclass_field_targets_for_stmt(stmt)


def _is_class_def_node(node: ast.AST) -> TypeGuard[ast.ClassDef]:
    return isinstance(node, ast.ClassDef)


def _class_def_nodes(tree: ast.AST) -> Iterator[ast.ClassDef]:
    return filter(_is_class_def_node, ast.walk(tree))


@singledispatch
def _dataclass_field_targets_for_stmt(stmt: ast.AST) -> Iterator[ast.Name]:
    del stmt
    return iter(())


@_dataclass_field_targets_for_stmt.register(ast.AnnAssign)
def _dataclass_field_targets_for_ann_assign(stmt: ast.AnnAssign) -> Iterator[ast.Name]:
    return _dataclass_assign_targets(stmt.target)


@_dataclass_field_targets_for_stmt.register(ast.Assign)
def _dataclass_field_targets_for_assign(stmt: ast.Assign) -> Iterator[ast.Name]:
    return chain.from_iterable(
        _dataclass_assign_targets(target) for target in stmt.targets
    )


@singledispatch
def _dataclass_assign_targets(target: ast.AST) -> Iterator[ast.Name]:
    del target
    return iter(())


@_dataclass_assign_targets.register(ast.Name)
def _dataclass_assign_targets_name(target: ast.Name) -> Iterator[ast.Name]:
    return _singleton_name_iterator(target)


def _singleton_name_iterator(target: ast.Name) -> Iterator[ast.Name]:
    yield target


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
    for class_node in _class_def_nodes(tree):
        deps.check_deadline_fn()
        decorators = {deps.decorator_text_fn(dec) for dec in class_node.decorator_list}
        if not any("dataclass" in dec for dec in decorators):
            continue
        fields: list[str] = []
        for target in _iter_dataclass_field_targets(class_node):
            deps.check_deadline_fn()
            raw_name = deps.simple_store_name_fn(target)
            match raw_name:
                case str() as field_name:
                    fields.append(field_name)
                case _:
                    never("unreachable wildcard match fall-through")
        if not fields:
            continue
        if module:
            registry[f"{module}.{class_node.name}"] = fields
        else:
            registry[class_node.name] = fields
    return registry
