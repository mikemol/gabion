# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=class_index_accumulator
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast


@dataclass(frozen=True)
class AccumulateClassIndexForTreeDeps:
    check_deadline_fn: Callable[[], None]
    parent_annotator_ctor: Callable[[], object]
    module_name_fn: Callable[..., str]
    enclosing_class_scopes_fn: Callable[..., list[str]]
    base_identifier_fn: Callable[..., object]
    class_info_ctor: Callable[..., object]


def _is_class_def_node(node: ast.AST) -> bool:
    return isinstance(node, ast.ClassDef)


def accumulate_class_index_for_tree(
    class_index: dict[str, object],
    path: Path,
    tree: ast.Module,
    *,
    project_root,
    deps: AccumulateClassIndexForTreeDeps,
) -> None:
    deps.check_deadline_fn()
    parents = deps.parent_annotator_ctor()
    parents.visit(tree)
    module = deps.module_name_fn(path, project_root)
    for class_node in filter(_is_class_def_node, ast.walk(tree)):
        deps.check_deadline_fn()
        scopes = deps.enclosing_class_scopes_fn(class_node, parents.parents)
        qual_parts = [module] if module else []
        qual_parts.extend(scopes)
        qual_parts.append(class_node.name)
        qual = ".".join(qual_parts)
        bases: list[str] = []
        for base in class_node.bases:
            deps.check_deadline_fn()
            base_name = deps.base_identifier_fn(base)
            if base_name:
                bases.append(base_name)
        methods: set[str] = set()
        for stmt in class_node.body:
            deps.check_deadline_fn()
            stmt_type = type(stmt)
            if stmt_type in {ast.FunctionDef, ast.AsyncFunctionDef}:
                methods.add(cast(ast.FunctionDef | ast.AsyncFunctionDef, stmt).name)
        class_index[qual] = deps.class_info_ctor(
            qual=qual,
            module=module,
            bases=bases,
            methods=methods,
        )
