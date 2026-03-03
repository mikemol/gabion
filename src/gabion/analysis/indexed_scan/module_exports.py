# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ModuleExportsCollectDeps:
    check_deadline_fn: Callable[[], None]
    string_list_fn: Callable[..., object]
    target_names_fn: Callable[..., set[str]]


def collect_module_exports(
    tree: ast.AST,
    *,
    module_name: str,
    import_map: dict[str, str],
    deps: ModuleExportsCollectDeps,
) -> tuple[set[str], dict[str, str]]:
    deps.check_deadline_fn()
    explicit_all: list[str] = []
    has_explicit_all = False
    for stmt in getattr(tree, "body", []):
        deps.check_deadline_fn()
        stmt_type = type(stmt)
        if stmt_type is ast.Assign:
            assign_stmt = stmt
            targets = assign_stmt.targets
            if any(type(target) is ast.Name and target.id == "__all__" for target in targets):
                values = deps.string_list_fn(assign_stmt.value)
                if values is not None:
                    explicit_all = list(values)
                    has_explicit_all = True
        elif stmt_type is ast.AnnAssign:
            ann_assign = stmt
            target = ann_assign.target
            if type(target) is ast.Name and target.id == "__all__":
                values = deps.string_list_fn(ann_assign.value) if ann_assign.value is not None else None
                if values is not None:
                    explicit_all = list(values)
                    has_explicit_all = True
        elif stmt_type is ast.AugAssign:
            aug_assign = stmt
            target = aug_assign.target
            if (
                type(target) is ast.Name
                and target.id == "__all__"
                and type(aug_assign.op) is ast.Add
            ):
                values = deps.string_list_fn(aug_assign.value)
                if values is not None:
                    if not has_explicit_all:
                        has_explicit_all = True
                        explicit_all = []
                    explicit_all.extend(values)

    local_defs: set[str] = set()
    for stmt in getattr(tree, "body", []):
        deps.check_deadline_fn()
        stmt_type = type(stmt)
        if stmt_type in {ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef}:
            stmt_name = str(getattr(stmt, "name", ""))
            if stmt_name and not stmt_name.startswith("_"):
                local_defs.add(stmt_name)
        elif stmt_type is ast.Assign:
            assign_stmt = stmt
            for target in assign_stmt.targets:
                deps.check_deadline_fn()
                local_defs.update(
                    name for name in deps.target_names_fn(target) if not name.startswith("_")
                )
        elif stmt_type is ast.AnnAssign:
            ann_assign = stmt
            local_defs.update(
                name
                for name in deps.target_names_fn(ann_assign.target)
                if not name.startswith("_")
            )

    if has_explicit_all:
        export_names = set(explicit_all)
    else:
        export_names = set(local_defs) | {
            name for name in import_map.keys() if not name.startswith("_")
        }
        export_names = {name for name in export_names if not name.startswith("_")}

    export_map: dict[str, str] = {}
    for name in export_names:
        deps.check_deadline_fn()
        if name in import_map:
            export_map[name] = import_map[name]
        elif name in local_defs:
            export_map[name] = f"{module_name}.{name}" if module_name else name
    return export_names, export_map
