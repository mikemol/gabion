from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ModuleExportsCollectDeps:
    check_deadline_fn: Callable[[], None]
    string_list_fn: Callable[..., object]
    target_names_fn: Callable[..., set[str]]


def _is_all_name_target(node: ast.AST) -> bool:
    match node:
        case ast.Name(id="__all__"):
            return True
        case _:
            return False


def _assign_targets_include_all(targets: list[ast.AST]) -> bool:
    return any(_is_all_name_target(target) for target in targets)


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
        match stmt:
            case ast.Assign(targets=targets, value=value):
                if _assign_targets_include_all(targets):
                    values = deps.string_list_fn(value)
                    if values is not None:
                        explicit_all = list(values)
                        has_explicit_all = True
            case ast.AnnAssign(target=target, value=value):
                if _is_all_name_target(target):
                    values = deps.string_list_fn(value) if value is not None else None
                    if values is not None:
                        explicit_all = list(values)
                        has_explicit_all = True
            case ast.AugAssign(target=target, op=ast.Add(), value=value):
                if _is_all_name_target(target):
                    values = deps.string_list_fn(value)
                    if values is not None:
                        if not has_explicit_all:
                            has_explicit_all = True
                            explicit_all = []
                        explicit_all.extend(values)
            case _:
                pass

    local_defs: set[str] = set()
    for stmt in getattr(tree, "body", []):
        deps.check_deadline_fn()
        match stmt:
            case ast.FunctionDef(name=stmt_name) | ast.AsyncFunctionDef(name=stmt_name) | ast.ClassDef(name=stmt_name):
                if stmt_name and not stmt_name.startswith("_"):
                    local_defs.add(stmt_name)
            case ast.Assign(targets=targets):
                for target in targets:
                    deps.check_deadline_fn()
                    local_defs.update(
                        name for name in deps.target_names_fn(target) if not name.startswith("_")
                    )
            case ast.AnnAssign(target=target):
                local_defs.update(
                    name
                    for name in deps.target_names_fn(target)
                    if not name.startswith("_")
                )
            case _:
                pass

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
