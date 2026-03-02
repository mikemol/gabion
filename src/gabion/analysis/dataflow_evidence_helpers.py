# gabion:boundary_normalization_module
from __future__ import annotations

"""Evidence/index helper surfaces consumed by test-evidence projections."""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, cast

from gabion.analysis.aspf import Alt, Forest, NodeId
from gabion.analysis.dataflow_contracts import ClassInfo, FunctionInfo, SymbolTable
from gabion.analysis.dataflow_function_index_helpers import (
    _build_function_index as _build_function_index_impl,
)
from gabion.analysis.dataflow_parse_helpers import (
    _ParseModuleFailure,
    _ParseModuleStage,
    _ParseModuleSuccess,
    _parse_module_tree,
)
from gabion.analysis.json_types import JSONObject, JSONValue
from gabion.analysis.timeout_context import check_deadline
from gabion.analysis.visitors import ImportVisitor, ParentAnnotator

def _is_test_path(path: Path) -> bool:
    if "tests" in path.parts:
        return True
    return path.name.startswith("test_")


def _module_name(path: Path, project_root=None) -> str:
    rel = path.with_suffix("")
    if project_root is not None:
        try:
            rel = rel.relative_to(project_root)
        except ValueError:
            pass
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)


def _enclosing_scopes(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[str]:
    check_deadline()
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        check_deadline()
        current_type = type(current)
        if current_type is ast.ClassDef:
            scopes.append(cast(ast.ClassDef, current).name)
        elif current_type is ast.FunctionDef or current_type is ast.AsyncFunctionDef:
            scopes.append(cast(ast.FunctionDef | ast.AsyncFunctionDef, current).name)
        current = parents.get(current)
    return list(reversed(scopes))


def _enclosing_class_scopes(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[str]:
    check_deadline()
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        check_deadline()
        if type(current) is ast.ClassDef:
            scopes.append(cast(ast.ClassDef, current).name)
        current = parents.get(current)
    return list(reversed(scopes))


def _string_list(node: ast.AST):
    check_deadline()
    node_type = type(node)
    if node_type is ast.List or node_type is ast.Tuple:
        container = cast(ast.List | ast.Tuple, node)
        values: list[str] = []
        for elt in container.elts:
            check_deadline()
            if type(elt) is ast.Constant and type(cast(ast.Constant, elt).value) is str:
                values.append(cast(str, cast(ast.Constant, elt).value))
            else:
                return None
        return values
    return None


def _target_names(target: ast.AST) -> set[str]:
    check_deadline()
    names: set[str] = set()
    target_type = type(target)
    if target_type is ast.Name:
        names.add(cast(ast.Name, target).id)
        return names
    if target_type is ast.Tuple or target_type is ast.List:
        for element in cast(ast.Tuple | ast.List, target).elts:
            check_deadline()
            names.update(_target_names(element))
    return names


def _collect_module_exports(
    tree: ast.AST,
    *,
    module_name: str,
    import_map: dict[str, str],
) -> tuple[set[str], dict[str, str]]:
    check_deadline()
    explicit_all: list[str] = []
    has_explicit_all = False
    for stmt in getattr(tree, "body", []):
        check_deadline()
        stmt_type = type(stmt)
        if stmt_type is ast.Assign:
            assign_stmt = cast(ast.Assign, stmt)
            targets = assign_stmt.targets
            if any(
                type(target) is ast.Name and cast(ast.Name, target).id == "__all__"
                for target in targets
            ):
                values = _string_list(assign_stmt.value)
                if values is not None:
                    explicit_all = list(values)
                    has_explicit_all = True
        elif stmt_type is ast.AnnAssign:
            ann_assign = cast(ast.AnnAssign, stmt)
            target = ann_assign.target
            if type(target) is ast.Name and cast(ast.Name, target).id == "__all__":
                values = _string_list(ann_assign.value) if ann_assign.value is not None else None
                if values is not None:
                    explicit_all = list(values)
                    has_explicit_all = True
        elif stmt_type is ast.AugAssign:
            aug_assign = cast(ast.AugAssign, stmt)
            target = aug_assign.target
            if (
                type(target) is ast.Name
                and cast(ast.Name, target).id == "__all__"
                and type(aug_assign.op) is ast.Add
            ):
                values = _string_list(aug_assign.value)
                if values is not None:
                    if not has_explicit_all:
                        has_explicit_all = True
                        explicit_all = []
                    explicit_all.extend(values)

    local_defs: set[str] = set()
    for stmt in getattr(tree, "body", []):
        check_deadline()
        stmt_type = type(stmt)
        if stmt_type in {ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef}:
            stmt_name = str(getattr(stmt, "name", ""))
            if stmt_name and not stmt_name.startswith("_"):
                local_defs.add(stmt_name)
        elif stmt_type is ast.Assign:
            for target in cast(ast.Assign, stmt).targets:
                check_deadline()
                local_defs.update(name for name in _target_names(target) if not name.startswith("_"))
        elif stmt_type is ast.AnnAssign:
            local_defs.update(
                name
                for name in _target_names(cast(ast.AnnAssign, stmt).target)
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
        check_deadline()
        if name in import_map:
            export_map[name] = import_map[name]
        elif name in local_defs:
            export_map[name] = f"{module_name}.{name}" if module_name else name
    return export_names, export_map


def _base_identifier(node: ast.AST):
    check_deadline()
    node_type = type(node)
    if node_type is ast.Name:
        return cast(ast.Name, node).id
    if node_type is ast.Attribute:
        try:
            return ast.unparse(node)
        except (AttributeError, TypeError, ValueError, RecursionError):
            return None
    if node_type is ast.Subscript:
        return _base_identifier(cast(ast.Subscript, node).value)
    if node_type is ast.Call:
        return _base_identifier(cast(ast.Call, node).func)
    return None


def _build_symbol_table(
    paths: list[Path],
    project_root,
    *,
    external_filter: bool,
    parse_failure_witnesses: list[JSONObject],
) -> SymbolTable:
    check_deadline()
    table = SymbolTable(external_filter=external_filter)
    for path in paths:
        check_deadline()
        parse_outcome = _parse_module_tree(
            path,
            stage=_ParseModuleStage.SYMBOL_TABLE,
            parse_failure_witnesses=parse_failure_witnesses,
        )
        match parse_outcome:
            case _ParseModuleSuccess(kind="parsed", tree=tree):
                module = _module_name(path, project_root)
                table.internal_roots.add(module.split(".")[0])
                visitor = ImportVisitor(module, table)
                visitor.visit(tree)
                import_map = {
                    local: fqn
                    for (mod, local), fqn in table.imports.items()
                    if mod == module
                }
                exports, export_map = _collect_module_exports(
                    tree,
                    module_name=module,
                    import_map=import_map,
                )
                table.module_exports[module] = exports
                table.module_export_map[module] = export_map
            case _ParseModuleFailure(kind="parse_failure"):
                pass
    return table


def _collect_class_index(
    paths: list[Path],
    project_root,
    *,
    parse_failure_witnesses: list[JSONObject],
) -> dict[str, ClassInfo]:
    check_deadline()
    class_index: dict[str, ClassInfo] = {}
    for path in paths:
        check_deadline()
        parse_outcome = _parse_module_tree(
            path,
            stage=_ParseModuleStage.CLASS_INDEX,
            parse_failure_witnesses=parse_failure_witnesses,
        )
        match parse_outcome:
            case _ParseModuleSuccess(kind="parsed", tree=tree):
                parents = ParentAnnotator()
                parents.visit(tree)
                module = _module_name(path, project_root)
                for node in ast.walk(tree):
                    check_deadline()
                    if type(node) is ast.ClassDef:
                        class_node = cast(ast.ClassDef, node)
                        scopes = _enclosing_class_scopes(class_node, parents.parents)
                        qual_parts = [module] if module else []
                        qual_parts.extend(scopes)
                        qual_parts.append(class_node.name)
                        qual = ".".join(qual_parts)
                        bases: list[str] = []
                        for base in class_node.bases:
                            check_deadline()
                            base_name = _base_identifier(base)
                            if base_name:
                                bases.append(base_name)
                        methods: set[str] = set()
                        for stmt in class_node.body:
                            check_deadline()
                            if type(stmt) in {ast.FunctionDef, ast.AsyncFunctionDef}:
                                methods.add(
                                    cast(ast.FunctionDef | ast.AsyncFunctionDef, stmt).name
                                )
                        class_index[qual] = ClassInfo(
                            qual=qual,
                            module=module,
                            bases=bases,
                            methods=methods,
                        )
            case _ParseModuleFailure(kind="parse_failure"):
                pass
    return class_index


def _build_function_index(
    paths: list[Path],
    project_root,
    ignore_params: set[str],
    strictness: str,
    transparent_decorators=None,
    *,
    parse_failure_witnesses: list[JSONObject],
) -> tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]]:
    return _build_function_index_impl(
        paths,
        project_root,
        ignore_params,
        strictness,
        transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
    )


def _callee_key(name: str) -> str:
    if not name:
        return name
    return name.split(".")[-1]


def _resolve_class_candidates(
    base: str,
    *,
    module: str,
    symbol_table,
    class_index: dict[str, ClassInfo],
) -> list[str]:
    check_deadline()
    if not base:
        return []
    candidates: list[str] = []
    if "." in base:
        parts = base.split(".")
        head = parts[0]
        tail = ".".join(parts[1:])
        if symbol_table is not None:
            resolved_head = symbol_table.resolve(module, head)
            if resolved_head:
                candidates.append(f"{resolved_head}.{tail}")
        if module:
            candidates.append(f"{module}.{base}")
        candidates.append(base)
    else:
        if symbol_table is not None:
            resolved = symbol_table.resolve(module, base)
            if resolved:
                candidates.append(resolved)
            resolved_star = symbol_table.resolve_star(module, base)
            if resolved_star:
                candidates.append(resolved_star)
        if module:
            candidates.append(f"{module}.{base}")
        candidates.append(base)
    seen: set[str] = set()
    resolved_candidates: list[str] = []
    for candidate in candidates:
        check_deadline()
        if candidate not in seen:
            seen.add(candidate)
            if candidate in class_index:
                resolved_candidates.append(candidate)
    return resolved_candidates


@dataclass(frozen=True)
class _MethodHierarchyResolutionFound:
    kind: Literal["found"]
    resolved: FunctionInfo


@dataclass(frozen=True)
class _MethodHierarchyResolutionMissing:
    kind: Literal["not_found"]


MethodHierarchyResolution = (
    _MethodHierarchyResolutionFound | _MethodHierarchyResolutionMissing
)


def _resolve_method_in_hierarchy_outcome(
    class_qual: str,
    method: str,
    *,
    class_index: dict[str, ClassInfo],
    by_qual: Mapping[str, FunctionInfo],
    symbol_table,
    seen: set[str],
) -> MethodHierarchyResolution:
    check_deadline()
    if class_qual in seen:
        return _MethodHierarchyResolutionMissing(kind="not_found")
    seen.add(class_qual)
    candidate = f"{class_qual}.{method}"
    resolved = by_qual.get(candidate)
    if resolved is not None:
        return _MethodHierarchyResolutionFound(kind="found", resolved=resolved)
    info = class_index.get(class_qual)
    if info is not None:
        for base in info.bases:
            check_deadline()
            for base_qual in _resolve_class_candidates(
                base,
                module=info.module,
                symbol_table=symbol_table,
                class_index=class_index,
            ):
                check_deadline()
                resolution = _resolve_method_in_hierarchy_outcome(
                    base_qual,
                    method,
                    class_index=class_index,
                    by_qual=by_qual,
                    symbol_table=symbol_table,
                    seen=seen,
                )
                if type(resolution) is _MethodHierarchyResolutionFound:
                    return resolution
    return _MethodHierarchyResolutionMissing(kind="not_found")


def _resolve_method_in_hierarchy(
    class_qual: str,
    method: str,
    *,
    class_index: dict[str, ClassInfo],
    by_qual: Mapping[str, FunctionInfo],
    symbol_table,
    seen: set[str],
) -> MethodHierarchyResolution:
    return _resolve_method_in_hierarchy_outcome(
        class_qual,
        method,
        class_index=class_index,
        by_qual=by_qual,
        symbol_table=symbol_table,
        seen=seen,
    )


def _resolve_callee(
    callee_key: str,
    caller: FunctionInfo,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table=None,
    project_root=None,
    class_index=None,
    call=None,
    ambiguity_sink=None,
    local_lambda_bindings=None,
):
    # Local import avoids semantic-core cycle while callee resolution is extracted.
    from .dataflow_callee_resolution import (
        CalleeResolutionContext,
        collect_callee_resolution_effects,
        resolve_callee_with_effects,
    )

    check_deadline()
    lambda_bindings = local_lambda_bindings
    if lambda_bindings is None:
        lambda_bindings = caller.local_lambda_bindings
    context = CalleeResolutionContext(
        callee_key=callee_key,
        caller=caller,
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=symbol_table,
        project_root=project_root,
        class_index=class_index,
        call=call,
        local_lambda_bindings=lambda_bindings,
        caller_module=_module_name(caller.path, project_root=project_root),
    )
    resolution = resolve_callee_with_effects(context)
    if ambiguity_sink is not None:
        for effect in collect_callee_resolution_effects(resolution):
            check_deadline()
            ambiguity_sink(
                caller,
                call,
                list(effect.candidates),
                effect.phase,
                effect.callee_key,
            )
    return resolution.resolved


def _alt_input(alt: Alt, kind: str):
    for node_id in alt.inputs:
        check_deadline()
        if node_id.kind == kind:
            return node_id
    return None


def _paramset_key(forest: Forest, paramset_id: NodeId) -> tuple[str, ...]:
    node = forest.nodes.get(paramset_id)
    if node is not None:
        params = node.meta.get("params")
        if type(params) is list:
            return tuple(str(p) for p in cast(list[JSONValue], params))
    return tuple(str(p) for p in paramset_id.key)

__all__ = [
    "ParentAnnotator",
    "_callee_key",
    "_alt_input",
    "_build_function_index",
    "_build_symbol_table",
    "_collect_class_index",
    "_enclosing_scopes",
    "_is_test_path",
    "_module_name",
    "_paramset_key",
    "_resolve_class_candidates",
    "_resolve_callee",
    "_resolve_method_in_hierarchy",
]
