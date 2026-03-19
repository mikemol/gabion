from __future__ import annotations

"""Evidence/index helper surfaces consumed by test-evidence projections."""

import ast
from functools import singledispatch
from pathlib import Path
from types import EllipsisType
from typing import Mapping, cast

from gabion.analysis.aspf.aspf import Alt, Forest, NodeId
from gabion.analysis.dataflow.engine.dataflow_callee_resolution import (
    CalleeResolutionContext,
    collect_callee_resolution_effects,
    resolve_callee_with_effects,
)
from gabion.analysis.dataflow.engine.dataflow_callee_resolution_support import (
    _callee_key,
    _resolve_class_candidates,
    _resolve_method_in_hierarchy,
    _resolve_method_in_hierarchy_outcome,
)
from gabion.analysis.dataflow.engine.dataflow_contracts import ClassInfo, FunctionInfo, SymbolTable
from gabion.analysis.dataflow.engine.dataflow_function_index_helpers import (
    _build_function_index as _build_function_index_impl)
from gabion.analysis.dataflow.io.dataflow_parse_helpers import (
    _ParseModuleFailure, _ParseModuleStage, _ParseModuleSuccess, _parse_module_tree)
from gabion.analysis.foundation.json_types import JSONObject, JSONValue, ParseFailureWitnesses
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.core.visitors import ImportVisitor, ParentAnnotator
from gabion.invariants import never

_NONE_TYPE = type(None)
_ELLIPSIS_TYPE = type(Ellipsis)
_ConstantValue = (
    str | bool | int | float | complex | bytes | bytearray | tuple | frozenset | None | EllipsisType
)


def _leaf_ast_subclasses(base_type: type[ast.AST]) -> tuple[type[ast.AST], ...]:
    node_types: list[type[ast.AST]] = []
    for candidate in vars(ast).values():
        try:
            if issubclass(candidate, base_type) and candidate is not base_type:
                node_types.append(candidate)
        except TypeError:
            continue
    node_types_tuple = tuple(node_types)
    return tuple(
        node_type
        for node_type in node_types_tuple
        if not any(
            candidate is not node_type and issubclass(candidate, node_type)
            for candidate in node_types_tuple
        )
    )


_AST_LEAF_NODE_TYPES = _leaf_ast_subclasses(ast.AST)
_AST_LEAF_OPERATOR_TYPES = _leaf_ast_subclasses(ast.operator)


@singledispatch
def _is_class_def(value: ast.AST) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_class_def.register(ast.ClassDef)
def _sd_reg_1(value: ast.ClassDef) -> bool:
    _ = value
    return True


def _is_not_class_def(value: ast.AST) -> bool:
    _ = value
    return False


for _runtime_type in _AST_LEAF_NODE_TYPES:
    if _runtime_type is ast.ClassDef:
        continue
    _is_class_def.register(_runtime_type)(_is_not_class_def)


@singledispatch
def _class_def_name(value: ast.AST) -> str:
    never("unregistered runtime type", value_type=type(value).__name__)


@_class_def_name.register(ast.ClassDef)
def _sd_reg_2(value: ast.ClassDef) -> str:
    return value.name


@singledispatch
def _class_def_bases(value: ast.AST) -> tuple[ast.expr, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_class_def_bases.register(ast.ClassDef)
def _sd_reg_3(value: ast.ClassDef) -> tuple[ast.expr, ...]:
    return tuple(value.bases)


@singledispatch
def _class_def_body(value: ast.AST) -> tuple[ast.stmt, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_class_def_body.register(ast.ClassDef)
def _sd_reg_4(value: ast.ClassDef) -> tuple[ast.stmt, ...]:
    return tuple(value.body)


@singledispatch
def _is_name_node(value: ast.AST) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_name_node.register(ast.Name)
def _sd_reg_5(value: ast.Name) -> bool:
    _ = value
    return True


def _is_not_name_node(value: ast.AST) -> bool:
    _ = value
    return False


for _runtime_type in _AST_LEAF_NODE_TYPES:
    if _runtime_type is ast.Name:
        continue
    _is_name_node.register(_runtime_type)(_is_not_name_node)


@singledispatch
def _name_node_id(value: ast.AST) -> str:
    never("unregistered runtime type", value_type=type(value).__name__)


@_name_node_id.register(ast.Name)
def _sd_reg_6(value: ast.Name) -> str:
    return value.id


@singledispatch
def _is_add_operator(value: ast.operator) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_add_operator.register(ast.Add)
def _sd_reg_7(value: ast.Add) -> bool:
    _ = value
    return True


def _is_not_add_operator(value: ast.operator) -> bool:
    _ = value
    return False


for _runtime_type in _AST_LEAF_OPERATOR_TYPES:
    if _runtime_type is ast.Add:
        continue
    _is_add_operator.register(_runtime_type)(_is_not_add_operator)


@singledispatch
def _is_list_or_tuple_node(value: ast.AST) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_list_or_tuple_node.register(ast.List)
def _sd_reg_8(value: ast.List) -> bool:
    _ = value
    return True


@_is_list_or_tuple_node.register(ast.Tuple)
def _sd_reg_9(value: ast.Tuple) -> bool:
    _ = value
    return True


def _is_not_list_or_tuple_node(value: ast.AST) -> bool:
    _ = value
    return False


for _runtime_type in _AST_LEAF_NODE_TYPES:
    if _runtime_type in {ast.List, ast.Tuple}:
        continue
    _is_list_or_tuple_node.register(_runtime_type)(_is_not_list_or_tuple_node)


@singledispatch
def _sequence_elements(value: ast.AST) -> tuple[ast.AST, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_sequence_elements.register(ast.List)
def _sd_reg_10(value: ast.List) -> tuple[ast.AST, ...]:
    return tuple(value.elts)


@_sequence_elements.register(ast.Tuple)
def _sd_reg_11(value: ast.Tuple) -> tuple[ast.AST, ...]:
    return tuple(value.elts)


@singledispatch
def _is_string_constant_node(value: ast.AST) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_string_constant_node.register(ast.Constant)
def _sd_reg_12(value: ast.Constant) -> bool:
    return _is_string_value(value.value)


def _is_not_string_constant_node(value: ast.AST) -> bool:
    _ = value
    return False


for _runtime_type in _AST_LEAF_NODE_TYPES:
    if _runtime_type is ast.Constant:
        continue
    _is_string_constant_node.register(_runtime_type)(_is_not_string_constant_node)


@singledispatch
def _string_constant_text(value: ast.AST) -> str:
    never("unregistered runtime type", value_type=type(value).__name__)


@_string_constant_text.register(ast.Constant)
def _sd_reg_13(value: ast.Constant) -> str:
    return _string_value_text(value.value)


@singledispatch
def _is_string_value(value: _ConstantValue) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_string_value.register(str)
def _sd_reg_14(value: str) -> bool:
    _ = value
    return True


def _is_not_string_value(value: _ConstantValue) -> bool:
    _ = value
    return False


for _runtime_type in (
    bool,
    int,
    float,
    complex,
    bytes,
    bytearray,
    tuple,
    frozenset,
    _NONE_TYPE,
    _ELLIPSIS_TYPE,
):
    _is_string_value.register(_runtime_type)(_is_not_string_value)


@singledispatch
def _string_value_text(value: _ConstantValue) -> str:
    never("unregistered runtime type", value_type=type(value).__name__)


@_string_value_text.register(str)
def _sd_reg_15(value: str) -> str:
    return value


@singledispatch
def _is_json_value_list(value: JSONValue) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_json_value_list.register(list)
def _sd_reg_16(value: list[JSONValue]) -> bool:
    _ = value
    return True


@singledispatch
def _json_value_list_items(value: JSONValue) -> tuple[JSONValue, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_json_value_list_items.register(list)
def _sd_reg_17(value: list[JSONValue]) -> tuple[JSONValue, ...]:
    return tuple(value)


def _is_not_json_value_list(value: JSONValue) -> bool:
    _ = value
    return False


for _runtime_type in (tuple, set, dict, str, int, float, bool, _NONE_TYPE):
    _is_json_value_list.register(_runtime_type)(_is_not_json_value_list)

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


module_name = _module_name


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
        if _is_class_def(current):
            scopes.append(_class_def_name(current))
        current = parents.get(current)
    return list(reversed(scopes))


def _string_list(node: ast.AST):
    check_deadline()
    if _is_list_or_tuple_node(node):
        values: list[str] = []
        for elt in _sequence_elements(node):
            check_deadline()
            if _is_string_constant_node(elt):
                values.append(_string_constant_text(elt))
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
                _is_name_node(target) and _name_node_id(target) == "__all__"
                for target in targets
            ):
                values = _string_list(assign_stmt.value)
                if values is not None:
                    explicit_all = list(values)
                    has_explicit_all = True
        elif stmt_type is ast.AnnAssign:
            ann_assign = cast(ast.AnnAssign, stmt)
            target = ann_assign.target
            if _is_name_node(target) and _name_node_id(target) == "__all__":
                values = _string_list(ann_assign.value) if ann_assign.value is not None else None
                if values is not None:
                    explicit_all = list(values)
                    has_explicit_all = True
        elif stmt_type is ast.AugAssign:
            aug_assign = cast(ast.AugAssign, stmt)
            target = aug_assign.target
            if (
                _is_name_node(target)
                and _name_node_id(target) == "__all__"
                and _is_add_operator(aug_assign.op)
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
    parse_failure_witnesses: ParseFailureWitnesses,
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
    parse_failure_witnesses: ParseFailureWitnesses,
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
                    if _is_class_def(node):
                        scopes = _enclosing_class_scopes(node, parents.parents)
                        qual_parts = [module] if module else []
                        qual_parts.extend(scopes)
                        qual_parts.append(_class_def_name(node))
                        qual = ".".join(qual_parts)
                        bases: list[str] = []
                        for base in _class_def_bases(node):
                            check_deadline()
                            base_name = _base_identifier(base)
                            if base_name:
                                bases.append(base_name)
                        methods: set[str] = set()
                        for stmt in _class_def_body(node):
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
    parse_failure_witnesses: ParseFailureWitnesses,
) -> tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]]:
    return _build_function_index_impl(
        paths,
        project_root,
        ignore_params,
        strictness,
        transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
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
        if _is_json_value_list(params):
            return tuple(str(p) for p in _json_value_list_items(params))
    return tuple(str(p) for p in paramset_id.key)

__all__ = [
    "ParentAnnotator",
    "_base_identifier",
    "_callee_key",
    "_alt_input",
    "_build_function_index",
    "_build_symbol_table",
    "_collect_class_index",
    "_collect_module_exports",
    "_enclosing_scopes",
    "_is_test_path",
    "_module_name",
    "module_name",
    "_paramset_key",
    "_resolve_class_candidates",
    "_resolve_callee",
    "_resolve_method_in_hierarchy",
    "_resolve_method_in_hierarchy_outcome",
    "_target_names",
]
