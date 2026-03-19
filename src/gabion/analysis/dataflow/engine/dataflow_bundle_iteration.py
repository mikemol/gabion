from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path
from typing import Literal, cast

from gabion.analysis.dataflow.engine.dataflow_contracts import SymbolTable
from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import _module_name
from gabion.analysis.dataflow.io.dataflow_parse_helpers import (
    ParseModuleFailure,
    ParseModuleStage,
    ParseModuleSuccess,
    forbid_adhoc_bundle_discovery,
    parse_module_tree,
)
from gabion.analysis.foundation.json_types import JSONObject, ParseFailureWitnesses
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never
from gabion.order_contract import sort_once

@dataclass(frozen=True)
class BundleIterationContext:
    path: Path
    module: "ModuleIdentifier"
    symbol_table: object
    local_dataclasses: Mapping[str, tuple[str, ...]]
    dataclass_registry: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class BundleIterationOutcome:
    bundles: frozenset[tuple[str, ...]]
    witness_effects: tuple[JSONObject, ...]


@dataclass(frozen=True)
class _ConstructorOperation:
    kind: str
    count: int = 0
    name: str = ""
    source: str = ""


@dataclass(frozen=True)
class _ConstructorPlan:
    operations: tuple[_ConstructorOperation, ...]
    witness_effects: tuple[JSONObject, ...]
    terminal_status: str


@dataclass(frozen=True)
class ModuleIdentifier:
    value: str


@dataclass(frozen=True)
class _ConstructorProjectionApplied:
    kind: Literal["applied"]
    names: tuple[str, ...]
    witness_effects: tuple[JSONObject, ...]


@dataclass(frozen=True)
class _ConstructorProjectionRejected:
    kind: Literal["rejected"]
    reason: str
    witness_effects: tuple[JSONObject, ...]


ConstructorProjectionResult = (
    _ConstructorProjectionApplied | _ConstructorProjectionRejected
)


@dataclass(frozen=True)
class _StarredElementsResolved:
    count: int


@dataclass(frozen=True)
class _StarredElementsRejected:
    detail: str


StarredElementsDecision = _StarredElementsResolved | _StarredElementsRejected


@dataclass(frozen=True)
class _KeywordMappingExpanded:
    operations: tuple[_ConstructorOperation, ...]


@dataclass(frozen=True)
class _KeywordMappingRejected:
    witness_effects: tuple[JSONObject, ...]


KeywordMappingDecision = _KeywordMappingExpanded | _KeywordMappingRejected


def _collect_local_dataclasses(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    local_dataclasses: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        check_deadline()
        match node:
            case ast.ClassDef() as class_node:
                decorators = {
                    ast.unparse(dec) if hasattr(ast, "unparse") else ""
                    for dec in class_node.decorator_list
                }
                if any("dataclass" in dec for dec in decorators):
                    fields: list[str] = []
                    for stmt in class_node.body:
                        check_deadline()
                        match stmt:
                            case ast.AnnAssign(target=ast.Name() as target_name):
                                fields.append(target_name.id)
                            case ast.Assign(targets=assign_targets):
                                for target in assign_targets:
                                    check_deadline()
                                    match target:
                                        case ast.Name() as target_name:
                                            fields.append(target_name.id)
                    if fields:
                        local_dataclasses[class_node.name] = tuple(fields)
    return local_dataclasses


def _module_identifier(module_name: object) -> ModuleIdentifier:
    match module_name:
        case str() as module_name_text:
            normalized = module_name_text.strip()
            if normalized:
                return ModuleIdentifier(value=normalized)
            raise ValueError("module identifier must be non-empty")
    raise ValueError("_module_name must return str")


def _effective_dataclass_registry(
    *,
    module: ModuleIdentifier,
    local_dataclasses: Mapping[str, tuple[str, ...]],
    dataclass_registry: object,
) -> dict[str, tuple[str, ...]]:
    match dataclass_registry:
        case dict() as dataclass_registry_payload:
            return {
                name: tuple(fields)
                for name, fields in dataclass_registry_payload.items()
            }
    effective_registry: dict[str, tuple[str, ...]] = {}
    for name, fields in local_dataclasses.items():
        check_deadline()
        effective_registry[f"{module.value}.{name}"] = fields
    return effective_registry


def _resolve_dataclass_fields(
    call: ast.Call,
    *,
    context: BundleIterationContext,
) -> tuple[str, ...]:
    symbol_table = cast(SymbolTable, context.symbol_table) if context.symbol_table is not None else None
    match call.func:
        case ast.Name(id=name):
            if name in context.local_dataclasses:
                return context.local_dataclasses[name]
            candidate = f"{context.module.value}.{name}"
            if candidate in context.dataclass_registry:
                return context.dataclass_registry[candidate]
            if symbol_table is not None:
                resolved = symbol_table.resolve(context.module.value, name)
                if resolved in context.dataclass_registry:
                    return context.dataclass_registry[resolved]
                resolved_star = symbol_table.resolve_star(context.module.value, name)
                if resolved_star in context.dataclass_registry:
                    return context.dataclass_registry[resolved_star]
            if name in context.dataclass_registry:
                return context.dataclass_registry[name]
        case ast.Attribute(value=ast.Name(id=base), attr=attr):
            if symbol_table is not None:
                base_fqn = symbol_table.resolve(context.module.value, base)
                if base_fqn:
                    candidate = f"{base_fqn}.{attr}"
                    if candidate in context.dataclass_registry:
                        return context.dataclass_registry[candidate]
                base_star = symbol_table.resolve_star(context.module.value, base)
                if base_star:
                    candidate = f"{base_star}.{attr}"
                    if candidate in context.dataclass_registry:
                        return context.dataclass_registry[candidate]
    return ()


def _unresolved_starred_witness(
    *,
    path: Path,
    call: ast.Call,
    category: str,
    detail: str,
) -> JSONObject:
    reason_by_category = {
        "dynamic_star_args": "unresolved_starred_positional",
        "positional_arity_overflow": "invalid_starred_positional_arity",
        "dynamic_star_kwargs": "unresolved_starred_keyword",
    }
    return {
        "path": str(path),
        "stage": ParseModuleStage.DATACLASS_CALL_BUNDLES.value,
        "reason": reason_by_category.get(category, category),
        "error_type": "UnresolvedStarredArgument",
        "error": f"{category}: {detail}",
        "line": int(getattr(call, "lineno", 0) or 0),
        "col": int(getattr(call, "col_offset", 0) or 0),
    }


@singledispatch
def _starred_elements_decision(node: object) -> StarredElementsDecision:
    return _StarredElementsRejected(
        detail=f"unsupported * payload={type(node).__name__}"
    )


@_starred_elements_decision.register(ast.List)
def _starred_list_elements_decision(node: ast.List) -> StarredElementsDecision:
    return _StarredElementsResolved(count=len(node.elts))


@_starred_elements_decision.register(ast.Tuple)
def _starred_tuple_elements_decision(node: ast.Tuple) -> StarredElementsDecision:
    return _StarredElementsResolved(count=len(node.elts))


@_starred_elements_decision.register(ast.Set)
def _starred_set_elements_decision(node: ast.Set) -> StarredElementsDecision:
    return _StarredElementsResolved(count=len(node.elts))


@singledispatch
def _keyword_mapping_projection(
    node: object,
    *,
    path: Path,
    call: ast.Call,
) -> KeywordMappingDecision:
    return _KeywordMappingRejected(
        witness_effects=(
            _unresolved_starred_witness(
                path=path,
                call=call,
                category="dynamic_star_kwargs",
                detail=f"unsupported ** payload={type(node).__name__}",
            ),
        )
    )


@_keyword_mapping_projection.register(ast.Dict)
def _keyword_mapping_dict_projection(
    node: ast.Dict,
    *,
    path: Path,
    call: ast.Call,
) -> KeywordMappingDecision:
    operations: list[_ConstructorOperation] = []
    for key in node.keys:
        check_deadline()
        match key:
            case None:
                return _KeywordMappingRejected(
                    witness_effects=(
                        _unresolved_starred_witness(
                            path=path,
                            call=call,
                            category="dynamic_star_kwargs",
                            detail="dict unpack inside ** literal is dynamic",
                        ),
                    )
                )
            case ast.Constant(value=str() as key_name):
                operations.append(
                    _ConstructorOperation(
                        kind="append_keyword",
                        name=key_name,
                        source="**dict",
                    )
                )
            case ast.expr():
                return _KeywordMappingRejected(
                    witness_effects=(
                        _unresolved_starred_witness(
                            path=path,
                            call=call,
                            category="dynamic_star_kwargs",
                            detail="non-string literal key in ** dict",
                        ),
                    )
                )
    return _KeywordMappingExpanded(operations=tuple(operations))


def _plan_constructor_operations(
    *,
    path: Path,
    call: ast.Call,
) -> _ConstructorPlan:
    operations: list[_ConstructorOperation] = []
    witness_effects: list[JSONObject] = []

    for arg in call.args:
        check_deadline()
        match arg:
            case ast.Starred(value=starred_value):
                match _starred_elements_decision(starred_value):
                    case _StarredElementsResolved(count=elements_count):
                        operations.append(
                            _ConstructorOperation(
                                kind="append_positional",
                                count=elements_count,
                                source=f"*{type(starred_value).__name__}",
                            )
                        )
                    case _StarredElementsRejected(detail=detail):
                        witness_effects.append(
                            _unresolved_starred_witness(
                                path=path,
                                call=call,
                                category="dynamic_star_args",
                                detail=detail,
                            )
                        )
                        return _ConstructorPlan(
                            operations=tuple(operations),
                            witness_effects=tuple(witness_effects),
                            terminal_status="stop",
                        )
            case ast.expr():
                operations.append(
                    _ConstructorOperation(
                        kind="append_positional",
                        count=1,
                        source="arg",
                    )
                )
    for keyword in call.keywords:
        check_deadline()
        if keyword.arg is not None:
            operations.append(
                _ConstructorOperation(
                    kind="append_keyword",
                    name=keyword.arg,
                    source="keyword",
                )
            )
            continue
        match _keyword_mapping_projection(keyword.value, path=path, call=call):
            case _KeywordMappingExpanded(operations=mapping_operations):
                operations.extend(mapping_operations)
            case _KeywordMappingRejected(witness_effects=mapping_witness_effects):
                witness_effects.extend(mapping_witness_effects)
                return _ConstructorPlan(
                    operations=tuple(operations),
                    witness_effects=tuple(witness_effects),
                    terminal_status="stop",
                )
    return _ConstructorPlan(
        operations=tuple(operations),
        witness_effects=tuple(witness_effects),
        terminal_status="apply",
    )


def _apply_constructor_plan(
    *,
    path: Path,
    call: ast.Call,
    fields: Sequence[str],
    plan: _ConstructorPlan,
) -> ConstructorProjectionResult:
    witness_effects = list(plan.witness_effects)
    if plan.terminal_status != "apply":
        return _ConstructorProjectionRejected(
            kind="rejected",
            reason="plan_terminal_status",
            witness_effects=tuple(witness_effects),
        )

    field_set = set(fields)
    names: list[str] = []
    position = 0
    for operation in plan.operations:
        check_deadline()
        if operation.kind == "append_positional":
            for _ in range(operation.count):
                check_deadline()
                if position >= len(fields):
                    witness_effects.append(
                        _unresolved_starred_witness(
                            path=path,
                            call=call,
                            category="positional_arity_overflow",
                            detail=f"source={operation.source} exceeds dataclass field count",
                        )
                    )
                    return _ConstructorProjectionRejected(
                        kind="rejected",
                        reason="positional_arity_overflow",
                        witness_effects=tuple(witness_effects),
                    )
                names.append(fields[position])
                position += 1
            continue
        if operation.kind == "append_keyword" and operation.name:
            names.append(operation.name)
            continue
        return _ConstructorProjectionRejected(
            kind="rejected",
            reason="unknown_operation",
            witness_effects=tuple(witness_effects),
        )

    if any(name not in field_set for name in names):
        return _ConstructorProjectionRejected(
            kind="rejected",
            reason="field_name_mismatch",
            witness_effects=tuple(witness_effects),
        )
    return _ConstructorProjectionApplied(
        kind="applied",
        names=tuple(names),
        witness_effects=tuple(witness_effects),
    )


def iter_dataclass_call_bundle_effects(
    path: Path,
    *,
    project_root: object = None,
    symbol_table: object = None,
    dataclass_registry: object = None,
    parse_failure_witnesses: ParseFailureWitnesses,
) -> BundleIterationOutcome:
    check_deadline()
    forbid_adhoc_bundle_discovery("_iter_dataclass_call_bundles")

    parse_outcome = parse_module_tree(
        path,
        stage=ParseModuleStage.DATACLASS_CALL_BUNDLES,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    match parse_outcome:
        case ParseModuleSuccess(kind="parsed", tree=tree):
            pass
        case ParseModuleFailure(kind="parse_failure"):
            return BundleIterationOutcome(bundles=frozenset(), witness_effects=())

    module = _module_identifier(_module_name(path, project_root))
    local_dataclasses = _collect_local_dataclasses(tree)
    effective_registry = _effective_dataclass_registry(
        module=module,
        local_dataclasses=local_dataclasses,
        dataclass_registry=dataclass_registry,
    )
    context = BundleIterationContext(
        path=path,
        module=module,
        symbol_table=symbol_table,
        local_dataclasses=local_dataclasses,
        dataclass_registry=effective_registry,
    )

    bundles: set[tuple[str, ...]] = set()
    witness_effects: list[JSONObject] = []
    for node in ast.walk(tree):
        check_deadline()
        match node:
            case ast.Call() as call_node:
                fields = _resolve_dataclass_fields(call_node, context=context)
                if fields:
                    plan = _plan_constructor_operations(path=path, call=call_node)
                    projection = _apply_constructor_plan(
                        path=path,
                        call=call_node,
                        fields=fields,
                        plan=plan,
                    )
                    match projection:
                        case _ConstructorProjectionApplied(
                            kind="applied",
                            names=names,
                            witness_effects=projection_witness_effects,
                        ):
                            witness_effects.extend(projection_witness_effects)
                            if len(names) >= 2:
                                bundles.add(
                                    tuple(
                                        sort_once(
                                            names,
                                            source=(
                                                "src/gabion/analysis/dataflow_bundle_iteration.py:"
                                                "iter_dataclass_call_bundle_effects"
                                            ),
                                        )
                                    )
                                )
                        case _ConstructorProjectionRejected(
                            kind="rejected", witness_effects=projection_witness_effects
                        ):
                            witness_effects.extend(projection_witness_effects)
                        case _:
                            never(
                                "unexpected constructor projection outcome",
                                projection_type=type(projection).__name__,
                            )
    return BundleIterationOutcome(
        bundles=frozenset(bundles),
        witness_effects=tuple(witness_effects),
    )
