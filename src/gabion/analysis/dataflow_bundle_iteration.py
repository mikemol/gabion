# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping, Sequence

from gabion.analysis.json_types import JSONObject
from gabion.order_contract import sort_once

_BOUND = False


def _bind_audit_symbols() -> None:
    global _BOUND
    if _BOUND:
        return
    from gabion.analysis import dataflow_audit as _audit

    module_globals = globals()
    for name, value in _audit.__dict__.items():
        module_globals.setdefault(name, value)
    _BOUND = True


@dataclass(frozen=True)
class BundleIterationContext:
    path: Path
    module: str
    symbol_table: SymbolTable | None
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
    name: str | None = None
    source: str = ""


@dataclass(frozen=True)
class _ConstructorPlan:
    operations: tuple[_ConstructorOperation, ...]
    witness_effects: tuple[JSONObject, ...]
    terminal_status: str


@dataclass(frozen=True)
class _ConstructorProjectionOutcome:
    names: tuple[str, ...] | None
    witness_effects: tuple[JSONObject, ...]


def _collect_local_dataclasses(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    local_dataclasses: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        check_deadline()
        if not isinstance(node, ast.ClassDef):
            continue
        decorators = {
            ast.unparse(dec) if hasattr(ast, "unparse") else ""
            for dec in node.decorator_list
        }
        if not any("dataclass" in dec for dec in decorators):
            continue
        fields: list[str] = []
        for stmt in node.body:
            check_deadline()
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                fields.append(stmt.target.id)
                continue
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    check_deadline()
                    if isinstance(target, ast.Name):
                        fields.append(target.id)
        if fields:
            local_dataclasses[node.name] = tuple(fields)
    return local_dataclasses


def _effective_dataclass_registry(
    *,
    module: str,
    local_dataclasses: Mapping[str, tuple[str, ...]],
    dataclass_registry: dict[str, list[str]] | None,
) -> dict[str, tuple[str, ...]]:
    if dataclass_registry is not None:
        return {
            name: tuple(fields)
            for name, fields in dataclass_registry.items()
        }

    effective_registry: dict[str, tuple[str, ...]] = {}
    for name, fields in local_dataclasses.items():
        check_deadline()
        if module:
            effective_registry[f"{module}.{name}"] = fields
            continue
        # pragma: no cover - module name is always non-empty for file paths
        effective_registry[name] = fields  # pragma: no cover
    return effective_registry


def _resolve_dataclass_fields(
    call: ast.Call,
    *,
    context: BundleIterationContext,
) -> tuple[str, ...] | None:
    if isinstance(call.func, ast.Name):
        name = call.func.id
        if name in context.local_dataclasses:
            return context.local_dataclasses[name]
        candidate = f"{context.module}.{name}"
        if candidate in context.dataclass_registry:
            return context.dataclass_registry[candidate]
        if context.symbol_table is not None:
            resolved = context.symbol_table.resolve(context.module, name)
            if resolved in context.dataclass_registry:
                return context.dataclass_registry[resolved]
            resolved_star = context.symbol_table.resolve_star(context.module, name)
            if resolved_star in context.dataclass_registry:
                return context.dataclass_registry[resolved_star]
        if name in context.dataclass_registry:
            return context.dataclass_registry[name]
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        base = call.func.value.id
        attr = call.func.attr
        if context.symbol_table is None:
            return None
        base_fqn = context.symbol_table.resolve(context.module, base)
        if base_fqn:
            candidate = f"{base_fqn}.{attr}"
            if candidate in context.dataclass_registry:
                return context.dataclass_registry[candidate]
        base_star = context.symbol_table.resolve_star(context.module, base)
        if base_star:
            candidate = f"{base_star}.{attr}"
            if candidate in context.dataclass_registry:
                return context.dataclass_registry[candidate]
    return None


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
        "stage": _ParseModuleStage.DATACLASS_CALL_BUNDLES.value,
        "reason": reason_by_category.get(category, category),
        "error_type": "UnresolvedStarredArgument",
        "error": f"{category}: {detail}",
        "line": int(getattr(call, "lineno", 0) or 0),
        "col": int(getattr(call, "col_offset", 0) or 0),
    }


def _plan_constructor_operations(
    *,
    path: Path,
    call: ast.Call,
) -> _ConstructorPlan:
    operations: list[_ConstructorOperation] = []
    witness_effects: list[JSONObject] = []

    for arg in call.args:
        check_deadline()
        if not isinstance(arg, ast.Starred):
            operations.append(_ConstructorOperation(kind="append_positional", count=1, source="arg"))
            continue
        value = arg.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            operations.append(
                _ConstructorOperation(
                    kind="append_positional",
                    count=len(value.elts),
                    source=f"*{type(value).__name__}",
                )
            )
            continue
        witness_effects.append(
            _unresolved_starred_witness(
                path=path,
                call=call,
                category="dynamic_star_args",
                detail=f"unsupported * payload={type(value).__name__}",
            )
        )
        return _ConstructorPlan(
            operations=tuple(operations),
            witness_effects=tuple(witness_effects),
            terminal_status="stop",
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
        mapping_node = keyword.value
        if not isinstance(mapping_node, ast.Dict):
            witness_effects.append(
                _unresolved_starred_witness(
                    path=path,
                    call=call,
                    category="dynamic_star_kwargs",
                    detail=f"unsupported ** payload={type(mapping_node).__name__}",
                )
            )
            return _ConstructorPlan(
                operations=tuple(operations),
                witness_effects=tuple(witness_effects),
                terminal_status="stop",
            )
        for key in mapping_node.keys:
            check_deadline()
            if key is None:
                witness_effects.append(
                    _unresolved_starred_witness(
                        path=path,
                        call=call,
                        category="dynamic_star_kwargs",
                        detail="dict unpack inside ** literal is dynamic",
                    )
                )
                return _ConstructorPlan(
                    operations=tuple(operations),
                    witness_effects=tuple(witness_effects),
                    terminal_status="stop",
                )
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                witness_effects.append(
                    _unresolved_starred_witness(
                        path=path,
                        call=call,
                        category="dynamic_star_kwargs",
                        detail="non-string literal key in ** dict",
                    )
                )
                return _ConstructorPlan(
                    operations=tuple(operations),
                    witness_effects=tuple(witness_effects),
                    terminal_status="stop",
                )
            operations.append(
                _ConstructorOperation(
                    kind="append_keyword",
                    name=key.value,
                    source="**dict",
                )
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
) -> _ConstructorProjectionOutcome:
    witness_effects = list(plan.witness_effects)
    if plan.terminal_status != "apply":
        return _ConstructorProjectionOutcome(  # pragma: no cover
            names=None,
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
                    return _ConstructorProjectionOutcome(
                        names=None,
                        witness_effects=tuple(witness_effects),
                    )
                names.append(fields[position])
                position += 1
            continue
        if operation.kind == "append_keyword" and operation.name is not None:
            names.append(operation.name)
            continue
        return _ConstructorProjectionOutcome(  # pragma: no cover
            names=None,
            witness_effects=tuple(witness_effects),
        )

    if any(name not in field_set for name in names):
        return _ConstructorProjectionOutcome(names=None, witness_effects=tuple(witness_effects))
    return _ConstructorProjectionOutcome(names=tuple(names), witness_effects=tuple(witness_effects))


def iter_dataclass_call_bundle_effects(
    path: Path,
    *,
    project_root: Path | None = None,
    symbol_table: SymbolTable | None = None,
    dataclass_registry: dict[str, list[str]] | None = None,
    parse_failure_witnesses: list[JSONObject],
) -> BundleIterationOutcome:
    _bind_audit_symbols()
    check_deadline()
    _forbid_adhoc_bundle_discovery("_iter_dataclass_call_bundles")

    tree = _parse_module_tree(
        path,
        stage=_ParseModuleStage.DATACLASS_CALL_BUNDLES,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    if tree is None:
        return BundleIterationOutcome(bundles=frozenset(), witness_effects=())

    module = _module_name(path, project_root)
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
        if not isinstance(node, ast.Call):
            continue
        fields = _resolve_dataclass_fields(node, context=context)
        if not fields:
            continue
        plan = _plan_constructor_operations(path=path, call=node)
        projection = _apply_constructor_plan(path=path, call=node, fields=fields, plan=plan)
        witness_effects.extend(projection.witness_effects)
        if projection.names is None or len(projection.names) < 2:
            continue
        bundles.add(
            tuple(
                sort_once(
                    projection.names,
                    source="src/gabion/analysis/dataflow_bundle_iteration.py:iter_dataclass_call_bundle_effects",
                )
            )
        )

    return BundleIterationOutcome(
        bundles=frozenset(bundles),
        witness_effects=tuple(witness_effects),
    )
