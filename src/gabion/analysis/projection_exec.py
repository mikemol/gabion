# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Iterable, Mapping, cast

from gabion.analysis.projection_normalize import normalize_spec
from gabion.analysis.projection_spec import ProjectionSpec
from gabion.json_types import JSONValue
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import OrderPolicy, sort_once

Relation = list[dict[str, JSONValue]]


@dataclass(frozen=True)
class SelectParams:
    predicates: tuple[str, ...]


@dataclass(frozen=True)
class ProjectParams:
    fields: tuple[JSONValue, ...]


@dataclass(frozen=True)
class CountByParams:
    fields: tuple[JSONValue, ...]


@dataclass(frozen=True)
class TraverseParams:
    field: str
    merge: bool = True
    keep: bool = False
    prefix: str = ""
    as_field: str = ""
    index_field: str = ""


@dataclass(frozen=True)
class SortKey:
    field: str
    order: str = "asc"


@dataclass(frozen=True)
class SortParams:
    keys: tuple[SortKey, ...]


@dataclass(frozen=True)
class LimitParams:
    count: int | None


def apply_spec(
    spec: ProjectionSpec,
    rows: Iterable[Mapping[str, JSONValue]],
    *,
    op_registry: Mapping[str, Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool]]
    | None = None,
    params_override: Mapping[str, JSONValue] | None = None,
    normalize: Callable[[ProjectionSpec], Mapping[str, JSONValue]] | None = None,
) -> Relation:
    check_deadline()
    normalize_fn = normalize or normalize_spec
    normalized = normalize_fn(spec)
    params: dict[str, JSONValue] = {}
    spec_params = normalized.get("params")
    if isinstance(spec_params, Mapping):
        params.update(spec_params)
    if params_override:
        params.update(params_override)
    op_registry = op_registry or {}

    current: Relation = [dict(row) for row in rows if isinstance(row, Mapping)]

    for op in normalized.get("pipeline") or []:
        check_deadline()
        if not isinstance(op, Mapping):
            continue
        op_name = op.get("op")
        params_map = op.get("params")
        if not isinstance(params_map, Mapping):
            params_map = {}

        if op_name == "select":
            select_params = _select_params_from_map(params_map)
            current = _apply_select(current, select_params, op_registry=op_registry, runtime_params=params)
            continue

        if op_name == "project":
            project_params = _project_params_from_map(params_map)
            if project_params is None:
                continue
            current = _apply_project(current, project_params)
            continue

        if op_name == "count_by":
            count_params = _count_by_params_from_map(params_map)
            if count_params is None:
                continue
            current = _apply_count_by(current, count_params)
            continue

        if op_name == "traverse":
            traverse_params = _traverse_params_from_map(params_map)
            if traverse_params is None:
                continue
            current = _apply_traverse(current, traverse_params)
            continue

        if op_name == "sort":
            sort_params = _sort_params_from_map(params_map)
            if sort_params is None:
                continue
            for key in reversed(sort_params.keys):
                check_deadline()
                current = sort_once(
                    current,
                    source=f"apply_spec.sort[{key.field}]",
                    policy=OrderPolicy.SORT,
                    key=lambda row, name=key.field: _sort_value(row.get(name)),
                    reverse=key.order == "desc",
                )
            continue

        if op_name == "limit":
            limit_params = _limit_params_from_map(params_map)
            if limit_params is None or limit_params.count is None or limit_params.count < 0:
                continue
            current = current[: limit_params.count]
            continue

    return current


def _sort_value(value: JSONValue) -> tuple[int, object]:
    if value is None:
        return (1, "")
    if isinstance(value, (int, float, str)):
        return (0, value)
    return (0, str(value))


def _hashable(value: JSONValue) -> object:
    try:
        hash(value)
    except TypeError:
        return json.dumps(value, sort_keys=False, separators=(",", ":"))
    return value


def _select_params_from_map(params_map: Mapping[str, JSONValue]) -> SelectParams:
    predicates = params_map.get("predicates", [])
    if isinstance(predicates, str):
        predicates = [predicates]
    if not isinstance(predicates, list):
        predicates = []
    names = tuple(name for name in predicates if isinstance(name, str))
    return SelectParams(predicates=names)


def _apply_select(
    rows: Relation,
    select_params: SelectParams,
    *,
    op_registry: Mapping[str, Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool]],
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    selected = rows
    for predicate_name in select_params.predicates:
        check_deadline()
        predicate = op_registry.get(predicate_name)
        if predicate is None:
            continue
        selected = [row for row in selected if predicate(row, runtime_params)]
    return selected


def _project_params_from_map(params_map: Mapping[str, JSONValue]) -> ProjectParams | None:
    fields = params_map.get("fields", [])
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list):
        return None
    return ProjectParams(fields=tuple(cast(JSONValue, field) for field in fields))


def _apply_project(rows: Relation, params: ProjectParams) -> Relation:
    projected: Relation = []
    for row in rows:
        check_deadline()
        projected.append({field: row.get(field) for field in params.fields})
    return projected


def _count_by_params_from_map(params_map: Mapping[str, JSONValue]) -> CountByParams | None:
    fields = params_map.get("fields", params_map.get("field"))
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list) or not fields:
        return None
    return CountByParams(fields=tuple(cast(JSONValue, field) for field in fields))


def _apply_count_by(rows: Relation, params: CountByParams) -> Relation:
    counts: dict[tuple[object, ...], dict[str, JSONValue]] = {}
    for row in rows:
        check_deadline()
        key_parts: list[object] = []
        for field in params.fields:
            check_deadline()
            if not isinstance(field, str):
                key_parts.append("")
                continue
            key_parts.append(_hashable(row.get(field)))
        key = tuple(key_parts)
        record = counts.get(key)
        if record is None:
            record = {str(field): row.get(field) for field in params.fields if isinstance(field, str)}
            record["count"] = 0
            counts[key] = record
        record["count"] = int(record.get("count", 0)) + 1
    ordered_group_keys = sort_once(
        counts,
        source="apply_spec.count_by.group_keys",
        policy=OrderPolicy.SORT,
        key=lambda key: tuple(_sort_value(cast(JSONValue, part)) for part in key),
    )
    return [counts[key] for key in ordered_group_keys]


def _traverse_params_from_map(params_map: Mapping[str, JSONValue]) -> TraverseParams | None:
    field = params_map.get("field")
    if not isinstance(field, str) or not field.strip():
        return None
    field = field.strip()
    merge = params_map.get("merge", True)
    if not isinstance(merge, bool):
        merge = True
    keep = params_map.get("keep", False)
    if not isinstance(keep, bool):
        keep = False
    prefix = params_map.get("prefix", "")
    if not isinstance(prefix, str):
        prefix = ""
    as_field = params_map.get("as", field)
    if not isinstance(as_field, str) or not as_field.strip():
        as_field = field
    index_field = params_map.get("index")
    if not isinstance(index_field, str) or not index_field.strip():
        index_field = ""
    return TraverseParams(
        field=field,
        merge=merge,
        keep=keep,
        prefix=prefix,
        as_field=as_field,
        index_field=index_field,
    )


def _apply_traverse(rows: Relation, params: TraverseParams) -> Relation:
    traversed: Relation = []
    for row in rows:
        check_deadline()
        seq = row.get(params.field)
        if not isinstance(seq, list):
            continue
        base = dict(row)
        if not params.keep:
            base.pop(params.field, None)
        for idx, element in enumerate(seq):
            check_deadline()
            out = dict(base)
            if params.index_field:
                out[params.index_field] = idx
            if params.merge and isinstance(element, Mapping):
                for key, value in element.items():
                    check_deadline()
                    merged_key = f"{params.prefix}{key}" if params.prefix else key
                    out[str(merged_key)] = value
            else:
                out[params.as_field] = element
            traversed.append(out)
    return traversed


def _sort_params_from_map(params_map: Mapping[str, JSONValue]) -> SortParams | None:
    by = params_map.get("by", [])
    if isinstance(by, Mapping):
        by = [by]
    if not isinstance(by, list):
        return None
    keys: list[SortKey] = []
    for entry in by:
        check_deadline()
        if not isinstance(entry, Mapping):
            continue
        field = entry.get("field")
        order = entry.get("order", "asc")
        if not isinstance(field, str):
            continue
        if not isinstance(order, str):
            order = "asc"
        keys.append(SortKey(field=field, order=order.strip().lower() or "asc"))
    return SortParams(keys=tuple(keys))


def _limit_params_from_map(params_map: Mapping[str, JSONValue]) -> LimitParams | None:
    count = params_map.get("count")
    try:
        limit = int(count) if count is not None else None
    except (TypeError, ValueError):
        limit = None
    return LimitParams(count=limit)
