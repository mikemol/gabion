# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Callable, Iterable, Mapping
from typing import cast

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
    fields: tuple[str, ...]


@dataclass(frozen=True)
class CountByParams:
    fields: tuple[str, ...]


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
    count: object = None


def apply_spec(
    spec: ProjectionSpec,
    rows: Iterable[Mapping[str, JSONValue]],
    *,
    op_registry = None,
    params_override = None,
    normalize = None,
) -> Relation:
    check_deadline()
    normalize_fn = normalize or normalize_spec
    normalized = normalize_fn(spec)
    params: dict[str, JSONValue] = {}
    spec_params = normalized.get("params")
    match spec_params:
        case Mapping() as spec_params_map:
            params.update(spec_params_map)
        case _:
            pass
    if params_override:
        params.update(params_override)
    op_registry = op_registry or {}

    current: Relation = []
    for row in rows:
        match row:
            case Mapping() as row_map:
                current.append(dict(row_map))
            case _:
                pass

    for op in normalized.get("pipeline") or []:
        check_deadline()
        match op:
            case Mapping() as op_map:
                op_name = op_map.get("op")
                params_payload = op_map.get("params")
                match params_payload:
                    case Mapping() as params_payload_map:
                        params_map = params_payload_map
                    case _:
                        params_map = {}

                if op_name == "select":
                    select_params = _select_params_from_map(params_map)
                    current = _apply_select(
                        current,
                        select_params,
                        op_registry=op_registry,
                        runtime_params=params,
                    )
                elif op_name == "project":
                    project_params = _project_params_from_map(params_map)
                    if project_params.fields:
                        current = _apply_project(current, project_params)
                elif op_name == "count_by":
                    count_params = _count_by_params_from_map(params_map)
                    if count_params.fields:
                        current = _apply_count_by(current, count_params)
                elif op_name == "traverse":
                    traverse_params = _traverse_params_from_map(params_map)
                    if traverse_params.field:
                        current = _apply_traverse(current, traverse_params)
                elif op_name == "sort":
                    sort_params = _sort_params_from_map(params_map)
                    for key in reversed(sort_params.keys):
                        check_deadline()
                        current = sort_once(
                            current,
                            source=f"apply_spec.sort[{key.field}]",
                            policy=OrderPolicy.SORT,
                            key=lambda row, name=key.field: _sort_value(row.get(name)),
                            reverse=key.order == "desc",
                        )
                elif op_name == "limit":
                    limit_params = _limit_params_from_map(params_map)
                    match limit_params.count:
                        case int() as limit_count if limit_count >= 0:
                            current = current[:limit_count]
                        case _:
                            pass
            case _:
                pass

    return current


def _sort_value(value: JSONValue) -> tuple[int, object]:
    if value is None:
        return (1, "")
    match value:
        case int() | float() | str():
            return (0, value)
        case _:
            return (0, str(value))


def _hashable(value: JSONValue) -> object:
    try:
        hash(value)
    except TypeError:
        return json.dumps(value, sort_keys=False, separators=(",", ":"))
    return value


def _select_params_from_map(params_map: Mapping[str, JSONValue]) -> SelectParams:
    predicates = params_map.get("predicates", [])
    match predicates:
        case str() as predicate_name:
            predicate_values: list[object] = [predicate_name]
        case list() as predicate_list:
            predicate_values = [value for value in predicate_list]
        case _:
            predicate_values = []
    names = tuple(name for name in predicate_values if type(name) is str)
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
        if predicate is not None:
            selected = [row for row in selected if predicate(row, runtime_params)]
    return selected


def _project_params_from_map(params_map: Mapping[str, JSONValue]) -> ProjectParams:
    fields = params_map.get("fields", [])
    match fields:
        case str() as field_name:
            field_values: list[object] = [field_name]
        case list() as field_list:
            field_values = [value for value in field_list]
        case _:
            field_values = []
    normalized_fields_list: list[str] = []
    for field in field_values:
        match field:
            case str() as field_text if field_text.strip():
                normalized_fields_list.append(field_text.strip())
            case _:
                pass
    normalized_fields = tuple(normalized_fields_list)
    return ProjectParams(fields=normalized_fields)


def _apply_project(rows: Relation, params: ProjectParams) -> Relation:
    projected: Relation = []
    for row in rows:
        check_deadline()
        projected.append({field: row.get(field) for field in params.fields})
    return projected


def _count_by_params_from_map(params_map: Mapping[str, JSONValue]) -> CountByParams:
    fields = params_map.get("fields", params_map.get("field"))
    match fields:
        case str() as field_name:
            field_values: list[object] = [field_name]
        case list() as field_list:
            field_values = [value for value in field_list]
        case _:
            field_values = []
    normalized_fields_list: list[str] = []
    for field in field_values:
        match field:
            case str() as field_text if field_text.strip():
                normalized_fields_list.append(field_text.strip())
            case _:
                pass
    normalized_fields = tuple(normalized_fields_list)
    return CountByParams(fields=normalized_fields)


def _apply_count_by(rows: Relation, params: CountByParams) -> Relation:
    counts: dict[tuple[object, ...], dict[str, JSONValue]] = {}
    for row in rows:
        check_deadline()
        key_parts: list[object] = []
        for field in params.fields:
            check_deadline()
            key_parts.append(_hashable(row.get(field)))
        key = tuple(key_parts)
        record = counts.get(key)
        if record is None:
            record = {field: row.get(field) for field in params.fields}
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


def _traverse_params_from_map(params_map: Mapping[str, JSONValue]) -> TraverseParams:
    field = params_map.get("field")
    match field:
        case str() as field_name if field_name.strip():
            field = field_name.strip()
        case _:
            return TraverseParams(field="")
    merge = params_map.get("merge", True)
    match merge:
        case bool() as merge_bool:
            merge = merge_bool
        case _:
            merge = True
    keep = params_map.get("keep", False)
    match keep:
        case bool() as keep_bool:
            keep = keep_bool
        case _:
            keep = False
    prefix = params_map.get("prefix", "")
    match prefix:
        case str() as prefix_text:
            prefix = prefix_text
        case _:
            prefix = ""
    as_field = params_map.get("as", field)
    match as_field:
        case str() as as_field_text if as_field_text.strip():
            as_field = as_field_text
        case _:
            as_field = field
    index_field = params_map.get("index")
    match index_field:
        case str() as index_field_text if index_field_text.strip():
            index_field = index_field_text
        case _:
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
        match seq:
            case list() as items:
                base = dict(row)
                if not params.keep:
                    base.pop(params.field, None)
                for idx, element in enumerate(items):
                    check_deadline()
                    out = dict(base)
                    if params.index_field:
                        out[params.index_field] = idx
                    match element:
                        case Mapping() as element_map if params.merge:
                            for key, value in element_map.items():
                                check_deadline()
                                merged_key = f"{params.prefix}{key}" if params.prefix else key
                                out[str(merged_key)] = value
                        case _:
                            out[params.as_field] = element
                    traversed.append(out)
            case _:
                pass
    return traversed


def _sort_params_from_map(params_map: Mapping[str, JSONValue]) -> SortParams:
    by = params_map.get("by", [])
    match by:
        case Mapping() as by_map:
            by_entries: list[object] = [by_map]
        case list() as by_list:
            by_entries = [entry for entry in by_list]
        case _:
            by_entries = []
    keys: list[SortKey] = []
    for entry in by_entries:
        check_deadline()
        match entry:
            case Mapping() as entry_map:
                field = entry_map.get("field")
                order = entry_map.get("order", "asc")
                match field:
                    case str() as field_text:
                        match order:
                            case str() as order_text:
                                normalized_order = order_text
                            case _:
                                normalized_order = "asc"
                        keys.append(
                            SortKey(
                                field=field_text,
                                order=normalized_order.strip().lower() or "asc",
                            )
                        )
                    case _:
                        pass
            case _:
                pass
    return SortParams(keys=tuple(keys))


def _limit_params_from_map(params_map: Mapping[str, JSONValue]) -> LimitParams:
    count = params_map.get("count")
    try:
        limit = int(count) if count is not None else None
    except (TypeError, ValueError):
        limit = None
    return LimitParams(count=limit)
