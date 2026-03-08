from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
import json
from collections.abc import Callable, Iterable, Mapping
from typing import cast

from gabion.analysis.foundation.resume_codec import str_list_from_sequence
from gabion.analysis.projection.projection_normalize import normalize_spec
from gabion.analysis.projection.projection_spec import ProjectionSpec
from gabion.json_types import JSONValue
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never
from gabion.order_contract import OrderPolicy, sort_once
from gabion.runtime_shape_dispatch import (
    int_or_none,
    json_list_or_none,
    json_mapping_or_none,
    str_or_none,
)

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


@singledispatch
def _string_sequence_payload(value: JSONValue) -> tuple[str, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_string_sequence_payload.register(str)
def _(value: str) -> tuple[str, ...]:
    return (value,)


@_string_sequence_payload.register(list)
def _(value: list[JSONValue]) -> tuple[str, ...]:
    return tuple(str_list_from_sequence(value))


@_string_sequence_payload.register(tuple)
def _(value: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(str_list_from_sequence(value))


@_string_sequence_payload.register(set)
def _(value: set[object]) -> tuple[str, ...]:
    return tuple(str_list_from_sequence(value))


def _none_string_sequence(value: JSONValue) -> tuple[str, ...]:
    _ = value
    return ()


for _runtime_type in (dict, int, float, bool, type(None)):
    _string_sequence_payload.register(_runtime_type)(_none_string_sequence)


@singledispatch
def _sort_entries_payload(value: JSONValue) -> tuple[Mapping[str, JSONValue], ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_sort_entries_payload.register(dict)
def _(value: dict[str, JSONValue]) -> tuple[Mapping[str, JSONValue], ...]:
    return (value,)


@_sort_entries_payload.register(list)
def _(value: list[JSONValue]) -> tuple[Mapping[str, JSONValue], ...]:
    entries: list[Mapping[str, JSONValue]] = []
    for entry in value:
        check_deadline()
        entry_map = json_mapping_or_none(entry)
        if entry_map is not None:
            entries.append(entry_map)
    return tuple(entries)


def _none_sort_entries(value: JSONValue) -> tuple[Mapping[str, JSONValue], ...]:
    _ = value
    return ()


for _runtime_type in (str, int, float, bool, tuple, set, type(None)):
    _sort_entries_payload.register(_runtime_type)(_none_sort_entries)


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
    spec_params_map = json_mapping_or_none(spec_params)
    if spec_params_map is not None:
        params.update(spec_params_map)
    if params_override:
        params.update(params_override)
    op_registry = op_registry or {}

    current: Relation = [
        dict(cast(Mapping[str, JSONValue], row))
        for row in rows
    ]

    for op in normalized.get("pipeline") or []:
        check_deadline()
        op_map = json_mapping_or_none(op)
        if op_map is not None:
            op_name = op_map.get("op")
            params_payload = op_map.get("params")
            params_map = json_mapping_or_none(params_payload) or {}

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
                limit_count = int_or_none(limit_params.count)
                if limit_count is not None and limit_count >= 0:
                    current = current[:limit_count]

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
    predicate_values = _string_sequence_payload(predicates)
    names: list[str] = []
    for name in predicate_values:
        predicate_name = str_or_none(name)
        if predicate_name is not None:
            names.append(predicate_name)
    return SelectParams(predicates=tuple(names))


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
    field_values = _string_sequence_payload(fields)
    normalized_fields_list = [field_text.strip() for field_text in field_values if field_text.strip()]
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
    field_values = _string_sequence_payload(fields)
    normalized_fields_list = [field_text.strip() for field_text in field_values if field_text.strip()]
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
    field_name = str_or_none(field)
    if field_name is None or not field_name.strip():
        return TraverseParams(field="")
    field = field_name.strip()
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
    prefix = str_or_none(prefix) or ""
    as_field = params_map.get("as", field)
    as_field_text = str_or_none(as_field)
    as_field = as_field_text if as_field_text and as_field_text.strip() else field
    index_field = params_map.get("index")
    index_field_text = str_or_none(index_field)
    index_field = (
        index_field_text
        if index_field_text and index_field_text.strip()
        else ""
    )
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
        items = json_list_or_none(seq)
        if items is not None:
            base = dict(row)
            if not params.keep:
                base.pop(params.field, None)
            for idx, element in enumerate(items):
                check_deadline()
                out = dict(base)
                if params.index_field:
                    out[params.index_field] = idx
                element_map = json_mapping_or_none(element)
                if element_map is not None and params.merge:
                    for key, value in element_map.items():
                        check_deadline()
                        merged_key = f"{params.prefix}{key}" if params.prefix else key
                        out[str(merged_key)] = value
                else:
                    out[params.as_field] = element
                traversed.append(out)
    return traversed


def _sort_params_from_map(params_map: Mapping[str, JSONValue]) -> SortParams:
    by = params_map.get("by", [])
    by_entries = _sort_entries_payload(by)
    keys: list[SortKey] = []
    for entry in by_entries:
        check_deadline()
        field_text = str_or_none(entry.get("field"))
        if field_text is not None:
            order_raw = str_or_none(entry.get("order", "asc"))
            normalized_order = (order_raw or "asc").strip().lower() or "asc"
            keys.append(
                SortKey(
                    field=field_text,
                    order=normalized_order,
                )
            )
    return SortParams(keys=tuple(keys))


def _limit_params_from_map(params_map: Mapping[str, JSONValue]) -> LimitParams:
    count = params_map.get("count")
    try:
        limit = int(count) if count is not None else None
    except (TypeError, ValueError):
        limit = None
    return LimitParams(count=limit)
