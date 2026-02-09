from __future__ import annotations

import json
from typing import Callable, Iterable, Mapping

from gabion.analysis.projection_normalize import normalize_spec
from gabion.analysis.projection_spec import ProjectionSpec
from gabion.json_types import JSONValue
from gabion.analysis.timeout_context import check_deadline

Relation = list[dict[str, JSONValue]]


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
        if not isinstance(op, Mapping):
            continue
        op_name = op.get("op")
        params_map = op.get("params")
        if not isinstance(params_map, Mapping):
            params_map = {}

        if op_name == "select":
            predicates = params_map.get("predicates", [])
            if isinstance(predicates, str):
                predicates = [predicates]
            if not isinstance(predicates, list):
                predicates = []
            for predicate_name in predicates:
                if not isinstance(predicate_name, str):
                    continue
                predicate = op_registry.get(predicate_name)
                if predicate is None:
                    continue
                current = [row for row in current if predicate(row, params)]
            continue

        if op_name == "project":
            fields = params_map.get("fields", [])
            if isinstance(fields, str):
                fields = [fields]
            if not isinstance(fields, list):
                continue
            projected: Relation = []
            for row in current:
                projected.append({field: row.get(field) for field in fields})
            current = projected
            continue

        if op_name == "count_by":
            fields = params_map.get("fields", params_map.get("field"))
            if isinstance(fields, str):
                fields = [fields]
            if not isinstance(fields, list) or not fields:
                continue
            counts: dict[tuple[object, ...], dict[str, JSONValue]] = {}
            for row in current:
                key_parts: list[object] = []
                for field in fields:
                    if not isinstance(field, str):
                        key_parts.append("")
                        continue
                    value = row.get(field)
                    key_parts.append(_hashable(value))
                key = tuple(key_parts)
                record = counts.get(key)
                if record is None:
                    record = {str(field): row.get(field) for field in fields if isinstance(field, str)}
                    record["count"] = 0
                    counts[key] = record
                record["count"] = int(record.get("count", 0)) + 1
            current = list(counts.values())
            continue

        if op_name == "sort":
            by = params_map.get("by", [])
            if isinstance(by, Mapping):
                by = [by]
            if not isinstance(by, list):
                continue
            sort_keys: list[tuple[str, str]] = []
            for entry in by:
                if not isinstance(entry, Mapping):
                    continue
                field = entry.get("field")
                order = entry.get("order", "asc")
                if not isinstance(field, str):
                    continue
                if not isinstance(order, str):
                    order = "asc"
                order_norm = order.strip().lower() or "asc"
                sort_keys.append((field, order_norm))
            for field, order in reversed(sort_keys):
                current = sorted(
                    current,
                    key=lambda row, name=field: _sort_value(row.get(name)),
                    reverse=order == "desc",
                )
            continue

        if op_name == "limit":
            count = params_map.get("count")
            try:
                limit = int(count) if count is not None else None
            except (TypeError, ValueError):
                limit = None
            if limit is None or limit < 0:
                continue
            current = current[:limit]
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
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value
