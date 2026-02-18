from __future__ import annotations

import json
from typing import Iterable, Mapping

from gabion.analysis.projection_spec import (
    ProjectionOp,
    ProjectionSpec,
    spec_from_dict,
)
from gabion.json_types import JSONValue
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import OrderPolicy, ordered_or_sorted


def normalize_spec(spec: ProjectionSpec) -> dict[str, JSONValue]:
    return {
        "spec_version": int(spec.spec_version) if spec.spec_version else 1,
        "name": str(spec.name),
        "domain": str(spec.domain),
        "params": _normalize_value(dict(spec.params)),
        "pipeline": _normalize_pipeline(spec.pipeline),
    }


def spec_canonical_json(spec: ProjectionSpec) -> str:
    payload = normalize_spec(spec)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def spec_hash(spec: ProjectionSpec | Mapping[str, JSONValue] | str) -> str:
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, ProjectionSpec):
        spec = spec_from_dict(spec)
    return spec_canonical_json(spec)


def _normalize_pipeline(pipeline: Iterable[ProjectionOp]) -> list[dict[str, JSONValue]]:
    check_deadline()
    normalized: list[dict[str, JSONValue]] = []
    pending_selects: list[str] = []

    def flush_selects() -> None:
        nonlocal pending_selects
        if not pending_selects:
            return
        predicates = _normalize_predicates(pending_selects)
        if not predicates:
            pending_selects = []
            return
        normalized.append({"op": "select", "params": {"predicates": predicates}})
        pending_selects = []

    for op in pipeline:
        check_deadline()
        op_name = str(op.op).strip()
        params = op.params if isinstance(op.params, Mapping) else {}
        if op_name == "select":
            pending_selects.extend(_extract_predicates(params))
            continue

        flush_selects()

        if op_name == "project":
            fields = _normalize_fields(params.get("fields"))
            if not fields:
                continue
            normalized.append({"op": "project", "params": {"fields": fields}})
            continue

        if op_name == "count_by":
            fields = _normalize_group_fields(
                params.get("fields", params.get("field"))
            )
            if not fields:
                continue
            normalized.append({"op": "count_by", "params": {"fields": fields}})
            continue

        if op_name == "sort":
            by = _normalize_sort_by(params.get("by"))
            if not by:
                continue
            normalized.append({"op": "sort", "params": {"by": by}})
            continue

        if op_name == "limit":
            count = _normalize_limit(params.get("count"))
            if count is None:
                continue
            normalized.append({"op": "limit", "params": {"count": count}})
            continue

        if op_name:
            normalized.append(
                {"op": op_name, "params": _normalize_value(dict(params))}
            )

    flush_selects()
    return normalized


def _extract_predicates(params: Mapping[str, JSONValue]) -> list[str]:
    check_deadline()
    predicates: list[str] = []
    legacy = params.get("predicate")
    if isinstance(legacy, str) and legacy.strip():
        predicates.append(legacy.strip())
    explicit = params.get("predicates")
    if isinstance(explicit, str) and explicit.strip():
        predicates.append(explicit.strip())
    elif isinstance(explicit, list):
        for entry in explicit:
            check_deadline()
            if isinstance(entry, str):
                cleaned = entry.strip()
                predicates.append(cleaned)
    return predicates


def _normalize_predicates(values: Iterable[str]) -> list[str]:
    # ordered internally; explicit sort only at edge.
    # We deduplicate with an insertion-preserving dict carrier, then hand the
    # keys to ordered_or_sorted(...) for canonical edge ordering.
    cleaned: dict[str, None] = {}
    for value in values:
        if not value:
            continue
        stripped = value.strip()
        if not stripped:
            continue
        cleaned.setdefault(stripped, None)
    return ordered_or_sorted(
        cleaned,
        source="_normalize_predicates.cleaned",
        policy=OrderPolicy.SORT,
    )


def _normalize_fields(value: JSONValue) -> list[str]:
    check_deadline()
    fields: list[str] = []
    if isinstance(value, str):
        if value.strip():
            fields.append(value.strip())
    elif isinstance(value, list):
        for entry in value:
            check_deadline()
            if isinstance(entry, str) and entry.strip():
                fields.append(entry.strip())
    seen: set[str] = set()
    ordered: list[str] = []
    for field in fields:
        check_deadline()
        if field in seen:
            continue
        seen.add(field)
        ordered.append(field)
    return ordered


def _normalize_group_fields(value: JSONValue) -> list[str]:
    fields = _normalize_fields(value)
    return ordered_or_sorted(
        fields,
        source="_normalize_group_fields.fields",
        policy=OrderPolicy.SORT,
    )


def _normalize_sort_by(value: JSONValue) -> list[dict[str, JSONValue]]:
    check_deadline()
    if value is None:
        return []
    items: list[dict[str, JSONValue]] = []
    if isinstance(value, str):
        if value.strip():
            items.append({"field": value.strip(), "order": "asc"})
        return items
    if isinstance(value, list):
        for entry in value:
            check_deadline()
            if isinstance(entry, str):
                if entry.strip():
                    items.append({"field": entry.strip(), "order": "asc"})
                continue
            if isinstance(entry, Mapping):
                field = entry.get("field")
                if not field:
                    field = entry.get("key")
                if not field:
                    field = entry.get("name")
                if not isinstance(field, str) or not field.strip():
                    continue
                order = entry.get("order", "asc")
                if not isinstance(order, str):
                    order = "asc"
                order_norm = order.strip().lower() or "asc"
                if order_norm not in {"asc", "desc"}:
                    order_norm = "asc"
                items.append({"field": field.strip(), "order": order_norm})
    return items


def _normalize_limit(value: JSONValue) -> int | None:
    if value is None:
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    if count < 0:
        return None
    return count


def _normalize_value(value: JSONValue) -> JSONValue:
    check_deadline()
    if isinstance(value, dict):
        ordered_keys = ordered_or_sorted(
            value,
            source="_normalize_value.dict_keys",
            policy=OrderPolicy.SORT,
        )
        return {str(k): _normalize_value(value[k]) for k in ordered_keys}
    if isinstance(value, list):
        return [_normalize_value(entry) for entry in value]
    return value
