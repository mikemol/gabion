# gabion:decision_protocol_module
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

from gabion.analysis.projection_spec import (
    ProjectionOp,
    ProjectionSpec,
    spec_from_dict,
)
from gabion.analysis.artifact_ordering import canonical_mapping_keys
from gabion.analysis.resume_codec import mapping_or_none
from gabion.json_types import JSONValue
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import OrderPolicy, sort_once

_NO_LIMIT = None


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
    return json.dumps(payload, sort_keys=False, separators=(",", ":"))


def spec_hash(spec) -> str:
    match spec:
        case str() as spec_text:
            return spec_text
        case ProjectionSpec() as projection_spec:
            return spec_canonical_json(projection_spec)
    spec_mapping = mapping_or_none(spec) or {}
    return spec_canonical_json(spec_from_dict(spec_mapping))


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
        params = op.params
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
            if count is not None:
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
    match legacy:
        case str() as legacy_text if legacy_text.strip():
            predicates.append(legacy_text.strip())
        case _:
            pass
    explicit = params.get("predicates")
    match explicit:
        case str() as explicit_text if explicit_text.strip():
            predicates.append(explicit_text.strip())
        case list() as explicit_list:
            for entry in explicit_list:
                check_deadline()
                match entry:
                    case str() as entry_text:
                        predicates.append(entry_text.strip())
                    case _:
                        pass
        case _:
            pass
    return predicates


def _normalize_predicates(values: Iterable[str]) -> list[str]:
    check_deadline()
    # ordered internally; explicit sort only at edge.
    # We deduplicate with an insertion-preserving dict carrier, then hand the
    # keys to sort_once(...) for canonical edge ordering.
    cleaned: dict[str, None] = {}
    for value in values:
        check_deadline()
        if not value:
            continue
        stripped = value.strip()
        if not stripped:
            continue
        cleaned.setdefault(stripped, None)
    return sort_once(
        cleaned,
        source="_normalize_predicates.cleaned",
        policy=OrderPolicy.SORT,
    )


def _normalize_fields(value: JSONValue) -> list[str]:
    check_deadline()
    fields: list[str] = []
    match value:
        case str() as value_text:
            if value_text.strip():
                fields.append(value_text.strip())
        case list() as value_list:
            for entry in value_list:
                check_deadline()
                match entry:
                    case str() as entry_text if entry_text.strip():
                        fields.append(entry_text.strip())
                    case _:
                        pass
        case _:
            pass
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
    return sort_once(
        fields,
        source="_normalize_group_fields.fields",
        policy=OrderPolicy.SORT,
    )


def _normalize_sort_by(value: JSONValue) -> list[dict[str, JSONValue]]:
    check_deadline()
    if value is None:
        return list()
    items: list[dict[str, JSONValue]] = []
    match value:
        case str() as value_text:
            if value_text.strip():
                items.append({"field": value_text.strip(), "order": "asc"})
        case list() as value_list:
            for entry in value_list:
                check_deadline()
                match entry:
                    case str() as entry_text:
                        if entry_text.strip():
                            items.append({"field": entry_text.strip(), "order": "asc"})
                    case Mapping() as entry_map:
                        field = entry_map.get("field") or entry_map.get("key") or entry_map.get("name")
                        match field:
                            case str() as field_text if field_text.strip():
                                order = entry_map.get("order", "asc")
                                match order:
                                    case str() as order_text:
                                        order_norm = order_text.strip().lower() or "asc"
                                    case _:
                                        order_norm = "asc"
                                if order_norm not in {"asc", "desc"}:
                                    order_norm = "asc"
                                items.append({"field": field_text.strip(), "order": order_norm})
                            case _:
                                pass
                    case _:
                        pass
        case _:
            pass
    return items


def _normalize_limit(value: JSONValue):
    if value is None:
        return _NO_LIMIT
    try:
        count = int(value)
    except (TypeError, ValueError):
        return _NO_LIMIT
    if count < 0:
        return _NO_LIMIT
    return count


def _normalize_value(value: JSONValue) -> JSONValue:
    check_deadline()
    match value:
        case dict() as value_map:
            ordered_keys = canonical_mapping_keys(value_map)
            return {str(key): _normalize_value(value_map[key]) for key in ordered_keys}
        case list() as value_list:
            return [_normalize_value(entry) for entry in value_list]
        case _:
            return value
