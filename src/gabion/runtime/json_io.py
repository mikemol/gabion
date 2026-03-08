from __future__ import annotations

import json
from pathlib import Path

from gabion.order_contract import enforce_ordered, sort_once
from gabion.runtime_shape_dispatch import (
    json_list_or_none as _json_list_or_none,
    json_mapping_or_none as _json_mapping_or_none,
)


def canonicalize_json(value: object) -> object:
    mapping_value = _json_mapping_or_none(value)
    if mapping_value is not None:
        normalized_items = [
            (str(key), canonicalize_json(item_value))
            for key, item_value in mapping_value.items()
        ]
        ordered_items = sort_once(
            normalized_items,
            source="json_io.canonicalize_json.mapping_items",
            # Sort key is lexical mapping-key text for canonical JSON shape.
            key=lambda item: item[0],
        )
        return {
            key: item_value
            for key, item_value in ordered_items
        }
    list_value = _json_list_or_none(value)
    if list_value is not None:
        return [canonicalize_json(item) for item in list_value]
    return value


def load_json_object_path(
    path: Path,
    *,
    encoding: str = "utf-8",
) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding=encoding))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    payload_mapping = _json_mapping_or_none(payload)
    if payload_mapping is None:
        return {}
    canonical = canonicalize_json(payload_mapping)
    canonical_mapping = _json_mapping_or_none(canonical)
    return canonical_mapping if canonical_mapping is not None else {}


def load_json_object_text(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    payload_mapping = _json_mapping_or_none(payload)
    if payload_mapping is None:
        return {}
    canonical = canonicalize_json(payload_mapping)
    canonical_mapping = _json_mapping_or_none(canonical)
    return canonical_mapping if canonical_mapping is not None else {}


def dump_json_pretty(payload: object) -> str:
    ordered = enforce_json_ordered(payload, source="json_io.dump_json_pretty")
    return json.dumps(ordered, indent=2, sort_keys=False)


def enforce_json_ordered(value: object, *, source: str) -> object:
    mapping_value = _json_mapping_or_none(value)
    if mapping_value is not None:
        normalized_items = [(str(key), mapping_value[key]) for key in mapping_value]
        ordered_items = enforce_ordered(
            normalized_items,
            source=f"{source}.mapping_items",
            # Sort key is lexical mapping-key text for canonical JSON shape.
            key=lambda item: item[0],
        )
        return {
            key: enforce_json_ordered(item_value, source=f"{source}.{key}")
            for key, item_value in ordered_items
        }
    list_value = _json_list_or_none(value)
    if list_value is not None:
        return [enforce_json_ordered(item, source=f"{source}.list_item") for item in list_value]
    return value
