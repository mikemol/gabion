# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from gabion.order_contract import enforce_ordered, sort_once


def canonicalize_json(value: object) -> object:
    if isinstance(value, Mapping):
        normalized_items = [
            (str(key), canonicalize_json(item_value))
            for key, item_value in value.items()
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
    if isinstance(value, list):
        return [canonicalize_json(item) for item in value]
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
    if not isinstance(payload, Mapping):
        return {}
    canonical = canonicalize_json(payload)
    return canonical if isinstance(canonical, dict) else {}


def load_json_object_text(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    canonical = canonicalize_json(payload)
    return canonical if isinstance(canonical, dict) else {}


def dump_json_pretty(payload: object) -> str:
    ordered = enforce_json_ordered(payload, source="json_io.dump_json_pretty")
    return json.dumps(ordered, indent=2, sort_keys=False)


def enforce_json_ordered(value: object, *, source: str) -> object:
    if isinstance(value, Mapping):
        normalized_items = [(str(key), value[key]) for key in value]
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
    if isinstance(value, list):
        return [enforce_json_ordered(item, source=f"{source}.list_item") for item in value]
    return value
