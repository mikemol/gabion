# gabion:decision_protocol_module
from __future__ import annotations

import json
from typing import Mapping

from gabion.order_contract import sort_once


def stable_compact_text(
    value: object,
    *,
    ensure_ascii: bool = True,
) -> str:
    """Deterministic text encoder for comparator/hash surfaces.

    Sort-contract note:
    - Mapping keys are sorted lexically exactly once per normalized carrier.
    - Sequence order is preserved; tuple/list normalize to JSON lists.
    - Set-like carriers are normalized into deterministically ordered lists.
    """
    normalized = stable_json_value(
        value,
        source="stable_encode.stable_compact_text",
    )
    return json.dumps(
        normalized,
        separators=(",", ":"),
        sort_keys=False,
        ensure_ascii=ensure_ascii,
    )


def stable_compact_bytes(
    value: object,
    *,
    ensure_ascii: bool = True,
) -> bytes:
    return stable_compact_text(value, ensure_ascii=ensure_ascii).encode("utf-8")


def stable_json_value(
    value: object,
    *,
    source: str,
) -> object:
    """Normalize arbitrary values into deterministic JSON-compatible carriers.

    Sort-contract note:
    - Mapping keys use lexical key text ordering.
    - List/tuple preserve sequence semantics and normalize recursively to lists.
    - Sets normalize to deterministically sorted lists.
    - Unsupported objects are rejected to prevent non-deterministic identity leaks.
    """
    return _normalize(value, source=source)


def _normalize(value: object, *, source: str) -> object:
    if isinstance(value, Mapping):
        keys = sort_once(
            (str(key) for key in value),
            source=f"{source}.mapping_keys",
            # Lexical mapping-key order defines canonical mapping identity.
            key=lambda item: item,
        )
        return {
            key: _normalize(value[key], source=f"{source}.{key}")
            for key in keys
        }
    if isinstance(value, tuple):
        return [
            _normalize(item, source=f"{source}.tuple_item")
            for item in value
        ]
    if isinstance(value, list):
        return [
            _normalize(item, source=f"{source}.list_item")
            for item in value
        ]
    if isinstance(value, set):
        normalized_items = [
            _normalize(item, source=f"{source}.set_item")
            for item in value
        ]
        return sort_once(
            normalized_items,
            source=f"{source}.set_items",
            # Non-lexical comparator: type-tag + deterministic stable text.
            key=lambda item: (type(item).__name__, stable_compact_text(item)),
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        "stable_json_value does not support value type "
        f"{type(value).__name__} at {source}"
    )
