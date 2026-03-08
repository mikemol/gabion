from __future__ import annotations

import json
from functools import singledispatch

from gabion.exceptions import NeverThrown
from gabion.invariants import never
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
    try:
        return _normalize(value, source=source)
    except NeverThrown as error:
        if str(error) != "unregistered runtime type":
            raise
        value_type = type(value).__name__
        match error.marker_payload.env.get("value_type"):
            case str() as nested_value_type if nested_value_type:
                value_type = nested_value_type
            case _:
                pass
        raise TypeError(
            "stable_json_value does not support value type "
            f"{value_type} at {source}"
        ) from None


@singledispatch
def _normalize(value: object, *, source: str) -> object:
    never(
        "unregistered runtime type",
        value_type=type(value).__name__,
    )


@_normalize.register(dict)
def _normalize_dict(value: dict[object, object], *, source: str) -> object:
    keyed_items = sort_once(
        ((str(key), key) for key in value),
        source=f"{source}.mapping_keys",
        # Lexical mapping-key order defines canonical mapping identity.
        key=lambda item: item[0],
    )
    return {
        key_text: _normalize(value[raw_key], source=f"{source}.{key_text}")
        for key_text, raw_key in keyed_items
    }


@_normalize.register(tuple)
def _normalize_tuple(value: tuple[object, ...], *, source: str) -> object:
    return [_normalize(item, source=f"{source}.tuple_item") for item in value]


@_normalize.register(list)
def _normalize_list(value: list[object], *, source: str) -> object:
    return [_normalize(item, source=f"{source}.list_item") for item in value]


@_normalize.register(set)
def _normalize_set(value: set[object], *, source: str) -> object:
    normalized_items = [_normalize(item, source=f"{source}.set_item") for item in value]
    return sort_once(
        normalized_items,
        source=f"{source}.set_items",
        # Non-lexical comparator: type-tag + deterministic stable text.
        key=lambda item: (type(item).__name__, stable_compact_text(item)),
    )


@_normalize.register(str)
@_normalize.register(int)
@_normalize.register(float)
@_normalize.register(bool)
@_normalize.register(type(None))
def _normalize_scalar(value: object, *, source: str) -> object:
    del source
    return value
