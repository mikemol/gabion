# gabion:decision_protocol_module
from __future__ import annotations

from typing import Mapping, cast

from gabion.invariants import never
from gabion.order_contract import (
    enforce_ordered,
    is_sorted_once_carrier,
    sort_once,
    sorted_once_source,
)
from gabion.runtime import stable_encode

# Sort-contract note:
# - Mapping items are ordered by `(stringified_key, _stable_value_key(value))`.
# - Sequence-like values are ordered by `_stable_value_key(value)`.
# - `_stable_value_key(value)` is non-lexical: `(type_name, canonical_json_or_repr)`.
# This keeps boundary canonicalization deterministic across mixed JSON-ish payloads.


class CanonicalBoundaryDict(dict[str, object]):
    """Boundary carrier marker: active sort already consumed."""

    __slots__ = ("_gabion_sorted_once", "_gabion_sort_source")

    def __init__(self, *, source: str) -> None:
        super().__init__()
        self._gabion_sorted_once = True
        self._gabion_sort_source = source


def canonicalize_boundary_mapping(
    payload: Mapping[str, object],
    *,
    source: str,
) -> dict[str, object]:
    if is_sorted_once_carrier(payload):
        never(
            "active sort attempted after first normalization",
            source=source,
            first_sort_source=sorted_once_source(payload),
        )
    return cast(dict[str, object], _canonicalize_value(payload, source=source))


def canonicalize_boundary_value(
    value: object,
    *,
    source: str,
) -> object:
    return _canonicalize_value(value, source=source)


def normalize_boundary_mapping_once(
    payload: Mapping[str, object],
    *,
    source: str,
) -> dict[str, object]:
    if is_sorted_once_carrier(payload):
        return enforce_boundary_mapping_ordered(payload, source=source)
    return canonicalize_boundary_mapping(payload, source=source)


def normalize_boundary_value_once(
    value: object,
    *,
    source: str,
) -> object:
    if is_sorted_once_carrier(value):
        return enforce_boundary_value_ordered(value, source=source)
    return canonicalize_boundary_value(value, source=source)


def apply_boundary_updates_once(
    payload: Mapping[str, object],
    updates: Mapping[str, object],
    *,
    source: str,
) -> dict[str, object]:
    """Merge updates into an ordered mapping without re-sorting the base carrier.

    Sort-contract note:
    - `payload` is normalized or enforced once at entry.
    - `updates` is normalized once as its own carrier.
    - merge is key-ordered linear merge; no second active sort on `payload`.
    """
    base = normalize_boundary_mapping_once(
        payload,
        source=f"{source}.base",
    )
    if not updates:
        return base
    normalized_updates = normalize_boundary_mapping_once(
        updates,
        source=f"{source}.updates",
    )
    if not normalized_updates:
        return base

    base_items = [(key, base[key]) for key in base if key not in normalized_updates]
    update_items = [(key, normalized_updates[key]) for key in normalized_updates]
    merged = CanonicalBoundaryDict(source=source)
    base_index = 0
    update_index = 0
    while base_index < len(base_items) and update_index < len(update_items):
        base_item = base_items[base_index]
        update_item = update_items[update_index]
        if _mapping_item_sort_key(base_item) <= _mapping_item_sort_key(update_item):
            merged[base_item[0]] = base_item[1]
            base_index += 1
            continue
        merged[update_item[0]] = update_item[1]
        update_index += 1
    while base_index < len(base_items):
        key, value = base_items[base_index]
        merged[key] = value
        base_index += 1
    while update_index < len(update_items):
        key, value = update_items[update_index]
        merged[key] = value
        update_index += 1
    return merged


def enforce_boundary_mapping_ordered(
    payload: Mapping[str, object],
    *,
    source: str,
) -> dict[str, object]:
    return cast(dict[str, object], _enforce_ordered_value(payload, source=source))


def enforce_boundary_value_ordered(
    value: object,
    *,
    source: str,
) -> object:
    return _enforce_ordered_value(value, source=source)


def _canonicalize_value(
    value: object,
    *,
    source: str,
) -> object:
    if isinstance(value, Mapping):
        if is_sorted_once_carrier(value):
            return _enforce_ordered_value(value, source=source)
        items = [(str(key), value[key]) for key in value]
        ordered_items = sort_once(
            items,
            source=f"{source}.mapping_items",
            key=_mapping_item_sort_key,
        )
        normalized = CanonicalBoundaryDict(source=source)
        for key, raw_value in ordered_items:
            normalized[key] = _canonicalize_value(
                raw_value,
                source=f"{source}.{key}",
            )
        return normalized
    if isinstance(value, list):
        if is_sorted_once_carrier(value):
            return _enforce_ordered_value(value, source=source)
        normalized_values = [
            _canonicalize_value(item, source=f"{source}.list_item")
            for item in value
        ]
        return sort_once(
            normalized_values,
            source=f"{source}.list_items",
            key=_stable_value_key,
        )
    if isinstance(value, tuple):
        normalized_values = [
            _canonicalize_value(item, source=f"{source}.tuple_item")
            for item in value
        ]
        ordered_values = sort_once(
            normalized_values,
            source=f"{source}.tuple_items",
            key=_stable_value_key,
        )
        return tuple(ordered_values)
    if isinstance(value, set):
        normalized_values = [
            _canonicalize_value(item, source=f"{source}.set_item")
            for item in value
        ]
        return sort_once(
            normalized_values,
            source=f"{source}.set_items",
            key=_stable_value_key,
        )
    return value


def _enforce_ordered_value(
    value: object,
    *,
    source: str,
) -> object:
    if isinstance(value, Mapping):
        items = [(str(key), value[key]) for key in value]
        ordered_items = enforce_ordered(
            items,
            source=f"{source}.mapping_items",
            key=_mapping_item_sort_key,
        )
        normalized = CanonicalBoundaryDict(source=source)
        for key, raw_value in ordered_items:
            normalized[key] = _enforce_ordered_value(raw_value, source=f"{source}.{key}")
        return normalized
    if isinstance(value, list):
        normalized_values = [
            _enforce_ordered_value(item, source=f"{source}.list_item")
            for item in value
        ]
        return enforce_ordered(
            normalized_values,
            source=f"{source}.list_items",
            key=_stable_value_key,
        )
    if isinstance(value, tuple):
        normalized_values = [
            _enforce_ordered_value(item, source=f"{source}.tuple_item")
            for item in value
        ]
        ordered_values = enforce_ordered(
            normalized_values,
            source=f"{source}.tuple_items",
            key=_stable_value_key,
        )
        return tuple(ordered_values)
    if isinstance(value, set):
        never(
            "unordered set cannot cross egress boundary",
            source=source,
        )
    return value


def _stable_value_key(value: object) -> tuple[str, str]:
    # Non-lexical sort key: primarily by runtime type, then stable text form.
    return (type(value).__name__, _stable_text(value))


def _mapping_item_sort_key(item: tuple[str, object]) -> tuple[str, tuple[str, str]]:
    # Mapping key comparator is shared across normalize/enforce/merge surfaces.
    # Primary order is lexical key text; value key is a deterministic tie-breaker.
    return (item[0], _stable_value_key(item[1]))


def _stable_text(value: object) -> str:
    return stable_encode.stable_compact_text(value, ensure_ascii=True)
