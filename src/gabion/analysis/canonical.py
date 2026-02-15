from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from gabion.json_types import JSONValue
from gabion.invariants import never
from gabion.order_contract import ordered_or_sorted
from gabion.analysis.timeout_context import check_deadline


def canon(value: object) -> JSONValue:
    check_deadline()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        keys = ordered_or_sorted(
            (str(key) for key in value.keys()),
            source="canonical.canon.dict_keys",
        )
        return {key: canon(value[key]) for key in keys}
    if isinstance(value, tuple):
        return [canon(item) for item in value]
    if isinstance(value, list):
        if _looks_multiset(value):
            return _canon_multiset(value)
        return [canon(item) for item in value]
    if isinstance(value, set):
        never(
            "canon() does not accept unordered set inputs",
            value_type=type(value).__name__,
        )
    never(
        "canon() received non-JSON value",
        value_type=type(value).__name__,
    )


def encode_canon(value: object) -> str:
    check_deadline()
    return json.dumps(
        canon(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def digest_index(value: object) -> str:
    check_deadline()
    encoded = encode_canon(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _looks_multiset(value: list[object]) -> bool:
    if len(value) != 2:
        return False
    marker = value[0]
    pairs = value[1]
    return marker == "ms" and isinstance(pairs, Sequence)


def _canon_multiset(value: list[object]) -> JSONValue:
    marker = value[0]
    if marker != "ms":
        never("multiset marker must be 'ms'", marker=marker)
    raw_pairs = value[1]
    if not isinstance(raw_pairs, Sequence):
        never("multiset payload must be a sequence", payload_type=type(raw_pairs).__name__)
    counts: dict[str, tuple[JSONValue, int]] = {}
    for raw in raw_pairs:
        check_deadline()
        if not isinstance(raw, Sequence) or len(raw) != 2:
            never("multiset pair must contain [value, count]", pair=raw)
        pair_value = canon(raw[0])
        try:
            pair_count = int(raw[1])
        except (TypeError, ValueError):
            never("multiset count must be an integer", count=raw[1])
        if pair_count <= 0:
            never("multiset count must be positive", count=pair_count)
        encoded = _encode_json(pair_value)
        previous = counts.get(encoded)
        if previous is None:
            counts[encoded] = (pair_value, pair_count)
            continue
        counts[encoded] = (previous[0], previous[1] + pair_count)
    ordered = ordered_or_sorted(
        counts.keys(),
        source="canonical.canon.multiset_keys",
    )
    normalized_pairs: list[JSONValue] = []
    for encoded in ordered:
        check_deadline()
        pair_value, pair_count = counts[encoded]
        normalized_pairs.append([pair_value, pair_count])
    return ["ms", normalized_pairs]


def _encode_json(value: JSONValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
