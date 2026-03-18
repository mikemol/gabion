from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from gabion.analysis.foundation.resume_codec import sequence_optional
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.json_types import JSONValue
from gabion.invariants import never
from gabion.order_contract import sort_once

# gabion:grade_boundary kind=semantic_carrier_adapter name=canonical.canon
def canon(value: object) -> JSONValue:
    check_deadline()
    match value:
        case None | str() | int() | float() | bool():
            return value
        case Mapping() as value_mapping:
            keys = sort_once(
                (str(key) for key in value_mapping.keys()),
                source="canonical.canon.dict_keys",
            )
            return {key: canon(value_mapping[key]) for key in keys}
        case tuple() as tuple_value:
            return [canon(item) for item in tuple_value]
        case ["ms", pairs] as multiset_value if sequence_optional(pairs) is not None:
            return _canon_multiset(multiset_value)
        case list() as list_value:
            return [canon(item) for item in list_value]
        case set():
            never("canon() does not accept unordered set inputs")
        case _:
            never("canon() received non-JSON value")


def encode_canon(value: object) -> str:
    check_deadline()
    return json.dumps(
        canon(value),
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def digest_index(value: object) -> str:
    check_deadline()
    encoded = encode_canon(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _canon_multiset(value: list[object]) -> JSONValue:
    marker = value[0] if value else None
    if marker != "ms":
        never("multiset marker must be 'ms'", marker=marker)
    raw_pairs = value[1] if len(value) > 1 else None
    pair_sequence = sequence_optional(raw_pairs)
    if pair_sequence is None:
        never(
            "multiset payload must be a sequence",
            payload_type=type(raw_pairs).__name__,
        )

    counts: dict[str, tuple[JSONValue, int]] = {}
    for raw in pair_sequence:
        check_deadline()
        pair = sequence_optional(raw)
        if pair is None or len(pair) != 2:
            never("multiset pair must contain [value, count]", pair=raw)
        pair_value = canon(pair[0])
        try:
            pair_count = int(pair[1])
        except (TypeError, ValueError):
            never("multiset count must be an integer", count=pair[1])
        if pair_count <= 0:
            never("multiset count must be positive", count=pair_count)
        encoded = _encode_json(pair_value)
        previous = counts.get(encoded)
        if previous is None:
            counts[encoded] = (pair_value, pair_count)
            continue
        counts[encoded] = (previous[0], previous[1] + pair_count)
    ordered = sort_once(
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
    return json.dumps(value, sort_keys=False, separators=(",", ":"), ensure_ascii=False)
