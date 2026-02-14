from __future__ import annotations

import json

import pytest

from gabion.analysis.canonical import canon, digest_index, encode_canon
from gabion.exceptions import NeverThrown


def test_canon_is_idempotent_and_order_invariant() -> None:
    left = {"b": 2, "a": {"z": 9, "y": [3, {"k": "v"}]}}
    right = {"a": {"y": [3, {"k": "v"}], "z": 9}, "b": 2}

    normalized = canon(left)
    assert normalized == canon(right)
    assert normalized == canon(normalized)


def test_canon_normalizes_multiset_encoding() -> None:
    value = [
        "ms",
        [
            [{"kind": "b"}, 1],
            [{"kind": "a"}, 2],
            [{"kind": "a"}, 3],
        ],
    ]
    normalized = canon(value)
    assert normalized == [
        "ms",
        [
            [{"kind": "a"}, 5],
            [{"kind": "b"}, 1],
        ],
    ]


def test_canon_rejects_set_inputs() -> None:
    with pytest.raises(NeverThrown):
        canon({"unordered": {1, 2, 3}})


def test_encode_and_digest_are_stable_for_equivalent_payloads() -> None:
    left = {"beta": [2, 1], "alpha": {"k": "v"}}
    right = {"alpha": {"k": "v"}, "beta": [2, 1]}
    assert encode_canon(left) == encode_canon(right)
    assert digest_index(left) == digest_index(right)
    assert json.loads(encode_canon(left)) == canon(left)
