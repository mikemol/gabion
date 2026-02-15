from __future__ import annotations

import json

import pytest

from gabion.analysis.canonical import _canon_multiset, canon, digest_index, encode_canon
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


def test_canon_converts_tuple_to_list() -> None:
    assert canon(("a", 1, {"b": 2})) == ["a", 1, {"b": 2}]


def test_canon_rejects_non_json_object() -> None:
    with pytest.raises(NeverThrown):
        canon(object())


@pytest.mark.parametrize(
    "value",
    [
        ["ms", [("x",)]],
        ["ms", [[{"k": "v"}, "nope"]]],
        ["ms", [[{"k": "v"}, 0]]],
    ],
)
def test_canon_rejects_invalid_multiset_shapes(value: object) -> None:
    with pytest.raises(NeverThrown):
        canon(value)


def test_canon_multiset_private_guards() -> None:
    with pytest.raises(NeverThrown):
        _canon_multiset(["ms", object()])
    with pytest.raises(NeverThrown):
        _canon_multiset(["bad", []])
