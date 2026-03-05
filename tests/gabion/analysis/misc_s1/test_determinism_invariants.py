from __future__ import annotations

import pytest

from gabion.analysis.core.determinism_invariants import (
    require_canonical_multiset, require_no_dupes, require_no_python_hash, require_sorted)
from gabion.exceptions import NeverThrown


def test_require_invariants_accept_valid_inputs() -> None:
    assert require_sorted("sorted", [1, 2]) is None
    assert require_no_dupes("dupes", ["a", "b"]) is None
    assert require_canonical_multiset("ms", [("a", 1)]) is None


def test_require_sorted_allows_reverse_sorted_in_reverse_mode() -> None:
    assert require_sorted("descending", [3, 2, 2, 1], reverse=True) is None
    assert require_sorted("empty", []) is None


def test_require_sorted_raises_and_reports_payload() -> None:
    observed: list[dict[str, object]] = []
    with pytest.raises(NeverThrown):
        require_sorted(
            "ascending",
            [1, 3, 2],
            on_violation=lambda payload: observed.append(payload),
            phase="collection",
        )
    assert observed
    assert observed[0]["constraint"] == "sorted"
    assert observed[0]["name"] == "ascending"
    assert observed[0]["phase"] == "collection"


def test_require_no_dupes_raises_and_reports_payload() -> None:
    observed: list[dict[str, object]] = []
    with pytest.raises(NeverThrown):
        require_no_dupes(
            "dupes",
            ["a", "b", "a"],
            on_violation=lambda payload: observed.append(payload),
            scope="wl",
        )
    assert observed
    assert observed[0]["constraint"] == "no_dupes"
    assert observed[0]["scope"] == "wl"


@pytest.mark.parametrize(
    "pairs",
    [
        [("a", 0)],
        [("a", 1), ("a", 1)],
        [("b", 1), ("a", 1)],
    ],
)
def test_require_canonical_multiset_rejects_invalid_inputs(
    pairs: list[tuple[str, int]],
) -> None:
    with pytest.raises(NeverThrown):
        require_canonical_multiset("ms", pairs)


def test_require_canonical_multiset_reports_payload_for_invalid_variants(
) -> None:
    for pairs in (
        [("a", 0)],
        [("a", 1), ("a", 1)],
        [("b", 1), ("a", 1)],
    ):
        observed: list[dict[str, object]] = []
        with pytest.raises(NeverThrown):
            require_canonical_multiset(
                "ms",
                pairs,
                on_violation=lambda payload: observed.append(payload),
                phase="wl",
            )
        assert observed
        assert observed[0]["constraint"] == "canonical_multiset"
        assert observed[0]["phase"] == "wl"


def test_require_no_python_hash_always_raises() -> None:
    observed: list[dict[str, object]] = []
    with pytest.raises(NeverThrown):
        require_no_python_hash(
            "hash-order",
            on_violation=lambda payload: observed.append(payload),
            spec="wl",
        )
    assert observed
    assert observed[0]["constraint"] == "no_python_hash"
    assert observed[0]["spec"] == "wl"


def test_require_invariants_raise_without_callbacks() -> None:
    with pytest.raises(NeverThrown):
        require_sorted("ascending", [2, 1])
    with pytest.raises(NeverThrown):
        require_no_dupes("dupes", ["x", "x"])
    with pytest.raises(NeverThrown):
        require_canonical_multiset("ms", [("k", 0)])
    with pytest.raises(NeverThrown):
        require_no_python_hash("hash-order")


def test_require_invariants_accept_empty_iterables() -> None:
    assert require_no_dupes("dupes", []) is None
    assert require_canonical_multiset("ms", []) is None
