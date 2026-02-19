from __future__ import annotations

import pytest

from gabion.analysis.determinism_invariants import (
    require_canonical_multiset,
    require_no_dupes,
    require_no_python_hash,
    require_sorted,
)
from gabion.exceptions import NeverThrown
from gabion.invariants import proof_mode_scope


# gabion:evidence E:call_footprint::tests/test_determinism_invariants.py::test_require_invariants_noop_when_proof_mode_disabled::determinism_invariants.py::gabion.analysis.determinism_invariants.require_canonical_multiset::determinism_invariants.py::gabion.analysis.determinism_invariants.require_no_dupes::determinism_invariants.py::gabion.analysis.determinism_invariants.require_no_python_hash::determinism_invariants.py::gabion.analysis.determinism_invariants.require_sorted
def test_require_invariants_noop_when_proof_mode_disabled() -> None:
    assert require_sorted("sorted", [2, 1]) is None
    assert require_no_dupes("dupes", ["a", "a"]) is None
    assert require_canonical_multiset("ms", [("a", 0)]) is None
    assert require_no_python_hash("hash") is None


# gabion:evidence E:call_footprint::tests/test_determinism_invariants.py::test_require_sorted_allows_reverse_sorted_in_reverse_mode::determinism_invariants.py::gabion.analysis.determinism_invariants.require_sorted::invariants.py::gabion.invariants.proof_mode_scope
def test_require_sorted_allows_reverse_sorted_in_reverse_mode() -> None:
    with proof_mode_scope(True):
        assert require_sorted("descending", [3, 2, 2, 1], reverse=True) is None
        assert require_sorted("empty", []) is None


# gabion:evidence E:call_footprint::tests/test_determinism_invariants.py::test_require_sorted_raises_and_reports_payload::determinism_invariants.py::gabion.analysis.determinism_invariants.require_sorted::invariants.py::gabion.invariants.proof_mode_scope
def test_require_sorted_raises_and_reports_payload() -> None:
    observed: list[dict[str, object]] = []
    with proof_mode_scope(True):
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


# gabion:evidence E:call_footprint::tests/test_determinism_invariants.py::test_require_no_dupes_raises_and_reports_payload::determinism_invariants.py::gabion.analysis.determinism_invariants.require_no_dupes::invariants.py::gabion.invariants.proof_mode_scope
def test_require_no_dupes_raises_and_reports_payload() -> None:
    observed: list[dict[str, object]] = []
    with proof_mode_scope(True):
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


# gabion:evidence E:call_footprint::tests/test_determinism_invariants.py::test_require_canonical_multiset_rejects_invalid_inputs::determinism_invariants.py::gabion.analysis.determinism_invariants.require_canonical_multiset::invariants.py::gabion.invariants.proof_mode_scope
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
    with proof_mode_scope(True):
        with pytest.raises(NeverThrown):
            require_canonical_multiset("ms", pairs)


# gabion:evidence E:call_footprint::tests/test_determinism_invariants.py::test_require_canonical_multiset_reports_payload_for_invalid_variants::determinism_invariants.py::gabion.analysis.determinism_invariants.require_canonical_multiset::invariants.py::gabion.invariants.proof_mode_scope
def test_require_canonical_multiset_reports_payload_for_invalid_variants(
) -> None:
    with proof_mode_scope(True):
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


# gabion:evidence E:call_footprint::tests/test_determinism_invariants.py::test_require_no_python_hash_always_raises_in_proof_mode::determinism_invariants.py::gabion.analysis.determinism_invariants.require_no_python_hash::invariants.py::gabion.invariants.proof_mode_scope
def test_require_no_python_hash_always_raises_in_proof_mode() -> None:
    observed: list[dict[str, object]] = []
    with proof_mode_scope(True):
        with pytest.raises(NeverThrown):
            require_no_python_hash(
                "hash-order",
                on_violation=lambda payload: observed.append(payload),
                spec="wl",
            )
    assert observed
    assert observed[0]["constraint"] == "no_python_hash"
    assert observed[0]["spec"] == "wl"


# gabion:evidence E:call_footprint::tests/test_determinism_invariants.py::test_require_invariants_raise_without_callbacks::determinism_invariants.py::gabion.analysis.determinism_invariants.require_canonical_multiset::determinism_invariants.py::gabion.analysis.determinism_invariants.require_no_dupes::determinism_invariants.py::gabion.analysis.determinism_invariants.require_no_python_hash::determinism_invariants.py::gabion.analysis.determinism_invariants.require_sorted::invariants.py::gabion.invariants.proof_mode_scope
def test_require_invariants_raise_without_callbacks() -> None:
    with proof_mode_scope(True):
        with pytest.raises(NeverThrown):
            require_sorted("ascending", [2, 1])
        with pytest.raises(NeverThrown):
            require_no_dupes("dupes", ["x", "x"])
        with pytest.raises(NeverThrown):
            require_canonical_multiset("ms", [("k", 0)])
        with pytest.raises(NeverThrown):
            require_no_python_hash("hash-order")


# gabion:evidence E:call_footprint::tests/test_determinism_invariants.py::test_require_invariants_accept_empty_iterables_in_proof_mode::determinism_invariants.py::gabion.analysis.determinism_invariants.require_canonical_multiset::determinism_invariants.py::gabion.analysis.determinism_invariants.require_no_dupes::invariants.py::gabion.invariants.proof_mode_scope
def test_require_invariants_accept_empty_iterables_in_proof_mode() -> None:
    with proof_mode_scope(True):
        assert require_no_dupes("dupes", []) is None
        assert require_canonical_multiset("ms", []) is None
