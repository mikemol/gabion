from __future__ import annotations

import pytest

from gabion.analysis.core.determinism_invariants import (
    require_canonical_multiset,
    require_no_dupes,
    require_no_python_hash,
    require_sorted,
)
from gabion.analysis.core.determinism_invariants_ingress import (
    CanonicalMultisetDuplicateKeyViolation,
    CanonicalMultisetInvalidCountViolation,
    CanonicalMultisetOrderViolation,
    NoDupesInvariantViolation,
    PythonHashInvariantViolation,
    SortedInvariantViolation,
    canonical_multiset_invariant_outcome,
    no_dupes_invariant_outcome,
    python_hash_invariant_violation,
    sorted_invariant_outcome,
)
from gabion.exceptions import NeverThrown


# gabion:behavior primary=desired
def test_require_invariants_accept_valid_inputs() -> None:
    assert require_sorted("sorted", [1, 2]) is None
    assert require_no_dupes("dupes", ["a", "b"]) is None
    assert require_canonical_multiset("ms", [("a", 1)]) is None


# gabion:behavior primary=desired
def test_require_sorted_allows_reverse_sorted_in_reverse_mode() -> None:
    assert require_sorted("descending", [3, 2, 2, 1], reverse=True) is None
    assert require_sorted("empty", []) is None


# gabion:behavior primary=verboten facets=raises
def test_require_sorted_raises_and_reports_payload() -> None:
    outcome = sorted_invariant_outcome("ascending", [1, 3, 2])
    assert isinstance(outcome, SortedInvariantViolation)
    assert outcome.name == "ascending"
    assert outcome.previous_key == "3"
    assert outcome.current_key == "2"


# gabion:behavior primary=verboten facets=raises
def test_require_no_dupes_raises_and_reports_payload() -> None:
    outcome = no_dupes_invariant_outcome("dupes", ["a", "b", "a"])
    assert isinstance(outcome, NoDupesInvariantViolation)
    assert outcome.name == "dupes"
    assert outcome.duplicate_key == "'a'"


# gabion:behavior primary=verboten facets=invalid
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


# gabion:behavior primary=verboten facets=invalid
def test_require_canonical_multiset_reports_payload_for_invalid_variants(
) -> None:
    invalid_count = canonical_multiset_invariant_outcome("ms", [("a", 0)])
    assert isinstance(invalid_count, CanonicalMultisetInvalidCountViolation)
    assert invalid_count.invalid_count == 0

    duplicate = canonical_multiset_invariant_outcome("ms", [("a", 1), ("a", 1)])
    assert isinstance(duplicate, CanonicalMultisetDuplicateKeyViolation)
    assert duplicate.duplicate_key == "a"

    order = canonical_multiset_invariant_outcome("ms", [("b", 1), ("a", 1)])
    assert isinstance(order, CanonicalMultisetOrderViolation)
    assert order.previous_key == "b"
    assert order.current_key == "a"


# gabion:behavior primary=verboten facets=raises
def test_require_no_python_hash_always_raises() -> None:
    outcome = python_hash_invariant_violation("hash-order")
    assert isinstance(outcome, PythonHashInvariantViolation)
    assert outcome.name == "hash-order"


# gabion:behavior primary=verboten facets=raises
def test_require_invariants_raise_without_callbacks() -> None:
    with pytest.raises(NeverThrown):
        require_sorted("ascending", [2, 1])
    with pytest.raises(NeverThrown):
        require_no_dupes("dupes", ["x", "x"])
    with pytest.raises(NeverThrown):
        require_canonical_multiset("ms", [("k", 0)])
    with pytest.raises(NeverThrown):
        require_no_python_hash("hash-order")


# gabion:behavior primary=verboten facets=empty
def test_require_invariants_accept_empty_iterables() -> None:
    assert require_no_dupes("dupes", []) is None
    assert require_canonical_multiset("ms", []) is None
