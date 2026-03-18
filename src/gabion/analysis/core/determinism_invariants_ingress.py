# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=determinism_invariants_ingress
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeVar

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never


T = TypeVar("T")


def _identity_key(value: T) -> object:
    return value


@dataclass(frozen=True)
class SortedInvariantSatisfied:
    pass


@dataclass(frozen=True)
class SortedInvariantViolation:
    name: str
    previous_key: str
    current_key: str
    reverse: bool


@dataclass(frozen=True)
class NoDupesInvariantSatisfied:
    pass


@dataclass(frozen=True)
class NoDupesInvariantViolation:
    name: str
    duplicate_key: str


@dataclass(frozen=True)
class CanonicalMultisetInvariantSatisfied:
    pass


@dataclass(frozen=True)
class CanonicalMultisetInvalidCountViolation:
    name: str
    invalid_count: int
    key: str


@dataclass(frozen=True)
class CanonicalMultisetDuplicateKeyViolation:
    name: str
    duplicate_key: str


@dataclass(frozen=True)
class CanonicalMultisetOrderViolation:
    name: str
    previous_key: str
    current_key: str


@dataclass(frozen=True)
class PythonHashInvariantViolation:
    name: str


def sorted_invariant_outcome(
    name: str,
    xs: Iterable[T],
    *,
    key: Callable[[T], object] = _identity_key,
    reverse: bool = False,
) -> SortedInvariantSatisfied | SortedInvariantViolation:
    check_deadline()
    iterator = iter(xs)
    try:
        previous = next(iterator)
    except StopIteration:
        return SortedInvariantSatisfied()
    previous_key = key(previous)
    for current in iterator:
        check_deadline()
        current_key = key(current)
        is_ordered = current_key <= previous_key if reverse else previous_key <= current_key
        if is_ordered:
            previous_key = current_key
            continue
        return SortedInvariantViolation(
            name=name,
            previous_key=repr(previous_key),
            current_key=repr(current_key),
            reverse=reverse,
        )
    return SortedInvariantSatisfied()


def no_dupes_invariant_outcome(
    name: str,
    xs: Iterable[T],
    *,
    key: Callable[[T], object] = _identity_key,
) -> NoDupesInvariantSatisfied | NoDupesInvariantViolation:
    check_deadline()
    seen: set[object] = set()
    for item in xs:
        check_deadline()
        entry_key = key(item)
        if entry_key not in seen:
            seen.add(entry_key)
            continue
        return NoDupesInvariantViolation(name=name, duplicate_key=repr(entry_key))
    return NoDupesInvariantSatisfied()


def canonical_multiset_invariant_outcome(
    name: str,
    pairs: Iterable[tuple[str, int]],
) -> (
    CanonicalMultisetInvariantSatisfied
    | CanonicalMultisetInvalidCountViolation
    | CanonicalMultisetDuplicateKeyViolation
    | CanonicalMultisetOrderViolation
):
    check_deadline()
    seen_keys: set[str] = set()
    previous_key = ""
    has_previous = False
    for key, count in pairs:
        check_deadline()
        if count <= 0:
            return CanonicalMultisetInvalidCountViolation(
                name=name,
                invalid_count=int(count),
                key=key,
            )
        if key in seen_keys:
            return CanonicalMultisetDuplicateKeyViolation(name=name, duplicate_key=key)
        seen_keys.add(key)
        if has_previous and key < previous_key:
            return CanonicalMultisetOrderViolation(
                name=name,
                previous_key=previous_key,
                current_key=key,
            )
        previous_key = key
        has_previous = True
    return CanonicalMultisetInvariantSatisfied()


def python_hash_invariant_violation(name: str) -> PythonHashInvariantViolation:
    return PythonHashInvariantViolation(name=name)


def require_sorted(
    name: str,
    xs: Iterable[T],
    *,
    key: Callable[[T], object] = _identity_key,
    reverse: bool = False,
) -> None:
    match sorted_invariant_outcome(name, xs, key=key, reverse=reverse):
        case SortedInvariantSatisfied():
            return
        case SortedInvariantViolation() as violation:
            never(
                "determinism invariant: sorted order violated",
                constraint="sorted",
                name=violation.name,
                previous_key=violation.previous_key,
                current_key=violation.current_key,
                reverse=violation.reverse,
            )


def require_no_dupes(
    name: str,
    xs: Iterable[T],
    *,
    key: Callable[[T], object] = _identity_key,
) -> None:
    match no_dupes_invariant_outcome(name, xs, key=key):
        case NoDupesInvariantSatisfied():
            return
        case NoDupesInvariantViolation() as violation:
            never(
                "determinism invariant: duplicate entry detected",
                constraint="no_dupes",
                name=violation.name,
                duplicate_key=violation.duplicate_key,
            )


def require_canonical_multiset(
    name: str,
    pairs: Iterable[tuple[str, int]],
) -> None:
    match canonical_multiset_invariant_outcome(name, pairs):
        case CanonicalMultisetInvariantSatisfied():
            return
        case CanonicalMultisetInvalidCountViolation() as violation:
            never(
                "determinism invariant: multiset count must be positive",
                constraint="canonical_multiset",
                name=violation.name,
                invalid_count=violation.invalid_count,
                key=violation.key,
            )
        case CanonicalMultisetDuplicateKeyViolation() as violation:
            never(
                "determinism invariant: multiset key duplicated",
                constraint="canonical_multiset",
                name=violation.name,
                duplicate_key=violation.duplicate_key,
            )
        case CanonicalMultisetOrderViolation() as violation:
            never(
                "determinism invariant: multiset keys out of order",
                constraint="canonical_multiset",
                name=violation.name,
                previous_key=violation.previous_key,
                current_key=violation.current_key,
            )


def require_no_python_hash(name: str) -> None:
    violation = python_hash_invariant_violation(name)
    never(
        "determinism invariant: Python hash/randomized ordering is forbidden in this path",
        constraint="no_python_hash",
        name=violation.name,
    )
