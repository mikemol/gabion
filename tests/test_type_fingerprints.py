from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import type_fingerprints as tf

    return tf


def test_canonical_type_key_normalizes_union_and_optional() -> None:
    tf = _load()
    assert tf.canonical_type_key("Optional[int]") == "Union[None, int]"
    assert tf.canonical_type_key("Union[str, int]") == "Union[int, str]"
    assert tf.canonical_type_key("int | None") == "Union[None, int]"


def test_canonical_type_key_normalizes_generics() -> None:
    tf = _load()
    assert tf.canonical_type_key("typing.List[int]") == "list[int]"
    assert tf.canonical_type_key("List[ str ]") == "list[str]"
    assert tf.canonical_type_key("Dict[str, List[int]]") == "dict[str, list[int]]"


def test_prime_registry_assigns_stable_primes() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    first = registry.get_or_assign("int")
    second = registry.get_or_assign("str")
    assert first != second
    assert registry.get_or_assign("int") == first


def test_bundle_fingerprint_multiplies_primes() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    fingerprint = tf.bundle_fingerprint(["int", "str", "int"], registry)
    assert fingerprint == registry.get_or_assign("int") * registry.get_or_assign("str") * registry.get_or_assign("int")


def test_fingerprint_arithmetic_ops() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    a = tf.bundle_fingerprint(["int", "str"], registry)
    b = tf.bundle_fingerprint(["str", "list[int]"], registry)
    shared = tf.fingerprint_gcd(a, b)
    assert shared == registry.get_or_assign("str")
    combined = tf.fingerprint_lcm(a, b)
    assert tf.fingerprint_contains(combined, a)
    assert tf.fingerprint_contains(combined, b)
    diff = tf.fingerprint_symmetric_diff(a, b)
    assert diff == registry.get_or_assign("int") * registry.get_or_assign("list[int]")
