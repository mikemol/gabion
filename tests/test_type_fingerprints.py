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


def test_fingerprint_to_type_keys_roundtrip() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    fingerprint = tf.bundle_fingerprint(["int", "str", "int"], registry)
    keys = tf.fingerprint_to_type_keys(fingerprint, registry)
    assert keys.count("int") == 2
    assert keys.count("str") == 1


def test_fingerprint_hybrid_bitmask() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    product, mask = tf.fingerprint_hybrid(["int", "str", "int"], registry)
    assert product == tf.bundle_fingerprint(["int", "str", "int"], registry)
    int_bit = registry.bit_for("int")
    str_bit = registry.bit_for("str")
    assert int_bit is not None and str_bit is not None
    assert mask & (1 << int_bit)
    assert mask & (1 << str_bit)


def test_constructor_registry_assigns_primes() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    ctor_registry = tf.TypeConstructorRegistry(registry)
    list_prime = ctor_registry.get_or_assign("List")
    dict_prime = ctor_registry.get_or_assign("Dict")
    assert list_prime != dict_prime
    assert registry.prime_for("ctor:list") == list_prime

def test_build_fingerprint_registry_deterministic_assignment() -> None:
    tf = _load()
    spec_a = {"user": ["int", "str"], "flags": ["list[int]"]}
    spec_b = {"flags": ["list[int]"], "user": ["str", "int"]}
    reg_a, index_a = tf.build_fingerprint_registry(spec_a)
    reg_b, index_b = tf.build_fingerprint_registry(spec_b)

    def _find_fingerprint(index, name):
        for fingerprint, names in index.items():
            if name in names:
                return fingerprint
        return None

    fp_a = _find_fingerprint(index_a, "user")
    fp_b = _find_fingerprint(index_b, "user")
    assert fp_a == fp_b
    assert reg_a.prime_for("int") == reg_b.prime_for("int")
    assert reg_a.prime_for("list[int]") == reg_b.prime_for("list[int]")
