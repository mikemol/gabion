from __future__ import annotations

from pathlib import Path

import pytest

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import type_fingerprints as tf

    return tf

# gabion:evidence E:function_site::test_type_fingerprints.py::tests.test_type_fingerprints._load
def test_prime_registry_assigns_stable_primes() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    first = registry.get_or_assign("int")
    second = registry.get_or_assign("str")
    assert first != second
    assert registry.get_or_assign("int") == first

def test_prime_registry_consumes_gas_ticks() -> None:
    tf = _load()
    from gabion.analysis.aspf import Forest
    from gabion.analysis.timeout_context import (
        Deadline,
        TimeoutExceeded,
        deadline_clock_scope,
        deadline_scope,
        forest_scope,
    )
    from gabion.deadline_clock import GasMeter

    registry = tf.PrimeRegistry()
    with forest_scope(Forest()):
        with deadline_scope(Deadline.from_timeout_ms(1_000)):
            meter = GasMeter(limit=1)
            with deadline_clock_scope(meter):
                with pytest.raises(TimeoutExceeded):
                    registry.get_or_assign("int")
    assert meter.current == 1

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_lcm::a,b E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_symmetric_diff::a,b E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_contains::part
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

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_to_type_keys::strict
def test_fingerprint_to_type_keys_roundtrip() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    fingerprint = tf.bundle_fingerprint(["int", "str", "int"], registry)
    keys = tf.fingerprint_to_type_keys(fingerprint, registry)
    assert keys.count("int") == 2
    assert keys.count("str") == 1

# gabion:evidence E:function_site::test_type_fingerprints.py::tests.test_type_fingerprints._load
def test_constructor_registry_assigns_primes() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    ctor_registry = tf.TypeConstructorRegistry(registry)
    list_prime = ctor_registry.get_or_assign("List")
    dict_prime = ctor_registry.get_or_assign("Dict")
    assert list_prime != dict_prime
    assert registry.prime_for("ctor:list") == list_prime

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._normalize_type_list::value
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
    assert reg_a.prime_for("ctor:list") == reg_b.prime_for("ctor:list")

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_to_type_keys::strict
def test_fingerprint_to_type_keys_with_remainder_and_strict() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    int_prime = registry.get_or_assign("int")
    fingerprint = int_prime * 97
    keys, remaining = tf.fingerprint_to_type_keys_with_remainder(
        fingerprint, registry
    )
    assert keys == ["int"]
    assert remaining == 97
    try:
        tf.fingerprint_to_type_keys(fingerprint, registry, strict=True)
    except ValueError as exc:
        assert "not in registry" in str(exc)
    else:
        raise AssertionError("Expected strict factorization failure")

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::payload,registry E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::registry
def test_synth_registry_payload_roundtrip() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    ctor_registry = tf.TypeConstructorRegistry(registry)
    fp = tf.bundle_fingerprint_dimensional(["list[int]"], registry, ctor_registry)
    synth_registry = tf.build_synth_registry([fp, fp], registry, min_occurrences=2)
    payload = tf.synth_registry_payload(synth_registry, registry, min_occurrences=2)
    assert "registry" in payload
    assert isinstance(payload["registry"], dict)
    assert "primes" in payload["registry"]
    assert "bit_positions" in payload["registry"]
    restored = tf.build_synth_registry_from_payload(payload, registry)
    assert restored.tails

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::payload,registry E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::registry
def test_build_synth_registry_from_payload_applies_registry_basis_to_empty_registry() -> None:
    tf = _load()
    registry_a = tf.PrimeRegistry()
    ctor_registry = tf.TypeConstructorRegistry(registry_a)
    fp = tf.bundle_fingerprint_dimensional(["list[int]"], registry_a, ctor_registry)
    synth_registry = tf.build_synth_registry([fp, fp], registry_a, min_occurrences=2)
    payload = tf.synth_registry_payload(synth_registry, registry_a, min_occurrences=2)

    registry_b = tf.PrimeRegistry()
    restored = tf.build_synth_registry_from_payload(payload, registry_b)

    assert registry_b.prime_for("int") == registry_a.prime_for("int")
    assert registry_b.bit_for("int") == registry_a.bit_for("int")
    assert registry_b.prime_for("ctor:list") == registry_a.prime_for("ctor:list")
    assert registry_b.bit_for("ctor:list") == registry_a.bit_for("ctor:list")

    synth_key = tf._synth_key(restored.version, fp)
    assert registry_b.prime_for(synth_key) == registry_a.prime_for(synth_key)
    assert registry_b.bit_for(synth_key) == registry_a.bit_for(synth_key)
    assert restored.primes.get(fp) == synth_registry.primes.get(fp)

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::payload,registry E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::registry
def test_build_synth_registry_from_payload_rejects_registry_mismatch() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    ctor_registry = tf.TypeConstructorRegistry(registry)
    fp = tf.bundle_fingerprint_dimensional(["list[int]"], registry, ctor_registry)
    synth_registry = tf.build_synth_registry([fp, fp], registry, min_occurrences=2)
    payload = tf.synth_registry_payload(synth_registry, registry, min_occurrences=2)

    other = tf.PrimeRegistry()
    other.get_or_assign("int")
    basis = payload.get("registry")
    assert isinstance(basis, dict)
    primes = basis.get("primes")
    assert isinstance(primes, dict)
    primes["int"] = 97
    try:
        tf.build_synth_registry_from_payload(payload, other)
    except ValueError as exc:
        assert "Registry basis mismatch" in str(exc)
    else:
        raise AssertionError("Expected registry basis mismatch error")

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._normalize_type_list::value
def test_build_fingerprint_registry_skips_empty_entries() -> None:
    tf = _load()
    registry, index = tf.build_fingerprint_registry({"empty": []})
    assert index == {}
    assert registry.primes == {}
    assert registry.bit_positions == {}

# gabion:evidence E:function_site::test_type_fingerprints.py::tests.test_type_fingerprints._load
def test_registry_helpers_cover_edges() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    try:
        registry.get_or_assign("")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty key")
    assert registry.key_for_prime(9999) is None
    assert registry.bit_for("unknown") is None

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_lcm::a,b E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_symmetric_diff::a,b E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_contains::part
def test_fingerprint_arithmetic_edges() -> None:
    tf = _load()
    assert tf.fingerprint_lcm(0, 3) == 0
    assert tf.fingerprint_contains(4, 0) is False
    assert tf.fingerprint_symmetric_diff(0, 5) == 5
    assert tf.fingerprint_symmetric_diff(7, 0) == 7

def test_split_top_level_handles_empty_segments() -> None:
    tf = _load()
    assert tf._split_top_level(",a", ",") == ["a"]
    assert tf._split_top_level("a,", ",") == ["a"]

def test_prime_registry_existing_bit_and_key_lookup_scan() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    registry.bit_positions["k"] = 7
    prime = registry.get_or_assign("k")
    assert registry.bit_for("k") == 7
    other_prime = registry.get_or_assign("other")
    assert registry.key_for_prime(other_prime) == "other"
    assert prime != other_prime

def test_normalize_type_list_ignores_non_string_entries() -> None:
    tf = _load()
    assert tf._normalize_type_list(123) == []
    assert tf._normalize_type_list(["a", 1, "b"]) == ["a", "b"]

def test_dimension_helpers_handle_missing_registry_bits() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    registry.get_or_assign("int")
    registry.bit_positions.pop("int", None)
    dim = tf._dimension_from_keys(["int"], registry)
    assert dim.mask == 0

    registry.get_or_assign("ctor:list")
    registry.bit_positions.pop("ctor:list", None)
    ctor_dim = tf._ctor_dimension_from_names(["list"], registry)
    assert ctor_dim.mask == 0

def test_apply_registry_payload_filters_invalid_registry_values() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    tf._apply_registry_payload({"primes": [], "bit_positions": []}, registry)
    assert registry.primes == {}
    assert registry.bit_positions == {}

    tf._apply_registry_payload(
        {
            "primes": {"a": "bad", "b": 5},
            "bit_positions": {"a": "bad", "b": 2},
        },
        registry,
    )
    assert registry.prime_for("a") is None
    assert registry.prime_for("b") == 5
    assert registry.bit_for("a") is None
    assert registry.bit_for("b") == 2

def test_bundle_fingerprint_dimensional_without_constructor_registry() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    fingerprint = tf.bundle_fingerprint_dimensional(["int"], registry, None)
    assert fingerprint.base.product == registry.get_or_assign("int")
    assert fingerprint.ctor.is_empty()

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._normalize_type_list::value
def test_build_fingerprint_registry_skips_empty_entries_with_valid() -> None:
    tf = _load()
    registry, index = tf.build_fingerprint_registry(
        {"empty": [], "valid": ["int"]}
    )
    assert registry.prime_for("int") is not None
    assert any("valid" in names for names in index.values())

def test_prime_registry_seed_payload_roundtrip_with_namespaces() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    registry.get_or_assign("int")
    registry.get_or_assign("str")
    registry.get_or_assign("ctor:list")
    seed = registry.seed_payload()

    loaded = tf.PrimeRegistry()
    loaded.load_seed_payload(seed)

    assert loaded.prime_for("int") == registry.prime_for("int")
    assert loaded.prime_for("str") == registry.prime_for("str")
    assert loaded.prime_for("ctor:list") == registry.prime_for("ctor:list")
    assert loaded.bit_for("ctor:list") == registry.bit_for("ctor:list")


def test_prime_registry_load_seed_payload_accepts_flat_legacy_payload() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()

    registry.load_seed_payload(
        {
            "primes": {"int": 2},
            "bit_positions": {"int": 0},
        }
    )

    assert registry.prime_for("int") == 2
    assert registry.bit_for("int") == 0


def test_prime_registry_load_seed_payload_ignores_invalid_namespace_entries() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()

    registry.load_seed_payload(
        {
            "namespaces": {
                "type_base": {
                    "primes": {"int": 2, "bad": "x", 3: 5},
                    "bit_positions": {"int": 0, "bad": "x", 4: 1},
                },
                "type_ctor": {
                    "primes": ["not-a-dict"],
                    "bit_positions": "not-a-dict",
                },
                "invalid_namespace": "not-a-dict",
                7: {"primes": {"ignored": 13}},
            }
        }
    )

    assert registry.prime_for("int") == 2
    assert registry.bit_for("int") == 0
    assert registry.prime_for("bad") is None
    assert registry.bit_for("bad") is None


def test_build_fingerprint_registry_seed_is_stable_under_reordered_inputs() -> None:
    tf = _load()
    spec_a = {
        "alpha": ["int", "list[str]"],
        "beta": ["dict[str, int]"],
    }
    spec_b = {
        "beta": ["dict[str, int]"],
        "alpha": ["list[str]", "int"],
    }
    seed_a = {
        "namespaces": {
            "type_ctor": {
                "primes": {"dict": 17, "list": 13},
                "bit_positions": {"dict": 3, "list": 2},
            },
            "type_base": {
                "primes": {"str": 5, "int": 2},
                "bit_positions": {"str": 1, "int": 0},
            },
        }
    }
    seed_b = {
        "namespaces": {
            "type_base": {
                "bit_positions": {"int": 0, "str": 1},
                "primes": {"int": 2, "str": 5},
            },
            "type_ctor": {
                "bit_positions": {"list": 2, "dict": 3},
                "primes": {"list": 13, "dict": 17},
            },
        }
    }

    reg_a, _ = tf.build_fingerprint_registry(spec_a, registry_seed=seed_a)
    reg_b, _ = tf.build_fingerprint_registry(spec_b, registry_seed=seed_b)

    assert reg_a.primes == reg_b.primes
    assert reg_a.bit_positions == reg_b.bit_positions
    assert reg_a.prime_for("int") == 2
    assert reg_a.prime_for("ctor:list") == 13
