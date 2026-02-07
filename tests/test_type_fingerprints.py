from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import type_fingerprints as tf

    return tf


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._split_top_level::sep
def test_canonical_type_key_normalizes_union_and_optional() -> None:
    tf = _load()
    assert tf.canonical_type_key("Optional[int]") == "Union[None, int]"
    assert tf.canonical_type_key("Union[str, int]") == "Union[int, str]"
    assert tf.canonical_type_key("int | None") == "Union[None, int]"


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._split_top_level::sep
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

def test_bundle_fingerprint_setlike_ignores_duplicates() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    multiset_fp = tf.bundle_fingerprint(["int", "int", "str"], registry)
    setlike_fp = tf.bundle_fingerprint_setlike(["int", "int", "str"], registry)
    assert multiset_fp != setlike_fp
    assert setlike_fp == tf.bundle_fingerprint_setlike(["str", "int"], registry)


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


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry
def test_dimensional_fingerprint_includes_constructors() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    ctor_registry = tf.TypeConstructorRegistry(registry)
    fingerprint = tf.bundle_fingerprint_dimensional(
        ["list[int]", "dict[str, int]"],
        registry,
        ctor_registry,
    )
    assert fingerprint.base.product != 1
    assert fingerprint.ctor.product != 1
    assert fingerprint.base.mask != 0
    assert fingerprint.ctor.mask != 0


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::a,b E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::a,b
def test_carrier_soundness_mask_disjoint_implies_gcd_one() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    ctor_registry = tf.TypeConstructorRegistry(registry)
    a = tf.bundle_fingerprint_dimensional(["int"], registry, ctor_registry).base
    b = tf.bundle_fingerprint_dimensional(["str"], registry, ctor_registry).base
    assert (a.mask & b.mask) == 0
    assert tf.fingerprint_carrier_soundness(a, b)


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry
def test_synth_registry_assigns_primes_deterministically() -> None:
    tf = _load()
    registry_a = tf.PrimeRegistry()
    registry_b = tf.PrimeRegistry()
    ctor_a = tf.TypeConstructorRegistry(registry_a)
    ctor_b = tf.TypeConstructorRegistry(registry_b)
    fp_a1 = tf.bundle_fingerprint_dimensional(["list[int]"], registry_a, ctor_a)
    fp_a2 = tf.bundle_fingerprint_dimensional(["list[int]"], registry_a, ctor_a)
    fp_b1 = tf.bundle_fingerprint_dimensional(["list[int]"], registry_b, ctor_b)
    fp_b2 = tf.bundle_fingerprint_dimensional(["list[int]"], registry_b, ctor_b)
    synth_a = tf.build_synth_registry([fp_a2, fp_a1], registry_a, min_occurrences=2)
    synth_b = tf.build_synth_registry([fp_b1, fp_b2], registry_b, min_occurrences=2)
    prime_a = synth_a.get_or_assign(fp_a1)
    prime_b = synth_b.get_or_assign(fp_b1)
    assert prime_a == prime_b


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry
def test_apply_synth_dimension_attaches_tail() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    ctor_registry = tf.TypeConstructorRegistry(registry)
    fp = tf.bundle_fingerprint_dimensional(["dict[str, int]"], registry, ctor_registry)
    synth_registry = tf.build_synth_registry([fp, fp], registry, min_occurrences=2)
    synthesized = tf.apply_synth_dimension(fp, synth_registry)
    assert synthesized.synth.product != 1
    assert synth_registry.tails.get(synthesized.synth.product) == fp


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


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::payload,registry E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::registry
def test_build_synth_registry_from_payload_assigns_bits_when_missing() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    payload = {
        "version": "synth@1",
        "entries": [],
        "registry": {"primes": {"b": 3, "a": 2}},
    }
    tf.build_synth_registry_from_payload(payload, registry)
    assert registry.bit_for("a") == 0
    assert registry.bit_for("b") == 1
    assert registry.get_or_assign("c") == 5


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::payload,registry E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::registry
def test_build_synth_registry_from_payload_rejects_duplicate_primes() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    payload = {
        "version": "synth@1",
        "entries": [],
        "registry": {"primes": {"a": 2, "b": 2}},
    }
    try:
        tf.build_synth_registry_from_payload(payload, registry)
    except ValueError as exc:
        assert "duplicate primes" in str(exc)
    else:
        raise AssertionError("Expected duplicate prime error")


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::payload,registry E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::registry
def test_build_synth_registry_from_payload_rejects_duplicate_bit_positions() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    payload = {
        "version": "synth@1",
        "entries": [],
        "registry": {
            "primes": {"a": 2, "b": 3},
            "bit_positions": {"a": 0, "b": 0},
        },
    }
    try:
        tf.build_synth_registry_from_payload(payload, registry)
    except ValueError as exc:
        assert "duplicate bit positions" in str(exc)
    else:
        raise AssertionError("Expected duplicate bit position error")


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::payload,registry E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::registry
def test_build_synth_registry_from_payload_ignores_non_string_registry_keys() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    payload = {
        "version": "synth@1",
        "entries": [],
        "registry": {
            "primes": {"a": 2, 1: 3},
            "bit_positions": {"a": 0, 2: 1},
        },
    }
    tf.build_synth_registry_from_payload(payload, registry)
    assert registry.prime_for("a") == 2
    assert registry.bit_for("a") == 0


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::payload,registry E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::registry
def test_build_synth_registry_from_payload_rejects_bit_mismatch() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    registry.get_or_assign("a")
    payload = {
        "version": "synth@1",
        "entries": [],
        "registry": {
            "primes": {"a": registry.prime_for("a")},
            "bit_positions": {"a": 99},
        },
    }
    try:
        tf.build_synth_registry_from_payload(payload, registry)
    except ValueError as exc:
        assert "Registry basis mismatch for bit" in str(exc)
    else:
        raise AssertionError("Expected bit position mismatch error")


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._strip_known_prefix::name E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._split_top_level::sep
def test_normalization_helpers_cover_edges() -> None:
    tf = _load()
    assert tf._split_top_level("A,B[C,D],E", ",") == ["A", "B[C,D]", "E"]
    assert tf._strip_known_prefix("typing.List") == "List"
    assert tf._strip_known_prefix("builtins.int") == "int"
    assert tf._normalize_base("LIST") == "list"
    assert tf._normalize_base("Optional") == "Optional"
    assert tf._normalize_base("Union") == "Union"
    assert tf._normalize_base("NoneType") == "None"
    assert tf.canonical_type_key(" ") == ""


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._split_top_level::sep
def test_canonical_type_key_with_constructor_handles_union_and_optional() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    ctor_registry = tf.TypeConstructorRegistry(registry)
    key = tf.canonical_type_key_with_constructor("Optional[int]", ctor_registry)
    assert key == "Union[None, int]"
    key = tf.canonical_type_key_with_constructor("Union[str, int]", ctor_registry)
    assert key == "Union[int, str]"
    key = tf.canonical_type_key_with_constructor("Dict[str, List[int]]", ctor_registry)
    assert key == "dict[str, list[int]]"


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._split_top_level::sep
def test_collect_base_atoms_and_constructors_cover_empty_and_unions() -> None:
    tf = _load()
    atoms: list[str] = []
    tf._collect_base_atoms("", atoms)
    assert atoms == []
    tf._collect_base_atoms("Optional[int]", atoms)
    assert "int" in atoms
    assert "None" in atoms
    constructors: list[str] = []
    tf._collect_constructors_multiset(" ", constructors)
    tf._collect_constructors_multiset("list[int] | dict[str, int]", constructors)
    tf._collect_constructors_multiset("Union[list[int], dict[str, int]]", constructors)
    assert "list" in constructors
    assert "dict" in constructors
    ctor_set: set[str] = set()
    tf._collect_constructors("", ctor_set)
    tf._collect_constructors("Union[list[int], dict[str, int]]", ctor_set)
    tf._collect_constructors("Optional[list[int]]", ctor_set)
    assert "list" in ctor_set
    assert "dict" in ctor_set


def test_bundle_fingerprint_with_constructors_skips_empty_keys() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    ctor_registry = tf.TypeConstructorRegistry(registry)
    fingerprint = tf.bundle_fingerprint_with_constructors([" ", "int"], registry, ctor_registry)
    assert fingerprint == registry.get_or_assign("int")


def test_fingerprint_bitmask_skips_empty_keys() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    mask = tf.fingerprint_bitmask([" ", "int"], registry)
    int_bit = registry.bit_for("int")
    assert int_bit is not None
    assert mask == (1 << int_bit)


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::payload,registry E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::registry
def test_build_synth_registry_from_payload_skips_non_dict_entries() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    payload = {"entries": ["bad"], "version": "synth@1", "min_occurrences": 2}
    synth_registry = tf.build_synth_registry_from_payload(payload, registry)
    assert synth_registry.tails == {}


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._normalize_type_list::value
def test_build_fingerprint_registry_skips_empty_entries() -> None:
    tf = _load()
    registry, index = tf.build_fingerprint_registry({"empty": []})
    assert index == {}
    assert registry.primes == {}
    assert registry.bit_positions == {}


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._split_top_level::sep
def test_collect_atoms_and_constructors() -> None:
    tf = _load()
    base_atoms: list[str] = []
    tf._collect_base_atoms("Union[int, Optional[str]]", base_atoms)
    assert "int" in base_atoms and "str" in base_atoms
    ctor_names: list[str] = []
    tf._collect_constructors_multiset("list[dict[str, int]]", ctor_names)
    assert "list" in ctor_names and "dict" in ctor_names
    ctor_set: set[str] = set()
    tf._collect_constructors("list[dict[str, int]]", ctor_set)
    assert "list" in ctor_set and "dict" in ctor_set


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.format_fingerprint::fingerprint
def test_dimension_helpers_and_formatting() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    dim = tf._dimension_from_keys(["int", ""], registry)
    ctor_dim = tf._ctor_dimension_from_names(["list", ""], registry)
    fingerprint = tf.Fingerprint(
        base=dim,
        ctor=ctor_dim,
        provenance=tf.FingerprintDimension(product=7, mask=0),
        synth=tf.FingerprintDimension(product=11, mask=0),
    )
    rendered = tf.format_fingerprint(fingerprint)
    assert "prov=" in rendered and "synth=" in rendered and "ctor=" in rendered
    assert tf._fingerprint_sort_key(fingerprint)


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


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._normalize_type_list::value
def test_normalize_type_list_variants() -> None:
    tf = _load()
    assert tf._normalize_type_list(None) == []
    assert tf._normalize_type_list("a, b") == ["a", "b"]
    assert tf._normalize_type_list(["a, b", "c"]) == ["a", "b", "c"]


def test_synth_registry_payload_handles_non_list_entries() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    payload = {"version": "synth@1", "entries": "bad"}
    entries, version, min_occ = tf.load_synth_registry_payload(payload)
    assert entries == []
    assert version == "synth@1"
    assert min_occ == 2


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::payload,registry E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints._apply_registry_payload::registry
def test_synth_registry_from_payload_overrides_prime() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    fp = tf.Fingerprint(
        base=tf.FingerprintDimension(product=registry.get_or_assign("int"), mask=0),
        ctor=tf.FingerprintDimension(),
    )
    synth_registry = tf.build_synth_registry([fp, fp], registry, min_occurrences=2)
    payload = tf.synth_registry_payload(synth_registry, registry, min_occurrences=2)
    payload["entries"][0]["prime"] = 97
    restored = tf.build_synth_registry_from_payload(payload, registry)
    assert 97 in restored.tails


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._is_prime::value
def test_prime_checks_and_key_lookup() -> None:
    tf = _load()
    assert tf._is_prime(1) is False
    assert tf._is_prime(25) is False
    assert tf._is_prime(29) is True
    registry = tf.PrimeRegistry()
    prime = registry.get_or_assign("int")
    assert registry.key_for_prime(prime) == "int"


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._split_top_level::sep
def test_canonical_type_key_with_constructor_pipe_union_and_empty() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    ctor_registry = tf.TypeConstructorRegistry(registry)
    assert tf.canonical_type_key_with_constructor(" ", ctor_registry) == ""
    key = tf.canonical_type_key_with_constructor("int | str", ctor_registry)
    assert key == "Union[int, str]"


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._split_top_level::sep
def test_collect_atoms_union_and_optional_paths() -> None:
    tf = _load()
    atoms: list[str] = []
    tf._collect_base_atoms("int | Optional[str]", atoms)
    assert "int" in atoms and "str" in atoms and "None" in atoms
    constructors: list[str] = []
    tf._collect_constructors_multiset("Optional[list[int]]", constructors)
    assert "list" in constructors
    ctor_set: set[str] = set()
    tf._collect_constructors("int | list[str]", ctor_set)
    assert "list" in ctor_set


def test_format_fingerprint_str_and_synth_dimension_none() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    dim = tf._dimension_from_keys(["int"], registry)
    fingerprint = tf.Fingerprint(base=dim)
    assert str(fingerprint).startswith("{base=")
    synth_registry = tf.SynthRegistry(registry=registry)
    assert synth_registry.synth_dimension_for(fingerprint) is None


def test_apply_synth_dimension_noop_when_missing() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    dim = tf._dimension_from_keys(["int"], registry)
    fingerprint = tf.Fingerprint(base=dim)
    synth_registry = tf.SynthRegistry(registry=registry)
    assert tf.apply_synth_dimension(fingerprint, synth_registry) == fingerprint


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::a,b E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::a,b
def test_carrier_soundness_mask_overlap_true() -> None:
    tf = _load()
    dim = tf.FingerprintDimension(product=2, mask=1)
    other = tf.FingerprintDimension(product=2, mask=1)
    assert tf.fingerprint_carrier_soundness(dim, other)


def test_bundle_fingerprint_with_empty_and_constructor_bitmask() -> None:
    tf = _load()
    registry = tf.PrimeRegistry()
    ctor_registry = tf.TypeConstructorRegistry(registry)
    assert tf.bundle_fingerprint([" ", ""], registry) == 1
    assert tf.bundle_fingerprint_setlike(["", "int"], registry) == registry.get_or_assign(
        "int"
    )
    product = tf.bundle_fingerprint_with_constructors(
        ["list[int]"], registry, ctor_registry
    )
    assert product == registry.get_or_assign("list[int]")
    registry.bit_positions.pop("int", None)
    assert tf.fingerprint_bitmask(["int"], registry) == 0


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints._normalize_type_list::value
def test_build_fingerprint_registry_skips_empty_entries_with_valid() -> None:
    tf = _load()
    registry, index = tf.build_fingerprint_registry(
        {"empty": [], "valid": ["int"]}
    )
    assert registry.prime_for("int") is not None
    assert any("valid" in names for names in index.values())
