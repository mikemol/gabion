from pathlib import Path

from gabion.analysis.dataflow_audit import (
    _compute_fingerprint_matches,
    _compute_fingerprint_provenance,
)
from gabion.analysis.type_fingerprints import (
    FingerprintDimension,
    PrimeRegistry,
    TypeConstructorRegistry,
    bundle_fingerprint_dimensional,
    fingerprint_to_type_keys_with_remainder,
    format_fingerprint,
)


def _legacy_decode(fingerprint, registry: PrimeRegistry) -> tuple[list[str], int, list[str], int]:
    base_keys, base_remaining = fingerprint_to_type_keys_with_remainder(
        fingerprint.base.product,
        registry,
    )
    ctor_keys, ctor_remaining = fingerprint_to_type_keys_with_remainder(
        fingerprint.ctor.product,
        registry,
    )
    ctor_keys = [
        key[len("ctor:") :] if key.startswith("ctor:") else key
        for key in ctor_keys
    ]
    return base_keys, base_remaining, ctor_keys, ctor_remaining


# gabion:evidence E:call_footprint::tests/test_type_fingerprints_sidecar.py::test_dimension_sidecar_decode_matches_legacy_division::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_to_type_keys_with_remainder
def test_dimension_sidecar_decode_matches_legacy_division() -> None:
    registry = PrimeRegistry()
    ctor_registry = TypeConstructorRegistry(registry)
    fingerprint = bundle_fingerprint_dimensional(
        ["list[int]", "dict[str, int]"],
        registry,
        ctor_registry,
    )

    expected_base = fingerprint_to_type_keys_with_remainder(fingerprint.base.product, registry)
    expected_ctor = fingerprint_to_type_keys_with_remainder(fingerprint.ctor.product, registry)

    base_keys, base_remaining = fingerprint.base.keys_with_remainder(registry)
    ctor_keys, ctor_remaining = fingerprint.ctor.keys_with_remainder(registry)
    assert (sorted(base_keys), base_remaining) == (sorted(expected_base[0]), expected_base[1])
    assert (sorted(ctor_keys), ctor_remaining) == (sorted(expected_ctor[0]), expected_ctor[1])


# gabion:evidence E:call_footprint::tests/test_type_fingerprints_sidecar.py::test_dimension_sidecar_falls_back_to_product_when_inconsistent::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_to_type_keys_with_remainder
def test_dimension_sidecar_falls_back_to_product_when_inconsistent() -> None:
    registry = PrimeRegistry()
    ctor_registry = TypeConstructorRegistry(registry)
    fingerprint = bundle_fingerprint_dimensional(["list[int]"], registry, ctor_registry)

    inconsistent = type(fingerprint.base)(
        product=fingerprint.base.product,
        mask=fingerprint.base.mask,
        exponents=(("list", 4),),
    )

    assert inconsistent.keys_with_remainder(registry) == fingerprint_to_type_keys_with_remainder(
        fingerprint.base.product,
        registry,
    )


# gabion:evidence E:call_footprint::tests/test_type_fingerprints_sidecar.py::test_dimension_sidecar_skips_non_positive_exponents::type_fingerprints.py::gabion.analysis.type_fingerprints.FingerprintDimension::type_fingerprints.py::gabion.analysis.type_fingerprints.PrimeRegistry
def test_dimension_sidecar_skips_non_positive_exponents() -> None:
    registry = PrimeRegistry()
    int_prime = registry.get_or_assign("int")
    dimension = FingerprintDimension(
        product=int_prime,
        mask=0,
        exponents=(("int", 0), ("int", 1), ("int", -2)),
    )

    assert dimension.keys_with_remainder(registry) == (["int"], 1)


# gabion:evidence E:call_footprint::tests/test_type_fingerprints_sidecar.py::test_dimension_sidecar_falls_back_on_product_mismatch::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_to_type_keys_with_remainder
def test_dimension_sidecar_falls_back_on_product_mismatch() -> None:
    registry = PrimeRegistry()
    int_prime = registry.get_or_assign("int")
    inconsistent = FingerprintDimension(
        product=int_prime,
        mask=0,
        exponents=(("int", 2),),
    )

    assert inconsistent.keys_with_remainder(registry) == fingerprint_to_type_keys_with_remainder(
        inconsistent.product,
        registry,
    )


# gabion:evidence E:call_footprint::tests/test_type_fingerprints_sidecar.py::test_dataflow_fingerprint_reporting_parity_with_legacy_decode::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::test_type_fingerprints_sidecar.py::tests.test_type_fingerprints_sidecar._legacy_decode::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::type_fingerprints.py::gabion.analysis.type_fingerprints.format_fingerprint
def test_dataflow_fingerprint_reporting_parity_with_legacy_decode() -> None:
    registry = PrimeRegistry()
    ctor_registry = TypeConstructorRegistry(registry)

    path = Path("pkg/mod.py")
    groups_by_path = {path: {"fn": [{"left", "right"}]}}
    annotations_by_path = {
        path: {
            "fn": {
                "left": "list[int]",
                "right": "dict[str, int]",
            }
        }
    }

    fingerprint = bundle_fingerprint_dimensional(
        ["list[int]", "dict[str, int]"],
        registry,
        ctor_registry,
    )
    index = {fingerprint: {"bundle.shape"}}
    legacy_base, legacy_base_remaining, legacy_ctor, legacy_ctor_remaining = _legacy_decode(
        fingerprint,
        registry,
    )
    legacy_base_sorted = sorted(legacy_base)
    legacy_ctor_sorted = sorted(legacy_ctor)

    matches = _compute_fingerprint_matches(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index=index,
        ctor_registry=ctor_registry,
    )
    expected_detail = (
        f"mod.py:fn bundle ['left', 'right'] fingerprint {format_fingerprint(fingerprint)} "
        "matches: bundle.shape "
        f"base={legacy_base_sorted} ctor={legacy_ctor_sorted}"
    )
    if legacy_base_remaining not in (0, 1) or legacy_ctor_remaining not in (0, 1):
        expected_detail += f" remainder=({legacy_base_remaining},{legacy_ctor_remaining})"
    assert expected_detail in matches

    provenance = _compute_fingerprint_provenance(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        project_root=None,
        index=index,
        ctor_registry=ctor_registry,
    )
    assert len(provenance) == 1
    entry = provenance[0]
    assert entry["base_keys"] == legacy_base_sorted
    assert entry["ctor_keys"] == legacy_ctor_sorted
    assert entry["remainder"] == {
        "base": legacy_base_remaining,
        "ctor": legacy_ctor_remaining,
    }
