from __future__ import annotations

from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import dataflow_audit as da
    from gabion.analysis import type_fingerprints as tf

    return da, tf

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::a,b E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::a,b E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::stale_81ea454a69a5
def test_fingerprint_soundness_issues_detects_overlap() -> None:
    da, tf = _load()
    fingerprint = tf.Fingerprint(
        base=tf.FingerprintDimension(product=2, mask=0),
        ctor=tf.FingerprintDimension(product=2, mask=0),
    )
    issues = da._fingerprint_soundness_issues(fingerprint)
    assert "base/ctor" in issues

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::a,b E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::a,b E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::stale_000d23f26126
def test_fingerprint_soundness_issues_skip_empty() -> None:
    da, tf = _load()
    fingerprint = tf.Fingerprint(
        base=tf.FingerprintDimension(product=2, mask=0),
    )
    assert da._fingerprint_soundness_issues(fingerprint) == []

# gabion:evidence E:function_site::tests/test_fingerprint_soundness.py::tests.test_fingerprint_soundness.test_fingerprint_identity_payload_marks_canonical_vs_derived
def test_fingerprint_identity_payload_marks_canonical_vs_derived() -> None:
    _, tf = _load()
    fingerprint = tf.Fingerprint(base=tf.FingerprintDimension(product=2, mask=0))
    payload = tf.fingerprint_identity_payload(fingerprint)
    assert payload["identity_layers"]["identity_layer"] == "canonical_aspf_path"
    assert payload["identity_layers"]["derived"]["scalar_prime_product"]["canonical"] is False
    assert payload["identity_layers"]["derived"]["digest_alias"]["canonical"] is False

# gabion:evidence E:function_site::tests/test_fingerprint_soundness.py::tests.test_fingerprint_soundness.test_fingerprint_identity_payload_handles_empty_cofibration_basis
def test_fingerprint_identity_payload_handles_empty_cofibration_basis() -> None:
    _, tf = _load()
    fingerprint = tf.Fingerprint(
        base=tf.FingerprintDimension(product=1, mask=0),
        ctor=tf.FingerprintDimension(product=1, mask=0),
        provenance=tf.FingerprintDimension(product=1, mask=0),
        synth=tf.FingerprintDimension(product=1, mask=0),
    )
    payload = tf.fingerprint_identity_payload(fingerprint)
    assert payload["witness_carriers"]["cofibration_witness"] == {"entries": []}
    assert "cofibration_witness" not in payload


def _normalize_provenance(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(entries, key=lambda entry: str(entry.get("provenance_id", "")))


# gabion:evidence E:call_footprint::tests/test_fingerprint_soundness.py::test_fingerprint_phase_outputs_are_stable_under_permuted_input_order::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_fingerprint_atom_keys::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth
def test_fingerprint_phase_outputs_are_stable_under_permuted_input_order() -> None:
    da, tf = _load()

    first = Path("pkg/a.py")
    second = Path("pkg/b.py")
    groups_a = {
        first: {"f": [{"left", "right"}]},
        second: {"g": [{"payload", "meta"}]},
    }
    annotations_a = {
        first: {"f": {"left": "list[int]", "right": "dict[str, int]"}},
        second: {"g": {"payload": "tuple[str, int]", "meta": "list[str]"}},
    }
    groups_b = {
        second: {"g": [{"meta", "payload"}]},
        first: {"f": [{"right", "left"}]},
    }
    annotations_b = {
        second: {"g": {"meta": "list[str]", "payload": "tuple[str, int]"}},
        first: {"f": {"right": "dict[str, int]", "left": "list[int]"}},
    }

    base_registry, base_index = tf.build_fingerprint_registry(
        {"shape.a": ["list[int]", "dict[str, int]"], "shape.b": ["tuple[str, int]", "list[str]"]}
    )
    seed = base_registry.seed_payload()

    def _run(groups, annotations):
        registry = tf.PrimeRegistry()
        registry.load_seed_payload(seed)
        ctor_registry = tf.TypeConstructorRegistry(registry)
        base_keys, ctor_keys = da._collect_fingerprint_atom_keys(groups, annotations)
        for key in base_keys:
            registry.get_or_assign(key)
        for key in ctor_keys:
            ctor_registry.get_or_assign(key)
        warnings = da._compute_fingerprint_warnings(
            groups,
            annotations,
            registry=registry,
            index=base_index,
            ctor_registry=ctor_registry,
        )
        matches = da._compute_fingerprint_matches(
            groups,
            annotations,
            registry=registry,
            index=base_index,
            ctor_registry=ctor_registry,
        )
        provenance = da._compute_fingerprint_provenance(
            groups,
            annotations,
            registry=registry,
            project_root=None,
            index=base_index,
            ctor_registry=ctor_registry,
        )
        synth_lines, synth_payload = da._compute_fingerprint_synth(
            groups,
            annotations,
            registry=registry,
            ctor_registry=ctor_registry,
            min_occurrences=2,
            version="synth-registry@v1",
            existing=None,
        )
        return warnings, matches, _normalize_provenance(provenance), synth_lines, synth_payload

    assert _run(groups_a, annotations_a) == _run(groups_b, annotations_b)
