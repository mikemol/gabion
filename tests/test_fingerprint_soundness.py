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
