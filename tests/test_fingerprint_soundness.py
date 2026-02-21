from __future__ import annotations

from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import dataflow_audit as da
    from gabion.analysis import type_fingerprints as tf

    return da, tf

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::a,b E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::a,b
def test_fingerprint_soundness_issues_detects_overlap() -> None:
    da, tf = _load()
    fingerprint = tf.Fingerprint(
        base=tf.FingerprintDimension(product=2, mask=0),
        ctor=tf.FingerprintDimension(product=2, mask=0),
    )
    issues = da._fingerprint_soundness_issues(fingerprint)
    assert "base/ctor" in issues

# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::a,b E:decision_surface/value_encoded::type_fingerprints.py::gabion.analysis.type_fingerprints.fingerprint_carrier_soundness::a,b
def test_fingerprint_soundness_issues_skip_empty() -> None:
    da, tf = _load()
    fingerprint = tf.Fingerprint(
        base=tf.FingerprintDimension(product=2, mask=0),
    )
    assert da._fingerprint_soundness_issues(fingerprint) == []
