from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da
    from gabion.analysis import type_fingerprints as tf

    return da, tf


def test_fingerprint_soundness_issues_detects_overlap() -> None:
    da, tf = _load()
    fingerprint = tf.Fingerprint(
        base=tf.FingerprintDimension(product=2, mask=0),
        ctor=tf.FingerprintDimension(product=2, mask=0),
    )
    issues = da._fingerprint_soundness_issues(fingerprint)
    assert "base/ctor" in issues


def test_fingerprint_soundness_issues_skip_empty() -> None:
    da, tf = _load()
    fingerprint = tf.Fingerprint(
        base=tf.FingerprintDimension(product=2, mask=0),
    )
    assert da._fingerprint_soundness_issues(fingerprint) == []
