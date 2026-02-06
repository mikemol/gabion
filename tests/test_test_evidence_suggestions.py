from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis import test_evidence_suggestions


def test_suggests_alias_invariance() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_alias_attribute.py::test_alias_attribute_forwarding",
        file="tests/test_alias_attribute.py",
        line=10,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.suggested == 1
    assert suggestions[0].suggested == ("E:bundle/alias_invariance",)
    assert suggestions[0].matches == ("alias_invariance",)


def test_suggests_aspf_bundle_pair() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_aspf.py::test_paramset_packed_reuse",
        file="tests/test_aspf.py",
        line=12,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.suggested == 1
    assert suggestions[0].suggested == (
        "E:forest/canonical_paramset",
        "E:forest/packed_reuse",
    )


def test_skips_mapped_entries() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_baseline_ratchet.py::test_baseline_write_and_apply",
        file="tests/test_baseline_ratchet.py",
        line=2,
        evidence=("E:baseline/ratchet_monotonicity",),
        status="mapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert suggestions == []
    assert summary.skipped_mapped == 1


def test_skips_no_match_entries() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_misc.py::test_smoke",
        file="tests/test_misc.py",
        line=1,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert suggestions == []
    assert summary.skipped_no_match == 1


def test_load_test_evidence_payload(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_policy_check.py::test_policy_check_runs",
                "file": "tests/test_policy_check.py",
                "line": 5,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    path = tmp_path / "test_evidence.json"
    path.write_text(json.dumps(payload))
    entries = test_evidence_suggestions.load_test_evidence(str(path))
    assert entries[0].test_id.endswith("test_policy_check.py::test_policy_check_runs")
