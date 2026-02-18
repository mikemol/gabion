from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis import semantic_coverage_map, test_evidence


def _write_test_module(root: Path) -> Path:
    path = root / "tests" / "test_semantic_case.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
# gabion:evidence INV.alpha

def test_alpha() -> None:
    assert True

# gabion:evidence INV.beta

def test_beta() -> None:
    assert True
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_semantic_coverage_payload_is_deterministic(tmp_path: Path) -> None:
    test_path = _write_test_module(tmp_path)
    evidence_payload = test_evidence.build_test_evidence_payload(
        [test_path],
        root=tmp_path,
    )
    evidence_path = tmp_path / "out" / "test_evidence.json"
    test_evidence.write_test_evidence(evidence_payload, evidence_path)
    mapping_path = tmp_path / "out" / "semantic_coverage_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {"obligation": "lemma.beta", "obligation_kind": "lemma", "evidence": "INV.beta"},
                    {"obligation": "invariant.alpha", "obligation_kind": "invariant", "evidence": "INV.alpha"},
                ],
            }
        ),
        encoding="utf-8",
    )

    first = semantic_coverage_map.build_semantic_coverage_payload(
        [test_path],
        root=tmp_path,
        mapping_path=mapping_path,
        evidence_path=evidence_path,
    )
    second = semantic_coverage_map.build_semantic_coverage_payload(
        [test_path],
        root=tmp_path,
        mapping_path=mapping_path,
        evidence_path=evidence_path,
    )

    assert first == second
    assert [entry["obligation"] for entry in first["mapped_obligations"]] == [
        "invariant.alpha",
        "lemma.beta",
    ]


def test_semantic_coverage_reports_unmapped_dead_and_duplicate_entries(tmp_path: Path) -> None:
    test_path = _write_test_module(tmp_path)
    evidence_payload = test_evidence.build_test_evidence_payload(
        [test_path],
        root=tmp_path,
    )
    evidence_path = tmp_path / "out" / "test_evidence.json"
    test_evidence.write_test_evidence(evidence_payload, evidence_path)
    mapping_path = tmp_path / "out" / "semantic_coverage_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {"obligation": "invariant.alpha", "obligation_kind": "invariant", "evidence": "INV.alpha"},
                    {"obligation": "invariant.alpha", "obligation_kind": "invariant", "evidence": "INV.alpha"},
                    {"obligation": "invariant.dead", "obligation_kind": "invariant", "evidence": "INV.missing"},
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = semantic_coverage_map.build_semantic_coverage_payload(
        [test_path],
        root=tmp_path,
        mapping_path=mapping_path,
        evidence_path=evidence_path,
    )

    assert [entry["obligation"] for entry in payload["unmapped_obligations"]] == [
        "invariant.dead"
    ]
    assert len(payload["dead_mapping_entries"]) == 1
    assert payload["duplicate_mapping_entries"][0]["count"] == 2
