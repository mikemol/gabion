from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis import test_obsolescence


def _write_test_evidence(tmp_path: Path, tests: list[dict[str, object]]) -> Path:
    payload = {
        "schema_version": 1,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": tests,
        "evidence_index": [],
    }
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "test_evidence.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_basic_dominance() -> None:
    evidence_by_test = {
        "tests/test_alpha.py::test_a": ["E:x"],
        "tests/test_beta.py::test_b": ["E:x", "E:y"],
    }
    status_by_test = {
        "tests/test_alpha.py::test_a": "mapped",
        "tests/test_beta.py::test_b": "mapped",
    }
    dominators = test_obsolescence.compute_dominators(evidence_by_test)
    assert dominators["tests/test_alpha.py::test_a"] == [
        "tests/test_beta.py::test_b"
    ]

    candidates, summary = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    by_id = {entry["test_id"]: entry for entry in candidates}
    assert by_id["tests/test_alpha.py::test_a"]["class"] == "redundant_by_evidence"
    assert by_id["tests/test_beta.py::test_b"]["class"] == "obsolete_candidate"
    assert summary == {
        "redundant_by_evidence": 1,
        "equivalent_witness": 0,
        "obsolete_candidate": 1,
        "unmapped": 0,
    }


def test_unmapped_classification(tmp_path: Path) -> None:
    evidence_path = _write_test_evidence(
        tmp_path,
        [
            {
                "test_id": "tests/test_unmapped.py::test_skip",
                "file": "tests/test_unmapped.py",
                "line": 1,
                "evidence": [],
                "status": "unmapped",
            },
            {
                "test_id": "tests/test_mapped.py::test_ok",
                "file": "tests/test_mapped.py",
                "line": 1,
                "evidence": ["E:ok"],
                "status": "mapped",
            },
        ],
    )
    evidence_by_test, status_by_test = test_obsolescence.load_test_evidence(
        str(evidence_path)
    )
    candidates, summary = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    by_id = {entry["test_id"]: entry for entry in candidates}
    assert by_id["tests/test_unmapped.py::test_skip"]["class"] == "unmapped"
    assert summary["unmapped"] == 1


def test_equivalent_witness_classification() -> None:
    evidence_by_test = {
        "tests/test_alpha.py::test_a": ["E:x"],
        "tests/test_beta.py::test_b": ["E:x"],
    }
    status_by_test = {
        "tests/test_alpha.py::test_a": "mapped",
        "tests/test_beta.py::test_b": "mapped",
    }
    candidates, summary = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    by_id = {entry["test_id"]: entry for entry in candidates}
    assert by_id["tests/test_alpha.py::test_a"]["class"] == "equivalent_witness"
    assert by_id["tests/test_beta.py::test_b"]["class"] == "equivalent_witness"
    assert summary == {
        "redundant_by_evidence": 0,
        "equivalent_witness": 2,
        "obsolete_candidate": 0,
        "unmapped": 0,
    }
