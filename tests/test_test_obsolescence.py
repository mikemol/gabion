from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis import evidence_keys, test_obsolescence


def _evidence_item(display: str) -> dict[str, object]:
    key = evidence_keys.make_opaque_key(display)
    return {"key": key, "display": display}


def _write_test_evidence(tmp_path: Path, tests: list[dict[str, object]]) -> Path:
    payload = {
        "schema_version": 2,
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
        "tests/test_alpha.py::test_a": [
            test_obsolescence.EvidenceRef(
                key=evidence_keys.make_opaque_key("E:x"),
                identity=evidence_keys.key_identity(evidence_keys.make_opaque_key("E:x")),
                display="E:x",
                opaque=True,
            )
        ],
        "tests/test_beta.py::test_b": [
            test_obsolescence.EvidenceRef(
                key=evidence_keys.make_opaque_key("E:x"),
                identity=evidence_keys.key_identity(evidence_keys.make_opaque_key("E:x")),
                display="E:x",
                opaque=True,
            ),
            test_obsolescence.EvidenceRef(
                key=evidence_keys.make_opaque_key("E:y"),
                identity=evidence_keys.key_identity(evidence_keys.make_opaque_key("E:y")),
                display="E:y",
                opaque=True,
            ),
        ],
    }
    status_by_test = {
        "tests/test_alpha.py::test_a": "mapped",
        "tests/test_beta.py::test_b": "mapped",
    }
    dominators = test_obsolescence.compute_dominators(
        {
            test_id: [ref.identity for ref in refs]
            for test_id, refs in evidence_by_test.items()
        }
    )
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
                "evidence": [_evidence_item("E:ok")],
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
    ref = test_obsolescence.EvidenceRef(
        key=evidence_keys.make_opaque_key("E:x"),
        identity=evidence_keys.key_identity(evidence_keys.make_opaque_key("E:x")),
        display="E:x",
        opaque=True,
    )
    evidence_by_test = {
        "tests/test_alpha.py::test_a": [ref],
        "tests/test_beta.py::test_b": [ref],
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


def test_render_markdown_includes_spec_metadata() -> None:
    candidates = [
        {
            "test_id": "tests/test_alpha.py::test_a",
            "class": "unmapped",
            "dominators": [],
            "reason": {"status": "unmapped"},
        }
    ]
    summary = {"redundant_by_evidence": 0, "equivalent_witness": 0, "obsolete_candidate": 0, "unmapped": 1}
    report = test_obsolescence.render_markdown(candidates, summary)
    assert "generated_by_spec_id:" in report
    assert "generated_by_spec:" in report


def test_render_json_payload_includes_spec_metadata() -> None:
    candidates = []
    summary = {
        "redundant_by_evidence": 0,
        "equivalent_witness": 0,
        "obsolete_candidate": 0,
        "unmapped": 0,
    }
    payload = test_obsolescence.render_json_payload(candidates, summary)
    assert payload["version"] == 3
    assert "generated_by_spec_id" in payload
    assert "generated_by_spec" in payload
