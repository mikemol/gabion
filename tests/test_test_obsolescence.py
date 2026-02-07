from __future__ import annotations

import json
from pathlib import Path

import pytest

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


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key
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


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key
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


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key
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


# gabion:evidence E:function_site::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_lines
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


# gabion:evidence E:function_site::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_payload
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


# gabion:evidence E:function_site::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_lines
def test_render_markdown_includes_suffix_details() -> None:
    candidates = [
        {
            "test_id": "tests/test_alpha.py::test_a",
            "class": "redundant_by_evidence",
            "dominators": ["tests/test_beta.py::test_b"],
            "reason": {
                "guardrail": "high-risk-last-witness",
                "guardrail_evidence": ["E:high"],
                "opaque_evidence": ["E:opaque"],
            },
        }
    ]
    summary = {
        "redundant_by_evidence": 1,
        "equivalent_witness": 0,
        "obsolete_candidate": 0,
        "unmapped": 0,
    }
    report = test_obsolescence.render_markdown(candidates, summary)
    assert "dominators" in report
    assert "guardrail" in report
    assert "opaque: 1" in report


# gabion:evidence E:function_site::test_obsolescence.py::gabion.analysis.test_obsolescence.RiskInfo.from_payload
def test_risk_info_from_payload_variants() -> None:
    assert test_obsolescence.RiskInfo.from_payload("nope") is None
    assert test_obsolescence.RiskInfo.from_payload({"risk": ""}) is None
    info = test_obsolescence.RiskInfo.from_payload(
        {"risk": "high", "owner": "me", "rationale": "why"}
    )
    assert info is not None
    assert info.risk == "high"


# gabion:evidence E:function_site::test_obsolescence.py::gabion.analysis.test_obsolescence.load_test_evidence
def test_load_test_evidence_errors(tmp_path: Path) -> None:
    bad_schema = tmp_path / "bad.json"
    bad_schema.write_text(json.dumps({"schema_version": 3, "tests": []}))
    with pytest.raises(ValueError):
        test_obsolescence.load_test_evidence(str(bad_schema))

    bad_tests = tmp_path / "bad_tests.json"
    bad_tests.write_text(json.dumps({"schema_version": 2, "tests": "nope"}))
    with pytest.raises(ValueError):
        test_obsolescence.load_test_evidence(str(bad_tests))


# gabion:evidence E:function_site::test_obsolescence.py::gabion.analysis.test_obsolescence.load_test_evidence
def test_load_test_evidence_skips_invalid_entries(tmp_path: Path) -> None:
    payload = {
        "schema_version": 2,
        "tests": [
            "bad",
            {"test_id": " ", "evidence": []},
            {"test_id": "t1", "evidence": ["E:x"]},
        ],
    }
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(payload))
    evidence_by_test, status_by_test = test_obsolescence.load_test_evidence(str(path))
    assert list(evidence_by_test.keys()) == ["t1"]
    assert status_by_test["t1"] == "mapped"


# gabion:evidence E:function_site::test_obsolescence.py::gabion.analysis.test_obsolescence._parse_risk_registry_payload E:function_site::test_obsolescence.py::gabion.analysis.test_obsolescence.load_risk_registry
def test_load_risk_registry_variants(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert test_obsolescence.load_risk_registry(str(missing)) == {}

    bad_version = tmp_path / "bad_version.json"
    bad_version.write_text(json.dumps({"version": 2, "evidence": {}}))
    with pytest.raises(ValueError):
        test_obsolescence.load_risk_registry(str(bad_version))

    bad_evidence = tmp_path / "bad_evidence.json"
    bad_evidence.write_text(json.dumps({"version": 1, "evidence": "nope"}))
    assert test_obsolescence.load_risk_registry(str(bad_evidence)) == {}

    bad_payload = tmp_path / "bad_payload.json"
    bad_payload.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        test_obsolescence.load_risk_registry(str(bad_payload))

    registry = test_obsolescence._parse_risk_registry_payload(
        {
            "version": 1,
            "evidence": {
                1: {"risk": "high"},
                "E:x": {"risk": ""},
                "E:ok": {"risk": "low"},
            },
        }
    )
    assert list(registry.keys()) == ["E:ok"]


# gabion:evidence E:function_site::test_obsolescence.py::gabion.analysis.test_obsolescence.compute_dominators
def test_compute_dominators_handles_empty_evidence() -> None:
    dominators = test_obsolescence.compute_dominators({"t1": []})
    assert dominators["t1"] == []


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.render_display
def test_guardrail_and_opaque_evidence() -> None:
    key = evidence_keys.make_paramset_key(["x"])
    identity = evidence_keys.key_identity(key)
    ref_guard = test_obsolescence.EvidenceRef(
        key=key,
        identity=identity,
        display="E:high",
        opaque=True,
    )
    ref_other = test_obsolescence.EvidenceRef(
        key=key,
        identity=identity,
        display=evidence_keys.render_display(key),
        opaque=False,
    )
    ref_extra = test_obsolescence.EvidenceRef(
        key=evidence_keys.make_opaque_key("E:extra"),
        identity=evidence_keys.key_identity(evidence_keys.make_opaque_key("E:extra")),
        display="E:extra",
        opaque=True,
    )
    evidence_by_test = {
        "t1": [ref_guard],
        "t2": [ref_other, ref_extra],
    }
    status_by_test = {"t1": "mapped", "t2": "mapped"}
    risk_registry = {"E:high": test_obsolescence.RiskInfo(risk="high", owner="", rationale="")}
    candidates, summary = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, risk_registry
    )
    by_id = {entry["test_id"]: entry for entry in candidates}
    assert by_id["t1"]["class"] == "obsolete_candidate"
    assert by_id["t1"]["reason"]["guardrail"] == "high-risk-last-witness"
    assert "opaque_evidence" in by_id["t1"]["reason"]
    assert summary["redundant_by_evidence"] == 0


# gabion:evidence E:function_site::test_obsolescence.py::gabion.analysis.test_obsolescence._summarize_candidates
def test_summarize_candidates_handles_bad_counts() -> None:
    def apply(_spec, _relation):
        return [{"class": "unmapped", "count": "bad"}]

    summary = test_obsolescence._summarize_candidates(
        [{"class": "unmapped"}], {"unmapped": 0}, apply=apply
    )
    assert summary["unmapped"] == 0


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key
def test_normalize_evidence_refs_variants() -> None:
    ref = test_obsolescence.EvidenceRef(
        key=evidence_keys.make_opaque_key("E:x"),
        identity=evidence_keys.key_identity(evidence_keys.make_opaque_key("E:x")),
        display="E:x",
        opaque=True,
    )
    assert test_obsolescence._normalize_evidence_refs(None) == []
    assert test_obsolescence._normalize_evidence_refs(ref) == [ref]
    assert test_obsolescence._normalize_evidence_refs("E:paramset::x")
    refs = test_obsolescence._normalize_evidence_refs(
        [
            ref,
            {"key": evidence_keys.make_paramset_key(["a"]), "display": "E:paramset::a"},
            {"display": "E:paramset::b"},
            {"display": "E:opaque"},
            {"display": 123},
            " ",
            "E:opaque",
        ]
    )
    assert [item.display for item in refs if item.display]  # non-empty list
