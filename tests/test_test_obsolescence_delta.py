from __future__ import annotations

from pathlib import Path

import pytest

from gabion.analysis import evidence_keys, test_obsolescence, test_obsolescence_delta


def _ref_from_display(display: str) -> test_obsolescence.EvidenceRef:
    key = evidence_keys.make_opaque_key(display)
    identity = evidence_keys.key_identity(key)
    return test_obsolescence.EvidenceRef(
        key=key,
        identity=identity,
        display=display,
        opaque=True,
    )


def _ref_paramset(value: str) -> test_obsolescence.EvidenceRef:
    key = evidence_keys.make_paramset_key([value])
    identity = evidence_keys.key_identity(key)
    return test_obsolescence.EvidenceRef(
        key=key,
        identity=identity,
        display=evidence_keys.render_display(key),
        opaque=False,
    )


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.render_display E:function_site::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_payload
def test_build_baseline_payload_roundtrip(tmp_path: Path) -> None:
    evidence_by_test = {
        "tests/test_alpha.py::test_a": [_ref_paramset("x")],
        "tests/test_beta.py::test_b": [_ref_from_display("E:opaque")],
    }
    status_by_test = {
        "tests/test_alpha.py::test_a": "mapped",
        "tests/test_beta.py::test_b": "mapped",
    }
    candidates, summary = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    payload = test_obsolescence_delta.build_baseline_payload(
        evidence_by_test, status_by_test, candidates, summary
    )
    baseline = test_obsolescence_delta.parse_baseline_payload(payload)
    assert payload["version"] == test_obsolescence_delta.BASELINE_VERSION
    assert payload["generated_by_spec_id"]
    assert payload["generated_by_spec"]
    assert baseline.summary["obsolete_candidate"] == 2
    assert baseline.tests["tests/test_alpha.py::test_a"] == "obsolete_candidate"
    assert baseline.opaque_evidence_count == 1
    assert len(baseline.evidence_index) == 2


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.render_display
def test_delta_payload_detects_changes() -> None:
    key_a = evidence_keys.make_paramset_key(["a"])
    key_b = evidence_keys.make_paramset_key(["b"])
    key_c = evidence_keys.make_paramset_key(["c"])
    baseline_payload = {
        "version": 1,
        "summary": {
            "redundant_by_evidence": 0,
            "equivalent_witness": 0,
            "obsolete_candidate": 1,
            "unmapped": 1,
        },
        "tests": [
            {"test_id": "t1", "class": "unmapped"},
            {"test_id": "t2", "class": "obsolete_candidate"},
        ],
        "evidence_index": [
            {
                "key": key_a,
                "display": evidence_keys.render_display(key_a),
                "witness_count": 2,
            },
            {
                "key": key_b,
                "display": evidence_keys.render_display(key_b),
                "witness_count": 1,
            },
        ],
        "opaque_evidence_count": 1,
        "generated_by_spec_id": "baseline",
        "generated_by_spec": {},
    }
    current_payload = {
        "version": 1,
        "summary": {
            "redundant_by_evidence": 0,
            "equivalent_witness": 1,
            "obsolete_candidate": 0,
            "unmapped": 0,
        },
        "tests": [
            {"test_id": "t2", "class": "equivalent_witness"},
            {"test_id": "t3", "class": "obsolete_candidate"},
        ],
        "evidence_index": [
            {
                "key": key_a,
                "display": evidence_keys.render_display(key_a),
                "witness_count": 1,
            },
            {
                "key": key_c,
                "display": evidence_keys.render_display(key_c),
                "witness_count": 1,
            },
        ],
        "opaque_evidence_count": 0,
        "generated_by_spec_id": "current",
        "generated_by_spec": {},
    }
    baseline = test_obsolescence_delta.parse_baseline_payload(baseline_payload)
    current = test_obsolescence_delta.parse_baseline_payload(current_payload)
    delta = test_obsolescence_delta.build_delta_payload(
        baseline, current, baseline_path="baselines/test_obsolescence_baseline.json"
    )
    summary = delta["summary"]["counts"]["delta"]
    assert summary["unmapped"] == -1
    tests = delta["tests"]
    assert tests["added"][0]["test_id"] == "t3"
    assert tests["removed"][0]["test_id"] == "t1"
    assert tests["changed_class"][0]["test_id"] == "t2"
    evidence = delta["evidence_keys"]
    assert evidence["added"][0]["display"] == evidence_keys.render_display(key_c)
    assert evidence["removed"][0]["display"] == evidence_keys.render_display(key_b)
    assert evidence["changed"][0]["delta"] == -1
    assert delta["summary"]["opaque_evidence"]["delta"] == -1


# gabion:evidence E:function_site::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_lines E:function_site::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_payload
def test_render_markdown_includes_spec_metadata() -> None:
    baseline = test_obsolescence_delta.parse_baseline_payload(
        {
            "version": 1,
            "summary": {},
            "tests": [],
            "evidence_index": [],
            "opaque_evidence_count": 0,
            "generated_by_spec_id": "base",
            "generated_by_spec": {},
        }
    )
    delta_payload = test_obsolescence_delta.build_delta_payload(baseline, baseline)
    report = test_obsolescence_delta.render_markdown(delta_payload)
    assert "generated_by_spec_id:" in report
    assert "generated_by_spec:" in report


def test_parse_baseline_payload_rejects_bad_version() -> None:
    with pytest.raises(ValueError):
        test_obsolescence_delta.parse_baseline_payload({"version": "bad"})


def test_load_baseline_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text("[1,2,3]")
    with pytest.raises(ValueError):
        test_obsolescence_delta.load_baseline(str(path))


def test_helpers_cover_edge_cases() -> None:
    assert test_obsolescence_delta._normalize_summary_counts("nope") == {
        "redundant_by_evidence": 0,
        "equivalent_witness": 0,
        "obsolete_candidate": 0,
        "unmapped": 0,
    }
    assert test_obsolescence_delta._tests_from_candidates(["bad", {"test_id": " ", "class": "x"}]) == []
    assert test_obsolescence_delta._section_list("nope", "added") == []
    assert test_obsolescence_delta._section_list({"added": "nope"}, "added") == []
    assert test_obsolescence_delta._format_delta(1, 2, None) == "1 -> 2 (+1)"


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key
def test_build_evidence_index_merges_displays() -> None:
    key = evidence_keys.make_paramset_key(["x"])
    identity = evidence_keys.key_identity(key)
    ref_a = test_obsolescence.EvidenceRef(
        key=key,
        identity=identity,
        display="Z",
        opaque=False,
    )
    ref_b = test_obsolescence.EvidenceRef(
        key=key,
        identity=identity,
        display="A",
        opaque=False,
    )
    evidence_by_test = {
        "t1": [ref_a],
        "t2": [ref_b],
        "t3": [],
    }
    status_by_test = {"t1": "mapped", "t2": "mapped", "t3": "mapped", "t4": "unmapped"}
    entries = test_obsolescence_delta._build_evidence_index(
        evidence_by_test, status_by_test
    )
    assert entries[0]["display"] == "A"
    assert entries[0]["witness_count"] == 2


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key
def test_parse_evidence_index_merges_duplicates() -> None:
    key = evidence_keys.make_paramset_key(["y"])
    entry_a = {"key": key, "display": "Z", "witness_count": 1}
    entry_b = {"key": key, "display": "A", "witness_count": 3}
    parsed = test_obsolescence_delta._parse_evidence_index(
        [entry_a, "bad", entry_b, {"display": "no-key"}]
    )
    identity = evidence_keys.key_identity(key)
    assert parsed[identity].witness_count == 3
    assert parsed[identity].display == "A"


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.render_display
def test_render_markdown_with_entries() -> None:
    key_a = evidence_keys.make_paramset_key(["a"])
    key_b = evidence_keys.make_paramset_key(["b"])
    baseline_payload = {
        "version": 1,
        "summary": {
            "redundant_by_evidence": 0,
            "equivalent_witness": 0,
            "obsolete_candidate": 1,
            "unmapped": 0,
        },
        "tests": [{"test_id": "t1", "class": "obsolete_candidate"}],
        "evidence_index": [
            {
                "key": key_a,
                "display": evidence_keys.render_display(key_a),
                "witness_count": 2,
            }
        ],
        "opaque_evidence_count": 0,
        "generated_by_spec_id": "spec",
        "generated_by_spec": {},
    }
    current_payload = {
        "version": 1,
        "summary": {
            "redundant_by_evidence": 0,
            "equivalent_witness": 1,
            "obsolete_candidate": 0,
            "unmapped": 0,
        },
        "tests": [
            {"test_id": "t1", "class": "equivalent_witness"},
            {"test_id": "t2", "class": "obsolete_candidate"},
        ],
        "evidence_index": [
            {
                "key": key_a,
                "display": evidence_keys.render_display(key_a),
                "witness_count": 1,
            },
            {
                "key": key_b,
                "display": evidence_keys.render_display(key_b),
                "witness_count": 1,
            },
        ],
        "opaque_evidence_count": 0,
        "generated_by_spec_id": "spec",
        "generated_by_spec": {},
    }
    baseline = test_obsolescence_delta.parse_baseline_payload(baseline_payload)
    current = test_obsolescence_delta.parse_baseline_payload(current_payload)
    delta = test_obsolescence_delta.build_delta_payload(baseline, current)
    report = test_obsolescence_delta.render_markdown(delta)
    assert "- `t2`" in report
    assert "Witness Count Changes" in report
    assert evidence_keys.render_display(key_b) in report


def test_parse_baseline_payload_filters_invalid_entries() -> None:
    payload = {
        "version": 1,
        "summary": {},
        "tests": ["bad", {"test_id": " ", "class": "x"}, {"test_id": "t1", "class": ""}],
        "evidence_index": "nope",
        "opaque_evidence_count": 0,
        "generated_by_spec_id": "spec",
        "generated_by_spec": {},
    }
    baseline = test_obsolescence_delta.parse_baseline_payload(payload)
    assert baseline.tests == {}
