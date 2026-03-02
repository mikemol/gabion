from __future__ import annotations

import json
from pathlib import Path

import pytest

from gabion.analysis import evidence_keys, test_obsolescence, test_obsolescence_delta
from gabion.analysis.delta_tools import TransitionPair


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.render_display E:function_site::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_payload
def test_build_baseline_payload_roundtrip(
    tmp_path: Path,
    make_obsolescence_paramset_ref,
    make_obsolescence_opaque_ref,
) -> None:
    evidence_by_test = {
        "tests/test_alpha.py::test_a": [make_obsolescence_paramset_ref("x")],
        "tests/test_beta.py::test_b": [make_obsolescence_opaque_ref("E:opaque")],
    }
    status_by_test = {
        "tests/test_alpha.py::test_a": "mapped",
        "tests/test_beta.py::test_b": "mapped",
    }
    classification = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    payload = test_obsolescence_delta.build_baseline_payload(
        evidence_by_test,
        status_by_test,
        classification.stale_candidates,
        classification.stale_summary,
        active_tests=classification.active_tests,
        active_summary=classification.active_summary,
    )
    baseline = test_obsolescence_delta.parse_baseline_payload(payload)
    assert payload["version"] == test_obsolescence_delta.BASELINE_VERSION
    assert payload["generated_by_spec_id"]
    assert payload["generated_by_spec"]
    assert baseline.summary["obsolete_candidate"] == 0
    assert baseline.tests == {}
    assert baseline.active["summary"]["active_total"] == 2
    assert baseline.opaque_evidence_count == 1
    assert len(baseline.evidence_index) == 2


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.render_display E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key::stale_d66185f4e6fa
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


# gabion:evidence E:function_site::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.parse_baseline_payload E:decision_surface/direct::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.parse_baseline_payload::stale_c572e23c93c8
def test_parse_baseline_payload_rejects_bad_version() -> None:
    with pytest.raises(ValueError):
        test_obsolescence_delta.parse_baseline_payload({"version": "bad"})


# gabion:evidence E:function_site::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.parse_baseline_payload E:decision_surface/direct::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.parse_baseline_payload::stale_cc8a9fb52fb7_b91548fb
def test_parse_baseline_payload_accepts_v1_and_v2() -> None:
    payload_v1 = {
        "version": 1,
        "summary": {"unmapped": 1},
        "tests": [{"test_id": "t1", "class": "unmapped"}],
        "evidence_index": [],
        "opaque_evidence_count": 0,
        "generated_by_spec_id": "v1",
        "generated_by_spec": {},
    }
    parsed_v1 = test_obsolescence_delta.parse_baseline_payload(payload_v1)
    assert parsed_v1.summary["unmapped"] == 1
    assert parsed_v1.active == {}

    payload_v2 = {
        "version": 2,
        "summary": {"unmapped": 0},
        "tests": [],
        "active": {"summary": {"active_total": 3}},
        "evidence_index": [],
        "opaque_evidence_count": 0,
        "generated_by_spec_id": "v2",
        "generated_by_spec": {},
    }
    parsed_v2 = test_obsolescence_delta.parse_baseline_payload(payload_v2)
    assert parsed_v2.summary["unmapped"] == 0
    assert parsed_v2.active["summary"]["active_total"] == 3


# gabion:evidence E:call_footprint::tests/test_test_obsolescence_delta.py::test_build_baseline_payload_omits_empty_active_metadata::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.build_baseline_payload
def test_build_baseline_payload_omits_empty_active_metadata() -> None:
    payload = test_obsolescence_delta.build_baseline_payload(
        evidence_by_test={"t1": []},
        status_by_test={"t1": "mapped"},
        candidates=[{"test_id": "t1", "class": "unmapped"}],
        summary_counts={"unmapped": 1},
    )
    assert payload["tests"] == [{"test_id": "t1", "class": "unmapped"}]
    assert "active" not in payload


# gabion:evidence E:call_footprint::tests/test_test_obsolescence_delta.py::test_normalize_active_metadata_handles_invalid_inputs::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta._normalize_active_metadata
def test_normalize_active_metadata_handles_invalid_inputs() -> None:
    assert test_obsolescence_delta._normalize_active_metadata([]) == {}
    assert (
        test_obsolescence_delta._normalize_active_metadata(
            {"tests": "bad", "summary": []}
        )
        == {}
    )
    normalized = test_obsolescence_delta._normalize_active_metadata(
        {
            "tests": [1, "", "b", "a", "b"],
            "summary": {1: 2, "active_total": "3"},
        }
    )
    assert normalized == {"tests": ["a", "b"], "summary": {"active_total": 3}}


# gabion:evidence E:function_site::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.load_baseline
def test_load_baseline_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text("[1,2,3]")
    with pytest.raises(ValueError):
        test_obsolescence_delta.load_baseline(str(path))


# gabion:evidence E:function_site::delta_tools.py::gabion.analysis.delta_tools.format_transition E:function_site::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta._normalize_summary_counts E:function_site::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta._section_list
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
    assert (
        test_obsolescence_delta.format_transition(TransitionPair(1, 2), None)
        == "1 -> 2 (+1)"
    )


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.key_identity::stale_8098d92df5ab
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


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.key_identity::stale_6d752f1b4bee
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


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.render_display E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key::stale_a19a5554d0bc
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


# gabion:evidence E:function_site::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.parse_baseline_payload E:decision_surface/direct::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.parse_baseline_payload::stale_368f8bfe951a
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


# gabion:evidence E:call_footprint::tests/test_test_obsolescence_delta.py::test_parse_baseline_payload_ignores_non_list_tests_payload::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.parse_baseline_payload
def test_parse_baseline_payload_ignores_non_list_tests_payload() -> None:
    baseline = test_obsolescence_delta.parse_baseline_payload(
        {
            "version": 1,
            "summary": {},
            "tests": {"bad": True},
            "evidence_index": [],
            "opaque_evidence_count": 0,
            "generated_by_spec_id": "spec",
            "generated_by_spec": {},
        }
    )
    assert baseline.tests == {}


# gabion:evidence E:call_footprint::tests/test_test_obsolescence_delta.py::test_render_markdown_handles_non_mapping_summary_and_meta_sections::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.render_markdown
def test_render_markdown_handles_non_mapping_summary_and_meta_sections() -> None:
    rendered = test_obsolescence_delta.render_markdown(
        {
            "summary": "bad",
            "tests": {},
            "evidence_keys": {},
            "baseline": [],
            "current": [],
            "generated_by_spec_id": "spec",
            "generated_by_spec": {},
        }
    )
    assert "Summary" in rendered


# gabion:evidence E:call_footprint::tests/test_test_obsolescence_delta.py::test_parse_evidence_index_prefers_existing_display_when_new_display_missing::evidence_keys.py::gabion.analysis.evidence_keys.key_identity::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta._parse_evidence_index
def test_parse_evidence_index_prefers_existing_display_when_new_display_missing() -> None:
    key = evidence_keys.make_paramset_key(["z"])
    entry_a = {"key": key, "display": "A", "witness_count": 1}
    entry_b = {"key": key, "display": "", "witness_count": 3}
    parsed = test_obsolescence_delta._parse_evidence_index([entry_a, entry_b])
    identity = evidence_keys.key_identity(key)
    assert parsed[identity].display == "A"
    assert parsed[identity].witness_count == 3


# gabion:evidence E:call_footprint::tests/test_test_obsolescence_delta.py::test_render_markdown_skips_empty_baseline_and_current_spec_ids::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.render_markdown
def test_render_markdown_skips_empty_baseline_and_current_spec_ids() -> None:
    rendered = test_obsolescence_delta.render_markdown(
        {
            "summary": {"counts": {}},
            "tests": {},
            "evidence_keys": {},
            "baseline": {"generated_by_spec_id": ""},
            "current": {"generated_by_spec_id": ""},
            "generated_by_spec_id": "spec",
            "generated_by_spec": {},
        }
    )
    assert "baseline_spec_id" not in rendered
    assert "current_spec_id" not in rendered


# gabion:evidence E:call_footprint::tests/test_test_obsolescence_delta.py::test_resolve_baseline_path_and_write_baseline::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.load_baseline::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.resolve_baseline_path::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.write_baseline
def test_resolve_baseline_path_and_write_baseline(tmp_path: Path) -> None:
    baseline_path = test_obsolescence_delta.resolve_baseline_path(tmp_path)
    assert baseline_path == tmp_path / test_obsolescence_delta.BASELINE_RELATIVE_PATH
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": test_obsolescence_delta.BASELINE_VERSION,
        "summary": {},
        "tests": [],
        "evidence_index": [],
        "opaque_evidence_count": 0,
        "generated_by_spec_id": "spec",
        "generated_by_spec": {},
    }
    test_obsolescence_delta.write_baseline(str(baseline_path), payload)
    loaded = test_obsolescence_delta.load_baseline(str(baseline_path))
    assert loaded.generated_by_spec_id == "spec"


# gabion:evidence E:call_footprint::tests/test_test_obsolescence_delta.py::test_render_markdown_includes_baseline_path_when_present::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.render_markdown
def test_render_markdown_includes_baseline_path_when_present() -> None:
    rendered = test_obsolescence_delta.render_markdown(
        {
            "summary": {"counts": {}},
            "tests": {},
            "evidence_keys": {},
            "baseline": {"path": "baselines/test_obsolescence_baseline.json"},
            "current": {},
            "generated_by_spec_id": "spec",
            "generated_by_spec": {},
        }
    )
    assert "- baseline: baselines/test_obsolescence_baseline.json" in rendered


# gabion:evidence E:call_footprint::tests/test_test_obsolescence_delta.py::test_build_baseline_payload_from_paths_calls_obsolescence_pipeline::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.build_baseline_payload_from_paths
def test_build_baseline_payload_from_paths_calls_obsolescence_pipeline() -> None:
    observed: dict[str, object] = {}
    expected_payload = {"version": 2}
    evidence_by_test = {"t": []}
    status_by_test = {"t": "mapped"}
    candidates = [{"test_id": "t", "class": "obsolete_candidate"}]
    summary = {"obsolete_candidate": 1}
    active_tests = ["a"]
    active_summary = {"active_total": 1, "pareto_frontier_total": 1}

    def _build(ev, st, cand, summ, **kwargs):
        observed["args"] = (ev, st, cand, summ, kwargs)
        return expected_payload

    payload = test_obsolescence_delta.build_baseline_payload_from_paths(
        "evidence.json",
        "risk.json",
        load_test_evidence_fn=lambda _path: (evidence_by_test, status_by_test),
        load_risk_registry_fn=lambda _path: {"E:x": object()},
        classify_candidates_fn=lambda ev, st, rr: test_obsolescence.ClassificationResult(
            stale_candidates=candidates,
            stale_summary=summary,
            active_tests=active_tests,
            active_summary=active_summary,
        ),
        build_baseline_payload_fn=_build,
    )
    assert payload == expected_payload
    assert observed["args"] == (
        evidence_by_test,
        status_by_test,
        candidates,
        summary,
        {"active_tests": active_tests, "active_summary": active_summary},
    )


# gabion:evidence E:call_footprint::tests/test_test_obsolescence_delta.py::test_build_evidence_index_skips_unmapped_and_keeps_existing_display_order::evidence_keys.py::gabion.analysis.evidence_keys.key_identity::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key::evidence_keys.py::gabion.analysis.evidence_keys.render_display::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta._build_evidence_index
def test_build_evidence_index_skips_unmapped_and_keeps_existing_display_order() -> None:
    key = evidence_keys.make_paramset_key(["x"])
    display = evidence_keys.render_display(key)
    mapped_ref = test_obsolescence.EvidenceRef(
        key=key,
        identity=evidence_keys.key_identity(key),
        display=display,
        opaque=False,
    )
    evidence_by_test = {
        "mapped": [mapped_ref],
        "unmapped": [mapped_ref],
    }
    status_by_test = {"mapped": "mapped", "unmapped": "unmapped"}
    rows = test_obsolescence_delta._build_evidence_index(evidence_by_test, status_by_test)
    assert len(rows) == 1
    assert rows[0]["display"] == display

    rich_ref = test_obsolescence.EvidenceRef(
        key=key,
        identity=evidence_keys.key_identity(key),
        display=display,
        opaque=False,
    )
    less_preferred = test_obsolescence.EvidenceRef(
        key=key,
        identity=evidence_keys.key_identity(key),
        display=display + "zzz",
        opaque=False,
    )
    rows = test_obsolescence_delta._build_evidence_index(
        {"t1": [rich_ref], "t2": [less_preferred]},
        {"t1": "mapped", "t2": "mapped"},
    )
    assert rows[0]["display"] == display


# gabion:evidence E:call_footprint::tests/test_test_obsolescence_delta.py::test_build_resolution_worklist_reports_dispositions_and_actions::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.build_resolution_worklist
def test_build_resolution_worklist_reports_dispositions_and_actions() -> None:
    baseline = test_obsolescence_delta.ObsolescenceBaseline(
        summary={
            "redundant_by_evidence": 1,
            "equivalent_witness": 1,
            "obsolete_candidate": 1,
            "unmapped": 1,
        },
        tests={
            "t_unmapped": "unmapped",
            "t_redundant": "redundant_by_evidence",
            "t_equivalent": "equivalent_witness",
            "t_obsolete": "obsolete_candidate",
            "t_unknown": "custom",
        },
        active={},
        evidence_index={},
        opaque_evidence_count=0,
        generated_by_spec_id="base",
        generated_by_spec={},
    )
    classification = test_obsolescence.ClassificationResult(
        stale_candidates=[
            {"test_id": "t_unmapped", "class": "unmapped"},
            {"test_id": "t_redundant", "class": "redundant_by_evidence"},
            {"test_id": "t_obsolete", "class": "obsolete_candidate"},
        ],
        stale_summary={
            "redundant_by_evidence": 1,
            "equivalent_witness": 0,
            "obsolete_candidate": 1,
            "unmapped": 1,
        },
        active_tests=["t_equivalent"],
        active_summary={},
    )
    payload = test_obsolescence_delta.build_resolution_worklist(baseline, classification)
    assert payload["summary"]["baseline_total"] == 5
    assert payload["summary"]["resolved_total"] == 2
    assert payload["summary"]["remaining_stale_total"] == 3
    rows = {row["test_id"]: row for row in payload["rows"]}
    assert rows["t_unmapped"]["proposed_action"] == "map_evidence"
    assert rows["t_unmapped"]["final_disposition"] == "stale_unmapped"
    assert rows["t_redundant"]["proposed_action"] == "merge_or_prune"
    assert rows["t_redundant"]["final_disposition"] == "stale_overlap"
    assert rows["t_equivalent"]["proposed_action"] == "pareto_keep_one"
    assert rows["t_equivalent"]["final_disposition"] == "retained_active"
    assert rows["t_obsolete"]["proposed_action"] == "resolve_or_remove"
    assert rows["t_obsolete"]["final_disposition"] == "stale_unresolved"
    assert rows["t_unknown"]["proposed_action"] == "review"
    assert rows["t_unknown"]["final_disposition"] == "removed_or_missing"


# gabion:evidence E:call_footprint::tests/test_test_obsolescence_delta.py::test_build_resolution_worklist_from_paths_end_to_end::test_obsolescence_delta.py::gabion.analysis.test_obsolescence_delta.build_resolution_worklist_from_paths
def test_build_resolution_worklist_from_paths_end_to_end(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "version": 2,
                "summary": {"unmapped": 1},
                "tests": [{"test_id": "t1", "class": "unmapped"}],
                "evidence_index": [],
                "opaque_evidence_count": 0,
                "generated_by_spec_id": "base",
                "generated_by_spec": {},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "test_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "scope": {"root": ".", "include": [], "exclude": []},
                "tests": [
                    {
                        "test_id": "t1",
                        "file": "tests/test_sample.py",
                        "line": 1,
                        "evidence": [],
                        "status": "unmapped",
                    }
                ],
                "evidence_index": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    risk_registry_path = tmp_path / "risk_registry.json"
    risk_registry_path.write_text(
        json.dumps(
            {"version": 1, "evidence": {}},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    payload = test_obsolescence_delta.build_resolution_worklist_from_paths(
        baseline_path=str(baseline_path),
        evidence_path=str(evidence_path),
        risk_registry_path=str(risk_registry_path),
    )
    assert payload["summary"]["baseline_total"] == 1
    assert payload["rows"] == [
        {
            "test_id": "t1",
            "baseline_class": "unmapped",
            "current_class": "unmapped",
            "proposed_action": "map_evidence",
            "final_disposition": "stale_unmapped",
        }
    ]
