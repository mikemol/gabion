from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gabion.analysis import evidence_keys, test_obsolescence


def _evidence_item(display: str) -> dict[str, object]:
    key = evidence_keys.make_opaque_key(display)
    return {"key": key, "display": display}


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.key_identity::stale_7b725bec393e
def test_basic_dominance(make_obsolescence_opaque_ref, obsolescence_summary_counts) -> None:
    evidence_by_test = {
        "tests/test_alpha.py::test_a": [make_obsolescence_opaque_ref("E:x")],
        "tests/test_beta.py::test_b": [
            make_obsolescence_opaque_ref("E:x"),
            make_obsolescence_opaque_ref("E:y"),
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

    result = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    candidates = result.stale_candidates
    summary = result.stale_summary
    by_id = {entry["test_id"]: entry for entry in candidates}
    assert by_id["tests/test_alpha.py::test_a"]["class"] == "redundant_by_evidence"
    assert "tests/test_beta.py::test_b" not in by_id
    assert summary == obsolescence_summary_counts(redundant_by_evidence=1)
    assert result.active_tests == ["tests/test_beta.py::test_b"]


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key::stale_0a7b59371188
def test_unmapped_classification(tmp_path: Path, write_test_evidence_payload) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = out_dir / "test_evidence.json"
    write_test_evidence_payload(
        evidence_path,
        entries=[
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
        include=[],
    )
    evidence_by_test, status_by_test = test_obsolescence.load_test_evidence(
        str(evidence_path)
    )
    result = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    candidates = result.stale_candidates
    summary = result.stale_summary
    by_id = {entry["test_id"]: entry for entry in candidates}
    assert by_id["tests/test_unmapped.py::test_skip"]["class"] == "unmapped"
    assert summary["unmapped"] == 1


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.key_identity::stale_b12b1958b067
def test_equivalent_witness_classification(
    make_obsolescence_opaque_ref,
    obsolescence_summary_counts,
) -> None:
    ref = make_obsolescence_opaque_ref("E:x")
    evidence_by_test = {
        "tests/test_alpha.py::test_a": [ref],
        "tests/test_beta.py::test_b": [ref],
    }
    status_by_test = {
        "tests/test_alpha.py::test_a": "mapped",
        "tests/test_beta.py::test_b": "mapped",
    }
    result = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    candidates = result.stale_candidates
    summary = result.stale_summary
    by_id = {entry["test_id"]: entry for entry in candidates}
    assert by_id["tests/test_beta.py::test_b"]["class"] == "equivalent_witness"
    assert "tests/test_alpha.py::test_a" in result.active_tests
    assert summary == obsolescence_summary_counts(equivalent_witness=1)


# gabion:evidence E:call_footprint::tests/test_test_obsolescence.py::test_equivalent_witness_pareto_tiebreak_prefers_branch_guard_then_runtime::test_obsolescence.py::gabion.analysis.test_obsolescence.classify_candidates
def test_equivalent_witness_pareto_tiebreak_prefers_branch_guard_then_runtime(
    make_obsolescence_opaque_ref,
) -> None:
    ref = make_obsolescence_opaque_ref("E:x")
    evidence_by_test = {
        "t_a": [ref],
        "t_b": [ref],
        "t_c": [ref],
    }
    status_by_test = {test_id: "mapped" for test_id in evidence_by_test}
    result = test_obsolescence.classify_candidates(
        evidence_by_test,
        status_by_test,
        {},
        options=test_obsolescence.ClassifierOptions(
            runtime_ms_by_test={"t_a": 5.0, "t_b": 10.0, "t_c": 1.0},
            branch_guard_by_test={"t_b": True, "t_c": True},
        ),
    )
    assert result.active_tests == ["t_c"]
    assert result.stale_summary["equivalent_witness"] == 2


# gabion:evidence E:call_footprint::tests/test_test_obsolescence.py::test_redundant_branch_guard_retains_active::test_obsolescence.py::gabion.analysis.test_obsolescence.classify_candidates
def test_redundant_branch_guard_retains_active(make_obsolescence_opaque_ref) -> None:
    evidence_by_test = {
        "t_redundant": [make_obsolescence_opaque_ref("E:x")],
        "t_dominator": [
            make_obsolescence_opaque_ref("E:x"),
            make_obsolescence_opaque_ref("E:y"),
        ],
    }
    status_by_test = {test_id: "mapped" for test_id in evidence_by_test}
    result = test_obsolescence.classify_candidates(
        evidence_by_test,
        status_by_test,
        {},
        options=test_obsolescence.ClassifierOptions(
            branch_guard_by_test={"t_redundant": True}
        ),
    )
    assert result.stale_candidates == []
    assert set(result.active_tests) == {"t_dominator", "t_redundant"}
    assert result.active_summary["branch_guard_retained"] == 1


# gabion:evidence E:call_footprint::tests/test_test_obsolescence.py::test_classify_candidates_marks_requested_unresolved_paths_as_obsolete::test_obsolescence.py::gabion.analysis.test_obsolescence.classify_candidates
def test_classify_candidates_marks_requested_unresolved_paths_as_obsolete(
    make_obsolescence_opaque_ref,
) -> None:
    shared_key = evidence_keys.make_paramset_key(["shared"])
    shared_identity = evidence_keys.key_identity(shared_key)
    redundant_ref = test_obsolescence.EvidenceRef(
        key=shared_key,
        identity=shared_identity,
        display="E:high-unique",
        opaque=False,
    )
    dominator_shared_ref = test_obsolescence.EvidenceRef(
        key=shared_key,
        identity=shared_identity,
        display="E:shared",
        opaque=False,
    )
    eq_ref = make_obsolescence_opaque_ref("E:eq")
    evidence_by_test = {
        "t_unmapped": [make_obsolescence_opaque_ref("E:unmapped")],
        "t_redundant": [redundant_ref],
        "t_dominator": [dominator_shared_ref, make_obsolescence_opaque_ref("E:dom-extra")],
        "t_eq_a": [eq_ref],
        "t_eq_b": [eq_ref],
        "t_unique": [make_obsolescence_opaque_ref("E:unique")],
    }
    status_by_test = {
        "t_unmapped": "unmapped",
        "t_redundant": "mapped",
        "t_dominator": "mapped",
        "t_eq_a": "mapped",
        "t_eq_b": "mapped",
        "t_unique": "mapped",
    }
    result = test_obsolescence.classify_candidates(
        evidence_by_test,
        status_by_test,
        {"E:high-unique": test_obsolescence.RiskInfo(risk="high", owner="", rationale="")},
        options=test_obsolescence.ClassifierOptions(
            unresolved_test_ids=frozenset(
                {"t_unmapped", "t_redundant", "t_eq_b", "t_unique"}
            )
        ),
    )
    by_id = {str(entry["test_id"]): entry for entry in result.stale_candidates}
    assert by_id["t_unmapped"]["class"] == "obsolete_candidate"
    assert by_id["t_unmapped"]["reason"]["stale_from"] == "unmapped"
    assert by_id["t_unmapped"]["reason"]["opaque_evidence"] == ["E:unmapped"]

    assert by_id["t_redundant"]["class"] == "obsolete_candidate"
    assert by_id["t_redundant"]["reason"]["stale_from"] == "redundant_by_evidence"
    assert by_id["t_redundant"]["reason"]["guardrail"] == "high-risk-last-witness"
    assert by_id["t_redundant"]["reason"]["guardrail_evidence"] == ["E:high-unique"]
    assert "opaque_evidence" not in by_id["t_redundant"]["reason"]

    assert by_id["t_eq_b"]["class"] == "obsolete_candidate"
    assert by_id["t_eq_b"]["reason"]["stale_from"] == "equivalent_witness"
    assert by_id["t_eq_b"]["reason"]["pareto_winner"] == "t_eq_a"

    assert by_id["t_unique"]["class"] == "obsolete_candidate"
    assert by_id["t_unique"]["reason"]["stale_from"] == "active_candidate"
    assert result.stale_summary["obsolete_candidate"] == 4
    assert result.active_summary["unresolved_obsolete"] == 4
    assert result.active_tests == ["t_dominator", "t_eq_a"]


# gabion:evidence E:call_footprint::tests/test_test_obsolescence.py::test_equivalent_classification_selects_single_active_peer::test_obsolescence.py::gabion.analysis.test_obsolescence.classify_candidates
def test_equivalent_classification_selects_single_active_peer(
    make_obsolescence_opaque_ref,
) -> None:
    ref = make_obsolescence_opaque_ref("E:pair")
    evidence_by_test = {"t1": [ref], "t2": [ref]}
    status_by_test = {"t1": "mapped", "t2": "mapped"}
    result = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    assert len(result.active_tests) == 1
    assert set(result.active_tests) <= {"t1", "t2"}
    assert result.stale_summary["equivalent_witness"] == 1


# gabion:evidence E:call_footprint::tests/test_test_obsolescence.py::test_pareto_helpers_cover_runtime_and_objective_fallback_paths::test_obsolescence.py::gabion.analysis.test_obsolescence._pareto_sort_key
def test_pareto_helpers_cover_runtime_and_objective_fallback_paths() -> None:
    options = test_obsolescence.ClassifierOptions(
        runtime_ms_by_test={"bad": "oops", "neg": -1},  # type: ignore[arg-type]
        objective_order=("unknown_metric",),  # type: ignore[arg-type]
        lexical_test_id_ascending=False,
    )
    assert math.isinf(test_obsolescence._runtime_ms("missing", options=options))
    assert math.isinf(test_obsolescence._runtime_ms("bad", options=options))
    assert math.isinf(test_obsolescence._runtime_ms("neg", options=options))
    assert test_obsolescence._pareto_winner([], options=options) == ""
    key = test_obsolescence._pareto_sort_key("ab", novelty=0, options=options)
    assert key[0] == 1
    assert math.isinf(key[1])
    assert key[2] == (-97, -98)


# gabion:evidence E:function_site::tests/test_test_obsolescence.py::test_branch_guard_and_runtime_defaults_cover_missing_option_maps
def test_branch_guard_and_runtime_defaults_cover_missing_option_maps() -> None:
    options = test_obsolescence.ClassifierOptions(
        branch_guard_by_test=None,
        runtime_ms_by_test=None,
        default_keep_for_branch_guard=True,
    )
    assert test_obsolescence._is_branch_guarded("missing", options=options) is True
    assert math.isinf(test_obsolescence._runtime_ms("missing", options=options))

    options_with_map = test_obsolescence.ClassifierOptions(
        branch_guard_by_test={},
        runtime_ms_by_test={},
        default_keep_for_branch_guard=True,
    )
    assert test_obsolescence._is_branch_guarded("missing", options=options_with_map) is True


# gabion:evidence E:function_site::tests/test_test_obsolescence.py::test_equivalent_classification_handles_empty_pareto_winner
def test_equivalent_classification_handles_empty_pareto_winner(
    make_obsolescence_opaque_ref,
) -> None:
    ref = make_obsolescence_opaque_ref("E:x")
    evidence_by_test = {
        "tests/test_alpha.py::test_a": [ref],
        "tests/test_beta.py::test_b": [ref],
    }
    status_by_test = {
        "tests/test_alpha.py::test_a": "mapped",
        "tests/test_beta.py::test_b": "mapped",
    }
    original = test_obsolescence._pareto_winner
    try:
        test_obsolescence._pareto_winner = (
            lambda peers, *, options: ""
        )
        result = test_obsolescence.classify_candidates(
            evidence_by_test,
            status_by_test,
            {},
        )
    finally:
        test_obsolescence._pareto_winner = original
    stale_by_id = {entry["test_id"]: entry for entry in result.stale_candidates}
    reason = stale_by_id["tests/test_beta.py::test_b"]["reason"]
    assert "pareto_winner" not in reason


# gabion:evidence E:call_footprint::tests/test_test_obsolescence.py::test_unresolved_active_candidate_without_opaque_evidence_reason::test_obsolescence.py::gabion.analysis.test_obsolescence.classify_candidates
def test_unresolved_active_candidate_without_opaque_evidence_reason() -> None:
    key = evidence_keys.make_paramset_key(["plain"])
    identity = evidence_keys.key_identity(key)
    ref = test_obsolescence.EvidenceRef(
        key=key,
        identity=identity,
        display=evidence_keys.render_display(key),
        opaque=False,
    )
    result = test_obsolescence.classify_candidates(
        {"t_plain": [ref]},
        {"t_plain": "mapped"},
        {},
        options=test_obsolescence.ClassifierOptions(
            unresolved_test_ids=frozenset({"t_plain"})
        ),
    )
    assert result.stale_summary["obsolete_candidate"] == 1
    assert "opaque_evidence" not in result.stale_candidates[0]["reason"]


# gabion:evidence E:function_site::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_lines E:decision_surface/direct::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_lines::stale_26799f525b41
def test_render_markdown_includes_spec_metadata(obsolescence_summary_counts) -> None:
    candidates = [
        {
            "test_id": "tests/test_alpha.py::test_a",
            "class": "unmapped",
            "dominators": [],
            "reason": {"status": "unmapped"},
        }
    ]
    summary = obsolescence_summary_counts(unmapped=1)
    report = test_obsolescence.render_markdown(candidates, summary)
    assert "generated_by_spec_id:" in report
    assert "generated_by_spec:" in report


# gabion:evidence E:function_site::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_payload E:decision_surface/direct::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_payload::stale_e830da863aed
def test_render_json_payload_includes_spec_metadata(obsolescence_summary_counts) -> None:
    candidates = []
    summary = obsolescence_summary_counts()
    payload = test_obsolescence.render_json_payload(candidates, summary)
    assert payload["version"] == 3
    assert "generated_by_spec_id" in payload
    assert "generated_by_spec" in payload


# gabion:evidence E:function_site::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_lines E:decision_surface/direct::projection_registry.py::gabion.analysis.projection_registry.spec_metadata_lines::stale_7626b84a7eed
def test_render_markdown_includes_suffix_details(obsolescence_summary_counts) -> None:
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
    summary = obsolescence_summary_counts(redundant_by_evidence=1)
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


# gabion:evidence E:function_site::test_obsolescence.py::gabion.analysis.test_obsolescence.load_test_evidence E:decision_surface/direct::test_obsolescence.py::gabion.analysis.test_obsolescence.load_test_evidence::stale_129fbdcd5daa_974b980e
def test_load_test_evidence_errors(tmp_path: Path) -> None:
    bad_payload = tmp_path / "bad_payload.json"
    bad_payload.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        test_obsolescence.load_test_evidence(str(bad_payload))

    bad_schema = tmp_path / "bad.json"
    bad_schema.write_text(json.dumps({"schema_version": 3, "tests": []}))
    with pytest.raises(ValueError):
        test_obsolescence.load_test_evidence(str(bad_schema))

    bad_tests = tmp_path / "bad_tests.json"
    bad_tests.write_text(json.dumps({"schema_version": 2, "tests": "nope"}))
    with pytest.raises(ValueError):
        test_obsolescence.load_test_evidence(str(bad_tests))


# gabion:evidence E:function_site::test_obsolescence.py::gabion.analysis.test_obsolescence.load_test_evidence E:decision_surface/direct::test_obsolescence.py::gabion.analysis.test_obsolescence.load_test_evidence::stale_6427b32d068f
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


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.render_display E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.key_identity::stale_8ca24934dfb5
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
    result = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, risk_registry
    )
    assert result.stale_summary["redundant_by_evidence"] == 0
    assert result.stale_candidates == []
    assert result.active_summary["high_risk_guardrail_retained"] == 1


# gabion:evidence E:function_site::test_obsolescence.py::gabion.analysis.test_obsolescence._summarize_candidates
def test_summarize_candidates_handles_bad_counts() -> None:
    def apply(_spec, _relation):
        return [{"class": "unmapped", "count": "bad"}]

    summary = test_obsolescence._summarize_candidates(
        [{"class": "unmapped"}], {"unmapped": 0}, apply=apply
    )
    assert summary["unmapped"] == 0


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.key_identity E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.key_identity::stale_3290faf8acae
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


# gabion:evidence E:call_footprint::tests/test_test_obsolescence.py::test_classify_candidates_handles_non_singleton_high_risk_witnesses::test_obsolescence.py::gabion.analysis.test_obsolescence.classify_candidates
def test_classify_candidates_handles_non_singleton_high_risk_witnesses(
    make_obsolescence_opaque_ref,
) -> None:
    ref = make_obsolescence_opaque_ref("E:high")
    evidence_by_test = {"t1": [ref], "t2": [ref]}
    status_by_test = {"t1": "mapped", "t2": "mapped"}
    risk_registry = {"E:high": test_obsolescence.RiskInfo(risk="high", owner="", rationale="")}
    result = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, risk_registry
    )
    assert result.stale_summary["equivalent_witness"] == 1
    assert all("guardrail" not in entry["reason"] for entry in result.stale_candidates)


# gabion:evidence E:call_footprint::tests/test_test_obsolescence.py::test_render_markdown_handles_empty_guardrail_and_suffix_parts::test_obsolescence.py::gabion.analysis.test_obsolescence.render_markdown
def test_render_markdown_handles_empty_guardrail_and_suffix_parts(
    obsolescence_summary_counts,
) -> None:
    report = test_obsolescence.render_markdown(
        [
            {
                "test_id": "t1",
                "class": "obsolete_candidate",
                "dominators": [],
                "reason": {"guardrail": "high-risk-last-witness", "guardrail_evidence": []},
            }
        ],
        obsolescence_summary_counts(obsolete_candidate=1),
    )
    assert "guardrail: high-risk-last-witness" in report


# gabion:evidence E:call_footprint::tests/test_test_obsolescence.py::test_summarize_candidates_ignores_unknown_rows_and_bad_input_types::test_obsolescence.py::gabion.analysis.test_obsolescence._summarize_candidates
def test_summarize_candidates_ignores_unknown_rows_and_bad_input_types() -> None:
    def apply(_spec, _relation):
        return [{"class": "custom", "count": 4}, {"class": "obsolete_candidate", "count": "bad"}]

    summary = test_obsolescence._summarize_candidates(
        [{"class": "obsolete_candidate"}],
        {"obsolete_candidate": 0},
        apply=apply,
    )
    assert summary["obsolete_candidate"] == 0


# gabion:evidence E:call_footprint::tests/test_test_obsolescence.py::test_normalize_evidence_refs_ignores_non_iterable_and_non_supported_entries::test_obsolescence.py::gabion.analysis.test_obsolescence._normalize_evidence_refs
def test_normalize_evidence_refs_ignores_non_iterable_and_non_supported_entries() -> None:
    assert test_obsolescence._normalize_evidence_refs(123) == []
    refs = test_obsolescence._normalize_evidence_refs([123, "E:opaque"])
    assert len(refs) == 1


# gabion:evidence E:call_footprint::tests/test_test_obsolescence.py::test_render_markdown_emits_entry_without_suffix_when_reason_is_empty::test_obsolescence.py::gabion.analysis.test_obsolescence.render_markdown
def test_render_markdown_emits_entry_without_suffix_when_reason_is_empty(
    obsolescence_summary_counts,
) -> None:
    report = test_obsolescence.render_markdown(
        [{"test_id": "t1", "class": "obsolete_candidate", "dominators": [], "reason": {}}],
        obsolescence_summary_counts(obsolete_candidate=1),
    )
    assert "- `t1`" in report
