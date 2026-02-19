from __future__ import annotations

from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import dataflow_audit as da

    return da

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._glossary_match_strata::matches
def test_glossary_match_strata_classification() -> None:
    da = _load()
    assert da._glossary_match_strata(None) == "none"
    assert da._glossary_match_strata([]) == "none"
    assert da._glossary_match_strata(["x"]) == "exact"
    assert da._glossary_match_strata(["x", "y"]) == "ambiguous"

# gabion:evidence E:function_site::test_rewrite_plan_verification.py::tests.test_rewrite_plan_verification._load
def test_normalize_bundle_key_covers_edges() -> None:
    da = _load()
    assert da.normalize_bundle_key("not-a-list") == ""
    assert da.normalize_bundle_key([]) == ""
    assert da.normalize_bundle_key(["a", 1, None]) == "a"
    assert da.normalize_bundle_key(["a", "b"]) == "a,b"

def _plan(**overrides: object) -> dict[str, object]:
    plan: dict[str, object] = {
        "plan_id": "p1",
        "site": {"path": "a.py", "function": "f", "bundle": ["a"]},
        "pre": {
            "base_keys": ["int"],
            "ctor_keys": [],
            "remainder": {"base": 1, "ctor": 1},
        },
        "rewrite": {"kind": "BUNDLE_ALIGN", "parameters": {"candidates": ["ctx_a", "ctx_b"]}},
        "evidence": {"provenance_id": "prov:a.py:f:a", "coherence_id": "coh:a.py:f:a"},
        "verification": {
            "predicates": [
                {"kind": "base_conservation", "expect": True},
                {"kind": "ctor_coherence", "expect": True},
                {"kind": "match_strata", "expect": "exact", "candidates": ["ctx_a", "ctx_b"]},
                {"kind": "remainder_non_regression", "expect": "no-new-remainder"},
            ]
        },
        "post_expectation": {"match_strata": "exact"},
    }
    plan.update(overrides)
    return plan

def _post_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": "a.py",
        "function": "f",
        "bundle": ["a"],
        "base_keys": ["int"],
        "ctor_keys": [],
        "remainder": {"base": 1, "ctor": 1},
        "glossary_matches": ["ctx_a"],
    }
    entry.update(overrides)
    return entry

# gabion:evidence E:decision_surface/direct::evidence.py::gabion.analysis.evidence.Site.from_payload::payload
def test_find_provenance_entry_for_site_covers_misses_and_hit() -> None:
    da = _load()
    provenance = [
        "nope",
        _post_entry(path="other.py"),
        _post_entry(function="other"),
        _post_entry(bundle="not-a-list"),
        _post_entry(),
    ]
    site = da.Site(path="a.py", function="f", bundle=("a",))
    assert (
        da._find_provenance_entry_for_site(
            provenance,
            site=site,
        )
        == provenance[-1]
    )
    missing_site = da.Site(path="missing.py", function="f", bundle=("a",))
    assert (
        da._find_provenance_entry_for_site(
            provenance,
            site=missing_site,
        )
        is None
    )

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._glossary_match_strata::matches E:decision_surface/direct::evidence.py::gabion.analysis.evidence.Site.from_payload::payload E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::post_exception_obligations
def test_verify_rewrite_plan_missing_post_entry() -> None:
    da = _load()
    result = da.verify_rewrite_plan(_plan(), post_provenance=[])
    assert result["accepted"] is False
    assert "missing post provenance entry for site" in result["issues"]

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._glossary_match_strata::matches E:decision_surface/direct::evidence.py::gabion.analysis.evidence.Site.from_payload::payload E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::post_exception_obligations
def test_verify_rewrite_plan_accepts_happy_path_and_list_helper() -> None:
    da = _load()
    plan = _plan()
    post = [_post_entry()]
    result = da.verify_rewrite_plan(plan, post_provenance=post)
    assert result["accepted"] is True

    results = da.verify_rewrite_plans([plan], post_provenance=post)
    assert results[0]["accepted"] is True

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._glossary_match_strata::matches E:decision_surface/direct::evidence.py::gabion.analysis.evidence.Site.from_payload::payload E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::post_exception_obligations
def test_verify_rewrite_plan_detects_candidate_mismatch() -> None:
    da = _load()
    plan = _plan()
    post = [_post_entry(glossary_matches=["not-a-candidate"])]
    result = da.verify_rewrite_plan(plan, post_provenance=post)
    assert result["accepted"] is False

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._glossary_match_strata::matches E:decision_surface/direct::evidence.py::gabion.analysis.evidence.Site.from_payload::payload E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::post_exception_obligations
def test_verify_rewrite_plan_detects_remainder_regression() -> None:
    da = _load()
    plan = _plan()
    post = [_post_entry(remainder={"base": 97, "ctor": 1})]
    result = da.verify_rewrite_plan(plan, post_provenance=post)
    assert result["accepted"] is False
    assert any(r.get("kind") == "remainder_non_regression" and not r.get("passed") for r in result["predicate_results"])

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._glossary_match_strata::matches E:decision_surface/direct::evidence.py::gabion.analysis.evidence.Site.from_payload::payload E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::post_exception_obligations
def test_verify_rewrite_plan_handles_non_dict_fields() -> None:
    da = _load()
    plan = _plan(
        pre="not-a-dict",
        rewrite="nope",
        post_expectation="nope",
        site={"path": "a.py", "function": "f", "bundle": "a"},
    )
    post = [_post_entry()]
    result = da.verify_rewrite_plan(plan, post_provenance=post)
    assert result["accepted"] is False

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._glossary_match_strata::matches E:decision_surface/direct::evidence.py::gabion.analysis.evidence.Site.from_payload::payload E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::post_exception_obligations
def test_verify_rewrite_plan_rejects_missing_or_invalid_site() -> None:
    da = _load()
    result = da.verify_rewrite_plan(_plan(site="nope"), post_provenance=[_post_entry()])
    assert result["accepted"] is False
    assert "missing or invalid plan site" in result["issues"]

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._glossary_match_strata::matches E:decision_surface/direct::evidence.py::gabion.analysis.evidence.Site.from_payload::payload E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::post_exception_obligations
def test_verify_rewrite_plan_handles_non_dict_remainder_and_params() -> None:
    da = _load()
    plan = _plan(
        pre={"base_keys": ["int"], "ctor_keys": [], "remainder": "oops"},
        rewrite={"parameters": "oops"},
    )
    post = [_post_entry(remainder="oops")]
    result = da.verify_rewrite_plan(plan, post_provenance=post)
    assert result["accepted"] is False

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._glossary_match_strata::matches E:decision_surface/direct::evidence.py::gabion.analysis.evidence.Site.from_payload::payload E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::post_exception_obligations
def test_verify_rewrite_plan_enforces_exception_obligation_non_regression_when_requested() -> None:
    da = _load()
    plan = _plan()
    plan["pre"] = {
        "base_keys": ["int"],
        "ctor_keys": [],
        "remainder": {"base": 1, "ctor": 1},
        "exception_obligations_summary": {"UNKNOWN": 1, "DEAD": 0, "HANDLED": 0, "total": 1},
    }
    plan["verification"] = {
        "predicates": [
            {"kind": "base_conservation", "expect": True},
            {"kind": "ctor_coherence", "expect": True},
            {"kind": "match_strata", "expect": "exact", "candidates": ["ctx_a", "ctx_b"]},
            {"kind": "remainder_non_regression", "expect": "no-new-remainder"},
            {"kind": "exception_obligation_non_regression", "expect": "XV1"},
        ]
    }
    post = [_post_entry()]

    obligations = [
        {
            "exception_path_id": "e1",
            "site": {"path": "a.py", "function": "f", "bundle": ["a"]},
            "source_kind": "E0",
            "status": "UNKNOWN",
            "witness_ref": None,
            "remainder": {},
            "environment_ref": None,
        }
    ]
    result = da.verify_rewrite_plan(
        plan,
        post_provenance=post,
        post_exception_obligations=obligations,
    )
    assert result["accepted"] is True

    result = da.verify_rewrite_plan(
        plan,
        post_provenance=post,
        post_exception_obligations=obligations + [dict(obligations[0], exception_path_id="e2")],
    )
    assert result["accepted"] is False
    assert any(
        r.get("kind") == "exception_obligation_non_regression" and not r.get("passed")
        for r in result["predicate_results"]
    )

    discharged_plan = _plan()
    discharged_plan["pre"] = {
        "base_keys": ["int"],
        "ctor_keys": [],
        "remainder": {"base": 1, "ctor": 1},
        "exception_obligations_summary": {"UNKNOWN": 0, "DEAD": 1, "HANDLED": 0, "total": 1},
    }
    discharged_plan["verification"] = plan["verification"]
    discharged = [
        dict(obligations[0], status="DEAD", witness_ref="deadness:e1")
    ]
    regression = [
        dict(obligations[0], status="UNKNOWN", exception_path_id="e3")
    ]
    result = da.verify_rewrite_plan(
        discharged_plan,
        post_provenance=post,
        post_exception_obligations=regression,
    )
    assert result["accepted"] is False

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._glossary_match_strata::matches E:decision_surface/direct::evidence.py::gabion.analysis.evidence.Site.from_payload::payload E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::post_exception_obligations
def test_verify_rewrite_plan_exception_predicate_missing_inputs_and_parse_errors() -> None:
    da = _load()
    plan = _plan()
    plan["pre"] = {
        "base_keys": ["int"],
        "ctor_keys": [],
        "remainder": {"base": 1, "ctor": 1},
        "exception_obligations_summary": {"UNKNOWN": "oops", "DEAD": 0, "HANDLED": 0, "total": 1},
    }
    plan["verification"] = {
        "predicates": [
            {"kind": "exception_obligation_non_regression", "expect": "XV1"},
            {"kind": "not-a-real-predicate", "expect": True},
        ]
    }
    post = [_post_entry()]

    missing_post = da.verify_rewrite_plan(plan, post_provenance=post)
    assert missing_post["accepted"] is False
    assert any(r.get("issue") == "missing post exception obligations" for r in missing_post["predicate_results"])
    assert any(r.get("issue") == "unknown predicate kind" for r in missing_post["predicate_results"])

    bad_pre = _plan(
        pre={"base_keys": ["int"], "ctor_keys": [], "remainder": {"base": 1, "ctor": 1}, "exception_obligations_summary": "oops"},
        verification={
            "predicates": [
                {"kind": "base_conservation", "expect": True},
                {"kind": "ctor_coherence", "expect": True},
                {"kind": "match_strata", "expect": "exact", "candidates": ["ctx_a", "ctx_b"]},
                {"kind": "remainder_non_regression", "expect": "no-new-remainder"},
                {"kind": "exception_obligation_non_regression"},
            ]
        },
    )
    result = da.verify_rewrite_plan(
        bad_pre,
        post_provenance=post,
        post_exception_obligations=[],
    )
    assert result["accepted"] is False
    assert any(r.get("issue") == "missing pre exception obligations summary" for r in result["predicate_results"])

    parse_plan = _plan(
        pre={
            "base_keys": ["int"],
            "ctor_keys": [],
            "remainder": {"base": 1, "ctor": 1},
            "exception_obligations_summary": {"UNKNOWN": "oops", "DEAD": 0, "HANDLED": 0, "total": 1},
        },
        verification={"predicates": [{"kind": "exception_obligation_non_regression"}]},
    )
    parsed = da.verify_rewrite_plan(
        parse_plan,
        post_provenance=post,
        post_exception_obligations=[],
    )
    assert parsed["accepted"] is True
    exc = next(r for r in parsed["predicate_results"] if r.get("kind") == "exception_obligation_non_regression")
    assert exc.get("expected") == {"UNKNOWN": 0, "DISCHARGED": 0}

# gabion:evidence E:call_footprint::tests/test_rewrite_plan_verification.py::test_verify_rewrite_plan_extended_kinds_have_deterministic_behavior::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_coherence::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::test_rewrite_plan_verification.py::tests.test_rewrite_plan_verification._load
def test_verify_rewrite_plan_extended_kinds_have_deterministic_behavior() -> None:
    da = _load()
    provenance = [
        {
            "path": "a.py",
            "function": "f",
            "bundle": ["a"],
            "provenance_id": "prov:a.py:f:a",
            "base_keys": ["int"],
            "ctor_keys": ["CtorA"],
            "remainder": {"base": 1, "ctor": 1},
            "glossary_matches": ["ctx_a", "ctx_b"],
        }
    ]
    coherence = da._compute_fingerprint_coherence(provenance, synth_version="synth@1")
    plans = da._compute_fingerprint_rewrite_plans(
        provenance,
        coherence,
        synth_version="synth@1",
    )
    by_kind = {plan["rewrite"]["kind"]: plan for plan in plans}

    post_good = [
        {
            "path": "a.py",
            "function": "f",
            "bundle": ["a"],
            "base_keys": ["int"],
            "ctor_keys": ["CtorA"],
            "remainder": {"base": 1, "ctor": 1},
            "glossary_matches": ["ctx_a"],
        }
    ]

    for kind in ("CTOR_NORMALIZE", "SURFACE_CANONICALIZE", "AMBIENT_REWRITE"):
        result = da.verify_rewrite_plan(by_kind[kind], post_provenance=post_good)
        assert result["accepted"] is True

    ctor_bad = [dict(post_good[0], ctor_keys=["CtorB"])]
    assert da.verify_rewrite_plan(by_kind["CTOR_NORMALIZE"], post_provenance=ctor_bad)["accepted"] is False

    mismatch = [dict(post_good[0], glossary_matches=["not-a-candidate"])]
    assert da.verify_rewrite_plan(by_kind["SURFACE_CANONICALIZE"], post_provenance=mismatch)["accepted"] is False
    assert da.verify_rewrite_plan(by_kind["AMBIENT_REWRITE"], post_provenance=mismatch)["accepted"] is False


# gabion:evidence E:decision_surface/direct::dataflow_decision_surfaces.py::gabion.analysis.dataflow_decision_surfaces.compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::post_exception_obligations
def test_rewrite_plan_precondition_abstentions_and_kind_predicates() -> None:
    da = _load()
    provenance = [
        {
            "path": "a.py",
            "function": "f",
            "bundle": ["a"],
            "provenance_id": "prov:a.py:f:a",
            "base_keys": ["int"],
            "ctor_keys": [],
            "remainder": {"base": 1, "ctor": 1},
            "glossary_matches": ["ctx_a", "ctx_b"],
        }
    ]
    coherence = da._compute_fingerprint_coherence(provenance, synth_version="synth@1")
    plans = da._compute_fingerprint_rewrite_plans(provenance, coherence, synth_version="synth@1")
    by_kind = {plan["rewrite"]["kind"]: plan for plan in plans}

    assert by_kind["CTOR_NORMALIZE"]["status"] == "ABSTAINED"
    assert by_kind["SURFACE_CANONICALIZE"]["status"] == "UNVERIFIED"
    assert by_kind["AMBIENT_REWRITE"]["status"] == "UNVERIFIED"

    abstained = da.verify_rewrite_plan(by_kind["CTOR_NORMALIZE"], post_provenance=provenance)
    assert abstained["accepted"] is False
    assert any("abstention reason" in issue for issue in abstained["issues"])


# gabion:evidence E:call_footprint::tests/test_rewrite_plan_verification.py::test_verify_rewrite_plan_rejects_missing_kind_payload_fields::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::test_rewrite_plan_verification.py::tests.test_rewrite_plan_verification._load::test_rewrite_plan_verification.py::tests.test_rewrite_plan_verification._plan::test_rewrite_plan_verification.py::tests.test_rewrite_plan_verification._post_entry
def test_verify_rewrite_plan_rejects_missing_kind_payload_fields() -> None:
    da = _load()
    plan = _plan(rewrite={"kind": "SURFACE_CANONICALIZE", "parameters": {"candidates": ["ctx_a"]}})
    result = da.verify_rewrite_plan(plan, post_provenance=[_post_entry()])
    assert result["accepted"] is False
    assert any("missing parameters" in issue for issue in result["issues"])
