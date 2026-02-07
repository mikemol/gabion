from __future__ import annotations

from gabion.analysis import evidence_keys


# gabion:evidence E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.make_never_sink_key::reason
def test_evidence_keys_normalize_and_render() -> None:
    assert evidence_keys.normalize_params([" b", "a", "a", ""]) == ["a", "b"]
    assert evidence_keys.normalize_param_string("b, a ,") == "a,b"
    assert evidence_keys.normalize_reason("  a   b ") == "a b"

    paramset = evidence_keys.make_paramset_key(["z", "y", "z"])
    assert paramset["k"] == "paramset"

    decision = evidence_keys.make_decision_surface_key(
        mode=" direct ",
        path="p",
        qual="q",
        param="x, y",
    )
    assert decision["m"] == "direct"

    never = evidence_keys.make_never_sink_key(
        path="p",
        qual="q",
        param="x",
        reason="  why  ",
    )
    assert never["reason"] == "why"

    site = evidence_keys.make_function_site_key(path="p", qual="q")
    assert site["k"] == "function_site"

    footprint = evidence_keys.make_call_footprint_key(
        path="p",
        qual="q",
        targets=[("a.py", "mod.fn"), {"path": "b.py", "qual": "mod.fn2"}],
    )
    assert footprint["k"] == "call_footprint"
    ambiguity = evidence_keys.make_ambiguity_set_key(
        path="p",
        qual="q",
        span=[1, 2, 3, 4],
        candidates=[("a.py", "mod.fn"), {"path": "b.py", "qual": "mod.fn2"}],
    )
    assert ambiguity["k"] == "ambiguity_set"
    witness = evidence_keys.make_partition_witness_key(
        kind="local_resolution_ambiguous",
        site={"path": "p", "qual": "q"},
        ambiguity=ambiguity,
        support={"phase": "resolve"},
        collapse={"hint": "qualify"},
    )
    assert witness["k"] == "partition_witness"

    assert evidence_keys.normalize_key({"k": "paramset", "params": "b,a"})["params"] == [
        "a",
        "b",
    ]
    assert evidence_keys.normalize_key(
        {"k": "decision_surface", "site": "bad", "param": "x"}
    )["site"]["path"] == ""
    assert evidence_keys.normalize_key(
        {"k": "never_sink", "site": "bad", "param": "x"}
    )["site"]["path"] == ""
    assert evidence_keys.normalize_key({"k": "function_site", "site": "bad"})["site"][
        "path"
    ] == ""
    assert evidence_keys.normalize_key({"k": "opaque", "s": "X"})["s"] == "X"
    assert evidence_keys.normalize_key({"k": ""})["k"] == "opaque"
    assert evidence_keys.normalize_key({"k": "custom", "value": 1})["k"] == "custom"

    assert evidence_keys.key_identity({"k": "opaque", "s": "X"}).startswith("{")

    assert evidence_keys.render_display(paramset).startswith("E:paramset")
    assert evidence_keys.render_display(decision).startswith("E:decision_surface/")
    assert evidence_keys.render_display(never).startswith("E:never/sink")
    assert evidence_keys.render_display(site).startswith("E:function_site")
    assert evidence_keys.render_display(footprint).startswith("E:call_footprint")
    assert evidence_keys.render_display(ambiguity).startswith("E:ambiguity_set::")
    assert evidence_keys.render_display(witness).startswith("E:partition_witness::")
    assert evidence_keys.render_display({"k": "custom"}) == "E:custom"


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.render_display
def test_render_display_handles_non_list_params() -> None:
    def fake_normalize_key(_key):
        return {"k": "paramset", "params": "oops"}

    assert (
        evidence_keys.render_display(
            {"k": "paramset"}, normalize=fake_normalize_key
        )
        == "E:paramset"
    )


# gabion:evidence E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.make_never_sink_key::reason
def test_parse_display_variants() -> None:
    assert evidence_keys.parse_display("nope") is None
    assert evidence_keys.parse_display("E:") is None
    assert evidence_keys.parse_display("E:paramset") is None
    assert evidence_keys.parse_display("E:unknown::x") is None
    assert evidence_keys.parse_display("E:paramset::a,b") == {
        "k": "paramset",
        "params": ["a", "b"],
    }
    assert evidence_keys.parse_display("E:decision_surface/direct::p::q") is None
    assert evidence_keys.parse_display("E:decision_surface/direct::p::q::x") == {
        "k": "decision_surface",
        "m": "direct",
        "site": {"path": "p", "qual": "q"},
        "param": "x",
    }
    assert evidence_keys.parse_display("E:never/sink::p::q") is None
    assert evidence_keys.parse_display("E:never/sink::p::q::x")["k"] == "never_sink"
    assert evidence_keys.parse_display("E:function_site::p") is None
    assert evidence_keys.parse_display("E:function_site::p::q") == {
        "k": "function_site",
        "site": {"path": "p", "qual": "q"},
    }
    assert evidence_keys.parse_display("E:call_footprint::p") is None
    assert evidence_keys.parse_display("E:call_footprint::p::q::x") is None
    assert evidence_keys.parse_display("E:call_footprint::p::q") == {
        "k": "call_footprint",
        "site": {"path": "p", "qual": "q"},
        "targets": [],
    }
    assert evidence_keys.parse_display(
        "E:call_footprint::p::q::a.py::mod.fn::b.py::mod.fn2"
    ) == {
        "k": "call_footprint",
        "site": {"path": "p", "qual": "q"},
        "targets": [
            {"path": "a.py", "qual": "mod.fn"},
            {"path": "b.py", "qual": "mod.fn2"},
        ],
    }
    ambiguity = evidence_keys.make_ambiguity_set_key(
        path="p",
        qual="q",
        span=[1, 2, 3, 4],
        candidates=[("a.py", "mod.fn")],
    )
    ambiguity_display = evidence_keys.render_display(ambiguity)
    assert evidence_keys.parse_display(ambiguity_display) == evidence_keys.normalize_key(
        ambiguity
    )
    witness = evidence_keys.make_partition_witness_key(
        kind="local_resolution_ambiguous",
        site={"path": "p", "qual": "q"},
        ambiguity=ambiguity,
        support={"phase": "resolve"},
        collapse={"hint": "qualify"},
    )
    witness_display = evidence_keys.render_display(witness)
    assert evidence_keys.parse_display(witness_display) == evidence_keys.normalize_key(
        witness
    )


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.is_opaque
def test_is_opaque() -> None:
    assert evidence_keys.is_opaque({"k": "opaque", "s": "X"}) is True
    assert evidence_keys.is_opaque({"k": "paramset", "params": ["a"]}) is False


# gabion:evidence E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.make_never_sink_key::reason E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys._normalize_target::target
def test_call_footprint_normalization_edges() -> None:
    targets = [
        ("a",),
        "bad",
        {"path": "", "qual": "q"},
        {"path": "p", "qual": ""},
        {"path": "p", "qual": "q"},
    ]
    normalized = evidence_keys.normalize_targets(targets)
    assert normalized == [{"path": "p", "qual": "q"}]

    normalized_key = evidence_keys.normalize_key(
        {"k": "call_footprint", "site": "bad", "targets": "bad"}
    )
    assert normalized_key["site"]["path"] == ""
    assert normalized_key["targets"] == []

    display = evidence_keys.render_display(
        {
            "k": "call_footprint",
            "site": {"path": "tests/test.py", "qual": "test"},
            "targets": ["bad", {"path": "", "qual": "q"}, {"path": "p", "qual": "q"}],
        }
    )
    assert display == "E:call_footprint::tests/test.py::test::p::q"

    def fake_normalize(_key):
        return {
            "k": "call_footprint",
            "site": {"path": "tests/test.py", "qual": "test"},
            "targets": ["bad", {"path": "", "qual": "q"}, {"path": "p", "qual": "q"}],
        }


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys._normalize_span
def test_normalize_span_and_site_edges() -> None:
    assert evidence_keys._normalize_span("bad") is None
    assert evidence_keys._normalize_site(["only"]) == {"path": "", "qual": ""}
    assert evidence_keys._normalize_site("bad") == {"path": "", "qual": ""}


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.render_display
def test_render_display_call_footprint_skips_invalid_targets() -> None:
    def fake_normalize(_key):
        return {
            "k": "call_footprint",
            "site": {"path": "tests/test.py", "qual": "test"},
            "targets": ["bad", {"path": "", "qual": ""}, {"path": "p", "qual": "q"}],
        }

    display = evidence_keys.render_display({"k": "call_footprint"}, normalize=fake_normalize)
    assert display == "E:call_footprint::tests/test.py::test::p::q"


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.parse_display
def test_parse_display_ambiguity_edges() -> None:
    assert evidence_keys.parse_display("E:ambiguity_set") is None
    assert evidence_keys.parse_display("E:ambiguity_set::123") is None
    assert evidence_keys.parse_display("E:partition_witness") is None
    assert evidence_keys.parse_display("E:partition_witness::123") is None

# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.normalize_key
def test_ambiguity_span_normalization_edges() -> None:
    key = evidence_keys.normalize_key(
        {
            "k": "ambiguity_set",
            "site": {"path": "p", "qual": "q", "span": {"line": "x"}},
            "candidates": "bad",
        }
    )
    assert "span" not in key["site"]
    assert key["candidates"] == []

    bad_span_key = evidence_keys.make_ambiguity_set_key(
        path="p",
        qual="q",
        span=[1, 2, 3],
        candidates=[],
    )
    assert "span" not in bad_span_key["site"]

    negative_span = evidence_keys.normalize_key(
        {
            "k": "ambiguity_set",
            "site": {"path": "p", "qual": "q", "span": [1, -1, 2, 3]},
            "candidates": [],
        }
    )
    assert "span" not in negative_span["site"]

    short_site = evidence_keys.normalize_key(
        {"k": "ambiguity_set", "site": ["only"], "candidates": []}
    )
    assert short_site["site"]["path"] == ""

    assert evidence_keys.parse_display("E:ambiguity_set::") is None
    assert evidence_keys.parse_display("E:ambiguity_set::{") is None
    assert evidence_keys.parse_display("E:partition_witness::") is None
    assert evidence_keys.parse_display("E:partition_witness::{") is None
