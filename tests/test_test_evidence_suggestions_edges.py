from __future__ import annotations

import json
from pathlib import Path

import pytest

from gabion.analysis import evidence_keys, test_evidence_suggestions
from gabion.analysis.aspf import Alt, Forest, Node, NodeId
from gabion.analysis.dataflow_audit import AuditConfig, FunctionInfo


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _minimal_entry(test_id: str, file_path: str) -> test_evidence_suggestions.TestEvidenceEntry:
    return test_evidence_suggestions.TestEvidenceEntry(
        test_id=test_id,
        file=file_path,
        line=1,
        evidence=(),
        status="unmapped",
    )


def test_load_test_evidence_errors_and_defaults(tmp_path: Path) -> None:
    bad_schema = tmp_path / "bad.json"
    _write_json(bad_schema, {"schema_version": 3})
    with pytest.raises(ValueError):
        test_evidence_suggestions.load_test_evidence(str(bad_schema))

    bad_tests = tmp_path / "bad_tests.json"
    _write_json(bad_tests, {"schema_version": 2, "tests": "nope"})
    with pytest.raises(ValueError):
        test_evidence_suggestions.load_test_evidence(str(bad_tests))

    good = tmp_path / "good.json"
    _write_json(
        good,
        {
            "schema_version": 2,
            "tests": [
                "not-a-mapping",
                {"test_id": " ", "file": "skip.py", "line": 1, "evidence": []},
                {"test_id": "t1", "file": "a.py", "line": 1, "evidence": []},
                {"test_id": "t2", "file": "b.py", "line": 2, "evidence": ["E:x"]},
            ],
        },
    )
    entries = test_evidence_suggestions.load_test_evidence(str(good))
    assert [entry.status for entry in entries] == ["unmapped", "mapped"]


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.render_display
def test_render_markdown_and_payload() -> None:
    key = evidence_keys.make_paramset_key(["x"])
    suggestion = test_evidence_suggestions.Suggestion(
        test_id="tests/test_sample.py::test_alpha",
        file="tests/test_sample.py",
        line=1,
        suggested=(
            test_evidence_suggestions.EvidenceSuggestion(
                key=key,
                display=evidence_keys.render_display(key),
            ),
        ),
        matches=("rule",),
        source=test_evidence_suggestions.HEURISTIC_SOURCE,
    )
    summary = test_evidence_suggestions.SuggestionSummary(
        total=1,
        suggested=1,
        suggested_graph=0,
        suggested_heuristic=1,
        skipped_mapped=0,
        skipped_no_match=0,
        graph_unresolved=1,
        unmapped_modules=(),
        unmapped_prefixes=(),
    )
    markdown = test_evidence_suggestions.render_markdown([suggestion], summary)
    assert "source: heuristic" in markdown
    assert "matched: rule" in markdown
    payload = test_evidence_suggestions.render_json_payload([suggestion], summary)
    assert payload["summary"]["suggested"] == 1

    summary_with_unmapped = test_evidence_suggestions.SuggestionSummary(
        total=2,
        suggested=0,
        suggested_graph=0,
        suggested_heuristic=0,
        skipped_mapped=0,
        skipped_no_match=2,
        graph_unresolved=2,
        unmapped_modules=(("tests/test_sample.py", 2),),
        unmapped_prefixes=(("test_alpha", 2),),
    )
    empty_markdown = test_evidence_suggestions.render_markdown([], summary_with_unmapped)
    assert "tests/test_sample.py" in empty_markdown


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path
def test_graph_suggestions_paths_filtered(tmp_path: Path) -> None:
    forest = Forest()
    entry = _minimal_entry("tests/test_sample.py::test_alpha", "tests/test_sample.py")
    config = AuditConfig(project_root=tmp_path, exclude_dirs={tmp_path.name})
    suggestions, resolved = test_evidence_suggestions._graph_suggestions(
        [entry],
        root=tmp_path,
        paths=[tmp_path],
        forest=forest,
        config=config,
        max_depth=1,
    )
    assert suggestions == {}
    assert resolved == set()


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path
def test_graph_suggestions_cache_and_unresolved(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("def helper(x):\n    return x\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_sample.py"
    test_file.write_text(
        "from app import helper\n"
        "\n"
        "def helper_local():\n"
        "    return 1\n"
        "\n"
        "def test_alpha():\n"
        "    helper_local()\n"
        "    helper(1)\n"
        "    missing()\n",
        encoding="utf-8",
    )

    forest = Forest()
    missing_site = NodeId(kind="FunctionSite", key=())
    forest.nodes[missing_site] = Node(node_id=missing_site, meta={})

    entry = _minimal_entry("tests/test_sample.py::test_alpha", "tests/test_sample.py")
    missing = _minimal_entry("tests/test_missing.py::test_missing", "tests/test_missing.py")
    suggestions, resolved = test_evidence_suggestions._graph_suggestions(
        [entry, entry, missing],
        root=tmp_path,
        paths=[tmp_path],
        forest=forest,
        config=None,
        max_depth=2,
    )
    assert suggestions == {}
    assert entry.test_id in resolved
    assert missing.test_id not in resolved


def test_collect_reachable_skips_visited() -> None:
    path = Path("sample.py")
    info = FunctionInfo(
        name="f",
        qual="mod.f",
        path=path,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
    )

    def resolve_callees(_info: FunctionInfo):
        return [info]

    reachable = test_evidence_suggestions._collect_reachable(
        info, max_depth=2, resolve_callees=resolve_callees
    )
    assert reachable == []


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._alt_input::kind E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.make_never_sink_key::reason
def test_build_indices_and_helper_functions(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    info = FunctionInfo(
        name="f",
        qual="mod.f",
        path=path,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
    )
    index = test_evidence_suggestions._build_test_index({"mod.f": info}, None)
    assert list(index.keys())[0].endswith("sample.py::f")
    assert test_evidence_suggestions._rel_path(path, tmp_path / "other") == str(path)

    forest = Forest()
    bad_site = NodeId(kind="FunctionSite", key=())
    forest.nodes[bad_site] = Node(node_id=bad_site, meta={})
    site_index, evidence_by_site = test_evidence_suggestions._build_forest_evidence_index(forest)
    assert site_index == {}
    assert evidence_by_site == {}

    alt = Alt(kind="Unknown", inputs=())
    assert test_evidence_suggestions._evidence_for_alt(alt, forest) is None

    paramset_id = forest.add_paramset(["x"])
    site_id = forest.add_site("a.py", "f")
    forest.add_alt("SignatureBundle", (site_id, paramset_id))
    forest.add_alt("SignatureBundle", (paramset_id,))
    forest.add_alt("DecisionSurface", (site_id,))
    site_index, evidence_by_site = test_evidence_suggestions._build_forest_evidence_index(
        forest
    )
    assert site_index
    signature_alt = Alt(kind="SignatureBundle", inputs=(site_id, paramset_id))
    suggestion = test_evidence_suggestions._evidence_for_alt(signature_alt, forest)
    assert suggestion is not None

    node_id = NodeId(kind="FunctionSite", key=("a.py", "f"))
    forest.nodes[node_id] = Node(node_id=node_id, meta={})
    assert test_evidence_suggestions._site_parts(node_id, forest) == ("a.py", "f")

    assert test_evidence_suggestions._normalize_evidence_list(None) == []
    assert test_evidence_suggestions._normalize_evidence_list("E:x") == ["E:x"]
    assert test_evidence_suggestions._normalize_evidence_list({"bad": "data"}) == []
    assert test_evidence_suggestions._normalize_evidence_list(
        [{"display": "E:y"}, " "]
    ) == ["E:y"]

    entries = [
        _minimal_entry("tests/test_one.py::test_alpha", "tests/test_one.py"),
        _minimal_entry("tests/test_two.py::test_beta", "tests/test_two.py"),
    ]
    modules, prefixes = test_evidence_suggestions._summarize_unmapped(entries)
    assert modules[0][1] == 1
    assert prefixes[0][0].startswith("test_")
    assert test_evidence_suggestions._test_prefix("tests/test.py::test_") == "test"
    assert test_evidence_suggestions._test_prefix("tests/test.py::helper") == ""

    rule = test_evidence_suggestions._SuggestionRule(
        rule_id="rule",
        evidence=("E:x",),
        needles=("alpha",),
        scope="name",
        exclude=("skip",),
    )
    assert rule.matches(file="file", name="alpha") is True
    assert rule.matches(file="file", name="skip_alpha") is False


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._alt_input::kind E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.make_never_sink_key::reason
def test_evidence_for_alt_variants() -> None:
    forest = Forest()
    site_id = forest.add_site("a.py", "f")
    paramset_id = forest.add_paramset(["x"])
    empty_paramset = forest.add_paramset([])

    for kind in ("DecisionSurface", "ValueDecisionSurface", "NeverInvariantSink"):
        alt = Alt(kind=kind, inputs=(site_id, paramset_id))
        suggestion = test_evidence_suggestions._evidence_for_alt(alt, forest)
        assert suggestion is not None

    missing_site_alt = Alt(kind="DecisionSurface", inputs=(paramset_id,))
    assert test_evidence_suggestions._evidence_for_alt(missing_site_alt, forest) is None

    missing_meta_site = NodeId(kind="FunctionSite", key=())
    forest.nodes[missing_meta_site] = Node(node_id=missing_meta_site, meta={})
    bad_site_alt = Alt(kind="DecisionSurface", inputs=(missing_meta_site, paramset_id))
    assert test_evidence_suggestions._evidence_for_alt(bad_site_alt, forest) is None

    empty_param_alt = Alt(kind="DecisionSurface", inputs=(site_id, empty_paramset))
    assert test_evidence_suggestions._evidence_for_alt(empty_param_alt, forest) is None

    custom_alt = Alt(kind="CustomAlt", inputs=(site_id, paramset_id))
    assert (
        test_evidence_suggestions._evidence_for_alt(
            custom_alt, forest, prefix_map={"CustomAlt": "custom"}
        )
        is None
    )
