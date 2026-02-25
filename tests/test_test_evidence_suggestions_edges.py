from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from gabion.analysis import evidence_keys, test_evidence_suggestions
from gabion.analysis.aspf import Alt, Forest, Node, NodeId
from gabion.analysis.dataflow_audit import AuditConfig, FunctionInfo, SymbolTable


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


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions.load_test_evidence E:function_site::test_test_evidence_suggestions_edges.py::tests.test_test_evidence_suggestions_edges._write_json
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


# gabion:evidence E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key E:function_site::evidence_keys.py::gabion.analysis.evidence_keys.render_display E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key::stale_e2f1950f5e6b
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


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::stale_c9d342c65475
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


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._graph_suggestions
def test_graph_suggestions_empty_entries_short_circuits(tmp_path: Path) -> None:
    suggestions, resolved = test_evidence_suggestions._graph_suggestions(
        [],
        root=tmp_path,
        paths=[tmp_path],
        forest=Forest(),
        config=None,
        max_depth=1,
    )
    assert suggestions == {}
    assert resolved == set()


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::stale_32b0c4fac550_22ff5a7a
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
    assert entry.test_id in suggestions
    assert suggestions[entry.test_id].source == "graph.call_footprint_fallback"
    assert entry.test_id in resolved
    assert missing.test_id not in resolved


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._collect_reachable
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
        function_span=(0, 0, 0, 1),
    )

    def resolve_callees(_info: FunctionInfo):
        return [info]

    reachable = test_evidence_suggestions._collect_reachable(
        info, max_depth=2, resolve_callees=resolve_callees
    )
    assert reachable == []


# gabion:evidence E:call_footprint::tests/test_test_evidence_suggestions_edges.py::test_test_qual_passthrough::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._test_qual
def test_test_qual_passthrough() -> None:
    assert test_evidence_suggestions._test_qual("no_delimiter") == "no_delimiter"


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._alt_input::kind E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.make_never_sink_key::reason E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._alt_input::stale_4cb8e03f700b_03483964
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
        function_span=(0, 0, 0, 1),
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


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._alt_input::kind E:decision_surface/direct::evidence_keys.py::gabion.analysis.evidence_keys.make_never_sink_key::reason E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._alt_input::stale_2e661a6266d4
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


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._collect_call_footprint_targets
def test_collect_call_footprint_targets_no_outer(tmp_path: Path) -> None:
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_sample.py"
    test_file.write_text("def test_alpha():\n    pass\n", encoding="utf-8")
    entry = _minimal_entry("tests/test_sample.py::test_alpha", "tests/test_sample.py")
    info = FunctionInfo(
        name="test_alpha",
        qual="tests.test_sample.test_alpha",
        path=test_file,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    targets = test_evidence_suggestions._collect_call_footprint_targets(
        info,
        entry=entry,
        direct_callees=(),
        node_cache={},
        module_cache={},
        symbol_table=SymbolTable(),
        by_name={},
        by_qual={},
        class_index=None,
        project_root=tmp_path,
    )
    assert targets == ()


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._find_module_level_calls E:decision_surface/direct::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._find_module_level_calls::stale_4b23852ce6db_be534f60
def test_find_module_level_calls_empty_and_missing(tmp_path: Path) -> None:
    info = FunctionInfo(
        name="test_alpha",
        qual="tests.test_sample.test_alpha",
        path=tmp_path / "tests" / "test_sample.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    empty_entry = _minimal_entry("tests/test_sample.py::test_alpha", "")
    assert (
        test_evidence_suggestions._find_module_level_calls(
            info,
            entry=empty_entry,
            node_cache={},
            module_cache={},
            symbol_table=SymbolTable(),
            by_name={},
            by_qual={},
            class_index=None,
            project_root=tmp_path,
        )
        == ()
    )
    missing_entry = _minimal_entry(
        "tests/missing.py::test_alpha", "tests/missing.py"
    )
    assert (
        test_evidence_suggestions._find_module_level_calls(
            info,
            entry=missing_entry,
            node_cache={},
            module_cache={},
            symbol_table=SymbolTable(),
            by_name={},
            by_qual={},
            class_index=None,
            project_root=tmp_path,
        )
        == ()
    )


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._find_module_level_calls E:decision_surface/direct::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._find_module_level_calls::stale_fc7abccd3cc6
def test_find_module_level_calls_node_missing(tmp_path: Path) -> None:
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_sample.py"
    test_file.write_text("def test_alpha():\n    pass\n", encoding="utf-8")
    entry = _minimal_entry("tests/test_sample.py::test_alpha", "tests/test_sample.py")
    info = FunctionInfo(
        name="test_beta",
        qual="tests.test_sample.test_beta",
        path=test_file,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    assert (
        test_evidence_suggestions._find_module_level_calls(
            info,
            entry=entry,
            node_cache={},
            module_cache={},
            symbol_table=SymbolTable(),
            by_name={},
            by_qual={},
            class_index=None,
            project_root=tmp_path,
        )
        == ()
    )


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._find_module_level_calls E:decision_surface/direct::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._find_module_level_calls::stale_94b65ff6ed2f
def test_find_module_level_calls_resolves_symbol_and_literal(tmp_path: Path) -> None:
    src_pkg = tmp_path / "src" / "pkg"
    src_pkg.mkdir(parents=True)
    (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    (src_pkg / "mod.py").write_text(
        "VALUE = 1\n"
        "def helper():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_sample.py"
    test_file.write_text(
        "from pkg import mod\n"
        "import runpy\n\n"
        "def test_alpha():\n"
        "    callable(mod.helper)\n"
        "    callable(mod.VALUE)\n"
        "    runpy.run_module('pkg.mod')\n",
        encoding="utf-8",
    )
    parse_failure_witnesses = []
    by_name, by_qual = test_evidence_suggestions._build_function_index(
        [test_file, src_pkg / "mod.py"],
        tmp_path,
        set(),
        "high",
        None,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    symbol_table = test_evidence_suggestions._build_symbol_table(
        [test_file, src_pkg / "mod.py"],
        tmp_path,
        external_filter=True,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    info = by_qual["tests.test_sample.test_alpha"]
    entry = _minimal_entry("tests/test_sample.py::test_alpha", "tests/test_sample.py")
    resolved = test_evidence_suggestions._find_module_level_calls(
        info,
        entry=entry,
        node_cache={},
        module_cache={},
        symbol_table=symbol_table,
        by_name=by_name,
        by_qual=by_qual,
        class_index=None,
        project_root=tmp_path,
    )
    assert ("mod.py", "pkg.mod.helper") in resolved
    assert ("mod.py", "pkg.mod.VALUE") in resolved
    assert ("mod.py", "pkg.mod") in resolved


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._iter_outer_calls
def test_iter_outer_calls_skips_nested() -> None:
    tree = ast.parse(
        "def test_alpha():\n"
        "    def inner():\n"
        "        helper()\n"
        "    helper()\n"
    )
    node = tree.body[0]
    calls = test_evidence_suggestions._iter_outer_calls(node)
    assert len(calls) == 1


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._call_symbol_refs
def test_call_symbol_and_literal_helpers() -> None:
    call = ast.Call(
        func=ast.Name(id="f", ctx=ast.Load()),
        args=[ast.Attribute(value=ast.Name(id="mod", ctx=ast.Load()), attr="x", ctx=ast.Load())],
        keywords=[ast.keyword(arg="kw", value=None)],
    )
    refs = test_evidence_suggestions._call_symbol_refs(call)
    assert refs == ["f", "mod.x"]
    assert test_evidence_suggestions._call_module_literals(call) == []
    call_kw = ast.Call(
        func=ast.Name(id="g", ctx=ast.Load()),
        args=[],
        keywords=[ast.keyword(arg="item", value=ast.Name(id="name", ctx=ast.Load()))],
    )
    assert test_evidence_suggestions._call_symbol_refs(call_kw) == ["g", "name"]
    literal_call = ast.Call(
        func=ast.Name(id="run_module", ctx=ast.Load()),
        args=[ast.Constant(value="pkg.mod")],
        keywords=[ast.keyword(arg="name", value=ast.Constant(value="pkg.core"))],
    )
    literals = test_evidence_suggestions._call_module_literals(literal_call)
    assert literals == ["pkg.mod", "pkg.core"]
    assert test_evidence_suggestions._expr_symbol_ref(ast.Name(id="name", ctx=ast.Load())) == "name"
    assert (
        test_evidence_suggestions._expr_symbol_ref(
            ast.Attribute(value=ast.Name(id="pkg", ctx=ast.Load()), attr="core", ctx=ast.Load())
        )
        == "pkg.core"
    )
    assert test_evidence_suggestions._expr_symbol_ref(ast.Constant(value=1)) is None
    bad_attr = ast.Attribute(value=ast.Call(func=ast.Name(id="f", ctx=ast.Load()), args=[]), attr="x")
    assert test_evidence_suggestions._attribute_chain(bad_attr) is None


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._resolve_symbol_target
def test_resolve_symbol_and_module_helpers(tmp_path: Path) -> None:
    src_pkg = tmp_path / "src" / "pkg"
    src_pkg.mkdir(parents=True)
    (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    module_path = src_pkg / "mod.py"
    module_path.write_text("", encoding="utf-8")

    table = SymbolTable()
    table.imports[("tests.sample", "Thing")] = "pkg.mod.Thing"
    table.internal_roots.add("pkg")
    module_cache: dict[str, Path | None] = {}
    assert (
        test_evidence_suggestions._resolve_symbol_target(
            "Thing", "tests.sample", table, module_cache, tmp_path
        )
        == ("mod.py", "pkg.mod.Thing")
    )
    assert (
        test_evidence_suggestions._resolve_symbol_target(
            "Thing.attr", "tests.sample", table, module_cache, tmp_path
        )
        == ("mod.py", "pkg.mod.Thing.attr")
    )
    assert (
        test_evidence_suggestions._resolve_symbol_target(
            "", "tests.sample", table, module_cache, tmp_path
        )
        is None
    )
    assert (
        test_evidence_suggestions._resolve_symbol_target(
            "Missing", "tests.sample", table, module_cache, tmp_path
        )
        is None
    )
    table.imports[("tests.sample", "Ghost")] = "ghost.mod.Ghost"
    table.internal_roots.add("ghost")
    assert (
        test_evidence_suggestions._resolve_symbol_target(
            "Ghost", "tests.sample", table, module_cache, tmp_path
        )
        is None
    )
    # cache hit + package init path
    assert (
        test_evidence_suggestions._resolve_module_file(
            "pkg.mod", tmp_path, module_cache
        )
        == module_path
    )
    assert (
        test_evidence_suggestions._resolve_module_file(
            "pkg.mod", tmp_path, module_cache
        )
        == module_path
    )
    init_path = src_pkg / "__init__.py"
    assert (
        test_evidence_suggestions._resolve_module_file("pkg", tmp_path, module_cache)
        == init_path
    )


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._resolve_module_literal
def test_resolve_module_literal_invalid(tmp_path: Path) -> None:
    module_cache: dict[str, Path | None] = {}
    assert (
        test_evidence_suggestions._resolve_module_literal(
            ".bad", tmp_path, module_cache
        )
        is None
    )
    assert (
        test_evidence_suggestions._resolve_module_literal(
            "pkg..mod", tmp_path, module_cache
        )
        is None
    )
    assert (
        test_evidence_suggestions._resolve_module_literal(
            "missing.mod", tmp_path, module_cache
        )
        is None
    )


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions.collect_call_footprints E:decision_surface/direct::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions.collect_call_footprints::stale_098851b92767
def test_collect_call_footprints_empty_entries() -> None:
    assert test_evidence_suggestions.collect_call_footprints([]) == {}


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions.collect_call_footprints E:decision_surface/direct::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions.collect_call_footprints::stale_538a74c2fcbd
def test_collect_call_footprints_empty_paths(tmp_path: Path) -> None:
    entry = _minimal_entry("tests/test_sample.py::test_alpha", "tests/test_sample.py")
    config = AuditConfig(project_root=tmp_path)
    footprints = test_evidence_suggestions.collect_call_footprints(
        [entry],
        root=tmp_path,
        paths=[tmp_path],
        config=config,
    )
    assert footprints == {}


# gabion:evidence E:function_site::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions.collect_call_footprints E:decision_surface/direct::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions.collect_call_footprints::stale_c18e05ce1806_bc86387e
def test_collect_call_footprints_cache_and_missing(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "app.py").write_text(
        "def helper(x):\n"
        "    return x\n",
        encoding="utf-8",
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        "from app import helper\n"
        "\n"
        "def test_alpha():\n"
        "    helper(1)\n"
        "    missing()\n",
        encoding="utf-8",
    )

    entry = _minimal_entry("tests/test_sample.py::test_alpha", "tests/test_sample.py")
    missing = _minimal_entry("tests/test_missing.py::test_missing", "tests/test_missing.py")
    config = AuditConfig(project_root=tmp_path)
    footprints = test_evidence_suggestions.collect_call_footprints(
        [entry, entry, missing],
        root=tmp_path,
        paths=[tmp_path],
        config=config,
    )
    assert entry.test_id in footprints


# gabion:evidence E:call_footprint::tests/test_test_evidence_suggestions_edges.py::test_suggest_evidence_without_heuristics_skips_no_match::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions.suggest_evidence::test_test_evidence_suggestions_edges.py::tests.test_test_evidence_suggestions_edges._minimal_entry
def test_suggest_evidence_without_heuristics_skips_no_match(tmp_path: Path) -> None:
    entry = _minimal_entry("tests/test_alpha.py::test_alpha", "tests/test_alpha.py")
    suggestions, summary = test_evidence_suggestions.suggest_evidence(
        [entry],
        root=tmp_path,
        forest=Forest(),
        include_heuristics=False,
        graph_suggestions_fn=lambda *args, **kwargs: ({}, set()),
        suggest_for_entry_fn=lambda _entry: ([], []),
    )
    assert suggestions == []
    assert summary.skipped_no_match == 1


# gabion:evidence E:call_footprint::tests/test_test_evidence_suggestions_edges.py::test_render_markdown_without_match_details::evidence_keys.py::gabion.analysis.evidence_keys.make_opaque_key::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions.render_markdown
def test_render_markdown_without_match_details() -> None:
    key = evidence_keys.make_opaque_key("opaque")
    suggestion = test_evidence_suggestions.Suggestion(
        test_id="tests/test_alpha.py::test_alpha",
        file="tests/test_alpha.py",
        line=1,
        suggested=(test_evidence_suggestions.EvidenceSuggestion(key=key, display="opaque"),),
        matches=(),
        source="graph",
    )
    summary = test_evidence_suggestions.SuggestionSummary(
        total=1,
        suggested=1,
        suggested_graph=1,
        suggested_heuristic=0,
        skipped_mapped=0,
        skipped_no_match=0,
        graph_unresolved=0,
        unmapped_modules=(),
        unmapped_prefixes=(),
    )
    rendered = test_evidence_suggestions.render_markdown([suggestion], summary)
    assert "matched:" not in rendered


# gabion:evidence E:call_footprint::tests/test_test_evidence_suggestions_edges.py::test_graph_suggestions_evidence_items_and_no_targets_branches::evidence_keys.py::gabion.analysis.evidence_keys.make_paramset_key::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._graph_suggestions::test_test_evidence_suggestions_edges.py::tests.test_test_evidence_suggestions_edges._minimal_entry
def test_graph_suggestions_evidence_items_and_no_targets_branches(tmp_path: Path) -> None:
    (tmp_path / "seed.py").write_text("def seed():\n    return 1\n", encoding="utf-8")
    entry = _minimal_entry("tests/test_alpha.py::test_alpha", "tests/test_alpha.py")
    info = FunctionInfo(
        name="test_alpha",
        qual="tests.test_alpha.test_alpha",
        path=tmp_path / "tests" / "test_alpha.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    callee = FunctionInfo(
        name="helper",
        qual="pkg.app.helper",
        path=tmp_path / "app.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    site_id = NodeId(kind="FunctionSite", key=("app.py", "pkg.app.helper"))
    evidence_item = test_evidence_suggestions.EvidenceSuggestion(
        key=evidence_keys.make_paramset_key(["x"]),
        display="E:paramset::x",
    )

    suggestions, resolved = test_evidence_suggestions._graph_suggestions(
        [entry],
        root=tmp_path,
        paths=[tmp_path],
        forest=Forest(),
        config=None,
        max_depth=1,
        iter_paths_fn=lambda _paths, _cfg: [tmp_path / "seed.py"],
        build_function_index_fn=lambda *args, **kwargs: ({}, {info.qual: info}),
        build_test_index_fn=lambda _by_qual, _root: {entry.test_id: info},
        build_symbol_table_fn=lambda *args, **kwargs: SymbolTable(),
        build_forest_evidence_index_fn=lambda _forest: (
            {("app.py", "pkg.app.helper"): site_id},
            {site_id: (evidence_item,)},
        ),
        collect_reachable_fn=lambda *_args, **_kwargs: [callee],
    )
    assert entry.test_id in suggestions
    assert suggestions[entry.test_id].source == test_evidence_suggestions.GRAPH_SOURCE
    assert entry.test_id in resolved

    suggestions, resolved = test_evidence_suggestions._graph_suggestions(
        [entry],
        root=tmp_path,
        paths=[tmp_path],
        forest=Forest(),
        config=None,
        max_depth=1,
        iter_paths_fn=lambda _paths, _cfg: [tmp_path / "seed.py"],
        build_function_index_fn=lambda *args, **kwargs: ({}, {info.qual: info}),
        build_test_index_fn=lambda _by_qual, _root: {entry.test_id: info},
        build_symbol_table_fn=lambda *args, **kwargs: SymbolTable(),
        build_forest_evidence_index_fn=lambda _forest: ({}, {}),
        collect_reachable_fn=lambda *_args, **_kwargs: [callee],
        collect_call_footprint_targets_fn=lambda *args, **kwargs: (),
    )
    assert suggestions == {}
    assert entry.test_id in resolved


# gabion:evidence E:call_footprint::tests/test_test_evidence_suggestions_edges.py::test_suggest_evidence_include_heuristics_no_match_branch::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions.suggest_evidence::test_test_evidence_suggestions_edges.py::tests.test_test_evidence_suggestions_edges._minimal_entry
def test_suggest_evidence_include_heuristics_no_match_branch(tmp_path: Path) -> None:
    entry = _minimal_entry("tests/test_alpha.py::test_alpha", "tests/test_alpha.py")
    suggestions, summary = test_evidence_suggestions.suggest_evidence(
        [entry],
        root=tmp_path,
        forest=Forest(),
        include_heuristics=True,
        graph_suggestions_fn=lambda *args, **kwargs: ({}, set()),
        suggest_for_entry_fn=lambda _entry: ([], []),
    )
    assert suggestions == []
    assert summary.skipped_no_match == 1


# gabion:evidence E:call_footprint::tests/test_test_evidence_suggestions_edges.py::test_collect_call_footprints_skips_empty_targets::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions.collect_call_footprints::test_test_evidence_suggestions_edges.py::tests.test_test_evidence_suggestions_edges._minimal_entry
def test_collect_call_footprints_skips_empty_targets(tmp_path: Path) -> None:
    seed = tmp_path / "seed.py"
    seed.write_text("def seed():\n    return 1\n", encoding="utf-8")
    entry = _minimal_entry("tests/test_alpha.py::test_alpha", "tests/test_alpha.py")
    info = FunctionInfo(
        name="test_alpha",
        qual="tests.test_alpha.test_alpha",
        path=seed,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    assert (
        test_evidence_suggestions.collect_call_footprints(
            [entry, entry],
            root=tmp_path,
            paths=[tmp_path],
            config=AuditConfig(project_root=tmp_path),
            iter_paths_fn=lambda _paths, _cfg: [seed],
            build_function_index_fn=lambda *args, **kwargs: ({}, {info.qual: info}),
            build_symbol_table_fn=lambda *args, **kwargs: SymbolTable(),
            collect_class_index_fn=lambda *args, **kwargs: {},
            build_test_index_fn=lambda _by_qual, _root: {entry.test_id: info},
            collect_call_footprint_targets_fn=lambda *args, **kwargs: (),
        )
        == {}
    )


# gabion:evidence E:call_footprint::tests/test_test_evidence_suggestions_edges.py::test_find_module_level_calls_relative_cache_and_symbol_helpers::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._call_module_literals::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._call_symbol_refs::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._find_module_level_calls::test_test_evidence_suggestions_edges.py::tests.test_test_evidence_suggestions_edges._minimal_entry
def test_find_module_level_calls_relative_cache_and_symbol_helpers(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_alpha.py"
    test_file.write_text("def test_alpha():\n    helper(mod.run, k=name, m='pkg.mod')\n", encoding="utf-8")
    entry = _minimal_entry("tests/test_alpha.py::test_alpha", "tests/test_alpha.py")
    info = FunctionInfo(
        name="test_alpha",
        qual="tests.test_alpha.test_alpha",
        path=test_file,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 1, 1),
    )
    node_cache: dict[Path, dict[tuple[tuple[str, ...], str], ast.AST]] = {}
    resolved = test_evidence_suggestions._find_module_level_calls(
        info,
        entry=entry,
        node_cache=node_cache,
        module_cache={},
        symbol_table=SymbolTable(),
        by_name={},
        by_qual={},
        class_index=None,
        project_root=tmp_path,
    )
    assert resolved == ()
    assert tmp_path.joinpath("tests", "test_alpha.py") in node_cache
    cached_resolved = test_evidence_suggestions._find_module_level_calls(
        info,
        entry=entry,
        node_cache=node_cache,
        module_cache={},
        symbol_table=SymbolTable(),
        by_name={},
        by_qual={},
        class_index=None,
        project_root=tmp_path,
    )
    assert cached_resolved == ()
    absolute_entry = _minimal_entry(
        "tests/test_alpha.py::test_alpha",
        str(tmp_path / "tests" / "test_alpha.py"),
    )
    absolute_resolved = test_evidence_suggestions._find_module_level_calls(
        info,
        entry=absolute_entry,
        node_cache=node_cache,
        module_cache={},
        symbol_table=SymbolTable(),
        by_name={},
        by_qual={},
        class_index=None,
        project_root=tmp_path,
    )
    assert absolute_resolved == ()

    call = ast.parse("pkg.run(mod.helper, k=name, m='pkg.mod')").body[0].value
    assert isinstance(call, ast.Call)
    refs = test_evidence_suggestions._call_symbol_refs(call)
    assert "pkg.run" in refs
    assert "mod.helper" in refs
    assert "name" in refs
    literals = test_evidence_suggestions._call_module_literals(call)
    assert "pkg.mod" in literals


# gabion:evidence E:call_footprint::tests/test_test_evidence_suggestions_edges.py::test_call_symbol_refs_attribute_none_and_suggest_for_entry_parse_fallback::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._call_symbol_refs::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._suggest_for_entry::test_test_evidence_suggestions_edges.py::tests.test_test_evidence_suggestions_edges._minimal_entry
def test_call_symbol_refs_attribute_none_and_suggest_for_entry_parse_fallback() -> None:
    call = ast.Call(
        func=ast.Attribute(
            value=ast.Call(func=ast.Name(id="factory", ctx=ast.Load()), args=[], keywords=[]),
            attr="run",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[],
    )
    assert test_evidence_suggestions._call_symbol_refs(call) == []

    entry = _minimal_entry("tests/test_alpha.py::test_alpha", "tests/test_alpha.py")
    rule = test_evidence_suggestions._SuggestionRule(
        rule_id="opaque",
        evidence=("opaque-display",),
        needles=("alpha",),
        scope="name",
    )
    suggested, _ = test_evidence_suggestions._suggest_for_entry(
        entry,
        rules_fn=lambda: [rule],
        parse_display_fn=lambda _value: None,
        make_opaque_key_fn=lambda display: {"kind": "opaque", "display": display},
        normalize_key_fn=lambda key: key,
        render_display_fn=lambda key: f"rendered::{key['display']}",
        is_opaque_fn=lambda _key: True,
    )
    assert suggested and suggested[0].display == "opaque-display"
    call_with_non_symbol_callee = ast.Call(
        func=ast.Call(func=ast.Name(id="factory", ctx=ast.Load()), args=[], keywords=[]),
        args=[],
        keywords=[],
    )
    assert test_evidence_suggestions._call_symbol_refs(call_with_non_symbol_callee) == []


# gabion:evidence E:call_footprint::tests/test_test_evidence_suggestions_edges.py::test_suggest_for_entry_parse_display_and_non_opaque_path::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._suggest_for_entry::test_test_evidence_suggestions_edges.py::tests.test_test_evidence_suggestions_edges._minimal_entry
def test_suggest_for_entry_parse_display_and_non_opaque_path() -> None:
    entry = _minimal_entry("tests/test_alpha.py::test_alpha", "tests/test_alpha.py")
    rule = test_evidence_suggestions._SuggestionRule(
        rule_id="parsed",
        evidence=("E:bundle/sample",),
        needles=("alpha",),
        scope="name",
    )
    suggested, matches = test_evidence_suggestions._suggest_for_entry(
        entry,
        rules_fn=lambda: [rule],
        parse_display_fn=lambda value: {"kind": "parsed", "value": value},
        normalize_key_fn=lambda key: key,
        render_display_fn=lambda key: f"rendered::{key['value']}",
        is_opaque_fn=lambda _key: False,
    )
    assert matches == ["parsed"]
    assert suggested and suggested[0].display == "rendered::E:bundle/sample"


# gabion:evidence E:call_footprint::tests/test_test_evidence_suggestions_edges.py::test_module_resolution_and_site_parts_edge_branches::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._resolve_module_file::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._site_parts
def test_module_resolution_and_site_parts_edge_branches(tmp_path: Path) -> None:
    module_dir = tmp_path / "src" / "pkg"
    module_dir.mkdir(parents=True)
    cache: dict[str, Path | None] = {}
    assert test_evidence_suggestions._resolve_module_file("pkg", tmp_path, cache) is None

    forest = Forest()
    node_id = NodeId(kind="FunctionSite", key=("a.py", "q"))
    assert test_evidence_suggestions._site_parts(node_id, forest) == ("a.py", "q")


# gabion:evidence E:call_footprint::tests/test_test_evidence_suggestions_edges.py::test_suggest_for_entry_opaque_and_normalize_mapping_edges::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._normalize_evidence_list::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._suggest_for_entry::test_evidence_suggestions.py::gabion.analysis.test_evidence_suggestions._summarize_unmapped::test_test_evidence_suggestions_edges.py::tests.test_test_evidence_suggestions_edges._minimal_entry
def test_suggest_for_entry_opaque_and_normalize_mapping_edges() -> None:
    entry = _minimal_entry("tests/test_alpha.py::test_alpha", "tests/test_alpha.py")
    rule = test_evidence_suggestions._SuggestionRule(
        rule_id="opaque",
        evidence=("opaque-display",),
        needles=("alpha",),
        scope="name",
    )
    suggested, matches = test_evidence_suggestions._suggest_for_entry(
        entry,
        rules_fn=lambda: [rule],
    )
    assert matches == ["opaque"]
    assert suggested and suggested[0].display == "opaque-display"

    assert test_evidence_suggestions._normalize_evidence_list(
        [{"display": 123}, {"not_display": "x"}]
    ) == []
    assert test_evidence_suggestions._normalize_evidence_list([1]) == []

    modules, prefixes = test_evidence_suggestions._summarize_unmapped(
        [
            _minimal_entry("tests/test_alpha.py::test_alpha", "tests/test_alpha.py"),
            _minimal_entry("tests/test_helper.py::helper", "tests/test_helper.py"),
        ]
    )
    assert modules
    assert prefixes
