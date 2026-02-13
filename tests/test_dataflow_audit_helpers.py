from __future__ import annotations

from pathlib import Path
import ast
import sys

import pytest


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def _write(path: Path, content: str) -> None:
    path.write_text(content)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._extract_invariant_from_expr::expr E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._invariant_term::expr,params
def test_extract_invariant_from_expr_edges() -> None:
    da = _load()
    expr = ast.parse("x").body[0].value
    assert (
        da._extract_invariant_from_expr(expr, {"x"}, scope="s", source="src")
        is None
    )
    expr = ast.parse("a == b == c").body[0].value
    assert (
        da._extract_invariant_from_expr(expr, {"a", "b"}, scope="s", source="src")
        is None
    )
    expr = ast.parse("a != b").body[0].value
    assert (
        da._extract_invariant_from_expr(expr, {"a", "b"}, scope="s", source="src")
        is None
    )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._scope_path::root
def test_invariant_collector_skips_nested_defs_and_lambda(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(
        path,
        "def outer(a, b):\n"
        "    assert a == b\n"
        "    def inner():\n"
        "        assert a == b\n"
        "    async def inner_async():\n"
        "        assert a == b\n"
        "    x = (lambda y: y)\n",
    )
    props = da._collect_invariant_propositions(
        path,
        ignore_params=set(),
        project_root=tmp_path,
    )
    assert len(props) == 1


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._scope_path::root
def test_collect_invariant_emitters_type_error(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a, b):\n    return a\n")

    def bad_emitter(_fn):
        return ["bad"]

    with pytest.raises(TypeError):
        da._collect_invariant_propositions(
            path,
            ignore_params=set(),
            project_root=tmp_path,
            emitters=[bad_emitter],
        )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._format_invariant_proposition::prop
def test_format_invariant_proposition_variants() -> None:
    da = _load()
    prop = da.InvariantProposition(
        form="Equal", terms=("a", "b"), scope="mod.py:f", source="assert"
    )
    assert "mod.py:f" in da._format_invariant_proposition(prop)
    other = da.InvariantProposition(form="LessThan", terms=("a",), scope="", source="")
    assert "LessThan" in da._format_invariant_proposition(other)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._mark_param_roots::params
def test_decision_helpers_cover_paths() -> None:
    da = _load()
    assert da._decision_root_name(ast.parse("1").body[0].value) is None
    assert da._decision_root_name(ast.parse("user.id").body[0].value) == "user"
    found: set[str] = set()
    da._mark_param_roots(ast.parse("user['id']").body[0].value, {"user"}, found)
    assert "user" in found
    assert da._contains_boolish(ast.parse("not flag").body[0].value)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params
def test_decision_surface_params_match_ifexp_and_match() -> None:
    da = _load()
    tree = ast.parse(
        "def f(a, b):\n"
        "    x = a if b else b\n"
        "    match a:\n"
        "        case _ if b:\n"
        "            return a\n"
        "    return x\n"
    )
    fn = tree.body[0]
    params = da._decision_surface_params(fn, ignore_params=set())
    assert "b" in params


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._mark_param_roots::params
def test_value_encoded_decision_params_branches() -> None:
    da = _load()
    tree = ast.parse(
        "def f(flag):\n"
        "    value = obj.max(flag, 1)\n"
        "    value2 = flag & 1\n"
        "    value3 = (flag == 1) * 2\n"
        "    return value + value2 + value3\n"
    )
    fn = tree.body[0]
    params, reasons = da._value_encoded_decision_params(fn, ignore_params=set())
    assert "flag" in params
    assert "min/max" in reasons
    assert "bitmask" in reasons
    assert "boolean arithmetic" in reasons


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decision_tier_for::tier_map
def test_analyze_decision_surfaces_repo_warnings(tmp_path: Path) -> None:
    da = _load()
    module_path = tmp_path / "module.py"
    _write(
        module_path,
        "def callee(a):\n"
        "    if a:\n"
        "        return 1\n"
        "\n"
        "def caller(a):\n"
        "    return callee(a)\n"
        "\n"
        "def orphan(c):\n"
        "    if c:\n"
        "        return 1\n"
        "\n"
        "def missing_call(a):\n"
        "    unknown(a)\n",
    )
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_path = test_dir / "test_sample.py"
    _write(
        test_path,
        "def callee(a):\n"
        "    if a:\n"
        "        return 1\n"
        "\n"
        "def caller(a):\n"
        "    return callee(a)\n",
    )
    surfaces, warnings, lint_lines = da.analyze_decision_surfaces_repo(
        [module_path, test_path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        decision_tiers={"a": 2},
        require_tiers=True,
        forest=da.Forest(),
    )
    assert surfaces
    assert any("missing decision tier metadata" in warning for warning in warnings)
    assert any("tier-2 decision param" in warning for warning in warnings)
    assert any("GABION_DECISION_TIER" in line for line in lint_lines)
    assert any("GABION_DECISION_SURFACE" in line for line in lint_lines)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decision_tier_for::tier_map
def test_analyze_value_encoded_decisions_repo_warnings(tmp_path: Path) -> None:
    da = _load()
    module_path = tmp_path / "module.py"
    _write(
        module_path,
        "def vcallee(flag):\n"
        "    return (flag == 1) * 2\n"
        "\n"
        "def vcaller(flag):\n"
        "    return vcallee(flag)\n"
        "\n"
        "def vmystery(x):\n"
        "    return (x == 1) * 2\n",
    )
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_path = test_dir / "test_value.py"
    _write(
        test_path,
        "def vcallee(flag):\n"
        "    return (flag == 1) * 2\n"
        "\n"
        "def vcaller(flag):\n"
        "    return vcallee(flag)\n",
    )
    surfaces, warnings, rewrites, lint_lines = da.analyze_value_encoded_decisions_repo(
        [module_path, test_path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        decision_tiers={"flag": 2},
        require_tiers=True,
        forest=da.Forest(),
    )
    assert surfaces
    assert rewrites
    assert any("missing decision tier metadata" in warning for warning in warnings)
    assert any("tier-2 value-encoded" in warning for warning in warnings)
    assert any("GABION_VALUE_DECISION_TIER" in line for line in lint_lines)
    assert any("GABION_VALUE_DECISION_SURFACE" in line for line in lint_lines)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_annotations::fn,ignore_params
def test_param_annotations_by_path_skips_parse_errors(tmp_path: Path) -> None:
    da = _load()
    bad = tmp_path / "bad.py"
    _write(bad, "def f(:\n")
    parse_failures: list[dict[str, object]] = []
    result = da._param_annotations_by_path(
        [bad],
        ignore_params=set(),
        parse_failure_witnesses=parse_failures,
    )
    assert bad not in result
    assert parse_failures
    assert parse_failures[0]["stage"] == "param_annotations"


def test_parse_failure_stage_taxonomy_is_canonical(tmp_path: Path) -> None:
    da = _load()
    bad = tmp_path / "bad.py"
    _write(bad, "def f(:\n")
    parse_failures: list[dict[str, object]] = []
    da._param_annotations_by_path(
        [bad],
        ignore_params=set(),
        parse_failure_witnesses=parse_failures,
    )
    da._collect_deadline_function_facts(
        [bad],
        project_root=tmp_path,
        ignore_params=set(),
        parse_failure_witnesses=parse_failures,
    )
    da._collect_call_nodes_by_path(
        [bad],
        parse_failure_witnesses=parse_failures,
    )
    da._build_symbol_table(
        [bad],
        tmp_path,
        external_filter=True,
        parse_failure_witnesses=parse_failures,
    )
    da._collect_class_index(
        [bad],
        tmp_path,
        parse_failure_witnesses=parse_failures,
    )
    da._build_function_index(
        [bad],
        tmp_path,
        set(),
        "high",
        None,
        parse_failure_witnesses=parse_failures,
    )
    da._iter_config_fields(bad, parse_failure_witnesses=parse_failures)
    da._collect_dataclass_registry(
        [bad],
        project_root=tmp_path,
        parse_failure_witnesses=parse_failures,
    )
    da._iter_dataclass_call_bundles(
        bad,
        project_root=tmp_path,
        parse_failure_witnesses=parse_failures,
    )
    seen = {str(entry["stage"]) for entry in parse_failures}
    expected = {stage.value for stage in da._ParseModuleStage}
    assert seen == expected


def test_parse_witness_contract_violations_detect_nullable_signature() -> None:
    da = _load()
    source = (
        "def _build_call_graph(*, parse_failure_witnesses: list[dict] | None = None):\n"
        "    return {}\n"
    )
    violations = da._parse_witness_contract_violations(
        source=source,
        source_path=Path("virtual_dataflow_audit.py"),
        target_helpers=frozenset({"_build_call_graph"}),
    )
    assert violations == [
        "virtual_dataflow_audit.py:_build_call_graph parse_sink_contract parse_failure_witnesses must be total list[JSONObject]",
        "virtual_dataflow_audit.py:_build_call_graph parse_sink_contract parse_failure_witnesses must not default to None",
    ]


def test_build_call_graph_reuses_prebuilt_analysis_index(tmp_path: Path) -> None:
    da = _load()
    module = tmp_path / "module.py"
    _write(
        module,
        "def callee(x):\n"
        "    return x\n"
        "\n"
        "def caller(x):\n"
        "    return callee(x)\n",
    )
    parse_failures: list[dict[str, object]] = []
    analysis_index = da._build_analysis_index(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=parse_failures,
    )
    by_name, by_qual, transitive_callers = da._build_call_graph(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=parse_failures,
        analysis_index=analysis_index,
    )
    assert by_name is analysis_index.by_name
    assert by_qual is analysis_index.by_qual
    assert transitive_callers.get("module.callee") == {"module.caller"}


def test_constant_flow_accepts_prebuilt_analysis_index(tmp_path: Path) -> None:
    da = _load()
    module = tmp_path / "module.py"
    _write(
        module,
        "def callee(x):\n"
        "    return x\n"
        "\n"
        "def caller():\n"
        "    return callee(1)\n",
    )
    parse_failures: list[dict[str, object]] = []
    analysis_index = da._build_analysis_index(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=parse_failures,
    )
    without_index = da.analyze_constant_flow_repo(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=parse_failures,
    )
    with_index = da.analyze_constant_flow_repo(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=parse_failures,
        analysis_index=analysis_index,
    )
    assert with_index == without_index
    assert any("callee.x only observed constant 1" in line for line in with_index)


def test_analysis_index_resolved_call_edges_cache_and_transparency(tmp_path: Path) -> None:
    da = _load()
    module = tmp_path / "module.py"
    _write(
        module,
        "def deny(fn):\n"
        "    return fn\n"
        "\n"
        "@deny\n"
        "def blocked(x):\n"
        "    return x\n"
        "\n"
        "def open_(x):\n"
        "    return x\n"
        "\n"
        "def caller(x):\n"
        "    blocked(x)\n"
        "    open_(x)\n",
    )
    parse_failures: list[dict[str, object]] = []
    analysis_index = da._build_analysis_index(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators={"allow"},
        parse_failure_witnesses=parse_failures,
    )
    all_edges = da._analysis_index_resolved_call_edges(
        analysis_index,
        project_root=tmp_path,
        require_transparent=False,
    )
    assert da._analysis_index_resolved_call_edges(
        analysis_index,
        project_root=tmp_path,
        require_transparent=False,
    ) is all_edges
    transparent_edges = da._analysis_index_resolved_call_edges(
        analysis_index,
        project_root=tmp_path,
        require_transparent=True,
    )
    assert da._analysis_index_resolved_call_edges(
        analysis_index,
        project_root=tmp_path,
        require_transparent=True,
    ) is transparent_edges
    assert sorted(edge.callee.name for edge in all_edges) == ["blocked", "open_"]
    assert [edge.callee.name for edge in transparent_edges] == ["open_"]


def test_constant_and_deadness_projections_share_constant_details(
    tmp_path: Path,
) -> None:
    da = _load()
    module = tmp_path / "module.py"
    _write(
        module,
        "def callee(x):\n"
        "    return x\n"
        "\n"
        "def caller():\n"
        "    return callee(1)\n",
    )
    parse_failures: list[dict[str, object]] = []
    analysis_index = da._build_analysis_index(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=parse_failures,
    )
    details = da._collect_constant_flow_details(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=parse_failures,
        analysis_index=analysis_index,
    )
    assert da._constant_smells_from_details(details) == da.analyze_constant_flow_repo(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=parse_failures,
        analysis_index=analysis_index,
    )
    assert da._deadness_witnesses_from_constant_details(
        details,
        project_root=tmp_path,
    ) == da.analyze_deadness_flow_repo(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=parse_failures,
        analysis_index=analysis_index,
    )


def test_caller_param_bindings_for_call_covers_low_strict_star_paths() -> None:
    da = _load()
    callee = da.FunctionInfo(
        name="f",
        qual="mod.f",
        path=Path("mod.py"),
        params=["a", "b", "kw"],
        annots={},
        calls=[],
        unused_params=set(),
        positional_params=("a", "b"),
        kwonly_params=("kw",),
        vararg="rest",
        kwarg="kwargs",
    )
    call = da.CallArgs(
        callee="f",
        pos_map={"0": "x"},
        kw_map={"kw": "k"},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[(2, "sx")],
        star_kw=["sk"],
        is_test=False,
        span=(0, 0, 0, 1),
    )
    strict = da._caller_param_bindings_for_call(call, callee, strictness="high")
    assert strict == {"a": {"x"}, "kw": {"k"}}
    low = da._caller_param_bindings_for_call(call, callee, strictness="low")
    assert low["a"] == {"x"}
    assert low["kw"] == {"k"}
    assert low["b"] == {"sx", "sk"}
    assert low["rest"] == {"sx", "sk"}
    assert low["kwargs"] == {"sx", "sk"}


def test_lint_rows_materialize_and_project_from_forest() -> None:
    da = _load()
    forest = da.Forest()
    da._materialize_lint_rows(
        forest=forest,
        rows=[
            {
                "path": "a.py",
                "line": 3,
                "col": 4,
                "code": "GABION_SAMPLE",
                "message": "example finding",
                "source": "x",
            },
            {
                "path": "a.py",
                "line": 3,
                "col": 4,
                "code": "GABION_SAMPLE",
                "message": "example finding",
                "source": "y",
            },
        ],
    )
    projected = da._project_lint_rows_from_forest(forest=forest)
    assert projected == [
        {
            "path": "a.py",
            "line": 3,
            "col": 4,
            "code": "GABION_SAMPLE",
            "message": "example finding",
        }
    ]
    facets = [
        alt
        for alt in forest.alts
        if alt.kind == "SpecFacet" and alt.evidence.get("spec_name") == "lint_findings"
    ]
    assert facets


def test_compute_lint_lines_uses_forest_projection() -> None:
    da = _load()
    forest = da.Forest()
    rendered = da._compute_lint_lines(
        forest=forest,
        groups_by_path={},
        bundle_sites_by_path={},
        type_callsite_evidence=[],
        ambiguity_witnesses=[],
        exception_obligations=[],
        never_invariants=[],
        deadline_obligations=[],
        decision_lint_lines=[
            "a.py:3:4: GABION_SAMPLE example finding",
            "a.py:3:4: GABION_SAMPLE example finding",
        ],
        broad_type_lint_lines=[],
        constant_smells=[],
        unused_arg_smells=[],
    )
    assert rendered == ["a.py:3:4: GABION_SAMPLE example finding"]
    finding_nodes = [node for node in forest.nodes if node.kind == "LintFinding"]
    assert finding_nodes


def test_project_report_section_lines_roundtrip() -> None:
    da = _load()
    forest = da.Forest()
    rendered = da._project_report_section_lines(
        forest=forest,
        run_id="report_run",
        section="demo",
        lines=["alpha", "beta"],
    )
    assert rendered == ["alpha", "beta"]
    section_nodes = [node for node in forest.nodes if node.kind == "ReportSectionLine"]
    assert section_nodes
    spec_facets = [
        alt
        for alt in forest.alts
        if alt.kind == "SpecFacet" and alt.evidence.get("spec_name") == "report_section_lines"
    ]
    assert spec_facets


def test_emit_report_materializes_report_section_specs(tmp_path: Path) -> None:
    da = _load()
    sample = tmp_path / "sample.py"
    _write(sample, "def f(a, b):\n    return a + b\n")
    analysis = da.analyze_paths(
        [sample],
        forest=da.Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=10,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_bundle_forest=True,
        include_lint_lines=False,
        config=da.AuditConfig(project_root=tmp_path),
    )
    forest = analysis.forest
    assert forest is not None
    report, _ = da.render_report(
        analysis.groups_by_path,
        max_components=5,
        report=da.ReportCarrier(
            forest=forest,
            bundle_sites_by_path=analysis.bundle_sites_by_path,
            type_suggestions=["sample.py:1:1: tighten type"],
        ),
    )
    assert "Dataflow grammar audit" in report
    spec_facets = [
        alt
        for alt in forest.alts
        if alt.kind == "SpecFacet" and alt.evidence.get("spec_name") == "report_section_lines"
    ]
    assert spec_facets


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path
def test_resolve_synth_registry_path_latest(tmp_path: Path) -> None:
    da = _load()
    root = tmp_path
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stamp = "20240203_000000"
    (out_dir / "LATEST.txt").write_text(stamp)
    (out_dir / stamp).mkdir()
    expected = (out_dir / stamp / "fingerprint_synth.json").resolve()
    expected.write_text("{}")
    resolved = da._resolve_synth_registry_path(
        "out/LATEST/fingerprint_synth.json", root
    )
    assert resolved == expected
    missing = da._resolve_synth_registry_path(
        "missing/LATEST/fingerprint_synth.json", root
    )
    assert missing is None
    relative = da._resolve_synth_registry_path("other.json", root)
    assert relative == (root / "other.json").resolve()
    assert da._resolve_synth_registry_path(" ", root) is None
    assert da._resolve_synth_registry_path(None, root) is None


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.format_fingerprint::fingerprint E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index
def test_compute_fingerprint_matches_skips_missing_types() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}], "g": [{"b"}]}}
    annotations_by_path = {
        path: {"f": {"a": None}, "g": {"b": "int"}},
    }
    registry = da.PrimeRegistry()
    fingerprint = da.bundle_fingerprint_dimensional(["str"], registry, None)
    index = {fingerprint: {"Known"}}
    matches = da._compute_fingerprint_matches(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index=index,
    )
    assert matches == []


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.format_fingerprint::fingerprint E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index
def test_compute_fingerprint_matches_skips_missing_params() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}]}}
    annotations_by_path = {path: {"f": {}}}
    registry = da.PrimeRegistry()
    fingerprint = da.bundle_fingerprint_dimensional(["int"], registry, None)
    index = {fingerprint: {"Known"}}
    matches = da._compute_fingerprint_matches(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index=index,
    )
    assert matches == []


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_compute_fingerprint_provenance_skips_none_types() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}]}}
    annotations_by_path = {path: {"f": {"a": None}}}
    registry = da.PrimeRegistry()
    entries = da._compute_fingerprint_provenance(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index={},
    )
    assert entries == []


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_compute_fingerprint_provenance_skips_missing_params() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}]}}
    annotations_by_path = {path: {"f": {}}}
    registry = da.PrimeRegistry()
    entries = da._compute_fingerprint_provenance(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index={},
    )
    assert entries == []


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::entries,max_examples
def test_summarize_fingerprint_provenance_empty() -> None:
    da = _load()
    assert da._summarize_fingerprint_provenance([]) == []


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences
def test_compute_fingerprint_synth_skips_none_types() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}]}}
    annotations_by_path = {path: {"f": {"a": None}}}
    registry = da.PrimeRegistry()
    ctor_registry = da.TypeConstructorRegistry(registry)
    lines, payload = da._compute_fingerprint_synth(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        ctor_registry=ctor_registry,
        min_occurrences=2,
        version="synth@1",
    )
    assert lines == []
    assert payload is None


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences
def test_compute_fingerprint_synth_skips_missing_params() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}]}}
    annotations_by_path = {path: {"f": {}}}
    registry = da.PrimeRegistry()
    ctor_registry = da.TypeConstructorRegistry(registry)
    lines, payload = da._compute_fingerprint_synth(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        ctor_registry=ctor_registry,
        min_occurrences=2,
        version="synth@1",
    )
    assert lines == []
    assert payload is None


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences
def test_compute_fingerprint_synth_includes_ctor_keys() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}], "g": [{"a"}]}}
    annotations_by_path = {
        path: {"f": {"a": "list[int]"}, "g": {"a": "list[int]"}}
    }
    registry = da.PrimeRegistry()
    ctor_registry = da.TypeConstructorRegistry(registry)
    lines, payload = da._compute_fingerprint_synth(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        ctor_registry=ctor_registry,
        min_occurrences=2,
        version="synth@1",
    )
    assert payload is not None
    assert any("ctor=" in line for line in lines)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_render_structure_snapshot_skips_invalid_invariant_scope(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["a", "b"])]}}
    invalid = da.InvariantProposition(
        form="Equal", terms=("a", "b"), scope="bad_scope", source="assert"
    )
    valid = da.InvariantProposition(
        form="Equal", terms=("a", "b"), scope="mod.py:f", source="assert"
    )
    snapshot = da.render_structure_snapshot(
        groups_by_path,
        project_root=tmp_path,
        forest=da.Forest(),
        invariant_propositions=[invalid, valid],
    )
    files = snapshot.get("files") or []
    assert files
    functions = files[0].get("functions") or []
    assert functions
    assert "invariants" in functions[0]


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.load_decision_snapshot
def test_load_decision_snapshot_errors(tmp_path: Path) -> None:
    da = _load()
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(ValueError):
        da.load_decision_snapshot(bad)
    not_obj = tmp_path / "list.json"
    not_obj.write_text("[]")
    with pytest.raises(ValueError):
        da.load_decision_snapshot(not_obj)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness
def test_exception_obligations_enum_and_handledness(tmp_path: Path) -> None:
    da = _load()
    module_path = tmp_path / "mod.py"
    _write(
        module_path,
        "def f(a):\n"
        "    raise ValueError('bad')\n"
        "\n"
        "def g(b):\n"
        "    try:\n"
        "        raise RuntimeError('oops')\n"
        "    except Exception:\n"
        "        return b\n",
    )
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
    )
    analysis = da.analyze_paths(
        forest=da.Forest(),
        paths=[tmp_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_exception_obligations=True,
        include_handledness_witnesses=True,
        config=config,
    )
    obligations = analysis.exception_obligations
    assert any(entry["status"] == "UNKNOWN" for entry in obligations)
    assert any(entry["status"] == "HANDLED" for entry in obligations)
    assert analysis.handledness_witnesses
