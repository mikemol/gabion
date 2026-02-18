from __future__ import annotations

from collections import Counter
from pathlib import Path
import ast

import pytest

from gabion.exceptions import NeverThrown
from gabion.analysis.timeout_context import (
    Deadline,
    TimeoutContext,
    deadline_scope,
    pack_call_stack,
)

def _load():
    repo_root = Path(__file__).resolve().parents[1]
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

def test_decision_surface_metafactory_parity(tmp_path: Path) -> None:
    da = _load()
    module_path = tmp_path / "module.py"
    _write(
        module_path,
        "def direct(a):\n"
        "    if a:\n"
        "        return 1\n"
        "    return 0\n"
        "\n"
        "def value(flag):\n"
        "    return (flag == 1) * 2\n",
    )
    parse_failures: list[dict[str, object]] = []
    analysis_index = da._build_analysis_index(
        [module_path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=parse_failures,
    )
    context = da._IndexedPassContext(
        paths=[module_path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        parse_failure_witnesses=parse_failures,
        analysis_index=analysis_index,
    )
    direct_helper = da._analyze_decision_surface_indexed(
        context,
        spec=da._DIRECT_DECISION_SURFACE_SPEC,
        decision_tiers=None,
        require_tiers=False,
        forest=da.Forest(),
    )
    value_helper = da._analyze_decision_surface_indexed(
        context,
        spec=da._VALUE_DECISION_SURFACE_SPEC,
        decision_tiers=None,
        require_tiers=False,
        forest=da.Forest(),
    )
    assert direct_helper[2] == []
    assert value_helper[2]

    direct_repo = da.analyze_decision_surfaces_repo(
        [module_path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        forest=da.Forest(),
        parse_failure_witnesses=parse_failures,
        analysis_index=analysis_index,
    )
    value_repo = da.analyze_value_encoded_decisions_repo(
        [module_path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        forest=da.Forest(),
        parse_failure_witnesses=parse_failures,
        analysis_index=analysis_index,
    )
    assert direct_repo == (direct_helper[0], direct_helper[1], direct_helper[3])
    assert value_repo == value_helper

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
    da._raw_sorted_contract_violations(
        [bad],
        parse_failure_witnesses=parse_failures,
    )
    da._materialize_structured_suite_sites(
        forest=da.Forest(),
        file_paths=[bad],
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
    all_edge_names = [edge.callee.name for edge in all_edges]
    assert len(all_edge_names) == 2
    assert set(all_edge_names) == {"blocked", "open_"}
    assert [edge.callee.name for edge in transparent_edges] == ["open_"]

def test_analysis_index_module_trees_reuses_cached_tree_across_stages(
    tmp_path: Path,
) -> None:
    da = _load()
    module = tmp_path / "module.py"
    _write(module, "def f(x):\n    return x\n")
    parse_failures: list[dict[str, object]] = []
    analysis_index = da.AnalysisIndex(
        by_name={},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    deadline_trees = da._analysis_index_module_trees(
        analysis_index,
        [module],
        stage=da._ParseModuleStage.DEADLINE_FUNCTION_FACTS,
        parse_failure_witnesses=parse_failures,
    )
    call_trees = da._analysis_index_module_trees(
        analysis_index,
        [module],
        stage=da._ParseModuleStage.CALL_NODES,
        parse_failure_witnesses=parse_failures,
    )
    assert parse_failures == []
    assert deadline_trees[module] is not None
    assert call_trees[module] is deadline_trees[module]

def test_analysis_index_module_trees_replays_parse_failure_by_stage(
    tmp_path: Path,
) -> None:
    da = _load()
    bad = tmp_path / "bad.py"
    _write(bad, "def f(:\n")
    parse_failures: list[dict[str, object]] = []
    analysis_index = da.AnalysisIndex(
        by_name={},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    deadline_trees = da._analysis_index_module_trees(
        analysis_index,
        [bad],
        stage=da._ParseModuleStage.DEADLINE_FUNCTION_FACTS,
        parse_failure_witnesses=parse_failures,
    )
    call_trees = da._analysis_index_module_trees(
        analysis_index,
        [bad],
        stage=da._ParseModuleStage.CALL_NODES,
        parse_failure_witnesses=parse_failures,
    )
    assert deadline_trees[bad] is None
    assert call_trees[bad] is None
    assert [entry["stage"] for entry in parse_failures] == [
        da._ParseModuleStage.DEADLINE_FUNCTION_FACTS.value,
        da._ParseModuleStage.CALL_NODES.value,
    ]
    assert bad in analysis_index.module_parse_errors_by_path

def test_build_module_artifacts_parses_each_path_once_across_specs(
    tmp_path: Path,
) -> None:
    da = _load()
    module = tmp_path / "module.py"
    _write(module, "def f(x):\n    return x\n")
    parse_failures: list[dict[str, object]] = []
    parse_calls = 0

    def _parse_module(path: Path) -> ast.Module:
        nonlocal parse_calls
        parse_calls += 1
        return ast.parse(path.read_text())

    specs = (
        da._ModuleArtifactSpec[list[str], tuple[str, ...]](
            artifact_id="first",
            stage=da._ParseModuleStage.FUNCTION_INDEX,
            init=list,
            fold=lambda acc, path, _tree: acc.append(f"first:{path.name}"),
            finish=tuple,
        ),
        da._ModuleArtifactSpec[list[str], tuple[str, ...]](
            artifact_id="second",
            stage=da._ParseModuleStage.SYMBOL_TABLE,
            init=list,
            fold=lambda acc, path, _tree: acc.append(f"second:{path.name}"),
            finish=tuple,
        ),
    )
    first, second = da._build_module_artifacts(
        [module],
        specs=specs,
        parse_failure_witnesses=parse_failures,
        parse_module=_parse_module,
    )
    assert parse_failures == []
    assert parse_calls == 1
    assert first == ("first:module.py",)
    assert second == ("second:module.py",)

def test_build_analysis_index_module_artifact_parse_stages(tmp_path: Path) -> None:
    da = _load()
    bad = tmp_path / "bad.py"
    _write(bad, "def f(:\n")
    parse_failures: list[dict[str, object]] = []
    analysis_index = da._build_analysis_index(
        [bad],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=parse_failures,
    )
    assert analysis_index.by_name == {}
    assert analysis_index.by_qual == {}
    assert analysis_index.class_index == {}
    assert [entry["stage"] for entry in parse_failures] == [
        da._ParseModuleStage.FUNCTION_INDEX.value,
        da._ParseModuleStage.SYMBOL_TABLE.value,
        da._ParseModuleStage.CLASS_INDEX.value,
    ]

def test_build_analysis_index_module_artifact_parity(tmp_path: Path) -> None:
    da = _load()
    module = tmp_path / "module.py"
    _write(
        module,
        "from collections import defaultdict as dd\n"
        "\n"
        "class Box:\n"
        "    def value(self, x):\n"
        "        return x\n"
        "\n"
        "def f(x):\n"
        "    return Box().value(x)\n",
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
    by_name, by_qual = da._build_function_index(
        [module],
        tmp_path,
        set(),
        "high",
        None,
        parse_failure_witnesses=[],
    )
    symbol_table = da._build_symbol_table(
        [module],
        tmp_path,
        external_filter=True,
        parse_failure_witnesses=[],
    )
    class_index = da._collect_class_index(
        [module],
        tmp_path,
        parse_failure_witnesses=[],
    )
    assert parse_failures == []
    assert analysis_index.by_name == by_name
    assert analysis_index.by_qual == by_qual
    assert analysis_index.class_index == class_index
    assert analysis_index.symbol_table.imports == symbol_table.imports
    assert analysis_index.symbol_table.internal_roots == symbol_table.internal_roots
    assert analysis_index.symbol_table.module_exports == symbol_table.module_exports
    assert (
        analysis_index.symbol_table.module_export_map
        == symbol_table.module_export_map
    )

def test_analysis_index_stage_cache_factory_reuses_builder(tmp_path: Path) -> None:
    da = _load()
    module = tmp_path / "module.py"
    _write(module, "def f(x):\n    return x\n")
    parse_failures: list[dict[str, object]] = []
    analysis_index = da.AnalysisIndex(
        by_name={},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    build_calls = 0

    def _build(tree: ast.Module, _path: Path) -> int:
        nonlocal build_calls
        build_calls += 1
        return len(list(ast.walk(tree)))

    spec = da._StageCacheSpec[int](
        stage=da._ParseModuleStage.CALL_NODES,
        cache_key=("demo-stage-cache",),
        build=_build,
    )
    first = da._analysis_index_stage_cache(
        analysis_index,
        [module],
        spec=spec,
        parse_failure_witnesses=parse_failures,
    )
    second = da._analysis_index_stage_cache(
        analysis_index,
        [module],
        spec=spec,
        parse_failure_witnesses=parse_failures,
    )
    assert parse_failures == []
    assert first[module] == second[module]
    assert build_calls == 1

def test_collect_config_and_dataclass_stage_caches_reuse_analysis_index(
    tmp_path: Path,
) -> None:
    da = _load()
    module = tmp_path / "module.py"
    _write(
        module,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class AppConfig:\n"
        "    timeout_fn: str\n"
        "\n"
        "@dataclass\n"
        "class Payload:\n"
        "    value: int\n",
    )
    parse_failures: list[dict[str, object]] = []
    analysis_index = da.AnalysisIndex(
        by_name={},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    bundles = da._collect_config_bundles(
        [module],
        parse_failure_witnesses=parse_failures,
        analysis_index=analysis_index,
    )
    registry = da._collect_dataclass_registry(
        [module],
        project_root=tmp_path,
        parse_failure_witnesses=parse_failures,
        analysis_index=analysis_index,
    )
    assert parse_failures == []
    assert module in bundles
    assert "AppConfig" in bundles[module]
    assert "module.AppConfig" in registry
    assert "module.Payload" in registry
    cache_keys = {key[1] for key in analysis_index.stage_cache_by_key}
    assert any(isinstance(key, tuple) and key[-1] == "config_fields" for key in cache_keys)
    assert any(isinstance(key, tuple) and key[-1] == "dataclass_registry" for key in cache_keys)

def test_run_indexed_pass_hydrates_index_and_sink() -> None:
    da = _load()
    sentinel_index = da.AnalysisIndex(
        by_name={},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    calls = 0

    def _build_analysis_index(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return sentinel_index

    result = da._run_indexed_pass(
        [Path("demo.py")],
        project_root=Path("."),
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        build_index=_build_analysis_index,
        spec=da._IndexedPassSpec(
            pass_id="demo",
            run=lambda context: (
                context.analysis_index is sentinel_index,
                context.parse_failure_witnesses == [],
            ),
        ),
    )
    assert calls == 1
    assert result == (True, True)

def test_run_indexed_pass_reuses_prebuilt_index() -> None:
    da = _load()
    sentinel_index = da.AnalysisIndex(
        by_name={},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    sink: list[dict[str, object]] = []

    def _unexpected_build(*_args, **_kwargs):
        raise AssertionError("unexpected index build")

    result = da._run_indexed_pass(
        [Path("demo.py")],
        project_root=Path("."),
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=sink,
        analysis_index=sentinel_index,
        build_index=_unexpected_build,
        spec=da._IndexedPassSpec(
            pass_id="demo",
            run=lambda context: (
                context.analysis_index is sentinel_index,
                context.parse_failure_witnesses is sink,
            ),
        ),
    )
    assert result == (True, True)

def test_reduce_resolved_call_edges_respects_transparency_filter(tmp_path: Path) -> None:
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

    def _spec() -> object:
        def _fold(acc: dict[str, int], _edge) -> None:
            acc["count"] += 1

        return da._ResolvedEdgeReducerSpec[dict[str, int], int](
            reducer_id="count_edges",
            init=lambda: {"count": 0},
            fold=_fold,
            finish=lambda acc: acc["count"],
        )

    all_count = da._reduce_resolved_call_edges(
        analysis_index,
        project_root=tmp_path,
        require_transparent=False,
        spec=_spec(),
    )
    transparent_count = da._reduce_resolved_call_edges(
        analysis_index,
        project_root=tmp_path,
        require_transparent=True,
        spec=_spec(),
    )
    assert all_count == 2
    assert transparent_count == 1

def test_iter_resolved_edge_param_events_low_strict_variadic_modes() -> None:
    da = _load()
    caller = da.FunctionInfo(
        name="caller",
        qual="mod.caller",
        path=Path("mod.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
    )
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
    edge = da._ResolvedCallEdge(caller=caller, call=call, callee=callee)
    named_only = list(
        da._iter_resolved_edge_param_events(
            edge,
            strictness="low",
            include_variadics_in_low_star=False,
        )
    )
    with_variadics = list(
        da._iter_resolved_edge_param_events(
            edge,
            strictness="low",
            include_variadics_in_low_star=True,
        )
    )

    named_counts = Counter(event.param for event in named_only)
    variadic_counts = Counter(event.param for event in with_variadics)
    assert named_counts == {"a": 1, "kw": 1, "b": 2, "rest": 1, "kwargs": 1}
    assert variadic_counts == {"a": 1, "kw": 1, "b": 2, "rest": 2, "kwargs": 2}
    assert all(event.kind == "non_const" for event in named_only)
    assert all(event.kind == "non_const" for event in with_variadics)
    assert all(not event.countable for event in named_only if event.param in {"rest", "kwargs"})
    assert all(
        not event.countable for event in with_variadics if event.param in {"rest", "kwargs"}
    )

def test_execution_pattern_suggestions_detect_indexed_pass_ingress() -> None:
    da = _load()
    source = (
        "def one(paths, *, project_root, ignore_params, strictness, external_filter, "
        "transparent_decorators=None, parse_failure_witnesses=None, analysis_index=None):\n"
        "    parse_failure_witnesses = _parse_failure_sink(parse_failure_witnesses)\n"
        "    if analysis_index is None:\n"
        "        analysis_index = _build_analysis_index(paths, project_root=project_root, "
        "ignore_params=ignore_params, strictness=strictness, external_filter=external_filter, "
        "transparent_decorators=transparent_decorators, parse_failure_witnesses=parse_failure_witnesses)\n"
        "    return analysis_index\n"
        "\n"
        "def two(paths, *, project_root, ignore_params, strictness, external_filter, "
        "transparent_decorators=None, parse_failure_witnesses=None, analysis_index=None):\n"
        "    return _build_call_graph(paths, project_root=project_root, ignore_params=ignore_params, "
        "strictness=strictness, external_filter=external_filter, transparent_decorators=transparent_decorators, "
        "parse_failure_witnesses=parse_failure_witnesses, analysis_index=analysis_index)\n"
        "\n"
        "def three(paths, *, project_root, ignore_params, strictness, external_filter, "
        "transparent_decorators=None, parse_failure_witnesses=None, analysis_index=None):\n"
        "    return _build_call_graph(paths, project_root=project_root, ignore_params=ignore_params, "
        "strictness=strictness, external_filter=external_filter, transparent_decorators=transparent_decorators, "
        "parse_failure_witnesses=parse_failure_witnesses, analysis_index=analysis_index)\n"
    )
    suggestions = da._execution_pattern_suggestions(source=source)
    assert any("indexed_pass_ingress" in line for line in suggestions)

def test_pattern_schema_suggestions_include_execution_and_dataflow_axes() -> None:
    da = _load()
    source = (
        "def one(paths, *, project_root, ignore_params, strictness, external_filter, "
        "transparent_decorators=None, parse_failure_witnesses=None, analysis_index=None):\n"
        "    return _build_analysis_index(paths, project_root=project_root, "
        "ignore_params=ignore_params, strictness=strictness, external_filter=external_filter, "
        "transparent_decorators=transparent_decorators, parse_failure_witnesses=parse_failure_witnesses)\n"
        "\n"
        "def two(paths, *, project_root, ignore_params, strictness, external_filter, "
        "transparent_decorators=None, parse_failure_witnesses=None, analysis_index=None):\n"
        "    return _build_call_graph(paths, project_root=project_root, ignore_params=ignore_params, "
        "strictness=strictness, external_filter=external_filter, transparent_decorators=transparent_decorators, "
        "parse_failure_witnesses=parse_failure_witnesses, analysis_index=analysis_index)\n"
        "\n"
        "def three(paths, *, project_root, ignore_params, strictness, external_filter, "
        "transparent_decorators=None, parse_failure_witnesses=None, analysis_index=None):\n"
        "    return _build_call_graph(paths, project_root=project_root, ignore_params=ignore_params, "
        "strictness=strictness, external_filter=external_filter, transparent_decorators=transparent_decorators, "
        "parse_failure_witnesses=parse_failure_witnesses, analysis_index=analysis_index)\n"
    )
    groups_by_path = {
        Path("mod.py"): {
            "f": [set(["a", "b"])],
            "g": [set(["a", "b"])],
        }
    }
    suggestions = da._pattern_schema_suggestions(
        groups_by_path=groups_by_path,
        source=source,
    )
    assert any(
        "pattern_schema axis=execution" in line and "indexed_pass_ingress" in line
        for line in suggestions
    )
    assert any(
        "pattern_schema axis=dataflow" in line and "bundle=a,b" in line
        for line in suggestions
    )

def test_pattern_schema_residue_entries_cover_both_axes() -> None:
    da = _load()
    source = (
        "def one(paths, *, project_root, ignore_params, strictness, external_filter, "
        "transparent_decorators=None, parse_failure_witnesses=None, analysis_index=None):\n"
        "    return _build_analysis_index(paths, project_root=project_root, "
        "ignore_params=ignore_params, strictness=strictness, external_filter=external_filter, "
        "transparent_decorators=transparent_decorators, parse_failure_witnesses=parse_failure_witnesses)\n"
        "\n"
        "def two(paths, *, project_root, ignore_params, strictness, external_filter, "
        "transparent_decorators=None, parse_failure_witnesses=None, analysis_index=None):\n"
        "    return _build_call_graph(paths, project_root=project_root, ignore_params=ignore_params, "
        "strictness=strictness, external_filter=external_filter, transparent_decorators=transparent_decorators, "
        "parse_failure_witnesses=parse_failure_witnesses, analysis_index=analysis_index)\n"
        "\n"
        "def three(paths, *, project_root, ignore_params, strictness, external_filter, "
        "transparent_decorators=None, parse_failure_witnesses=None, analysis_index=None):\n"
        "    return _build_call_graph(paths, project_root=project_root, ignore_params=ignore_params, "
        "strictness=strictness, external_filter=external_filter, transparent_decorators=transparent_decorators, "
        "parse_failure_witnesses=parse_failure_witnesses, analysis_index=analysis_index)\n"
    )
    groups_by_path = {
        Path("mod.py"): {
            "f": [set(["a", "b"])],
            "g": [set(["a", "b"])],
        }
    }
    instances = da._pattern_schema_matches(groups_by_path=groups_by_path, source=source)
    residue_entries = da._pattern_schema_residue_entries(instances)
    assert any(entry.reason == "unreified_metafactory" for entry in residue_entries)
    assert any(entry.reason == "unreified_protocol" for entry in residue_entries)
    residue_lines = da._pattern_schema_residue_lines(residue_entries)
    assert any("reason=unreified_metafactory" in line for line in residue_lines)
    assert any("reason=unreified_protocol" in line for line in residue_lines)


def test_pattern_schema_identity_is_stable_for_permuted_fixtures() -> None:
    da = _load()
    groups_a = {
        Path("a.py"): {"f": [set(["b", "a"])], "g": [set(["a", "b"])]},
    }
    groups_b = {
        Path("a.py"): {"g": [set(["b", "a"])], "f": [set(["a", "b"])]},
    }
    instances_a = da._pattern_schema_matches(groups_by_path=groups_a, include_execution=False)
    instances_b = da._pattern_schema_matches(groups_by_path=groups_b, include_execution=False)
    ids_a = [entry.schema.schema_id for entry in instances_a]
    ids_b = [entry.schema.schema_id for entry in instances_b]
    assert ids_a == ids_b
    assert [entry.schema.normalized_signature for entry in instances_a] == [
        entry.schema.normalized_signature for entry in instances_b
    ]


def test_pattern_schema_residue_is_deterministic_for_fixed_fixture() -> None:
    da = _load()
    groups_by_path = {
        Path("mod.py"): {
            "f": [set(["b", "a"])],
            "g": [set(["a", "b"])],
        }
    }
    first = da._pattern_schema_residue_lines(
        da._pattern_schema_residue_entries(
            da._pattern_schema_matches(groups_by_path=groups_by_path, include_execution=False)
        )
    )
    second = da._pattern_schema_residue_lines(
        da._pattern_schema_residue_entries(
            da._pattern_schema_matches(groups_by_path=groups_by_path, include_execution=False)
        )
    )
    assert first == second
    assert any("reason=unreified_protocol" in line for line in first)


def test_pattern_schema_normalize_signature_handles_nested_dict_values() -> None:
    from gabion.analysis import pattern_schema

    normalized = pattern_schema.normalize_signature(
        {
            "z": {"b": 2, "a": 1},
            "a": [{"b": 2, "a": 1}, {"d": 4, "c": 3}],
        }
    )
    assert list(normalized) == ["a", "z"]
    z_block = normalized["z"]
    assert isinstance(z_block, dict)
    assert list(z_block) == ["a", "b"]
    list_block = normalized["a"]
    assert isinstance(list_block, list)
    assert list_block == [
        {"a": 1, "b": 2},
        {"c": 3, "d": 4},
    ]


def test_pattern_schema_normalize_signature_keeps_unsortable_list_order() -> None:
    from gabion.analysis import pattern_schema

    normalized = pattern_schema.normalize_signature(
        {
            "items": [
                {"x": 1, "a": 2},
                {"b": 3, "a": 4},
            ],
        }
    )
    items = normalized["items"]
    assert isinstance(items, list)
    assert items == [
        {"a": 2, "x": 1},
        {"a": 4, "b": 3},
    ]


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
        section_key=da._ReportSectionKey(run_id="report_run", section="demo"),
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
        "        return b\n"
        "\n"
        "def h(c):\n"
        "    try:\n"
        "        raise ValueError(c)\n"
        "    except CustomError:\n"
        "        return c\n"
        "\n"
        "def j(d):\n"
        "    try:\n"
        "        raise ValueError(d)\n"
        "    except CustomError:\n"
        "        return d\n"
        "    except ValueError:\n"
        "        return d\n"
        "\n"
        "def k(e):\n"
        "    try:\n"
        "        raise e\n"
        "    except ValueError:\n"
        "        return e\n"
        "    except TypeError:\n"
        "        return e\n",
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
    witnesses_by_function = {
        str(entry.get("site", {}).get("function", "")): entry
        for entry in analysis.handledness_witnesses
    }
    assert witnesses_by_function["g"]["result"] == "HANDLED"
    assert witnesses_by_function["g"]["type_compatibility"] == "compatible"
    assert witnesses_by_function["h"]["result"] == "UNKNOWN"
    assert witnesses_by_function["h"]["type_compatibility"] == "unknown"
    assert witnesses_by_function["j"]["result"] == "HANDLED"
    assert witnesses_by_function["j"]["type_compatibility"] == "compatible"
    assert witnesses_by_function["j"]["handler_boundary"] == "except ValueError"
    assert witnesses_by_function["j"]["handler_types"] == ["ValueError"]
    assert witnesses_by_function["k"]["result"] == "UNKNOWN"
    assert witnesses_by_function["k"]["type_compatibility"] == "unknown"
    assert witnesses_by_function["k"]["handler_boundary"] == "except ValueError"
    assert witnesses_by_function["k"]["handler_types"] == ["ValueError"]
    obligations_by_function = {
        str(entry.get("site", {}).get("function", "")): entry
        for entry in obligations
    }
    assert obligations_by_function["g"]["status"] == "HANDLED"
    assert obligations_by_function["h"]["status"] == "UNKNOWN"
    assert obligations_by_function["j"]["status"] == "HANDLED"
    assert obligations_by_function["k"]["status"] == "UNKNOWN"

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_analysis_collection_resume_payload::bundle_sites_by_path,completed_paths,groups_by_path,invariant_propositions,param_spans_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._load_analysis_collection_resume_payload::file_paths,include_invariant_propositions,payload E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::collection_resume,file_paths_override,on_collection_progress
def test_analyze_paths_collection_resume_roundtrip(tmp_path: Path) -> None:
    da = _load()
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    _write(first, "def a(x):\n    return x\n")
    _write(second, "def b(y):\n    return y\n")
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
    )
    snapshots: list[dict[str, object]] = []
    baseline = da.analyze_paths(
        forest=da.Forest(),
        paths=[tmp_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_invariant_propositions=True,
        config=config,
        on_collection_progress=snapshots.append,
    )
    assert snapshots
    assert any(
        isinstance(snapshot.get("in_progress_scan_by_path"), dict)
        and bool(snapshot.get("in_progress_scan_by_path"))
        for snapshot in snapshots
    )
    resume_payload = snapshots[-1]
    resumed_updates = 0

    def _capture(_: dict[str, object]) -> None:
        nonlocal resumed_updates
        resumed_updates += 1

    resumed = da.analyze_paths(
        forest=da.Forest(),
        paths=[tmp_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_invariant_propositions=True,
        config=config,
        collection_resume=resume_payload,
        on_collection_progress=_capture,
    )
    assert resumed_updates == 0
    assert resumed.groups_by_path == baseline.groups_by_path
    assert resumed.param_spans_by_path == baseline.param_spans_by_path
    assert resumed.bundle_sites_by_path == baseline.bundle_sites_by_path
    assert resumed.invariant_propositions == baseline.invariant_propositions

def test_collection_resume_roundtrip_preserves_analysis_index_resume_payload() -> None:
    da = _load()
    payload = da._build_analysis_collection_resume_payload(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        invariant_propositions=[],
        completed_paths=set(),
        in_progress_scan_by_path={},
        analysis_index_resume={
            "format_version": 1,
            "phase": "analysis_index_hydration",
            "hydrated_paths": ["a.py"],
            "hydrated_paths_count": 1,
            "function_count": 1,
            "class_count": 0,
            "functions_by_qual": {},
            "symbol_table": {},
            "class_index": {},
        },
    )
    (
        _groups_by_path,
        _param_spans_by_path,
        _bundle_sites_by_path,
        _invariant_propositions,
        _completed_paths,
        _in_progress_scan_by_path,
        analysis_index_resume,
    ) = da._load_analysis_collection_resume_payload(
        payload=payload,
        file_paths=[],
        include_invariant_propositions=False,
    )
    assert isinstance(analysis_index_resume, dict)
    assert analysis_index_resume.get("hydrated_paths_count") == 1

def test_build_analysis_index_resumes_hydrated_payload(tmp_path: Path) -> None:
    da = _load()
    module_path = tmp_path / "sample.py"
    _write(
        module_path,
        "class C:\n"
        "    def f(self, x: int) -> int:\n"
        "        return x\n",
    )
    parse_failure_witnesses: list[dict[str, object]] = []
    progress_payloads: list[dict[str, object]] = []
    baseline = da._build_analysis_index(
        [module_path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=False,
        transparent_decorators=set(),
        parse_failure_witnesses=parse_failure_witnesses,
        on_progress=progress_payloads.append,
    )
    assert progress_payloads
    resume_payload = progress_payloads[-1]
    assert resume_payload.get("hydrated_paths_count") == 1
    resumed_progress_payloads: list[dict[str, object]] = []
    resumed = da._build_analysis_index(
        [module_path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=False,
        transparent_decorators=set(),
        parse_failure_witnesses=[],
        resume_payload=resume_payload,
        on_progress=resumed_progress_payloads.append,
    )
    assert resumed_progress_payloads
    assert resumed_progress_payloads[-1].get("hydrated_paths_count") == 1
    assert baseline.by_qual.keys() == resumed.by_qual.keys()
    assert baseline.symbol_table.imports == resumed.symbol_table.imports
    assert baseline.class_index.keys() == resumed.class_index.keys()

def test_build_analysis_index_resume_stable_under_hydrated_path_reorder(tmp_path: Path) -> None:
    da = _load()
    module_a = tmp_path / "a.py"
    module_b = tmp_path / "b.py"
    _write(module_a, "def fa(x):\n    return x\n")
    _write(module_b, "def fb(y):\n    return y\n")
    progress_payloads: list[dict[str, object]] = []
    da._build_analysis_index(
        [module_a, module_b],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=set(),
        parse_failure_witnesses=[],
        on_progress=progress_payloads.append,
    )
    resume_payload = dict(progress_payloads[-1])
    hydrated = list(resume_payload.get("hydrated_paths", []))
    resume_payload["hydrated_paths"] = list(reversed(hydrated))
    calls = 0

    def _accumulate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return da._accumulate_function_index_for_tree(*args, **kwargs)

    da._build_analysis_index(
        [module_a, module_b],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=set(),
        parse_failure_witnesses=[],
        resume_payload=resume_payload,
        accumulate_function_index_for_tree_fn=_accumulate,
    )
    assert calls == 0


def test_build_analysis_index_resume_misses_on_semantic_key_change(tmp_path: Path) -> None:
    da = _load()
    module = tmp_path / "sample.py"
    _write(module, "def f(x):\n    return x\n")
    progress_payloads: list[dict[str, object]] = []
    da._build_analysis_index(
        [module],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=set(),
        parse_failure_witnesses=[],
        on_progress=progress_payloads.append,
        fingerprint_seed_revision="rev-1",
    )
    resume_payload = dict(progress_payloads[-1])
    calls = 0

    def _accumulate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return da._accumulate_function_index_for_tree(*args, **kwargs)

    da._build_analysis_index(
        [module],
        project_root=tmp_path,
        ignore_params={"x"},
        strictness="high",
        external_filter=True,
        transparent_decorators=set(),
        parse_failure_witnesses=[],
        resume_payload=resume_payload,
        accumulate_function_index_for_tree_fn=_accumulate,
        fingerprint_seed_revision="rev-2",
    )
    assert calls == 1

def test_collection_resume_payload_persists_file_stage_timings() -> None:
    da = _load()
    payload = da._build_analysis_collection_resume_payload(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        invariant_propositions=[],
        completed_paths=set(),
        in_progress_scan_by_path={},
        file_stage_timings_v1_by_path={
            Path("a.py"): {
                "format_version": 1,
                "stage_ns": {"file_scan.read_parse": 11},
                "counters": {"file_scan.functions_total": 1},
            }
        },
    )
    stored = payload.get("file_stage_timings_v1_by_path")
    assert isinstance(stored, dict)
    assert stored["a.py"]["stage_ns"]["file_scan.read_parse"] == 11


def test_analyze_paths_emits_profiling_v1(tmp_path: Path) -> None:
    da = _load()
    sample = tmp_path / "sample.py"
    _write(sample, "def f(a, b):\n    return a + b\n")
    analysis = da.analyze_paths(
        forest=da.Forest(),
        paths=[tmp_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_invariant_propositions=False,
        config=da.AuditConfig(project_root=tmp_path),
    )
    assert isinstance(analysis.profiling_v1, dict)
    assert analysis.profiling_v1.get("format_version") == 1
    assert "file_stage_timings_v1_by_path" in analysis.profiling_v1


def test_analyze_paths_hydrates_file_stage_timings_from_resume(tmp_path: Path) -> None:
    da = _load()
    sample = tmp_path / "sample.py"
    _write(sample, "def f(a, b):\n    return a + b\n")
    resume_payload = da._build_analysis_collection_resume_payload(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        invariant_propositions=[],
        completed_paths={sample},
        in_progress_scan_by_path={},
        file_stage_timings_v1_by_path={
            sample: {
                "format_version": 1,
                "stage_ns": {"file_scan.read_parse": 11},
                "counters": {"file_scan.functions_total": 1},
            }
        },
    )
    raw_timings = resume_payload.get("file_stage_timings_v1_by_path")
    assert isinstance(raw_timings, dict)
    sample_key = str(sample)
    assert isinstance(raw_timings.get(sample_key), dict)
    raw_timings[sample_key][7] = "sentinel"
    analysis = da.analyze_paths(
        forest=da.Forest(),
        paths=[tmp_path],
        file_paths_override=[sample],
        collection_resume=resume_payload,
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_invariant_propositions=False,
        config=da.AuditConfig(project_root=tmp_path),
    )
    assert isinstance(analysis.profiling_v1, dict)
    timings_by_path = analysis.profiling_v1.get("file_stage_timings_v1_by_path")
    assert isinstance(timings_by_path, dict)
    assert sample_key in timings_by_path
    assert timings_by_path[sample_key]["7"] == "sentinel"


def test_analyze_paths_ignores_non_mapping_stage_timing_entries(tmp_path: Path) -> None:
    da = _load()
    sample = tmp_path / "sample.py"
    _write(sample, "def f(a, b):\n    return a + b\n")
    resume_payload = da._build_analysis_collection_resume_payload(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        invariant_propositions=[],
        completed_paths={sample},
        in_progress_scan_by_path={},
        file_stage_timings_v1_by_path={},
    )
    resume_payload["file_stage_timings_v1_by_path"] = {str(sample): 5}
    analysis = da.analyze_paths(
        forest=da.Forest(),
        paths=[tmp_path],
        file_paths_override=[sample],
        collection_resume=resume_payload,
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_invariant_propositions=False,
        config=da.AuditConfig(project_root=tmp_path),
    )
    assert isinstance(analysis.profiling_v1, dict)
    timings_by_path = analysis.profiling_v1.get("file_stage_timings_v1_by_path")
    assert isinstance(timings_by_path, dict)
    assert str(sample) not in timings_by_path


def test_analyze_paths_ignores_non_mapping_stage_timing_table(tmp_path: Path) -> None:
    da = _load()
    sample = tmp_path / "sample.py"
    _write(sample, "def f(a, b):\n    return a + b\n")
    resume_payload = da._build_analysis_collection_resume_payload(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        invariant_propositions=[],
        completed_paths={sample},
        in_progress_scan_by_path={},
        file_stage_timings_v1_by_path={},
    )
    resume_payload["file_stage_timings_v1_by_path"] = []
    analysis = da.analyze_paths(
        forest=da.Forest(),
        paths=[tmp_path],
        file_paths_override=[sample],
        collection_resume=resume_payload,
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_invariant_propositions=False,
        config=da.AuditConfig(project_root=tmp_path),
    )
    assert isinstance(analysis.profiling_v1, dict)
    timings_by_path = analysis.profiling_v1.get("file_stage_timings_v1_by_path")
    assert isinstance(timings_by_path, dict)
    assert str(sample) not in timings_by_path

def test_build_collection_resume_rejects_path_order_regression() -> None:
    da = _load()
    with pytest.raises(NeverThrown):
        da._build_analysis_collection_resume_payload(
            groups_by_path={},
            param_spans_by_path={},
            bundle_sites_by_path={},
            invariant_propositions=[],
            completed_paths=set(),
            in_progress_scan_by_path={
                Path("b.py"): {"phase": "scan_pending"},
                Path("a.py"): {"phase": "scan_pending"},
            },
        )

def test_iter_monotonic_paths_rejects_path_order_regression() -> None:
    da = _load()
    with pytest.raises(NeverThrown):
        da._iter_monotonic_paths(
            [Path("b.py"), Path("a.py")],
            source="test",
        )

def test_iter_monotonic_paths_accepts_monotonic_order() -> None:
    da = _load()
    ordered = da._iter_monotonic_paths(
        [Path("a.py"), Path("b.py")],
        source="test",
    )
    assert ordered == [Path("a.py"), Path("b.py")]

def test_analyze_paths_rejects_unsorted_file_paths_override(tmp_path: Path) -> None:
    da = _load()
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    _write(first, "def a(x):\n    return x\n")
    _write(second, "def b(y):\n    return y\n")
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
    )
    with pytest.raises(NeverThrown):
        da.analyze_paths(
            forest=da.Forest(),
            paths=[tmp_path],
            recursive=True,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=0,
            include_constant_smells=False,
            include_unused_arg_smells=False,
            config=config,
            file_paths_override=[second, first],
        )

def test_extract_report_sections_parses_marked_sections() -> None:
    da = _load()
    report = "\n".join(
        [
            "<!-- report-section:intro -->",
            "header",
            "<!-- report-section:components -->",
            "component-a",
            "component-b",
            "<!-- report-section:violations -->",
            "violation-1",
        ]
    )
    sections = da.extract_report_sections(report)
    assert sections["intro"] == ["header"]
    assert sections["components"] == ["component-a", "component-b"]
    assert sections["violations"] == ["violation-1"]

def test_report_projection_specs_rows() -> None:
    da = _load()
    rows = da.report_projection_spec_rows()
    assert rows
    section_ids = {str(row.get("section_id", "")) for row in rows}
    assert "intro" in section_ids
    assert "components" in section_ids
    components_row = next(
        row for row in rows if str(row.get("section_id", "")) == "components"
    )
    assert components_row["phase"] == "forest"
    assert "intro" in (components_row.get("deps") or [])
    assert components_row["has_preview"] is True
    type_flow_row = next(
        row for row in rows if str(row.get("section_id", "")) == "type_flow"
    )
    assert type_flow_row["has_preview"] is True
    deadline_summary_row = next(
        row for row in rows if str(row.get("section_id", "")) == "deadline_summary"
    )
    assert deadline_summary_row["has_preview"] is True
    fingerprint_warnings_row = next(
        row for row in rows if str(row.get("section_id", "")) == "fingerprint_warnings"
    )
    assert fingerprint_warnings_row["has_preview"] is True

def test_report_projection_specs_are_topologically_ordered() -> None:
    da = _load()
    rows = da.report_projection_spec_rows()
    index_by_section = {
        str(row.get("section_id", "")): idx for idx, row in enumerate(rows)
    }
    for idx, row in enumerate(rows):
        section_id = str(row.get("section_id", ""))
        deps = row.get("deps", [])
        if not isinstance(deps, list):
            continue
        for dep in deps:
            if not isinstance(dep, str):
                continue
            assert dep in index_by_section, (section_id, dep)
            assert index_by_section[dep] < idx, (section_id, dep)

def test_project_report_sections_preview_only() -> None:
    da = _load()
    sections = da.project_report_sections(
        {},
        da.ReportCarrier(
            forest=da.Forest(),
            parse_failure_witnesses=[],
        ),
        max_phase="post",
        include_previews=True,
        preview_only=True,
    )
    assert "components" in sections
    assert "violations" in sections
    assert "type_flow" in sections
    assert "deadline_summary" in sections
    assert "constant_smells" in sections
    assert "unused_arg_smells" in sections
    assert "parse_failure_witnesses" in sections
    assert "fingerprint_warnings" in sections
    assert sections["components"][0].startswith("Component preview")
    assert sections["violations"][0].startswith("Violations preview")
    assert sections["constant_smells"][0].startswith(
        "Constant-propagation smells preview"
    )

def test_report_projection_phase_rank_order() -> None:
    da = _load()
    assert da.report_projection_phase_rank("collection") < da.report_projection_phase_rank(
        "forest"
    )
    assert da.report_projection_phase_rank("forest") < da.report_projection_phase_rank(
        "edge"
    )
    assert da.report_projection_phase_rank("edge") < da.report_projection_phase_rank(
        "post"
    )

def test_resume_map_harness_uses_injected_parser() -> None:
    da = _load()
    seen: list[object] = []

    def _parser(value: object) -> int | None:
        seen.append(value)
        if value == "keep":
            return 1
        return None

    out = da.load_resume_map(
        payload={"k1": "keep", "k2": "drop", "k3": "ignored"},
        valid_keys={"k1", "k2"},
        parser=_parser,
    )
    assert seen == ["keep", "drop"]
    assert out == {"k1": 1}

def test_iter_valid_resume_entries_and_str_sequence_helpers() -> None:
    da = _load()
    entries = list(
        da.iter_valid_key_entries(
            payload={"k1": "v1", "k2": 2, "k3": "v3"},
            valid_keys={"k1", "k3"},
        )
    )
    assert entries == [("k1", "v1"), ("k3", "v3")]
    assert da.str_list_from_sequence(["a", 1, "b"]) == ["a", "b"]
    assert da.str_list_from_sequence("bad") == []
    assert da.str_tuple_from_sequence(["a", 1, "b"]) == ("a", "b")

def test_deserialize_param_use_filters_malformed_values() -> None:
    da = _load()
    use = da._deserialize_param_use(
        {
            "direct_forward": [["callee", "slot"], ["bad"], [1, 2], ["callee2", 3]],
            "non_forward": 1,
            "unknown_key_carrier": True,
            "current_aliases": ["a", 2, "b"],
            "unknown_key_sites": [[9, 8, 7, 6], [1, 2, 3], [1, 2, 3, "x"]],
            "forward_sites": [
                {
                    "callee": "callee",
                    "slot": "slot",
                    "spans": [[1, 2, 3, 4], [1, 2, 3], [1, 2, 3, "x"], (4, 5, 6, 7)],
                },
                {"callee": 1, "slot": "x", "spans": []},
                "bad",
            ],
        }
    )
    assert ("callee", "slot") in use.direct_forward
    assert use.current_aliases == {"a", "b"}
    assert use.non_forward is True
    assert use.unknown_key_carrier is True
    assert use.unknown_key_sites == {(9, 8, 7, 6)}
    assert use.forward_sites[("callee", "slot")] == {(1, 2, 3, 4), (4, 5, 6, 7)}


def test_normalize_key_expr_unary_non_int_literal_is_none() -> None:
    da = _load()
    node = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1.5))
    assert da._normalize_key_expr(node, const_bindings={}) is None

def test_deserialize_call_args_handles_invalid_shapes() -> None:
    da = _load()
    assert da._deserialize_call_args({"callee": 1}) is None

    call = da._deserialize_call_args(
        {
            "callee": "mod.fn",
            "pos_map": {"a": "p", 1: "bad", "b": 2},
            "kw_map": {"k": "v"},
            "const_pos": {"a": "1"},
            "const_kw": {"k": "2"},
            "non_const_pos": ["x", 1],
            "non_const_kw": ["y", 2],
            "star_pos": [[0, "p"], ["bad", "q"], [1, 2], ["3", "r"], [9]],
            "star_kw": ["kw", 2],
            "is_test": 1,
            "span": [1, 2, 3, "x"],
        }
    )
    assert call is not None
    assert call.callee == "mod.fn"
    assert call.pos_map == {"a": "p"}
    assert call.non_const_pos == {"x"}
    assert call.non_const_kw == {"y"}
    assert call.star_pos == [(0, "p"), (3, "r")]
    assert call.star_kw == ["kw"]
    assert call.span is None
    assert call.is_test is True

def test_serialize_call_and_function_info_resume_omit_optional_spans(tmp_path: Path) -> None:
    da = _load()
    call = da.CallArgs(
        callee="m.f",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=None,
    )
    call_payload = da._serialize_call_args(call)
    assert "span" not in call_payload

    info = da.FunctionInfo(
        name="f",
        qual="m.f",
        path=tmp_path / "m.py",
        params=["a"],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=None,
    )
    info_payload = da._serialize_function_info_for_resume(info)
    assert "function_span" not in info_payload

def test_deserialize_call_args_list_skips_non_call_mappings() -> None:
    da = _load()
    calls = da._deserialize_call_args_list(
        [
            {"callee": "m.ok"},
            {"callee": 1},
            "bad",
        ]
    )
    assert [call.callee for call in calls] == ["m.ok"]

def test_deserialize_function_info_for_resume_filters_malformed_fields(tmp_path: Path) -> None:
    da = _load()
    allowed_paths = {"m.py": tmp_path / "m.py"}
    assert (
        da._deserialize_function_info_for_resume(
            {"name": "f", "qual": "m.f", "path": "m.py", "params": "bad"},
            allowed_paths=allowed_paths,
        )
        is None
    )
    info = da._deserialize_function_info_for_resume(
        {
            "name": "f",
            "qual": "m.f",
            "path": "m.py",
            "params": ["a", 1],
            "annots": {"a": "int", "b": None, 1: "bad", "c": 3},
            "calls": [{"callee": "m.g"}, "bad"],
            "unused_params": ["u", 1],
            "defaults": ["d", 1],
            "transparent": False,
            "class_name": 123,
            "scope": ["S", 1],
            "lexical_scope": ["L", 1],
            "decision_params": ["x", 1],
            "value_decision_params": ["y", 1],
            "value_decision_reasons": ["r", 1],
            "positional_params": ["p", 1],
            "kwonly_params": ["k", 1],
            "vararg": 2,
            "kwarg": 3,
            "param_spans": {
                "a": [1, 2, 3, 4],
                "badlen": [1, 2, 3],
                "badtype": [1, 2, 3, "x"],
            },
            "function_span": [1, 2, 3, "x"],
        },
        allowed_paths=allowed_paths,
    )
    assert info is not None
    assert info.path == tmp_path / "m.py"
    assert info.params == ["a"]
    assert info.annots == {"a": "int", "b": None}
    assert len(info.calls) == 1
    assert info.class_name is None
    assert info.scope == ("S",)
    assert info.lexical_scope == ("L",)
    assert info.decision_params == {"x"}
    assert info.value_decision_params == {"y"}
    assert info.value_decision_reasons == {"r"}
    assert info.positional_params == ("p",)
    assert info.kwonly_params == ("k",)
    assert info.vararg is None
    assert info.kwarg is None
    assert info.param_spans == {"a": (1, 2, 3, 4)}
    assert info.function_span is None

def test_deserialize_symbol_table_for_resume_filters_malformed_entries() -> None:
    da = _load()
    table = da._deserialize_symbol_table_for_resume(
        {
            "external_filter": 0,
            "imports": [["m", "n", "m.n"], ["bad"], [1, 2, 3]],
            "internal_roots": ["pkg", 1],
            "star_imports": {"mod": ["a", 1], 2: ["x"]},
            "module_exports": {"mod": ["x", 1], 2: ["y"]},
            "module_export_map": {"mod": {"a": "b", "x": 1}, 2: {"q": "r"}},
        }
    )
    assert table.external_filter is False
    assert table.imports == {("m", "n"): "m.n"}
    assert table.internal_roots == {"pkg"}
    assert table.star_imports == {"mod": {"a"}}
    assert table.module_exports == {"mod": {"x"}}
    assert table.module_export_map == {"mod": {"a": "b"}}

def test_load_file_scan_resume_state_handles_invalid_shapes() -> None:
    da = _load()
    empty = ({}, {}, {}, {}, {}, {}, {}, set())
    assert da._load_file_scan_resume_state(payload=None, valid_fn_keys=set()) == empty
    assert (
        da._load_file_scan_resume_state(
            payload={"phase": "wrong"},
            valid_fn_keys={"f"},
        )
        == empty
    )
    assert (
        da._load_file_scan_resume_state(
            payload={
                "phase": "function_scan",
                "fn_use": [],
                "fn_calls": {},
                "fn_param_orders": {},
                "fn_param_spans": {},
                "fn_names": {},
                "fn_lexical_scopes": {},
                "fn_class_names": {},
            },
            valid_fn_keys={"f"},
        )
        == empty
    )

def test_load_file_scan_resume_state_parses_valid_entries() -> None:
    da = _load()
    payload = {
        "phase": "function_scan",
        "fn_use": {"f": {"p": {"direct_forward": [["g", "x"]]}}},
        "fn_calls": {"f": [{"callee": "m.g"}], "other": [{"callee": "ignored"}]},
        "fn_param_orders": {"f": ["a", 1], "other": ["x"]},
        "fn_param_spans": {"f": {"a": [1, 2, 3, 4]}},
        "fn_names": {"f": "name", "other": "ignored"},
        "fn_lexical_scopes": {"f": ["L", 1], "other": ["X"]},
        "fn_class_names": {"f": None, "other": "C"},
        "opaque_callees": ["f", "other", 1],
    }
    (
        fn_use,
        fn_calls,
        fn_param_orders,
        fn_param_spans,
        fn_names,
        fn_lexical_scopes,
        fn_class_names,
        opaque_callees,
    ) = da._load_file_scan_resume_state(payload=payload, valid_fn_keys={"f"})
    assert set(fn_use) == {"f"}
    assert set(fn_calls) == {"f"}
    assert fn_param_orders == {"f": ["a"]}
    assert fn_param_spans == {"f": {"a": (1, 2, 3, 4)}}
    assert fn_names == {"f": "name"}
    assert fn_lexical_scopes == {"f": ("L",)}
    assert fn_class_names == {"f": None}
    assert opaque_callees == {"f"}

def test_load_file_scan_resume_state_filters_class_name_and_opaque_shapes() -> None:
    da = _load()
    payload = {
        "phase": "function_scan",
        "fn_use": {},
        "fn_calls": {},
        "fn_param_orders": {},
        "fn_param_spans": {},
        "fn_names": {},
        "fn_lexical_scopes": {},
        "fn_class_names": {"f": 3},
        "opaque_callees": {"f": True},
    }
    result = da._load_file_scan_resume_state(payload=payload, valid_fn_keys={"f"})
    assert result[6] == {}
    assert result[7] == set()

def test_deserialize_invariants_for_resume_filters_malformed_entries() -> None:
    da = _load()
    invariants = da._deserialize_invariants_for_resume(
        [
            {"form": "eq", "terms": ["a", 1], "scope": "s", "source": "src"},
            {"form": "bad", "terms": "not-seq"},
            {"terms": ["x"]},
            "bad",
        ]
    )
    assert len(invariants) == 1
    invariant = invariants[0]
    assert invariant.form == "eq"
    assert invariant.terms == ("a",)
    assert invariant.scope == "s"
    assert invariant.source == "src"

def test_load_analysis_collection_resume_payload_invalid_shapes() -> None:
    da = _load()
    empty = ({}, {}, {}, [], set(), {}, None)
    assert (
        da._load_analysis_collection_resume_payload(
            payload=None,
            file_paths=[],
            include_invariant_propositions=False,
        )
        == empty
    )
    assert (
        da._load_analysis_collection_resume_payload(
            payload={"format_version": 0},
            file_paths=[],
            include_invariant_propositions=False,
        )
        == empty
    )
    assert (
        da._load_analysis_collection_resume_payload(
            payload={
                "format_version": 2,
                "groups_by_path": [],
                "param_spans_by_path": {},
                "bundle_sites_by_path": {},
                "in_progress_scan_by_path": {},
            },
            file_paths=[],
            include_invariant_propositions=False,
        )
        == empty
    )
    assert (
        da._load_analysis_collection_resume_payload(
            payload={
                "format_version": 2,
                "groups_by_path": {},
                "param_spans_by_path": [],
                "bundle_sites_by_path": {},
                "in_progress_scan_by_path": {},
            },
            file_paths=[],
            include_invariant_propositions=False,
        )
        == empty
    )

def test_load_analysis_collection_resume_payload_filters_entries(tmp_path: Path) -> None:
    da = _load()
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    payload = {
        "format_version": 2,
        "completed_paths": [str(first), str(second), "missing.py", 1],
        "groups_by_path": {
            str(first): {"fn": [["a"]]},
            str(second): [],
        },
        "param_spans_by_path": {
            str(first): {"fn": {"a": [1, 2, 3, 4]}},
            str(second): [],
        },
        "bundle_sites_by_path": {
            str(first): {"fn": [[{"kind": "k"}]]},
            str(second): [],
        },
        "in_progress_scan_by_path": {
            str(first): {"phase": "scan_pending"},
            str(second): {"phase": "function_scan"},
            "missing.py": {"phase": "scan_pending"},
            str(tmp_path / "bad.py"): "bad",
        },
        "invariant_propositions": [
            {"form": "eq", "terms": ["a"]},
            "bad",
        ],
        "analysis_index_resume": {"phase": "analysis_index_hydration"},
    }
    (
        groups_by_path,
        param_spans_by_path,
        bundle_sites_by_path,
        invariant_propositions,
        completed_paths,
        in_progress_scan_by_path,
        analysis_index_resume,
    ) = da._load_analysis_collection_resume_payload(
        payload=payload,
        file_paths=[first, second],
        include_invariant_propositions=True,
    )
    assert completed_paths == {first}
    assert groups_by_path[first]["fn"] == [{"a"}]
    assert param_spans_by_path[first]["fn"]["a"] == (1, 2, 3, 4)
    assert bundle_sites_by_path[first]["fn"][0][0]["kind"] == "k"
    assert second in in_progress_scan_by_path
    assert first not in in_progress_scan_by_path
    assert len(invariant_propositions) == 1
    assert isinstance(analysis_index_resume, dict)
    assert analysis_index_resume["phase"] == "analysis_index_hydration"

def test_load_analysis_collection_resume_payload_handles_non_sequence_completion_and_invariants(
    tmp_path: Path,
) -> None:
    da = _load()
    path = tmp_path / "a.py"
    payload = {
        "format_version": 2,
        "completed_paths": {"not": "a-sequence"},
        "groups_by_path": {},
        "param_spans_by_path": {},
        "bundle_sites_by_path": {},
        "invariant_propositions": {"bad": True},
    }
    loaded = da._load_analysis_collection_resume_payload(
        payload=payload,
        file_paths=[path],
        include_invariant_propositions=True,
    )
    assert loaded[4] == set()
    assert loaded[3] == []

def test_runtime_obligation_violation_lines_and_preview_helpers() -> None:
    da = _load()
    obligations = [
        {
            "status": "SATISFIED",
            "contract": "resume_contract",
            "kind": "ok",
            "detail": "done",
        },
        {
            "status": "VIOLATION",
            "contract": "resume_contract",
            "kind": "missing_checkpoint",
            "section_id": "intro",
            "phase": "collection",
            "detail": "checkpoint missing",
        },
    ]
    violations = da._runtime_obligation_violation_lines(obligations)
    assert violations == [
        "resume_contract missing_checkpoint section=intro phase=collection detail=checkpoint missing"
    ]
    lines = da._preview_runtime_obligations_section(
        title="Resumability obligations",
        obligations=obligations,
    )
    assert lines[0] == "Resumability obligations preview (provisional)."
    assert any("`violations`: `1`" in line for line in lines)
    assert any("sample_violation" in line for line in lines)

def test_known_violation_and_preview_violations_sections() -> None:
    da = _load()
    report = da.ReportCarrier(
        forest=da.Forest(),
        parse_failure_witnesses=[
            {
                "path": "a.py",
                "stage": "parse",
                "error_type": "SyntaxError",
                "error": "bad",
            }
        ],
        decision_warnings=["warn"],
        fingerprint_warnings=["warn"],
        resumability_obligations=[
            {
                "status": "VIOLATION",
                "contract": "resume_contract",
                "kind": "missing",
            }
        ],
    )
    known = da._known_violation_lines(report)
    assert any("resume_contract missing" in line for line in known)
    assert any("parse_failure" in line for line in known)
    assert known.count("warn") == 1

    preview = da._preview_violations_section(report, {})
    assert preview[0] == "Violations preview (provisional)."
    assert any("known_violations" in line for line in preview)
    assert any(line.startswith("- ") for line in preview[2:])

    empty_report = da.ReportCarrier(forest=da.Forest(), parse_failure_witnesses=[])
    empty_preview = da._preview_violations_section(empty_report, {})
    assert "- none observed yet" in empty_preview

def test_preview_parse_failure_witnesses_section_counts_stage() -> None:
    da = _load()
    report = da.ReportCarrier(
        forest=da.Forest(),
        parse_failure_witnesses=[
            {"path": "a.py", "stage": "parse"},
            {"path": "b.py", "stage": ""},
            {"path": "c.py"},
        ],
    )
    lines = da._preview_parse_failure_witnesses_section(report, {})
    assert lines[0] == "Parse failure witnesses preview (provisional)."
    assert any("stage[parse]" in line for line in lines)
    assert any("stage[unknown]" in line for line in lines)

def test_load_analysis_index_resume_payload_filters_entries(tmp_path: Path) -> None:
    da = _load()
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    hydrated_paths, by_qual, symbol_table, class_index = da._load_analysis_index_resume_payload(
        payload=None,
        file_paths=[first, second],
    )
    assert hydrated_paths == set()
    assert by_qual == {}
    assert symbol_table.imports == {}
    assert class_index == {}

    payload = {
        "format_version": 1,
        "hydrated_paths": [str(first), "missing.py", 1],
        "functions_by_qual": {
            "m.f": {
                "name": "f",
                "qual": "m.f",
                "path": str(first),
                "params": ["a"],
                "annots": {"a": "int"},
                "calls": [{"callee": "m.g"}],
                "unused_params": ["u"],
                "defaults": [],
                "transparent": True,
                "class_name": None,
                "scope": [],
                "lexical_scope": [],
                "decision_params": [],
                "value_decision_params": [],
                "value_decision_reasons": [],
                "positional_params": [],
                "kwonly_params": [],
                "vararg": None,
                "kwarg": None,
                "param_spans": {"a": [1, 2, 3, 4]},
            },
            "bad": "skip",
        },
        "symbol_table": {
            "imports": [["m", "n", "m.n"]],
            "internal_roots": ["pkg"],
            "external_filter": True,
            "star_imports": {},
            "module_exports": {},
            "module_export_map": {},
        },
        "class_index": {
            "m.C": {"qual": "m.C", "module": "m", "bases": [], "methods": ["x"]},
            "bad": "skip",
        },
    }
    hydrated_paths, by_qual, symbol_table, class_index = da._load_analysis_index_resume_payload(
        payload=payload,
        file_paths=[first, second],
    )
    assert hydrated_paths == {first}
    assert set(by_qual) == {"m.f"}
    assert symbol_table.imports == {("m", "n"): "m.n"}
    assert set(class_index) == {"m.C"}

def test_report_projection_spec_topology_guards() -> None:
    da = _load()
    first = da._report_section_spec(section_id="intro", phase="collection")
    second = da._report_section_spec(
        section_id="components",
        phase="forest",
        deps=("intro",),
    )
    ordered = da._topologically_order_report_projection_specs((second, first))
    assert [spec.section_id for spec in ordered] == ["intro", "components"]

    with pytest.raises(NeverThrown):
        da._topologically_order_report_projection_specs(
            (
                da._report_section_spec(section_id="x", phase="collection", deps=("missing",)),
            )
        )
    with pytest.raises(NeverThrown):
        da._topologically_order_report_projection_specs(
            (
                da._report_section_spec(section_id="x", phase="collection", deps=("x",)),
            )
        )
    with pytest.raises(NeverThrown):
        da._topologically_order_report_projection_specs(
            (
                da._report_section_spec(section_id="x", phase="collection"),
                da._report_section_spec(section_id="x", phase="forest"),
            )
        )
    with pytest.raises(NeverThrown):
        da._topologically_order_report_projection_specs(
            (
                da._report_section_spec(section_id="x", phase="collection", deps=("y",)),
                da._report_section_spec(section_id="y", phase="forest", deps=("x",)),
            )
        )


def test_report_projection_spec_topology_stable_for_unsorted_deps() -> None:
    da = _load()
    root = da._report_section_spec(section_id="root", phase="collection")
    left = da._report_section_spec(
        section_id="left",
        phase="forest",
        deps=("root",),
    )
    right = da._report_section_spec(
        section_id="right",
        phase="forest",
        deps=("root",),
    )
    sink = da._report_section_spec(
        section_id="sink",
        phase="post",
        deps=("right", "left", "left"),
    )

    ordered = da._topologically_order_report_projection_specs((sink, right, left, root))

    assert [spec.section_id for spec in ordered] == ["root", "right", "left", "sink"]

def test_report_preview_helpers_cover_samples() -> None:
    da = _load()
    report = da.ReportCarrier(
        forest=da.Forest(),
        parse_failure_witnesses=[],
        type_ambiguities=["ambiguous[x]"],
        type_suggestions=["s"],
        type_callsite_evidence=["e"],
        constant_smells=["const"],
        deadline_obligations=[{"kind": "k", "status": "VIOLATION", "detail": "d"}],
        resumability_obligations=[{"status": "PENDING", "contract": "resume", "kind": "k"}],
    )
    type_preview = da._preview_type_flow_section(report, {})
    assert any("sample_type_ambiguity" in line for line in type_preview)

    deadline_preview = da._preview_deadline_summary_section(report, {})
    assert deadline_preview[0] == "Deadline propagation preview (provisional)."
    empty_deadline_preview = da._preview_deadline_summary_section(
        da.ReportCarrier(forest=da.Forest(), parse_failure_witnesses=[]),
        {},
    )
    assert "- no deadline obligations yet" in empty_deadline_preview

    const_preview = da._preview_constant_smells_section(report, {})
    assert any("sample_constant_smell" in line for line in const_preview)

    obligations_preview = da._preview_runtime_obligations_section(
        title="Resumability obligations",
        obligations=report.resumability_obligations,
    )
    assert any("`pending`: `1`" in line for line in obligations_preview)
    assert da._report_section_no_violations(["x"]) == []

def test_parse_witness_contract_violations_read_and_parse_errors(tmp_path: Path) -> None:
    da = _load()
    missing = da._parse_witness_contract_violations(source_path=tmp_path / "missing.py")
    assert missing and "read_error" in missing[0]
    parse_error = da._parse_witness_contract_violations(
        source="def broken(:\n",
        source_path=tmp_path / "broken.py",
    )
    assert parse_error and "parse_error" in parse_error[0]

def test_parse_witness_contract_violations_missing_helper_and_param(tmp_path: Path) -> None:
    da = _load()
    source = (
        "def helper(parse_failure_witnesses: list[dict]):\n"
        "    return None\n"
        "def missing_param(x: int):\n"
        "    return x\n"
    )
    violations = da._parse_witness_contract_violations(
        source=source,
        source_path=tmp_path / "module.py",
        target_helpers=frozenset({"helper", "missing", "missing_param"}),
    )
    assert any("missing helper definition" in line for line in violations)
    assert any("missing parse_failure_witnesses" in line for line in violations)

def test_annotation_allows_none_and_parameter_default_map_edges() -> None:
    da = _load()
    assert da._annotation_allows_none(None) is True
    assert da._annotation_allows_none(ast.parse("Optional[int]").body[0].value) is True
    assert da._annotation_allows_none(ast.parse("int").body[0].value) is False

    fn = ast.parse(
        "def f(a, b=1, *, c=None, d=2):\n"
        "    return a\n"
    ).body[0]
    mapping = da._parameter_default_map(fn)
    assert mapping["b"] is not None
    assert mapping["c"] is not None
    assert mapping["d"] is not None

def test_raw_sorted_contract_violations_strict_and_baseline(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "m.py"
    _write(path, "def f(xs):\n    return sorted(xs)\n")
    baseline_key = da._raw_sorted_baseline_key(path)
    assert baseline_key.endswith("m.py")

    strict = da._raw_sorted_contract_violations(
        [path],
        parse_failure_witnesses=[],
        strict_forbid=True,
    )
    assert any("raw sorted() forbidden" in line for line in strict)

    exceeds = da._raw_sorted_contract_violations(
        [path],
        parse_failure_witnesses=[],
        baseline_counts={baseline_key: 0},
    )
    assert any("raw_sorted exceeded baseline" in line for line in exceeds)

def test_report_projection_render_paths_and_dedup_dep_edges(tmp_path: Path) -> None:
    da = _load()
    report = da.ReportCarrier(forest=da.Forest(), parse_failure_witnesses=[])
    groups = {tmp_path / "m.py": {"f": [{"x"}]}}
    section_lines = da._report_section_text(report, groups, section_id="intro")
    assert isinstance(section_lines, list)
    assert da.report_projection_specs()
    rendered = da.project_report_sections(
        groups,
        report,
        max_phase="collection",
        include_previews=False,
        preview_only=False,
    )
    assert "components" not in rendered

    ordered = da._topologically_order_report_projection_specs(
        (
            da._report_section_spec(section_id="root", phase="collection"),
            da._report_section_spec(
                section_id="child",
                phase="forest",
                deps=("root", "root"),
            ),
        )
    )
    assert [spec.section_id for spec in ordered] == ["root", "child"]

def test_decision_surface_indexed_rewrite_guard_and_annotation_unparse_failure() -> None:
    da = _load()

    def _patched_run(*_args, **_kwargs):
        return (["s"], [], ["rewrite"], [])

    with pytest.raises(NeverThrown):
        da._analyze_decision_surfaces_indexed(
            da._IndexedPassContext(
                paths=[],
                project_root=None,
                ignore_params=set(),
                strictness="high",
                external_filter=True,
                transparent_decorators=None,
                parse_failure_witnesses=[],
                analysis_index=da.AnalysisIndex(
                    by_name={},
                    by_qual={},
                    symbol_table=da.SymbolTable(),
                    class_index={},
                ),
            ),
            decision_tiers=None,
            require_tiers=False,
            forest=da.Forest(),
            run_fn=_patched_run,
        )

    def _raise_unparse(*_args: object, **_kwargs: object) -> str:
        raise ValueError("boom")

    assert (
        da._annotation_allows_none(
            ast.parse("int").body[0].value,
            unparse_fn=_raise_unparse,
        )
        is True
    )

def test_raw_sorted_key_and_callsite_count_non_py(tmp_path: Path) -> None:
    da = _load()
    src_path = tmp_path / "src" / "pkg" / "mod.py"
    src_path.parent.mkdir(parents=True)
    src_path.write_text("def f(xs):\n    return sorted(xs)\n")
    assert da._raw_sorted_baseline_key(src_path).startswith("src/")
    txt = tmp_path / "notes.txt"
    txt.write_text("sorted([1])\n")
    counts = da._raw_sorted_callsite_counts([txt], parse_failure_witnesses=[])
    assert counts == {}

def test_detect_execution_pattern_matches_read_and_parse_and_filter_paths(
    tmp_path: Path,
) -> None:
    da = _load()
    missing = da._detect_execution_pattern_matches(source=None, source_path=tmp_path / "missing.py")
    assert missing == []
    parse_fail = da._detect_execution_pattern_matches(
        source="def broken(:\n",
        source_path=tmp_path / "broken.py",
    )
    assert parse_fail == []
    filtered = da._detect_execution_pattern_matches(
        source=(
            "x = 1\n"
            "def f(a):\n"
            "    return a\n"
            "def g(paths, project_root, ignore_params, strictness, external_filter, transparent_decorators, parse_failure_witnesses, analysis_index):\n"
            "    return 1\n"
        ),
        source_path=tmp_path / "m.py",
    )
    assert filtered == []

def test_lint_and_report_section_projection_edge_filters() -> None:
    da = _load()
    assert da._parse_lint_remainder("") == ("GABION_UNKNOWN", "")
    rows = da._lint_rows_from_lines(
        ["no-location", "a.py:1:2: CODE message"],
        source="src",
    )
    assert len(rows) == 1

    forest = da.Forest()
    da._materialize_lint_rows(
        forest=forest,
        rows=[
            {"path": "", "line": 1, "col": 1, "code": "X", "message": "m"},
            {"path": "a.py", "line": "bad", "col": 1, "code": "X", "message": "m"},
            {"path": "a.py", "line": 1, "col": 1, "code": "", "message": "m"},
            {"path": "a.py", "line": 1, "col": 1, "code": "X", "message": "m"},
        ],
    )
    relation = da._lint_relation_from_forest(forest)
    assert relation

    assert da._project_lint_rows_from_forest(
        forest=da.Forest(),
        relation_fn=lambda _forest: [
            {
                "path": "",
                "line": 1,
                "col": 1,
                "code": "X",
                "message": "",
                "sources": [],
            }
        ],
        apply_spec_fn=lambda _spec, relation: relation,
    ) == [
        {"path": "", "line": 1, "col": 1, "code": "X", "message": "", "sources": []}
    ]

    key = da._ReportSectionKey(run_id="r", section="s")
    section_forest = da.Forest()
    assert da._project_report_section_lines(forest=section_forest, section_key=key, lines=[]) == []

def test_suite_order_and_suite_span_and_async_for_materialization(tmp_path: Path) -> None:
    da = _load()
    forest = da.Forest()
    da._materialize_suite_order_spec(forest=forest)
    assert not forest.alts

    assert da._suite_span_from_statements([]) is None
    expr_only = ast.parse("1\n").body
    assert da._suite_span_from_statements(expr_only) is not None
    body = ast.parse("x = 1\ny = 2\n").body
    span = da._suite_span_from_statements(body)
    assert span is not None

    module = ast.parse(
        "async def f():\n"
        "    async for item in xs:\n"
        "        y = item\n"
        "    else:\n"
        "        z = 1\n"
    )
    async_fn = module.body[0]
    async_for = async_fn.body[0]
    parent = forest.add_suite_site("m.py", "m.f", "function_body", span=(1, 1, 4, 1))
    da._materialize_statement_suite_contains(
        forest=forest,
        path_name="m.py",
        qual="m.f",
        statements=[async_for],
        parent_suite=parent,
    )
    assert any(alt.kind == "SuiteContains" for alt in forest.alts)

def test_materialize_statement_suite_contains_handles_all_statement_kinds() -> None:
    da = _load()
    tree = ast.parse(
        "def f(xs):\n"
        "    if xs:\n"
        "        a = 1\n"
        "    else:\n"
        "        a = 2\n"
        "    for item in xs:\n"
        "        b = item\n"
        "    else:\n"
        "        b = 0\n"
        "    while False:\n"
        "        c = 1\n"
        "    else:\n"
        "        c = 2\n"
        "    try:\n"
        "        d = 1\n"
        "    except Exception:\n"
        "        d = 2\n"
        "    else:\n"
        "        d = 3\n"
        "    finally:\n"
        "        d = 4\n"
        "    pass\n"
    )
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    forest = da.Forest()
    parent = forest.add_suite_site("m.py", "m.f", "function_body", span=(1, 1, 24, 1))
    da._materialize_statement_suite_contains(
        forest=forest,
        path_name="m.py",
        qual="m.f",
        statements=fn.body,
        parent_suite=parent,
    )
    suite_kinds = {
        str(node.meta.get("suite_kind", ""))
        for node in forest.nodes.values()
        if node.node_id.kind == "SuiteSite"
    }
    expected_kinds = {
        "if_body",
        "if_else",
        "for_body",
        "for_else",
        "while_body",
        "while_else",
        "try_body",
        "except_body",
        "try_else",
        "try_finally",
    }
    assert expected_kinds.issubset(suite_kinds)
    assert any(alt.kind == "SuiteContains" for alt in forest.alts)

def test_parse_failure_and_runtime_summary_edges() -> None:
    da = _load()
    assert da._summarize_parse_failure_witnesses([]) == []
    lines = da._summarize_parse_failure_witnesses(
        [
            {"path": "a.py", "stage": "parse", "error_type": "SyntaxError", "error": "bad"},
            {"path": "b.py", "stage": "parse", "error_type": "ValueError"},
            {"path": "c.py", "stage": "parse", "error_type": "TypeError"},
        ],
        max_entries=2,
    )
    assert any("more" in line for line in lines)
    violation_lines = da._parse_failure_violation_lines(
        [{"path": "a.py", "stage": "s", "error_type": "E", "error": "x"}]
    )
    assert any("parse_failure" in line for line in violation_lines)
    assert da._summarize_runtime_obligations([]) == []
    runtime = da._summarize_runtime_obligations(
        [{"contract": "c", "kind": "k", "status": "SATISFIED", "detail": "d"}] * 3,
        max_entries=2,
    )
    assert any("more" in line for line in runtime)

def test_resume_deserialize_helpers_cover_invalid_rows(tmp_path: Path) -> None:
    da = _load()
    assert da._parse_report_section_marker("no marker") is None
    assert da._parse_report_section_marker("<!-- report-section: -->") is None
    extracted = da.extract_report_sections("line")
    assert extracted == {}

    tree = ast.parse("def f(a):\n    return a\n")
    marker_lines = [
        "<!-- report-section:intro -->",
        "ok",
    ]
    assert da.extract_report_sections("\n".join(marker_lines)) == {"intro": ["ok"]}

    assert da._deserialize_param_use_map({"x": 1}) == {}
    allowed = {str(tmp_path / "m.py"): tmp_path / "m.py"}
    assert da._deserialize_function_info_for_resume({}, allowed_paths=allowed) is None
    assert da._deserialize_function_info_for_resume(
        {
            "name": "f",
            "qual": "m.f",
            "path": "missing.py",
            "params": [],
        },
        allowed_paths=allowed,
    ) is None
    info = da._deserialize_function_info_for_resume(
        {
            "name": "f",
            "qual": "m.f",
            "path": str(tmp_path / "m.py"),
            "params": [],
            "param_spans": {"x": ["bad", 1, 2, 3], 1: [1, 2, 3, 4]},
        },
        allowed_paths=allowed,
    )
    assert info is not None
    assert info.param_spans == {}
    assert da._deserialize_class_info_for_resume({"qual": 1, "module": "m"}) is None

def test_resume_payload_loaders_and_serializers_cover_edges(tmp_path: Path) -> None:
    da = _load()
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    hydrated_paths, by_qual, symbol_table, class_index = da._load_analysis_index_resume_payload(
        payload={"format_version": 0},
        file_paths=[first, second],
    )
    assert hydrated_paths == set()
    assert by_qual == {}
    assert symbol_table.imports == {}
    assert class_index == {}

    payload = da._load_analysis_collection_resume_payload(
        payload={"in_progress_scan_by_path": None, "analysis_index_resume": {"bad": 1}},
        file_paths=[first],
        include_invariant_propositions=False,
    )
    assert len(payload) == 7

    assert da._deserialize_groups_for_resume({1: []}) == {}
    assert da._deserialize_param_spans_for_resume(
        {"f": {"x": [1, 2, 3], 1: [1, 2, 3, 4]}}
    ) == {"f": {}}
    serialized_sites = da._serialize_bundle_sites_for_resume({"f": [[{"kind": "k"}, "bad"]]})
    assert serialized_sites == {"f": [[{"kind": "k"}]]}
    assert da._deserialize_bundle_sites_for_resume({"f": [["bad"]]}) == {"f": [[]]}
    assert da._serialize_invariants_for_resume(
        [da.InvariantProposition(form="Equal", terms=("a", "b"), scope="s", source="src")]
    )

def test_analysis_index_cache_and_build_edges(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "m.py"
    path.write_text("def f():\n    return 1\n")
    assert da._build_module_artifacts(
        [path],
        specs=(),
        parse_failure_witnesses=[],
    ) == ()

    index = da.AnalysisIndex(by_name={}, by_qual={}, symbol_table=da.SymbolTable(), class_index={})
    spec = da._StageCacheSpec(
        stage=da._ParseModuleStage.FUNCTION_INDEX,
        cache_key=("k",),
        build=lambda _tree, _path: "ok",
    )
    assert da._analysis_index_stage_cache(
        index,
        [path],
        spec=spec,
        parse_failure_witnesses=[],
        module_trees_fn=lambda *_args, **_kwargs: {path: None},
    ) == {path: None}

    index.resolved_transparent_edges_by_caller = {"m.f": ()}
    assert da._analysis_index_resolved_call_edges_by_caller(
        index,
        project_root=None,
        require_transparent=True,
    ) == {"m.f": ()}

def test_deadline_function_facts_cache_and_tree_path_edges(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "m.py"
    path.write_text("def f(deadline):\n    return deadline\n")
    index = da.AnalysisIndex(by_name={}, by_qual={}, symbol_table=da.SymbolTable(), class_index={})
    assert da._collect_deadline_function_facts(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        parse_failure_witnesses=[],
        analysis_index=index,
        stage_cache_fn=lambda *_args, **_kwargs: {path: None},
    ) == {}
    facts = da._collect_deadline_function_facts(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        parse_failure_witnesses=[],
        trees={path: ast.parse(path.read_text())},
    )
    assert isinstance(facts, dict)

def test_call_edge_helper_filters_and_report_line_filters() -> None:
    da = _load()
    forest = da.Forest()
    missing_suite = da.NodeId("SuiteSite", ("p", "q", "call"))
    forest.add_alt("CallCandidate", (missing_suite, da.NodeId("FunctionSite", ("p", "q"))))
    assert da._collect_call_edges_from_forest(forest, by_name={}) == {}

    bad_call_suite = forest.add_suite_site("p.py", "", "call", span=(1, 1, 1, 2))
    forest.add_alt("CallResolutionObligation", (bad_call_suite,), evidence={"callee": "x"})
    with pytest.raises(NeverThrown):
        da._collect_call_resolution_obligations_from_forest(forest)

    key = da._ReportSectionKey(run_id="r", section="s")
    report_forest = da.Forest()
    report_node = report_forest.add_node("Other", ("x",), {})
    report_forest.add_alt("ReportSectionLine", (da.NodeId("FileSite", ("<report>",)), report_node), evidence={"run_id": "r", "section": "s"})
    assert da._report_section_line_relation(forest=report_forest, section_key=key) == []

def test_materialize_call_candidates_span_requirements_and_duplicate_edges() -> None:
    da = _load()
    candidate = da.FunctionInfo(
        name="target",
        qual="pkg.target",
        path=Path("pkg/target.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.caller",
        path=Path("pkg/mod.py"),
        params=[],
        annots={},
        calls=[da.CallArgs(callee="x", pos_map={}, kw_map={}, const_pos={}, const_kw={}, non_const_pos=set(), non_const_kw=set(), star_pos=[], star_kw=[], is_test=False, span=None)],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    forest = da.Forest()

    def _internal(*_args, **_kwargs):
        return da._CalleeResolutionOutcome(
            status="unresolved_internal",
            phase="internal",
            callee_key="x",
            candidates=(candidate,),
        )

    with pytest.raises(NeverThrown):
        da._materialize_call_candidates(
            forest=forest,
            by_name={"caller": [caller]},
            by_qual={caller.qual: caller},
            symbol_table=da.SymbolTable(),
            project_root=Path("."),
            class_index={},
            resolve_callee_outcome_fn=_internal,
        )

    caller.calls = [
        da.CallArgs(callee="x", pos_map={}, kw_map={}, const_pos={}, const_kw={}, non_const_pos=set(), non_const_kw=set(), star_pos=[], star_kw=[], is_test=False, span=(1, 1, 1, 2)),
        da.CallArgs(callee="x", pos_map={}, kw_map={}, const_pos={}, const_kw={}, non_const_pos=set(), non_const_kw=set(), star_pos=[], star_kw=[], is_test=False, span=(1, 1, 1, 2)),
    ]
    seen_forest = da.Forest()
    suite = seen_forest.add_suite_site("mod.py", "pkg.caller", "call", span=(1, 1, 1, 2))
    target_site = da._call_candidate_target_site(forest=seen_forest, candidate=candidate)
    seen_forest.add_alt("CallCandidate", (suite, target_site))

    def _resolved(*_args, **_kwargs):
        return da._CalleeResolutionOutcome(
            status="resolved",
            phase="resolved",
            callee_key="x",
            candidates=(candidate,),
        )

    da._materialize_call_candidates(
        forest=seen_forest,
        by_name={"caller": [caller]},
        by_qual={caller.qual: caller},
        symbol_table=da.SymbolTable(),
        project_root=Path("."),
        class_index={},
        resolve_callee_outcome_fn=_resolved,
    )
    assert len([alt for alt in seen_forest.alts if alt.kind == "CallCandidate"]) == 1

def test_build_analysis_index_timeout_and_resolve_outcome_edges(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "m.py"
    path.write_text("def f():\n    return 1\n")
    timeout_exc = da.TimeoutExceeded(
        TimeoutContext(call_stack=pack_call_stack([{"path": "m.py", "qual": "m.f"}]))
    )
    with deadline_scope(Deadline.from_timeout_ms(10_000)):
        with pytest.raises(da.TimeoutExceeded):
            da._build_analysis_index(
                [path],
                project_root=tmp_path,
                ignore_params=set(),
                strictness="high",
                external_filter=True,
                parse_failure_witnesses=[],
                accumulate_function_index_for_tree_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(timeout_exc),
            )

    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.caller",
        path=Path("pkg/mod.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    candidate = da.FunctionInfo(
        name="target",
        qual="pkg.target",
        path=Path("pkg/target.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )

    def _sink_resolve(*_args, ambiguity_sink, **_kwargs):
        ambiguity_sink(caller, None, [candidate], "phase", "pkg.target")
        return None

    outcome = da._resolve_callee_outcome(
        "pkg.target",
        caller,
        {"target": [candidate]},
        {caller.qual: caller, candidate.qual: candidate},
        resolve_callee_fn=_sink_resolve,
    )
    assert outcome.status == "ambiguous"

    outcome = da._resolve_callee_outcome(
        "pkg.target",
        caller,
        {"target": [candidate]},
        {caller.qual: caller, candidate.qual: candidate},
        resolve_callee_fn=lambda *_args, **_kwargs: None,
    )
    assert outcome.status == "unresolved_internal"

def test_constant_flow_and_dataclass_registry_cache_none_edges() -> None:
    da = _load()
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.caller",
        path=Path("pkg/mod.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    callee = da.FunctionInfo(
        name="callee",
        qual="pkg.callee",
        path=Path("pkg/mod.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    edge = da._ResolvedCallEdge(
        caller=caller,
        call=da.CallArgs(callee="pkg.callee", pos_map={}, kw_map={}, const_pos={}, const_kw={}, non_const_pos=set(), non_const_kw=set(), star_pos=[], star_kw=[], is_test=False, span=(1, 1, 1, 2)),
        callee=callee,
    )
    def _reduce(_index, *, spec, **_kwargs):
        acc = spec.init()
        spec.fold(acc, edge)
        return spec.finish(acc)
    index = da.AnalysisIndex(
        by_name={"callee": [callee]},
        by_qual={callee.qual: callee},
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    details = da._collect_constant_flow_details(
        [Path("pkg/mod.py")],
        project_root=None,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=[],
        analysis_index=index,
        iter_resolved_edge_param_events_fn=lambda *_args, **_kwargs: [
            da._ResolvedEdgeParamEvent(
                kind="const",
                param="p",
                value=None,
                countable=True,
            )
        ],
        reduce_resolved_call_edges_fn=_reduce,
    )
    assert details == []

    assert da._collect_dataclass_registry(
        [Path("x.py")],
        project_root=None,
        parse_failure_witnesses=[],
        analysis_index=index,
        stage_cache_fn=lambda *_args, **_kwargs: {Path("x.py"): None},
    ) == {}

def test_resume_payload_loader_format_v1_invalid_rows(tmp_path: Path) -> None:
    da = _load()
    file_path = tmp_path / "a.py"
    payload = {
        "format_version": 1,
        "groups_by_path": {str(file_path): {"f": [["x"], "bad"]}},
        "param_spans_by_path": {str(file_path): {"f": {"x": ["a", 1, 2, 3]}}},
        "bundle_sites_by_path": {str(file_path): {"f": [["bad"], "bad"]}},
        "completed_paths": [str(file_path)],
        "in_progress_scan_by_path": None,
        "analysis_index_resume": {"phase": "x"},
    }
    (
        groups,
        spans,
        sites,
        _invariants,
        completed,
        in_progress,
        analysis_index_resume,
    ) = da._load_analysis_collection_resume_payload(
        payload=payload,
        file_paths=[file_path],
        include_invariant_propositions=False,
    )
    assert completed == set()
    assert groups == {}
    assert spans == {}
    assert sites == {}
    assert in_progress == {}
    assert analysis_index_resume is None

def test_detection_and_node_identity_and_graph_cycle_edges(tmp_path: Path) -> None:
    da = _load()
    noisy_source = (
        "def runner(paths, project_root, ignore_params, strictness, external_filter, transparent_decorators, parse_failure_witnesses, analysis_index):\n"
        + "\n".join(["    x = 1" for _ in range(70)])
        + "\n    obj.run()\n"
    )
    assert da._detect_execution_pattern_matches(source=noisy_source, source_path=tmp_path / "m.py") == []

    forest = da.Forest()
    assert da._node_to_function_suite_id(forest, da.NodeId("Missing", ("x",))) is None
    suite = forest.add_suite_site("a.py", "q", "call", span=(1, 1, 1, 2))
    forest.nodes[suite] = da.Node(node_id=suite, meta={"suite_kind": "call", "path": "a.py", "qual": "q"})
    assert da._node_to_function_suite_id(forest, suite) is None

    graph = {"a": {"b"}, "b": {"a"}}
    assert da._reachable_from_roots(graph, {"a"}) == {"a", "b"}

def test_lint_and_report_relation_skip_rows_edges() -> None:
    da = _load()
    forest = da.Forest()
    file_site = forest.add_file_site("a.py")
    other = forest.add_node("Other", ("x",), {})
    lint = forest.add_node("LintFinding", ("a.py", 1, 1, "X", "m"), meta={"path": "a.py", "line": "x", "col": 1, "code": "X", "message": "m"})
    forest.add_alt("LintFinding", (file_site, other), evidence={"source": "s"})
    forest.add_alt("LintFinding", (file_site, lint), evidence={"source": "s"})
    relation = da._lint_relation_from_forest(forest)
    assert relation == []

    section_key = da._ReportSectionKey(run_id="r", section="s")
    report_forest = da.Forest()
    line_node = report_forest.add_node(
        "ReportSectionLine",
        ("r", "s", "x", "text"),
        meta={"run_id": "r", "section": "other", "line_index": "x", "text": "text"},
    )
    report_forest.add_alt("ReportSectionLine", (report_forest.add_file_site("<report>"), line_node), evidence={"run_id": "r", "section": "s"})
    assert da._report_section_line_relation(forest=report_forest, section_key=section_key) == []

def test_suite_span_none_and_lint_render_filter_edges() -> None:
    da = _load()
    assert da._suite_span_from_statements([ast.Pass()]) is None
    forest = da.Forest()
    parent = forest.add_suite_site("m.py", "m.f", "function_body", span=(1, 1, 1, 2))
    stmt = ast.parse("if True:\n    pass\n").body[0]
    da._materialize_statement_suite_contains(
        forest=forest,
        path_name="m.py",
        qual="m.f",
        statements=[stmt],
        parent_suite=parent,
    )
    assert any(alt.kind == "SuiteContains" for alt in forest.alts)

    projected_rows = [
        {"path": "", "line": 1, "col": 1, "code": "X", "message": "m"},
        {"path": "a.py", "line": "x", "col": 1, "code": "X", "message": "m"},
    ]
    rendered = da._compute_lint_lines(
        forest=da.Forest(),
        groups_by_path={},
        bundle_sites_by_path={},
        type_callsite_evidence=[],
        ambiguity_witnesses=[],
        exception_obligations=[],
        never_invariants=[],
        deadline_obligations=[],
        decision_lint_lines=[],
        broad_type_lint_lines=[],
        constant_smells=[],
        unused_arg_smells=[],
    )
    assert isinstance(rendered, list)
    assert da._parse_failure_violation_lines([{"path": "a.py", "stage": "parse", "error_type": "SyntaxError"}])

def test_emit_report_parse_contract_section_and_marker_edges() -> None:
    da = _load()
    assert da._parse_report_section_marker("missing marker") is None
    lines, violations = da._emit_report(
        {},
        max_components=1,
        report=da.ReportCarrier(forest=da.Forest(), parse_failure_witnesses=[]),
        parse_witness_contract_violations_fn=lambda: ["violation"],
    )
    assert "Parse witness contract violations:" in lines
    assert violations

def test_resume_payload_loader_edge_rows_and_in_progress_defaults(tmp_path: Path) -> None:
    da = _load()
    file_path = tmp_path / "a.py"
    payload = {
        "format_version": 2,
        "groups_by_path": {str(file_path): {"f": [["x"], [1]]}},
        "param_spans_by_path": {str(file_path): {"f": {"x": [1, 2, 3, 4], "bad": ["a", 2, 3, 4]}}},
        "bundle_sites_by_path": {str(file_path): {"f": [["ok"], "bad"]}},
        "completed_paths": [str(file_path)],
        "in_progress_scan_by_path": None,
        "analysis_index_resume": {"format_version": 1, "hydrated_paths": [str(file_path)], "functions_by_qual": {"bad": []}, "class_index": {"bad": []}},
    }
    (
        groups,
        spans,
        sites,
        _invariants,
        completed,
        in_progress,
        _analysis_index_resume,
    ) = da._load_analysis_collection_resume_payload(
        payload=payload,
        file_paths=[file_path],
        include_invariant_propositions=False,
    )
    assert completed == {file_path}
    assert groups[file_path] == {"f": [{"x"}, {"1"}]}
    assert spans[file_path]["f"]["x"] == (1, 2, 3, 4)
    assert sites[file_path] == {"f": [[]]}
    assert in_progress == {}

def test_resume_index_and_collection_loader_additional_edge_rows(tmp_path: Path) -> None:
    da = _load()
    file_path = tmp_path / "a.py"
    index_payload = {
        "format_version": 1,
        "hydrated_paths": [str(file_path)],
        "functions_by_qual": {"bad": []},
        "symbol_table": {},
        "class_index": {"bad": []},
    }
    hydrated, by_qual, _symbol_table, class_index = da._load_analysis_index_resume_payload(
        payload=index_payload,
        file_paths=[file_path],
    )
    assert hydrated == {file_path}
    assert by_qual == {}
    assert class_index == {}

    assert da._deserialize_groups_for_resume({"f": ["bad"]}) == {"f": []}
    assert da._deserialize_param_spans_for_resume({1: {"x": [1, 2, 3, 4]}, "f": {"x": ["a", 2, 3, 4]}}) == {"f": {}}
    assert da._serialize_bundle_sites_for_resume({"f": ["bad"]}) == {"f": []}
    assert da._deserialize_bundle_sites_for_resume({1: [], "f": ["bad"]}) == {"f": []}

    collection_payload = {
        "format_version": 2,
        "groups_by_path": {str(file_path): {"f": [["x"]]}},
        "param_spans_by_path": {str(file_path): []},
        "bundle_sites_by_path": {str(file_path): {"f": []}},
        "completed_paths": [str(file_path)],
        "in_progress_scan_by_path": [],
    }
    loaded = da._load_analysis_collection_resume_payload(
        payload=collection_payload,
        file_paths=[file_path],
        include_invariant_propositions=False,
    )
    assert loaded[4] == set()


def test_load_analysis_index_resume_payload_rejects_projection_identity_mismatch(
    tmp_path: Path,
) -> None:
    da = _load()
    file_path = tmp_path / "m.py"
    payload = {
        "format_version": 1,
        "index_cache_identity": "index-ok",
        "projection_cache_identity": "projection-old",
        "hydrated_paths": [str(file_path)],
    }
    hydrated_paths, by_qual, symbol_table, class_index = da._load_analysis_index_resume_payload(
        payload=payload,
        file_paths=[file_path],
        expected_index_cache_identity="index-ok",
        expected_projection_cache_identity="projection-new",
    )
    assert hydrated_paths == set()
    assert by_qual == {}
    assert symbol_table.imports == {}
    assert class_index == {}


def test_serialize_analysis_index_resume_payload_omits_non_mapping_profiling() -> None:
    da = _load()
    payload = da._serialize_analysis_index_resume_payload(
        hydrated_paths={Path("a.py")},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
        index_cache_identity="index-id",
        projection_cache_identity="projection-id",
        profiling_v1=None,
    )
    assert payload["index_cache_identity"] == "index-id"
    assert payload["projection_cache_identity"] == "projection-id"
    assert "profiling_v1" not in payload


def test_call_resolution_and_lint_compute_filter_edges() -> None:
    da = _load()
    forest = da.Forest()
    missing_suite = da.NodeId("SuiteSite", ("p", "q", "call"))
    forest.add_alt("CallResolutionObligation", (missing_suite,), evidence={"callee": "x"})
    assert da._collect_call_resolution_obligations_from_forest(forest) == []
    assert da._dedupe_resolution_candidates(
        [
            da.FunctionInfo(
                name="f",
                qual="pkg.f",
                path=Path("tests/test_mod.py"),
                params=[],
                annots={},
                calls=[],
                unused_params=set(),
                function_span=(0, 0, 0, 1),
            )
        ]
    ) == ()

    rendered = da._compute_lint_lines(
        forest=da.Forest(),
        groups_by_path={},
        bundle_sites_by_path={},
        type_callsite_evidence=[],
        ambiguity_witnesses=[],
        exception_obligations=[],
        never_invariants=[],
        deadline_obligations=[],
        decision_lint_lines=[],
        broad_type_lint_lines=[],
        constant_smells=[],
        unused_arg_smells=[],
        project_lint_rows_from_forest_fn=lambda **_kwargs: [
            {"path": "", "line": 1, "col": 1, "code": "X", "message": "m"},
            {"path": "a.py", "line": "x", "col": 1, "code": "X", "message": "m"},
        ],
    )
    assert rendered == []

def test_statement_suite_contains_body_without_span_and_parse_marker_suffix() -> None:
    da = _load()
    parent = da.Forest().add_suite_site("m.py", "m.f", "function_body", span=(1, 1, 1, 2))
    stmt = ast.If(test=ast.Constant(value=True), body=[ast.Pass()], orelse=[])
    forest = da.Forest()
    parent = forest.add_suite_site("m.py", "m.f", "function_body", span=(1, 1, 1, 2))
    da._materialize_statement_suite_contains(
        forest=forest,
        path_name="m.py",
        qual="m.f",
        statements=[stmt],
        parent_suite=parent,
    )
    assert da._parse_report_section_marker("text") is None
    assert da._parse_report_section_marker("<!-- report-section:intro") is None

def test_lint_and_report_relations_skip_missing_nodes_and_bad_payloads() -> None:
    da = _load()
    forest = da.Forest()
    file_site = forest.add_file_site("a.py")
    missing_lint_node = da.NodeId("LintFinding", ("missing",))
    forest.add_alt("LintFinding", (file_site, missing_lint_node), evidence={"source": "s"})
    bad_lint_node = forest.add_node(
        "LintFinding",
        ("bad",),
        meta={"path": "", "code": "", "message": "m", "line": 1, "col": 1},
    )
    forest.add_alt("LintFinding", (file_site, bad_lint_node), evidence={"source": "s"})
    assert da._lint_relation_from_forest(forest) == []

    section_key = da._ReportSectionKey(run_id="run", section="intro")
    report_forest = da.Forest()
    report_file = report_forest.add_file_site("<report>")
    missing_report_node = da.NodeId("ReportSectionLine", ("run", "intro", 0, "x"))
    report_forest.add_alt(
        "ReportSectionLine",
        (report_file, missing_report_node),
        evidence={"run_id": "run", "section": "intro"},
    )
    bad_report_node = report_forest.add_node(
        "ReportSectionLine",
        ("run", "intro", 1, "x"),
        meta={"run_id": "run", "section": "intro", "line_index": "bad", "text": "x"},
    )
    report_forest.add_alt(
        "ReportSectionLine",
        (report_file, bad_report_node),
        evidence={"run_id": "run", "section": "intro"},
    )
    assert da._report_section_line_relation(forest=report_forest, section_key=section_key) == []

def test_resume_loaders_skip_invalid_function_and_class_rows(tmp_path: Path) -> None:
    da = _load()
    file_path = tmp_path / "m.py"
    payload = {
        "format_version": 1,
        "hydrated_paths": [str(file_path)],
        "functions_by_qual": {"pkg.bad": {"name": "bad"}},
        "symbol_table": {},
        "class_index": {"pkg.C": {"qual": 1}},
    }
    hydrated_paths, by_qual, _symbol_table, class_index = da._load_analysis_index_resume_payload(
        payload=payload,
        file_paths=[file_path],
    )
    assert hydrated_paths == {file_path}
    assert by_qual == {}
    assert class_index == {}

    loaded = da._load_analysis_collection_resume_payload(
        payload={
            "format_version": 2,
            "groups_by_path": {str(file_path): {"f": [["x"]]}},
            "param_spans_by_path": {str(file_path): []},
            "bundle_sites_by_path": {str(file_path): {"f": []}},
            "completed_paths": [str(file_path)],
            "in_progress_scan_by_path": [],
        },
        file_paths=[file_path],
        include_invariant_propositions=False,
    )
    assert loaded[4] == set()

def test_analyze_file_internal_timeout_re_emits_scan_progress(
    tmp_path: Path,
) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text("def f(x):\n    return x\n", encoding="utf-8")
    config = da.AuditConfig(project_root=tmp_path)
    emitted: list[dict[str, object]] = []

    def _raise_timeout(*_args: object, **_kwargs: object):
        raise da.TimeoutExceeded(
            TimeoutContext(call_stack=pack_call_stack([{"path": str(path), "qual": "mod.f"}]))
        )

    with pytest.raises(da.TimeoutExceeded):
        with deadline_scope(Deadline.from_timeout_ms(10_000)):
            da._analyze_file_internal(
                path,
                recursive=True,
                config=config,
                on_progress=lambda payload: emitted.append(dict(payload)),
                analyze_function_fn=_raise_timeout,
            )
    assert emitted

def test_analyze_file_internal_emits_scan_progress_on_interval(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    fn_count = da._FILE_SCAN_PROGRESS_EMIT_INTERVAL + 1
    lines: list[str] = []
    for index in range(fn_count):
        lines.append(f"def fn_{index}(value):")
        lines.append("    return value")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    emitted: list[dict[str, object]] = []

    with deadline_scope(Deadline.from_timeout_ms(10_000)):
        da._analyze_file_internal(
            path,
            recursive=True,
            config=da.AuditConfig(project_root=tmp_path),
            on_progress=lambda payload: emitted.append(dict(payload)),
        )

    assert len(emitted) >= 2

def test_suite_span_and_statement_suite_contains_no_span_branches() -> None:
    da = _load()
    spanned_stmt = ast.parse("x = 1\n").body[0]
    assert da._suite_span_from_statements([spanned_stmt, ast.Pass()]) == da._suite_span_from_statements([spanned_stmt])

    forest = da.Forest()
    parent = forest.add_suite_site("m.py", "m.f", "function_body", span=(1, 1, 1, 2))
    synthetic_statements: list[ast.stmt] = [
        ast.If(test=ast.Constant(value=True), body=[ast.Pass()], orelse=[ast.Pass()]),
        ast.For(
            target=ast.Name(id="item", ctx=ast.Store()),
            iter=ast.Name(id="items", ctx=ast.Load()),
            body=[ast.Pass()],
            orelse=[ast.Pass()],
            type_comment=None,
        ),
        ast.AsyncFor(
            target=ast.Name(id="item", ctx=ast.Store()),
            iter=ast.Name(id="items", ctx=ast.Load()),
            body=[ast.Pass()],
            orelse=[],
            type_comment=None,
        ),
        ast.AsyncFor(
            target=ast.Name(id="item", ctx=ast.Store()),
            iter=ast.Name(id="items", ctx=ast.Load()),
            body=[ast.Pass()],
            orelse=[ast.Pass()],
            type_comment=None,
        ),
        ast.While(test=ast.Constant(value=True), body=[ast.Pass()], orelse=[]),
        ast.While(test=ast.Constant(value=True), body=[ast.Pass()], orelse=[ast.Pass()]),
        ast.Try(
            body=[ast.Pass()],
            handlers=[ast.ExceptHandler(type=None, name=None, body=[ast.Pass()])],
            orelse=[],
            finalbody=[],
        ),
        ast.Try(
            body=[ast.Pass()],
            handlers=[ast.ExceptHandler(type=None, name=None, body=[ast.Pass()])],
            orelse=[ast.Pass()],
            finalbody=[ast.Pass()],
        ),
    ]
    da._materialize_statement_suite_contains(
        forest=forest,
        path_name="m.py",
        qual="m.f",
        statements=synthetic_statements,
        parent_suite=parent,
    )
    # Bodies without spans should not produce suite containment edges.
    assert [alt for alt in forest.alts if alt.kind == "SuiteContains"] == []

def test_fallback_deadline_arg_info_vararg_kwarg_and_low_strictness_edges() -> None:
    da = _load()
    call = da.CallArgs(
        callee="pkg.target",
        pos_map={"1": "pos_param"},
        kw_map={"extra": "kw_param"},
        const_pos={"2": "None"},
        const_kw={"extra_const": "None"},
        non_const_pos={"3"},
        non_const_kw={"extra_unknown"},
        star_pos=[(0, "spread_a"), (1, "spread_b")],
        star_kw=["spread_kw_a", "spread_kw_b"],
        is_test=False,
        span=(1, 1, 1, 2),
    )
    callee = da.FunctionInfo(
        name="target",
        qual="pkg.target",
        path=Path("pkg/target.py"),
        params=["p0"],
        annots={},
        calls=[],
        unused_params=set(),
        positional_params=("p0",),
        kwonly_params=("k0",),
        vararg="rest",
        kwarg="kwargs",
        function_span=(1, 1, 1, 2),
    )
    info_map = da._fallback_deadline_arg_info(call, callee, strictness="low")
    assert info_map["rest"].kind == "param"
    assert info_map["kwargs"].kind == "param"
    assert "k0" not in info_map

def test_collect_deadline_obligations_call_node_lookup_edge_cases(tmp_path: Path) -> None:
    da = _load()
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.caller",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[
            da.CallArgs(
                callee="pkg.callee",
                pos_map={},
                kw_map={},
                const_pos={},
                const_kw={},
                non_const_pos=set(),
                non_const_kw=set(),
                star_pos=[],
                star_kw=[],
                is_test=False,
                span=None,
            ),
            da.CallArgs(
                callee="pkg.callee",
                pos_map={},
                kw_map={},
                const_pos={},
                const_kw={},
                non_const_pos=set(),
                non_const_kw=set(),
                star_pos=[],
                star_kw=[],
                is_test=False,
                span=(1, 1, 1, 2),
            ),
        ],
        unused_params=set(),
        function_span=(1, 1, 1, 2),
    )
    callee = da.FunctionInfo(
        name="callee",
        qual="pkg.callee",
        path=tmp_path / "callee.py",
        params=["value"],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(1, 1, 1, 2),
    )
    index = da.AnalysisIndex(
        by_name={"caller": [caller]},
        by_qual={caller.qual: caller, callee.qual: callee},
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    forest = da.Forest()
    config = da.AuditConfig(project_root=tmp_path, deadline_roots={"pkg.root"})

    def _resolve(*_args, **_kwargs):
        return da._CalleeResolutionOutcome(
            status="resolved",
            phase="resolved",
            callee_key="pkg.callee",
            candidates=(callee,),
        )

    obligations = da._collect_deadline_obligations(
        [caller.path],
        project_root=tmp_path,
        config=config,
        forest=forest,
        parse_failure_witnesses=[],
        analysis_index=index,
        extra_deadline_params={"pkg.root": set()},
        materialize_call_candidates_fn=lambda **_kwargs: None,
        collect_call_nodes_by_path_fn=lambda *_args, **_kwargs: {
            caller.path: {(1, 1, 1, 2): []}
        },
        collect_deadline_function_facts_fn=lambda *_args, **_kwargs: {},
        collect_call_edges_from_forest_fn=lambda *_args, **_kwargs: {},
        collect_call_resolution_obligations_from_forest_fn=lambda _forest: [],
        reachable_from_roots_fn=lambda *_args, **_kwargs: set(),
        collect_recursive_nodes_fn=lambda _edges: set(),
        resolve_callee_outcome_fn=_resolve,
    )
    assert obligations == []

def test_collect_deadline_obligations_unknown_kind_fallthrough(tmp_path: Path) -> None:
    da = _load()
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.caller",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(1, 1, 1, 2),
    )
    callee = da.FunctionInfo(
        name="callee",
        qual="pkg.callee",
        path=tmp_path / "callee.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(1, 1, 1, 2),
    )
    index = da.AnalysisIndex(
        by_name={},
        by_qual={caller.qual: caller, callee.qual: callee},
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    call = da.CallArgs(
        callee="pkg.callee",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(1, 1, 1, 2),
    )
    obligations = da._collect_deadline_obligations(
        [caller.path],
        project_root=tmp_path,
        config=da.AuditConfig(project_root=tmp_path, deadline_roots={"pkg.caller"}),
        forest=da.Forest(),
        parse_failure_witnesses=[],
        analysis_index=index,
        extra_deadline_params={"pkg.callee": {"deadline"}},
        extra_call_infos={
            caller.qual: [
                (
                    call,
                    callee,
                    {"deadline": da._DeadlineArgInfo(kind="mystery")},
                )
            ]
        },
        materialize_call_candidates_fn=lambda **_kwargs: None,
        collect_call_nodes_by_path_fn=lambda *_args, **_kwargs: {},
        collect_deadline_function_facts_fn=lambda *_args, **_kwargs: {},
        collect_call_edges_from_forest_fn=lambda *_args, **_kwargs: {},
        collect_call_resolution_obligations_from_forest_fn=lambda _forest: [],
        reachable_from_roots_fn=lambda *_args, **_kwargs: set(),
        collect_recursive_nodes_fn=lambda _edges: set(),
        resolve_callee_outcome_fn=lambda *_args, **_kwargs: da._CalleeResolutionOutcome(
            status="resolved",
            phase="resolved",
            callee_key="pkg.callee",
            candidates=(callee,),
        ),
    )
    assert obligations == []

def test_resolve_class_candidates_symbol_table_and_module_edge_cases() -> None:
    da = _load()
    class_index = {
        "pkg.Base": da.ClassInfo(qual="pkg.Base", module="pkg", bases=[], methods=set()),
        "Local": da.ClassInfo(qual="Local", module="", bases=[], methods=set()),
    }
    assert da._resolve_class_candidates(
        "pkg.Base",
        module="",
        symbol_table=None,
        class_index=class_index,
    ) == ["pkg.Base"]

    symbol_table = da.SymbolTable(
        imports={("", "Local"): "ext.Local"},
        internal_roots=set(),
        external_filter=True,
    )
    assert da._resolve_class_candidates(
        "Local",
        module="",
        symbol_table=symbol_table,
        class_index=class_index,
    ) == ["Local"]

def test_callsite_evidence_for_bundle_skips_non_bundle_slots() -> None:
    da = _load()
    call = da.CallArgs(
        callee="pkg.fn",
        pos_map={"0": "x"},
        kw_map={"name": "y"},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[(1, "z")],
        star_kw=["w"],
        is_test=False,
        span=(1, 1, 1, 2),
    )
    evidence = da._callsite_evidence_for_bundle([call], {"bundle_only"})
    assert evidence == []

def test_collect_never_invariants_dead_env_edge_cases(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "never_edges.py"
    path.write_text(
        "from gabion.invariants import never\n"
        "\n"
        "def f(flag):\n"
        "    if 0:\n"
        "        never('const-false')\n"
        "\n"
        "def g(flag):\n"
        "    if flag + 1:\n"
        "        never('undecidable-with-env')\n"
    )
    normalized_path = da._normalize_snapshot_path(path, tmp_path)
    deadness = [
        {
            "path": normalized_path,
            "function": "f",
            "bundle": ["flag"],
            "environment": {"flag": "False"},
            "deadness_id": "dead:f",
        },
        {
            "path": normalized_path,
            "function": "g",
            "bundle": ["flag"],
            "environment": {"flag": "True"},
            "deadness_id": "dead:g",
        },
    ]
    invariants = da._collect_never_invariants(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        deadness_witnesses=deadness,
        forest=da.Forest(),
    )
    by_reason = {str(entry.get("reason", "")): entry for entry in invariants}
    assert by_reason["const-false"]["status"] == "OBLIGATION"
    assert "undecidable_reason" not in by_reason["undecidable-with-env"]


def test_resolve_callee_outcome_classifies_dynamic_dispatch_deterministically() -> None:
    da = _load()
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.caller",
        path=Path("pkg/mod.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    outcome1 = da._resolve_callee_outcome(
        "getattr(service, name)",
        caller,
        {},
        {caller.qual: caller},
        resolve_callee_fn=lambda *_args, **_kwargs: None,
    )
    outcome2 = da._resolve_callee_outcome(
        "getattr(service, name)",
        caller,
        {},
        {caller.qual: caller},
        resolve_callee_fn=lambda *_args, **_kwargs: None,
    )
    assert outcome1.status == "unresolved_dynamic"
    assert outcome1 == outcome2


def test_pattern_schema_matches_are_stable_across_repeated_runs() -> None:
    da = _load()
    groups_by_path = {Path("pkg/mod.py"): {"f": [{"a", "b"}, {"a", "b"}]}}
    source = (
        "def one(paths, strictness, project_root):\n"
        "    return _build_analysis_index(paths, strictness, project_root)\n"
        "def two(paths, strictness, project_root):\n"
        "    return _build_call_graph(paths, strictness, project_root)\n"
        "def three(paths, strictness, project_root):\n"
        "    return _build_analysis_index(paths, strictness, project_root)\n"
    )
    first = da._pattern_schema_snapshot_entries(
        da._pattern_schema_matches(groups_by_path=groups_by_path, source=source)
    )
    second = da._pattern_schema_snapshot_entries(
        da._pattern_schema_matches(groups_by_path=groups_by_path, source=source)
    )
    assert first == second


def test_build_function_index_indexes_lambda_sites_deterministically(tmp_path: Path) -> None:
    da = _load()
    mod = tmp_path / "mod.py"
    mod.write_text(
        "def outer(x):\n"
        "    f = lambda y: y\n"
        "    return f(x), (lambda z: z)(x)\n"
    )
    args = dict(
        paths=[mod],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        parse_failure_witnesses=[],
    )
    by_name1, by_qual1 = da._build_function_index(**args)
    by_name2, by_qual2 = da._build_function_index(**args)
    caller1 = by_qual1["mod.outer"]
    caller2 = by_qual2["mod.outer"]
    assert "f" in caller1.local_lambda_bindings
    assert caller1.local_lambda_bindings["f"] == caller2.local_lambda_bindings["f"]
    all_infos = [info for infos in by_name1.values() for info in infos]
    lambda_infos = [info for info in all_infos if info.name.startswith("<lambda:")]
    assert len(lambda_infos) >= 2
    assert tuple(by_qual1) == tuple(by_qual2)
