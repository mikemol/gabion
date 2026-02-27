from __future__ import annotations

import argparse
import ast
import pytest
from pathlib import Path

from gabion.analysis import dataflow_audit as da
from gabion.exceptions import NeverThrown
from tests.env_helpers import env_scope


def _load():
    return da


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_merge_counts_by_knobs_skips_larger_superset_after_first_merge::dataflow_audit.py::gabion.analysis.dataflow_audit._merge_counts_by_knobs::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_merge_counts_by_knobs_skips_larger_superset_after_first_merge() -> None:
    da = _load()
    counts = {
        ("a",): 1,
        ("a", "k1"): 1,
        ("a", "k1", "k2"): 1,
    }
    merged = da._merge_counts_by_knobs(counts, {"k1", "k2"})
    assert merged[("a", "k1")] >= 1


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_build_synthesis_plan_ignores_non_literal_const_hints::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_build_synthesis_plan_ignores_non_literal_const_hints(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "class M:\n"
        "    CONST = 1\n"
        "\n"
        "def callee(p, k=None):\n"
        "    return p\n"
        "\n"
        "def caller():\n"
        "    return callee(M.CONST, k=M.CONST)\n",
        encoding="utf-8",
    )
    groups_by_path: dict[Path, dict[str, list[set[str]]]] = {
        path: {"callee": [set(["p", "k"])]}
    }
    plan = da.build_synthesis_plan(
        groups_by_path,
        project_root=tmp_path,
        min_bundle_size=1,
        allow_singletons=True,
    )
    assert "protocols" in plan


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_render_synthesis_section_ignores_blank_field_names::dataflow_audit.py::gabion.analysis.dataflow_audit.render_synthesis_section::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_render_synthesis_section_ignores_blank_field_names() -> None:
    da = _load()
    plan = {
        "protocols": [
            {
                "name": "Bundle",
                "tier": 2,
                "fields": [
                    {"name": "", "type_hint": "int", "source_params": []},
                    {"type_hint": "str", "source_params": []},
                ],
                "bundle": [],
                "rationale": "test",
                "evidence": [],
            }
        ],
        "warnings": [],
        "errors": [],
    }
    text = da.render_synthesis_section(plan)
    assert "(no fields)" in text


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_invariant_proposition_and_projection_order_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._topologically_order_report_projection_specs::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_invariant_proposition_and_projection_order_edges() -> None:
    da = _load()
    assert da.InvariantProposition(form="Eq", terms=("a", "b")).as_dict() == {
        "form": "Eq",
        "terms": ["a", "b"],
    }
    assert da.InvariantProposition(
        form="Eq",
        terms=("a", "b"),
        scope="mod.f",
        source="assert",
    ).as_dict() == {
        "form": "Eq",
        "terms": ["a", "b"],
        "scope": "mod.f",
        "source": "assert",
    }

    noop = lambda *_args, **_kwargs: []  # noqa: E731
    spec_a = da.ReportProjectionSpec(
        section_id="a",
        phase="collection",
        deps=(),
        build=noop,
        render=lambda _value: [],
        violation_extract=lambda _value: [],
    )
    spec_b = da.ReportProjectionSpec(
        section_id="b",
        phase="collection",
        deps=(),
        build=noop,
        render=lambda _value: [],
        violation_extract=lambda _value: [],
    )
    spec_c = da.ReportProjectionSpec(
        section_id="c",
        phase="collection",
        deps=("a", "b"),
        build=noop,
        render=lambda _value: [],
        violation_extract=lambda _value: [],
    )
    ordered = da._topologically_order_report_projection_specs((spec_a, spec_b, spec_c))
    assert tuple(spec.section_id for spec in ordered) == ("a", "b", "c")


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_project_sections_and_invariant_helpers_misc_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._invariant_term::dataflow_audit.py::gabion.analysis.dataflow_audit._scope_path::dataflow_audit.py::gabion.analysis.dataflow_audit.project_report_sections::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_project_sections_and_invariant_helpers_misc_edges(tmp_path: Path) -> None:
    da = _load()
    report = da.ReportCarrier(forest=da.Forest())
    selected = da.project_report_sections(
        {},
        report,
        max_phase="collection",
        include_previews=True,
        preview_only=True,
    )
    assert isinstance(selected, dict)

    len_term = da._invariant_term(ast.parse("len(a)").body[0].value, {"a"})
    assert len_term == "a.length"

    fn = ast.parse(
        "def f(a):\n"
        "    assert a == a\n"
        "    assert a == a\n"
    ).body[0]
    collector = da._InvariantCollector({"a"}, "m.f")
    for stmt in fn.body:
        collector.visit(stmt)
    assert len(collector.propositions) == 1
    assert da._scope_path(Path("/outside/mod.py"), root=tmp_path) == "/outside/mod.py"


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_param_spans_deadline_reason_and_local_info_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_deadline_local_info::dataflow_audit.py::gabion.analysis.dataflow_audit._never_reason::dataflow_audit.py::gabion.analysis.dataflow_audit._param_spans::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_param_spans_deadline_reason_and_local_info_edges() -> None:
    da = _load()
    synthetic_fn = ast.FunctionDef(
        name="f",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="a"), ast.arg(arg="b")],
            vararg=ast.arg(arg="args"),
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=ast.arg(arg="kwargs"),
            defaults=[],
        ),
        body=[],
        decorator_list=[],
    )
    assert da._param_spans(synthetic_fn) == {}

    assert da._never_reason(ast.parse("never(1)").body[0].value) is None
    assert da._never_reason(ast.parse("never(x=1)").body[0].value) is None
    assert da._never_reason(ast.parse("never(reason=1)").body[0].value) is None

    origin_call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="Deadline", ctx=ast.Load()),
            attr="from_timeout_ms",
            ctx=ast.Load(),
        ),
        args=[ast.Constant(value=10)],
        keywords=[],
    )
    local_info = da._collect_deadline_local_info(
        assignments=[
            ([ast.Name(id="d", ctx=ast.Store())], origin_call, (1, 0, 1, 5)),
            ([ast.Name(id="e", ctx=ast.Store())], ast.Name(id="d", ctx=ast.Load()), (2, 0, 2, 1)),
            ([ast.Name(id="skip", ctx=ast.Store())], None, None),
            ([ast.Name(id="skip", ctx=ast.Store())], ast.Name(id="deadline", ctx=ast.Load()), None),
        ],
        params={"deadline"},
    )
    assert "d" in local_info.origin_vars
    assert "e" in local_info.origin_vars
    assert local_info.alias_to_param["deadline"] == "deadline"


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_deadline_collector_call_and_bind_args_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._bind_call_args::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_deadline_collector_call_and_bind_args_edges() -> None:
    da = _load()
    fn = ast.parse(
        "def f(deadline):\n"
        "    obj.check_deadline(1)\n"
        "    obj.check_deadline()\n"
        "    check_deadline(1)\n"
        "    require_deadline()\n"
        "    return deadline\n"
    ).body[0]
    collector = da._DeadlineFunctionCollector(fn, {"deadline"})
    collector.visit(fn)
    assert collector.ambient_check is True

    call_node = ast.parse("fn(1, *[2], **{'x': 3}, named=4)").body[0].value
    callee = da.FunctionInfo(
        name="fn",
        qual="pkg.fn",
        path=Path("pkg/mod.py"),
        params=["a", "named"],
        annots={},
        calls=[],
        unused_params=set(),
        positional_params=("a",),
        kwonly_params=("named",),
        function_span=(1, 1, 1, 2),
    )
    bound = da._bind_call_args(call_node, callee, strictness="low")
    assert "a" in bound
    assert "named" in bound


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_deadline_collector_deadline_loop_iter_branches::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_deadline_collector_deadline_loop_iter_branches() -> None:
    da = _load()
    fn = ast.parse(
        "def f(items):\n"
        "    for x in items:\n"
        "        pass\n"
        "    for y in obj.deadline_loop_iter(items):\n"
        "        pass\n"
        "    for z in (lambda seq: seq)(items):\n"
        "        pass\n"
        "    obj.deadline_loop_iter(items)\n"
        "    deadline_loop_iter(items)\n"
    ).body[0]
    collector = da._DeadlineFunctionCollector(fn, set())
    collector.visit(fn)
    assert collector.loop is True
    assert len(collector.loop_sites) == 3
    assert collector.loop_sites[0].ambient_check is False
    assert collector.loop_sites[1].ambient_check is True
    assert collector.loop_sites[2].ambient_check is False
    assert collector.ambient_check is True


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_load_analysis_index_resume_payload_edge_shapes::dataflow_audit.py::gabion.analysis.dataflow_audit._load_analysis_index_resume_payload::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_load_analysis_index_resume_payload_edge_shapes(tmp_path: Path) -> None:
    da = _load()
    file_path = tmp_path / "mod.py"
    payload = {
        "format_version": 1,
        "hydrated_paths": str(file_path),  # Sequence[str] branch with path miss hits
        "functions_by_qual": {"pkg.bad": [], 1: {}},
        "symbol_table": {"imports": []},
        "class_index": {"pkg.C": []},
    }
    hydrated, by_qual, symbol_table, class_index = da._load_analysis_index_resume_payload(
        payload=payload,
        file_paths=[file_path],
    )
    assert hydrated == set()
    assert by_qual == {}
    assert isinstance(symbol_table, da.SymbolTable)
    assert class_index == {}


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_load_analysis_index_resume_payload_hydrates_valid_sections::dataflow_audit.py::gabion.analysis.dataflow_audit._load_analysis_index_resume_payload::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_load_analysis_index_resume_payload_hydrates_valid_sections(tmp_path: Path) -> None:
    da = _load()
    file_path = tmp_path / "mod.py"
    file_path.write_text("def f(x):\n    return x\n", encoding="utf-8")
    payload = {
        "format_version": 1,
        "hydrated_paths": [str(file_path)],
        "functions_by_qual": {
            "mod.f": {
                "name": "f",
                "qual": "mod.f",
                "path": str(file_path),
                "params": ["x"],
                "annots": {},
                "calls": [],
                "unused_params": [],
                "transparent": True,
                "class_name": None,
                "scope": [],
                "lexical_scope": [],
                "decision_params": [],
                "value_decision_params": [],
                "value_decision_reasons": [],
                "positional_params": ["x"],
                "kwonly_params": [],
                "vararg": None,
                "kwarg": None,
                "param_spans": {},
                "function_span": [1, 0, 1, 8],
            }
        },
        "symbol_table": {
            "imports": [],
            "internal_roots": [],
            "external_filter": True,
            "star_imports": {},
            "module_exports": {},
            "module_export_map": {},
        },
        "class_index": {
            "mod.C": {
                "qual": "mod.C",
                "module": "mod",
                "bases": [],
                "methods": [],
            }
        },
    }
    hydrated, by_qual, symbol_table, class_index = da._load_analysis_index_resume_payload(
        payload=payload,
        file_paths=[file_path],
    )
    assert hydrated == {file_path}
    assert set(by_qual) == {"mod.f"}
    assert isinstance(symbol_table, da.SymbolTable)
    assert set(class_index) == {"mod.C"}


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_scope_path_relative_and_none_root_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._scope_path::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_scope_path_relative_and_none_root_edges(tmp_path: Path) -> None:
    da = _load()
    inside = tmp_path / "pkg" / "mod.py"
    outside = Path("/outside/mod.py")
    assert da._scope_path(inside, tmp_path) == "pkg/mod.py"
    assert da._scope_path(outside, tmp_path) == "/outside/mod.py"
    assert da._scope_path(inside, None).endswith("pkg/mod.py")


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_resolve_local_method_in_hierarchy_recurses_to_base::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_local_method_in_hierarchy::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_resolve_local_method_in_hierarchy_recurses_to_base() -> None:
    da = _load()
    resolved = da._resolve_local_method_in_hierarchy(
        "Child",
        "act",
        class_bases={"Child": ["Base"], "Base": []},
        local_functions={"Base.act"},
        seen=set(),
    )
    assert resolved == "Base.act"


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_fallback_deadline_arg_info_skips_vararg_kwarg_when_absent::dataflow_audit.py::gabion.analysis.dataflow_audit._fallback_deadline_arg_info::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_fallback_deadline_arg_info_skips_vararg_kwarg_when_absent() -> None:
    da = _load()
    call = da.CallArgs(
        callee="pkg.target",
        pos_map={"1": "extra_pos"},
        kw_map={"extra": "extra_kw"},
        const_pos={"2": "1"},
        const_kw={"extra_const": "2"},
        non_const_pos={"3"},
        non_const_kw={"extra_unknown"},
        star_pos=[],
        star_kw=[],
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
        kwonly_params=(),
        vararg=None,
        kwarg=None,
        function_span=(1, 1, 1, 2),
    )
    info_map = da._fallback_deadline_arg_info(call, callee, strictness="high")
    assert set(info_map) == set()


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_analyze_decision_surface_indexed_lint_none_paths::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_decision_surface_indexed::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_analyze_decision_surface_indexed_lint_none_paths(tmp_path: Path) -> None:
    da = _load()
    fn = da.FunctionInfo(
        name="f",
        qual="pkg.f",
        path=tmp_path / "mod.py",
        params=["flag"],
        annots={},
        calls=[],
        unused_params=set(),
        decision_params={"flag"},
        function_span=(1, 0, 1, 10),
    )
    by_qual = {fn.qual: fn}
    index = da.AnalysisIndex(
        by_name={"f": [fn]},
        by_qual=by_qual,
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    index.transitive_callers = {fn.qual: set()}
    context = da._IndexedPassContext(
        paths=[fn.path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        parse_failure_witnesses=[],
        analysis_index=index,
    )
    _surfaces, warnings, _rewrites, lint_lines = da._analyze_decision_surface_indexed(
        context,
        spec=da._DIRECT_DECISION_SURFACE_SPEC,
        decision_tiers={"other": 1},
        require_tiers=True,
        forest=da.Forest(),
    )
    assert warnings
    assert lint_lines == []

    # Internal-caller branch with tiered warning and lint=None.
    index.transitive_callers = {fn.qual: {"pkg.caller"}}
    _surfaces2, warnings2, _rewrites2, lint_lines2 = da._analyze_decision_surface_indexed(
        context,
        spec=da._DIRECT_DECISION_SURFACE_SPEC,
        decision_tiers={"flag": 2},
        require_tiers=False,
        forest=da.Forest(),
    )
    assert warnings2
    assert lint_lines2 == []


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_collect_exception_obligations_dead_reachability_branch::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_collect_exception_obligations_dead_reachability_branch(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def f(flag):\n"
        "    if flag:\n"
        "        raise ValueError(flag)\n",
        encoding="utf-8",
    )
    path_value = da._normalize_snapshot_path(path, tmp_path)
    obligations = da._collect_exception_obligations(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        handledness_witnesses=[{"exception_path_id": "", "handledness_id": "skip"}],
        deadness_witnesses=[
            {
                "path": path_value,
                "function": "f",
                "bundle": ["flag"],
                "environment": {"flag": "False"},
                "deadness_id": "dead:f",
            }
        ],
    )
    assert obligations
    assert obligations[0]["status"] == "DEAD"


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_build_synthesis_plan_duplicate_counts_and_hint_branches::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_build_synthesis_plan_duplicate_counts_and_hint_branches(tmp_path: Path) -> None:
    da = _load()
    module = tmp_path / "mod.py"
    module.write_text(
        "def target(flag):\n"
        "    if flag:\n"
        "        return 1\n"
        "    return (flag == 1) * 2\n"
        "\n"
        "def helper(flag):\n"
        "    return flag\n"
        "\n"
        "def caller_pos():\n"
        "    return helper(1)\n"
        "\n"
        "def caller_kw():\n"
        "    return helper(flag=2)\n",
        encoding="utf-8",
    )
    groups_by_path = {module: {"target": [set(["flag"])]}}
    plan = da.build_synthesis_plan(
        groups_by_path,
        project_root=tmp_path,
        merge_overlap_threshold=1.0,
    )
    assert isinstance(plan, dict)
    assert "protocols" in plan


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_build_synthesis_plan_handles_empty_bundle_memberless_merge_branch::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_build_synthesis_plan_handles_empty_bundle_memberless_merge_branch(
    tmp_path: Path,
) -> None:
    da = _load()
    module = tmp_path / "mod.py"
    module.write_text(
        "def first(x):\n"
        "    return x\n"
        "\n"
        "def second(x):\n"
        "    return x\n",
        encoding="utf-8",
    )
    groups_by_path = {
        module: {
            "first": [set(), {"x"}],
            "second": [{"x"}],
        }
    }
    plan = da.build_synthesis_plan(
        groups_by_path,
        project_root=tmp_path,
        min_bundle_size=0,
        allow_singletons=True,
        merge_overlap_threshold=0.5,
    )
    intern = plan.get("forest_signature", {}).get("nodes", {}).get("intern", [])
    assert ["ParamSet", []] in intern
    assert any(protocol.get("bundle") == ["x"] for protocol in plan["protocols"])


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_project_report_sections_phase_and_preview_branches::dataflow_audit.py::gabion.analysis.dataflow_audit.project_report_sections::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_project_report_sections_phase_and_preview_branches() -> None:
    da = _load()
    report = da.ReportCarrier(forest=da.Forest(), constant_smells=["const smell"])
    selected = da.project_report_sections(
        {},
        report,
        max_phase="post",
        include_previews=True,
        preview_only=True,
    )
    assert selected
    selected_without_max = da.project_report_sections(
        {},
        report,
        max_phase=None,
        include_previews=True,
        preview_only=True,
    )
    assert selected_without_max


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_invariant_term_len_call_branch::dataflow_audit.py::gabion.analysis.dataflow_audit._invariant_term::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_invariant_term_len_call_branch() -> None:
    da = _load()
    term = da._invariant_term(ast.parse("len(flag)").body[0].value, {"flag"})
    assert term == "flag.length"


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_resolve_local_method_in_hierarchy_unresolved_branch::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_local_method_in_hierarchy::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_resolve_local_method_in_hierarchy_unresolved_branch() -> None:
    da = _load()
    resolved = da._resolve_local_method_in_hierarchy(
        "Child",
        "missing",
        class_bases={"Child": ["Base"], "Base": []},
        local_functions={"Base.other"},
        seen=set(),
    )
    assert resolved is None


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_collect_deadline_local_info_multi_source_alias_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_deadline_local_info::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_collect_deadline_local_info_multi_source_alias_edges() -> None:
    da = _load()
    origin_call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="Deadline", ctx=ast.Load()),
            attr="from_timeout_ms",
            ctx=ast.Load(),
        ),
        args=[ast.Constant(value=10)],
        keywords=[],
    )
    local_info = da._collect_deadline_local_info(
        assignments=[
            ([ast.Name(id="origin", ctx=ast.Store())], origin_call, (1, 0, 1, 8)),
            ([ast.Name(id="origin", ctx=ast.Store())], origin_call, (2, 0, 2, 8)),
            ([ast.Name(id="x", ctx=ast.Store())], ast.Name(id="a", ctx=ast.Load()), None),
            ([ast.Name(id="x", ctx=ast.Store())], ast.Name(id="b", ctx=ast.Load()), None),
            ([ast.Name(id="from_origin", ctx=ast.Store())], ast.Name(id="origin", ctx=ast.Load()), None),
        ],
        params={"a", "b"},
    )
    assert "origin" in local_info.origin_vars
    assert "x" not in local_info.alias_to_param
    assert "from_origin" in local_info.origin_vars


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_summarize_never_invariants_evidence_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_summarize_never_invariants_evidence_edges() -> None:
    da = _load()
    entries = [
        {
            "status": "VIOLATION",
            "site": {"path": "a.py", "function": "f", "bundle": ["x"]},
            "witness_ref": None,
            "environment_ref": {"x": "1"},
            "span": [1, 1, 1, 2],
        },
        {
            "status": "PROVEN_UNREACHABLE",
            "site": {"path": "a.py", "function": "f", "bundle": ["x"]},
            "witness_ref": "dead:1",
            "environment_ref": {"x": "0"},
            "span": [2, 1, 2, 2],
        },
    ]
    lines = da._summarize_never_invariants(entries)
    assert any("env=" in line for line in lines)
    assert any("deadness=dead:1" in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_load_analysis_index_resume_payload_non_mapping_sections::dataflow_audit.py::gabion.analysis.dataflow_audit._load_analysis_index_resume_payload::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_load_analysis_index_resume_payload_non_mapping_sections(tmp_path: Path) -> None:
    da = _load()
    file_path = tmp_path / "mod.py"
    payload = {
        "format_version": 1,
        "hydrated_paths": None,
        "functions_by_qual": [],
        "symbol_table": [],
        "class_index": [],
    }
    hydrated, by_qual, symbol_table, class_index = da._load_analysis_index_resume_payload(
        payload=payload,
        file_paths=[file_path],
    )
    assert hydrated == set()
    assert by_qual == {}
    assert isinstance(symbol_table, da.SymbolTable)
    assert class_index == {}


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_verify_rewrite_plan_verification_and_remainder_edges::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_verify_rewrite_plan_verification_and_remainder_edges() -> None:
    da = _load()
    plan = {
        "plan_id": "p1",
        "site": {"path": "a.py", "function": "f", "bundle": ["x"]},
        "pre": {"base_keys": ["int"], "ctor_keys": [], "remainder": {"base": 2, "ctor": 2}},
        "rewrite": {"parameters": {"candidates": ["ctx"]}},
        "verification": [],
    }
    post = [
        {
            "path": "a.py",
            "function": "f",
            "bundle": ["x"],
            "base_keys": ["int"],
            "ctor_keys": [],
            "remainder": {"base": 1, "ctor": 1},
            "glossary_matches": ["ctx"],
        }
    ]
    result = da.verify_rewrite_plan(plan, post_provenance=post)
    assert result["accepted"] is True


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_eval_bool_expr_or_gte_and_branch_reachability_else_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._branch_reachability_under_env::dataflow_audit.py::gabion.analysis.dataflow_audit._eval_bool_expr::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_eval_bool_expr_or_gte_and_branch_reachability_else_edges() -> None:
    da = _load()
    or_expr = ast.parse("a or b").body[0].value
    or_outcome = da._eval_bool_expr(or_expr, {"a": False, "b": True})
    assert or_outcome.is_unknown() is False
    assert or_outcome.as_bool() is True
    gte_expr = ast.parse("x >= 1").body[0].value
    gte_outcome = da._eval_bool_expr(gte_expr, {"x": 2})
    assert gte_outcome.is_unknown() is False
    assert gte_outcome.as_bool() is True

    tree = ast.parse(
        "if flag:\n"
        "    a = 1\n"
        "else:\n"
        "    raise ValueError(flag)\n"
    )
    parent = da.ParentAnnotator()
    parent.visit(tree)
    raise_node = tree.body[0].orelse[0]
    reach = da._branch_reachability_under_env(
        raise_node,
        parent.parents,
        {"flag": True},
    )
    assert reach is da._EvalDecision.FALSE


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_collect_module_exports_all_assignment_none_values_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_collect_module_exports_all_assignment_none_values_edges() -> None:
    da = _load()
    tree = ast.parse(
        "__all__ = [name]\n"
        "__all__ = [\"public\"]\n"
        "public = 1\n"
    )
    exports, export_map = da._collect_module_exports(
        tree,
        module_name="pkg.mod",
        import_map={},
    )
    assert "public" in exports
    assert export_map.get("public") == "pkg.mod.public"


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_collect_module_exports_annassign_and_augassign_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_collect_module_exports_annassign_and_augassign_edges() -> None:
    da = _load()
    tree = ast.parse(
        "__all__: list[str] = [\"first\"]\n"
        "__all__ += [\"second\"]\n"
        "first = 1\n"
        "second = 2\n",
    )
    exports, _ = da._collect_module_exports(
        tree,
        module_name="pkg.mod",
        import_map={},
    )
    assert "first" in exports
    assert "second" in exports


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_accumulate_function_index_vararg_and_kwarg_ignored::dataflow_audit.py::gabion.analysis.dataflow_audit._accumulate_function_index_for_tree::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_accumulate_function_index_vararg_and_kwarg_ignored() -> None:
    da = _load()
    tree = ast.parse("def f(*skip_a, **skip_k):\n    return 1\n")
    acc = da._FunctionIndexAccumulator()
    da._accumulate_function_index_for_tree(
        acc,
        Path("mod.py"),
        tree,
        project_root=Path("."),
        ignore_params={"skip_a", "skip_k"},
        strictness="low",
        transparent_decorators=None,
    )
    info = acc.by_name["f"][0]
    assert info.vararg is None
    assert info.kwarg is None


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_bundle_name_registry_non_empty_keys::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_name_registry::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_bundle_name_registry_non_empty_keys(tmp_path: Path) -> None:
    da = _load()
    (tmp_path / "mod.py").write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class DemoData:\n"
        "    x: int\n"
        "\n"
        "class DemoConfig:\n"
        "    value: int\n",
        encoding="utf-8",
    )
    registry = da._bundle_name_registry(tmp_path)
    assert ("x",) in registry
    assert ("value",) in registry


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_bundle_projection_skips_empty_evidence_paths::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_projection_from_forest::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_bundle_projection_skips_empty_evidence_paths(tmp_path: Path) -> None:
    da = _load()
    forest = da.Forest()
    site = forest.add_site("a.py", "f")
    paramset = forest.add_paramset(["p"])
    forest.add_alt("SignatureBundle", (site, paramset))
    forest.add_alt("ConfigBundle", (paramset,), evidence={"path": ""})
    forest.add_alt("MarkerBundle", (paramset,), evidence={"path": ""})
    projection = da._bundle_projection_from_forest(
        forest,
        file_paths=[tmp_path / "a.py"],
    )
    assert projection.declared_global == {("p",)}
    assert projection.declared_by_path == {}
    assert projection.documented_by_path == {}


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_render_mermaid_component_empty_component_branch::dataflow_audit.py::gabion.analysis.dataflow_audit._render_mermaid_component::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_render_mermaid_component_empty_component_branch() -> None:
    da = _load()
    mermaid, summary = da._render_mermaid_component(
        nodes={},
        bundle_map={},
        bundle_counts={},
        adj={},
        component=[],
        declared_global=set(),
        declared_by_path={},
        documented_by_path={},
    )
    assert "flowchart LR" in mermaid
    assert "Observed bundles:" in summary


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_split_top_level_empty_part_and_tail_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._split_top_level::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_split_top_level_empty_part_and_tail_edges() -> None:
    da = _load()
    assert da._split_top_level("a,,", ",") == ["a"]


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_summarize_never_invariants_missing_evidence_branches::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_summarize_never_invariants_missing_evidence_branches() -> None:
    da = _load()
    entries = [
        {
            "status": "VIOLATION",
            "site": {"path": "a.py", "function": "f", "bundle": ["x"]},
            "witness_ref": "w:1",
            "environment_ref": None,
            "span": [1, 1, 1, 2],
        },
        {
            "status": "PROVEN_UNREACHABLE",
            "site": {"path": "a.py", "function": "f", "bundle": ["x"]},
            "witness_ref": None,
            "environment_ref": None,
            "span": [2, 1, 2, 2],
        },
    ]
    lines = da._summarize_never_invariants(entries)
    assert any("witness=w:1" in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_resolve_callee_self_and_hierarchy_none_branches::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_resolve_callee_self_and_hierarchy_none_branches(tmp_path: Path) -> None:
    da = _load()
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.mod.caller",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name=None,
    )
    # self/cls branch with no class_name should fall through.
    assert (
        da._resolve_callee(
            "self.run",
            caller,
            {"caller": [caller]},
            {},
            da.SymbolTable(),
            tmp_path,
            {},
        )
        is None
    )

    caller_with_class = da.FunctionInfo(
        name="caller",
        qual="pkg.mod.Caller.caller",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name="Caller",
    )
    class_index = {
        "pkg.mod.Caller": da.ClassInfo(
            qual="pkg.mod.Caller",
            module="pkg.mod",
            bases=[],
            methods={"other"},
        )
    }
    # Candidate path exists but method resolution yields None.
    assert (
        da._resolve_callee(
            "pkg.mod.Caller.missing",
            caller_with_class,
            {"caller": [caller_with_class]},
            {},
            da.SymbolTable(),
            tmp_path,
            class_index,
        )
        is None
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_iter_dataclass_call_bundles_assign_and_attribute_branches::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_iter_dataclass_call_bundles_assign_and_attribute_branches(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Empty:\n"
        "    pass\n"
        "@dataclass\n"
        "class Item:\n"
        "    a: int\n"
        "    b = 1\n"
        "    c, d = (1, 2)\n"
        "def make(alias):\n"
        "    Item(1, 2)\n"
        "    alias.Item(1, 2)\n"
        "    (alias()).Item(1, 2)\n",
        encoding="utf-8",
    )
    symbol_table = da.SymbolTable(
        imports={("mod", "alias"): "external.pkg"},
        internal_roots={"mod"},
        external_filter=True,
    )
    bundles = da._iter_dataclass_call_bundles(
        path,
        project_root=tmp_path,
        dataclass_registry={"mod.Item": ["a", "b"]},
        symbol_table=symbol_table,
        parse_failure_witnesses=[],
    )
    assert ("a", "b") in bundles


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_analyze_decision_surface_indexed_missing_tier_without_require::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_decision_surface_indexed::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_analyze_decision_surface_indexed_missing_tier_without_require(tmp_path: Path) -> None:
    da = _load()
    fn = da.FunctionInfo(
        name="f",
        qual="pkg.f",
        path=tmp_path / "mod.py",
        params=["flag"],
        annots={},
        calls=[],
        unused_params=set(),
        decision_params={"flag"},
        function_span=(1, 0, 1, 8),
    )
    index = da.AnalysisIndex(
        by_name={"f": [fn]},
        by_qual={fn.qual: fn},
        symbol_table=da.SymbolTable(),
        class_index={},
        transitive_callers={fn.qual: set()},
    )
    context = da._IndexedPassContext(
        paths=[fn.path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        parse_failure_witnesses=[],
        analysis_index=index,
    )
    _surfaces, warnings, _rewrites, lint_lines = da._analyze_decision_surface_indexed(
        context,
        spec=da._DIRECT_DECISION_SURFACE_SPEC,
        decision_tiers={"other": 1},
        require_tiers=False,
        forest=da.Forest(),
    )
    assert warnings == []
    assert lint_lines == []


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_project_report_sections_preview_selects_non_empty_preview::dataflow_audit.py::gabion.analysis.dataflow_audit.project_report_sections::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_project_report_sections_preview_selects_non_empty_preview(tmp_path: Path) -> None:
    da = _load()
    report = da.ReportCarrier(
        forest=da.Forest(),
        decision_warnings=["w1"],
        parse_failure_witnesses=[],
    )
    groups_by_path = {tmp_path / "m.py": {"f": [{"x"}]}}
    projected = da.project_report_sections(
        groups_by_path,
        report,
        include_previews=True,
        preview_only=True,
        max_phase="post",
    )
    assert "violations" in projected
    assert any("known_violations" in line for line in projected["violations"])


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_internal_broad_type_lint_lines_indexed_appends_multiple::dataflow_audit.py::gabion.analysis.dataflow_audit._internal_broad_type_lint_lines_indexed::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_internal_broad_type_lint_lines_indexed_appends_multiple(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    info = da.FunctionInfo(
        name="f",
        qual="pkg.f",
        path=path,
        params=["a", "b", "c"],
        annots={"a": "Any", "b": "Any", "c": "Any"},
        calls=[],
        unused_params=set(),
        param_spans={"a": (0, 0, 0, 1), "b": (0, 2, 0, 3)},
    )
    index = da.AnalysisIndex(
        by_name={"f": [info]},
        by_qual={info.qual: info},
        symbol_table=da.SymbolTable(),
        class_index={},
        transitive_callers={info.qual: {"pkg.caller"}},
    )
    context = da._IndexedPassContext(
        paths=[path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        parse_failure_witnesses=[],
        analysis_index=index,
    )
    lines = da._internal_broad_type_lint_lines_indexed(context)
    assert len(lines) == 2
    assert all("GABION_BROAD_TYPE" in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_raw_sorted_contract_violations_multi_path_exceeded_loop::dataflow_audit.py::gabion.analysis.dataflow_audit._raw_sorted_baseline_key::dataflow_audit.py::gabion.analysis.dataflow_audit._raw_sorted_contract_violations::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_raw_sorted_contract_violations_multi_path_exceeded_loop(tmp_path: Path) -> None:
    da = _load()
    left = tmp_path / "a.py"
    right = tmp_path / "b.py"
    left.write_text("def f(xs):\n    return sorted(xs)\n", encoding="utf-8")
    right.write_text("def g(xs):\n    return sorted(xs)\n", encoding="utf-8")
    baseline = {
        da._raw_sorted_baseline_key(left): 0,
        da._raw_sorted_baseline_key(right): 0,
    }
    lines = da._raw_sorted_contract_violations(
        [left, right],
        parse_failure_witnesses=[],
        baseline_counts=baseline,
    )
    assert len(lines) == 2
    assert all("raw_sorted exceeded baseline" in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_fingerprint_warning_provenance_and_rewrite_verification_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_fingerprint_warning_provenance_and_rewrite_verification_edges() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    groups = {path: {"f": [{"x"}], "g": [{"x"}]}}
    annots = {path: {"f": {"x": "int"}, "g": {"x": "int"}}}
    registry = da.PrimeRegistry()
    index = {
        da.bundle_fingerprint_dimensional(["str"], registry, None): {"OtherCtx"},
    }
    warnings = da._compute_fingerprint_warnings(
        groups,
        annots,
        registry=registry,
        index=index,
    )
    assert warnings
    provenance = da._compute_fingerprint_provenance(
        groups,
        annots,
        registry=registry,
        index=index,
    )
    assert provenance
    summary = da._summarize_fingerprint_provenance(provenance, max_examples=1)
    assert any("base=" in line for line in summary)

    rewrite_source = [
        {
            "path": "pkg/mod.py",
            "function": "f",
            "bundle": ["x"],
            "glossary_matches": ["CtxA", "CtxB"],
            "base_keys": ["int"],
            "ctor_keys": [],
            "remainder": {"base": 1, "ctor": 1},
        }
    ]
    plans = da._compute_fingerprint_rewrite_plans(
        rewrite_source,
        coherence=[{"site": {"path": "pkg/mod.py", "function": "f", "bundle": ["x"]}}],
        synth_version="synth@1",
    )
    assert plans
    plan = dict(plans[0])
    plan["verification"] = {"predicates": "not-a-list"}
    verified = da.verify_rewrite_plan(plan, post_provenance=rewrite_source)
    assert isinstance(verified.get("accepted"), bool)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_invariant_term_len_with_non_param_argument_returns_none::dataflow_audit.py::gabion.analysis.dataflow_audit._invariant_term::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_invariant_term_len_with_non_param_argument_returns_none() -> None:
    da = _load()
    expr = ast.parse("len(1)").body[0].value
    assert da._invariant_term(expr, {"data"}) is None


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_parameter_default_map_multiple_defaults_runs_single_check_once::dataflow_audit.py::gabion.analysis.dataflow_audit._parameter_default_map::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_parameter_default_map_multiple_defaults_runs_single_check_once() -> None:
    da = _load()
    fn = ast.parse("def f(a=1, b=2):\n    return a + b\n").body[0]
    mapping = da._parameter_default_map(fn)
    assert set(mapping) == {"a", "b"}


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_raw_sorted_contract_violations_mixed_baseline_paths::dataflow_audit.py::gabion.analysis.dataflow_audit._raw_sorted_baseline_key::dataflow_audit.py::gabion.analysis.dataflow_audit._raw_sorted_contract_violations::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_raw_sorted_contract_violations_mixed_baseline_paths(tmp_path: Path) -> None:
    da = _load()
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("def f(xs):\n    return sorted(xs)\n", encoding="utf-8")
    second.write_text("def g(xs):\n    return sorted(xs)\n", encoding="utf-8")
    baseline = {
        da._raw_sorted_baseline_key(first): 0,
        da._raw_sorted_baseline_key(second): 2,
    }
    lines = da._raw_sorted_contract_violations(
        [first, second],
        parse_failure_witnesses=[],
        baseline_counts=baseline,
    )
    assert len(lines) == 1
    assert "raw_sorted exceeded baseline" in lines[0]


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_fingerprint_provenance_index_lookup_and_types_summary_branch::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_fingerprint_provenance_index_lookup_and_types_summary_branch() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    groups = {path: {"f": [{"x"}], "g": [{"x"}]}}
    annots = {path: {"f": {"x": "int"}, "g": {"x": "int"}}}
    registry = da.PrimeRegistry()
    fp = da.bundle_fingerprint_dimensional(["int"], registry, None)
    provenance = da._compute_fingerprint_provenance(
        groups,
        annots,
        registry=registry,
        index={fp: {"KnownCtx"}},
    )
    assert provenance
    # No glossary matches => "types" grouping branch.
    for entry in provenance:
        entry["glossary_matches"] = []
    lines = da._summarize_fingerprint_provenance(provenance, max_examples=2)
    assert any("base=" in line for line in lines)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_eval_value_expr_unary_non_numeric_and_parse_range_branch::dataflow_audit.py::gabion.analysis.dataflow_audit._eval_value_expr::dataflow_audit.py::gabion.analysis.dataflow_audit._parse_lint_location::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_eval_value_expr_unary_non_numeric_and_parse_range_branch() -> None:
    da = _load()
    unary_outcome = da._eval_value_expr(ast.parse("-'x'").body[0].value, {})
    assert unary_outcome.is_unknown() is True
    assert da._parse_lint_location("a.py:1:2:-3:4: GABION_X message") is not None


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_collect_exception_obligations_names_loop_without_env_match::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_collect_exception_obligations_names_loop_without_env_match(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "m.py"
    path.write_text(
        "def f(x):\n"
        "    if 0 and missing:\n"
        "        return 1\n"
        "    raise ValueError(x)\n",
        encoding="utf-8",
    )
    deadness = [
        {
            "path": "m.py",
            "function": "f",
            "bundle": ["x"],
            "environment": {"x": "0"},
            "deadness_id": "dead:f:x",
        }
    ]
    obligations = da._collect_exception_obligations(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        deadness_witnesses=deadness,
    )
    assert obligations and obligations[0]["status"] in {"UNKNOWN", "DEAD"}


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_collect_call_resolution_obligations_invalid_span_list_raises::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_call_resolution_obligations_from_forest::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_collect_call_resolution_obligations_invalid_span_list_raises() -> None:
    da = _load()
    forest = da.Forest()
    call_suite = forest.add_suite_site(
        "a.py",
        "pkg.f",
        "call",
        span=(1, 0, 1, 1),
    )
    # Replace span metadata with invalid list to take coercion-failure branch.
    forest.nodes[call_suite].meta["span"] = [1, "x", 1, 1]
    forest.add_alt(
        "CallResolutionObligation",
        (call_suite,),
        evidence={"callee": "target"},
    )
    with pytest.raises(NeverThrown):
        da._collect_call_resolution_obligations_from_forest(forest)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_collect_recursive_nodes_singleton_self_loop_false_branch::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_recursive_nodes::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_collect_recursive_nodes_singleton_self_loop_false_branch() -> None:
    da = _load()
    assert da._collect_recursive_nodes({"a": set()}) == set()


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_analysis_index_resolved_edges_by_caller_require_transparent_branch::dataflow_audit.py::gabion.analysis.dataflow_audit._analysis_index_resolved_call_edges_by_caller::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_analysis_index_resolved_edges_by_caller_require_transparent_branch() -> None:
    da = _load()
    index = da.AnalysisIndex(
        by_name={},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_transparent_call_edges=(),
    )
    assert (
        da._analysis_index_resolved_call_edges_by_caller(
            index,
            project_root=None,
            require_transparent=True,
        )
        == {}
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_collect_bundle_evidence_lines_with_component_evidence::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_bundle_evidence_lines::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_collect_bundle_evidence_lines_with_component_evidence() -> None:
    da = _load()
    forest = da.Forest()
    site = forest.add_site("a.py", "f")
    bundle = forest.add_paramset(["x", "y"])
    forest.add_alt("SignatureBundle", (site, bundle))
    groups = {Path("a.py"): {"f": [{"x", "y"}]}}
    bundle_sites = {
        Path("a.py"): {
            "f": [
                [
                    {
                        "span": [0, 0, 0, 1],
                        "callee": "pkg.target",
                        "params": ["x", "y"],
                        "slots": ["x", "y"],
                    }
                ]
            ]
        }
    }
    lines = da._collect_bundle_evidence_lines(
        forest=forest,
        groups_by_path=groups,
        bundle_sites_by_path=bundle_sites,
    )
    assert lines


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_class_index_and_resolve_candidates_with_symbol_table_branches::dataflow_audit.py::gabion.analysis.dataflow_audit._accumulate_class_index_for_tree::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_class_index_and_resolve_candidates_with_symbol_table_branches() -> None:
    da = _load()
    tree = ast.parse("class C(A, B):\n    pass\n")
    class_index: dict[str, da.ClassInfo] = {}
    da._accumulate_class_index_for_tree(
        class_index,
        Path("mod.py"),
        tree,
        project_root=Path("."),
    )
    assert class_index

    symbol_table = da.SymbolTable(imports={("pkg.mod", "pkg"): "pkg"}, internal_roots={"pkg"})
    resolved = da._resolve_class_candidates(
        "pkg.Base",
        module="pkg.mod",
        symbol_table=symbol_table,
        class_index={"pkg.Base": da.ClassInfo("pkg.Base", "pkg", [], set())},
    )
    assert "pkg.Base" in resolved


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_collect_module_exports_augassign_initializes_explicit_all::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_collect_module_exports_augassign_initializes_explicit_all() -> None:
    da = _load()
    exports, _ = da._collect_module_exports(
        ast.parse("__all__ += ['x']\nx=1\n"),
        module_name="pkg.mod",
        import_map={},
    )
    assert "x" in exports


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_render_reuse_stubs_and_refactor_plan_order_branches::dataflow_audit.py::gabion.analysis.dataflow_audit.render_refactor_plan::dataflow_audit.py::gabion.analysis.dataflow_audit.render_reuse_lemma_stubs::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_render_reuse_stubs_and_refactor_plan_order_branches() -> None:
    da = _load()
    reuse = {
        "suggested_lemmas": [
            {
                "kind": "bundle",
                "suggested_name": "lemma_name",
                "count": 2,
                "value": ["x"],
            }
        ]
    }
    stubs = da.render_reuse_lemma_stubs(reuse)
    assert "reuse_rewrite_plan_bundle" in stubs

    text = da.render_refactor_plan(
        {
            "bundles": [
                {"bundle": ["x"], "order": ["a", "b"], "cycles": [["a", "b"]]},
            ],
            "warnings": [],
        }
    )
    assert "Order (callee-first):" in text


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_eval_expr_and_branch_reachability_else_constraint_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._branch_reachability_under_env::dataflow_audit.py::gabion.analysis.dataflow_audit._eval_bool_expr::dataflow_audit.py::gabion.analysis.dataflow_audit._eval_value_expr::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_eval_expr_and_branch_reachability_else_constraint_edges() -> None:
    da = _load()
    unary_plus_outcome = da._eval_value_expr(ast.parse("+2").body[0].value, {})
    assert unary_plus_outcome.is_unknown() is False
    assert unary_plus_outcome.value == 2
    or_outcome = da._eval_bool_expr(
        ast.parse("a or b").body[0].value,
        {"a": False, "b": False},
    )
    assert or_outcome.is_unknown() is False
    assert or_outcome.as_bool() is False
    gte_outcome = da._eval_bool_expr(ast.parse("'x' >= 'a'").body[0].value, {})
    assert gte_outcome.is_unknown() is True

    tree = ast.parse(
        "if flag:\n"
        "    a = 1\n"
        "else:\n"
        "    raise ValueError(flag)\n"
    )
    parent = da.ParentAnnotator()
    parent.visit(tree)
    raise_node = tree.body[0].orelse[0]
    assert (
        da._branch_reachability_under_env(
            raise_node,
            parent.parents,
            {"flag": False},
        )
        is da._EvalDecision.TRUE
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_collect_exception_obligations_dead_env_name_filter_branches::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_collect_exception_obligations_dead_env_name_filter_branches(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "dead_env.py"
    path.write_text(
        "def f(x):\n"
        "    if 0 and missing:\n"
        "        return\n"
        "    raise ValueError(x)\n"
        "def g(y):\n"
        "    if 0:\n"
        "        return\n"
        "    raise RuntimeError(y)\n",
        encoding="utf-8",
    )
    deadness = [
        {
            "path": "dead_env.py",
            "function": "f",
            "bundle": ["x"],
            "environment": {"x": "0"},
            "deadness_id": "dead:f:x",
        },
        {
            "path": "dead_env.py",
            "function": "g",
            "bundle": ["y"],
            "environment": {"y": "0"},
            "deadness_id": "dead:g:y",
        },
    ]
    obligations = da._collect_exception_obligations(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        deadness_witnesses=deadness,
    )
    assert obligations
    assert any(entry.get("status") in {"DEAD", "UNKNOWN"} for entry in obligations)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_deadline_local_info_call_resolution_and_recursive_node_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_call_resolution_obligations_from_forest::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_deadline_local_info::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_recursive_nodes::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_deadline_local_info_call_resolution_and_recursive_node_edges() -> None:
    da = _load()
    fn = ast.parse(
        "def f(deadline):\n"
        "    token = Deadline.from_timeout_ms(1)\n"
        "    alias = token\n"
        "    check_deadline(0)\n"
    ).body[0]
    visitor = da._DeadlineFunctionCollector(fn, {"deadline"})
    visitor.visit(fn)
    local = da._collect_deadline_local_info(visitor.assignments, {"deadline"})
    assert "token" in local.origin_vars or "alias" in local.origin_vars

    forest = da.Forest()
    call_suite = forest.add_suite_site(
        "a.py",
        "pkg.f",
        "call",
        span=(1, 0, 1, 10),
    )
    forest.add_alt(
        "CallResolutionObligation",
        (call_suite,),
        evidence={"callee": "target"},
    )
    assert da._collect_call_resolution_obligations_from_forest(forest)
    assert da._collect_recursive_nodes({"a": set()}) == set()


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_bind_call_args_classify_deadline_and_forward_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._bind_call_args::dataflow_audit.py::gabion.analysis.dataflow_audit._classify_deadline_expr::dataflow_audit.py::gabion.analysis.dataflow_audit._deadline_loop_forwarded_params::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_bind_call_args_classify_deadline_and_forward_edges() -> None:
    da = _load()
    call = ast.parse("fn(1, extra=2, extra2=3)").body[0].value
    callee = da.FunctionInfo(
        name="callee",
        qual="pkg.callee",
        path=Path("m.py"),
        params=["x"],
        annots={},
        calls=[],
        unused_params=set(),
        kwarg="kwargs",
    )
    mapping = da._bind_call_args(call, callee, strictness="high")
    assert "kwargs" in mapping

    assert da._classify_deadline_expr(
        ast.parse("origin_deadline").body[0].value,
        alias_to_param={},
        origin_vars={"origin_deadline"},
    ) == da._DeadlineArgInfo(kind="origin", param="origin_deadline")

    call_map = {"deadline": da._DeadlineArgInfo(kind="param", param="deadline")}
    loop_fact = da._DeadlineLoopFacts(span=(0, 0, 0, 1), kind="for")
    loop_fact.call_spans.add((0, 0, 0, 1))
    call = da.CallArgs(
        callee="callee",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(0, 0, 0, 1),
    )
    callee = da.FunctionInfo(
        name="callee",
        qual="pkg.callee",
        path=Path("m.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
    )
    assert da._deadline_loop_forwarded_params(
        qual="pkg.caller",
        loop_fact=loop_fact,
        deadline_params={"pkg.caller": {"deadline"}, "pkg.callee": {"deadline"}},
        call_infos={"pkg.caller": [(call, callee, call_map)]},
    ) == {"deadline"}


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_deadline_summary_parse_location_and_lint_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._deadline_lint_lines::dataflow_audit.py::gabion.analysis.dataflow_audit._parse_lint_location::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_deadline_obligations::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_deadline_summary_parse_location_and_lint_edges() -> None:
    da = _load()
    forest = da.Forest()
    summary = da._summarize_deadline_obligations(
        [
            {
                "site": {"path": "src/a.py", "function": "f", "bundle": ["x"]},
                "span": [0, 0, 0, 1],
                "status": "UNKNOWN",
                "kind": "k",
                "detail": "d",
                "deadline_id": "id:1",
            }
        ],
        forest=forest,
    )
    assert summary

    lint = da._deadline_lint_lines(
        [{"site": {"path": "a.py"}, "span": [0, 1, 0, 2], "status": "S", "kind": "K"}]
    )
    assert lint and lint[0].startswith("a.py:1:2:")

    parsed = da._parse_lint_location("a.py:1:2:-9:10: GABION_X message")
    assert parsed is not None
    assert parsed[0] == "a.py"


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_build_module_artifacts_and_lint_helper_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._build_module_artifacts::dataflow_audit.py::gabion.analysis.dataflow_audit._lint_lines_from_call_ambiguities::dataflow_audit.py::gabion.analysis.dataflow_audit._lint_lines_from_constant_smells::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_build_module_artifacts_and_lint_helper_edges(tmp_path: Path) -> None:
    da = _load()
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("x = 1\n", encoding="utf-8")
    second.write_text("y = 2\n", encoding="utf-8")

    spec = da._ModuleArtifactSpec(
        artifact_id="count_nodes",
        stage=da._ParseModuleStage.FUNCTION_INDEX,
        init=lambda: [],
        fold=lambda acc, path, tree: acc.append((path.name, len(list(ast.walk(tree))))),
        finish=lambda acc: tuple(acc),
    )
    built = da._build_module_artifacts(
        [first, second],
        specs=(spec,),
        parse_failure_witnesses=[],
    )
    assert len(built[0]) == 2

    ambiguity = da._lint_lines_from_call_ambiguities(
        [{"site": {"path": "a.py", "span": [0, 0, 0, 1]}, "candidate_count": 3}]
    )
    const_lines = da._lint_lines_from_constant_smells(
        ["constant smell (e.g. a.py:2:3: from sample)"]
    )
    assert ambiguity and "GABION_AMBIGUITY" in ambiguity[0]
    assert const_lines and "GABION_CONST_FLOW" in const_lines[0]


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_materialize_structured_suites_populate_runtime_and_exports_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::dataflow_audit.py::gabion.analysis.dataflow_audit._materialize_structured_suite_sites_for_tree::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_runtime_obligations::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_materialize_structured_suites_populate_runtime_and_exports_edges(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def f(x):\n"
        "    if x:\n"
        "        return 1\n"
        "    return 0\n",
        encoding="utf-8",
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forest = da.Forest()
    da._materialize_structured_suite_sites_for_tree(
        forest=forest,
        path=path,
        tree=tree,
        project_root=tmp_path,
    )
    assert any(node.kind == "SuiteSite" for node in forest.nodes)

    groups = {path: {"f": [{"x", "y"}]}}
    da._populate_bundle_forest(
        forest,
        groups_by_path=groups,
        file_paths=[path],
        project_root=tmp_path,
        include_all_sites=True,
        parse_failure_witnesses=[],
    )
    assert forest.alts

    runtime_lines = da._summarize_runtime_obligations(
        [
            {"status": "VIOLATION", "contract": "c", "kind": "k", "detail": "d1"},
            {"status": "SATISFIED", "contract": "c", "kind": "k", "detail": "d2"},
        ],
        max_entries=1,
    )
    assert any(line.startswith("... ") for line in runtime_lines)

    exports, _ = da._collect_module_exports(
        ast.parse("__all__ += ['a']\na=1\n"),
        module_name="pkg.mod",
        import_map={},
    )
    assert "a" in exports


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_class_resolution_type_flow_and_refactor_render_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._constant_smells_from_details::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_type_flow::dataflow_audit.py::gabion.analysis.dataflow_audit._render_type_mermaid::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_class_resolution_type_flow_and_refactor_render_edges(tmp_path: Path) -> None:
    da = _load()
    # _resolve_class_candidates dotted + module path branch.
    class_index = {"pkg.mod.Base": da.ClassInfo("pkg.mod.Base", "pkg.mod", [], set())}
    candidates = da._resolve_class_candidates(
        "Base",
        module="pkg.mod",
        symbol_table=None,
        class_index=class_index,
    )
    assert "pkg.mod.Base" in candidates

    # _resolve_method_in_hierarchy unresolved return path.
    unresolved = da._resolve_method_in_hierarchy(
        "pkg.mod.Base",
        "missing",
        class_index=class_index,
        by_qual={},
        symbol_table=None,
        seen=set(),
    )
    assert unresolved is None

    # _resolve_callee len(parts) == 2 branch.
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.mod.caller",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
    )
    target = da.FunctionInfo(
        name="m",
        qual="mod.Base.m",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name="Base",
    )
    resolved = da._resolve_callee(
        "Base.m",
        caller,
        by_name={"m": [target]},
        by_qual={target.qual: target},
        symbol_table=da.SymbolTable(),
        project_root=tmp_path,
        class_index={"mod.Base": da.ClassInfo("mod.Base", "mod", [], {"m"})},
    )
    assert resolved is target

    # _infer_type_flow changed=True branch and constant smell site suffix.
    callee = da.FunctionInfo(
        name="callee",
        qual="pkg.mod.callee",
        path=tmp_path / "mod.py",
        params=["p"],
        annots={"p": "int"},
        calls=[],
        unused_params=set(),
    )
    caller_flow = da.FunctionInfo(
        name="caller",
        qual="pkg.mod.caller_flow",
        path=tmp_path / "mod.py",
        params=["p"],
        annots={"p": "Any"},
        calls=[],
        unused_params=set(),
    )
    edge = da._ResolvedCallEdge(
        caller=caller_flow,
        callee=callee,
        call=da.CallArgs(
            callee="callee",
            pos_map={"0": "p"},
            kw_map={},
            const_pos={},
            const_kw={},
            non_const_pos=set(),
            non_const_kw=set(),
            star_pos=[],
            star_kw=[],
            is_test=False,
            span=(0, 0, 0, 1),
        ),
    )
    index = da.AnalysisIndex(
        by_name={"caller_flow": [caller_flow], "callee": [callee]},
        by_qual={caller_flow.qual: caller_flow, callee.qual: callee},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_transparent_edges_by_caller={caller_flow.qual: (edge,)},
    )
    _inferred, suggestions, _ambiguities, _evidence = da._infer_type_flow(
        [tmp_path / "mod.py"],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=[],
        analysis_index=index,
    )
    assert suggestions

    smells = da._constant_smells_from_details(
        [
            da.ConstantFlowDetail(
                path=tmp_path / "mod.py",
                qual="pkg.mod.caller_flow",
                name="caller_flow",
                param="p",
                value="1",
                count=2,
                sites=("a.py:1:2: loc",),
            )
        ]
    )
    assert "(e.g." in smells[0]

    # _render_type_mermaid item filtering branch.
    mermaid = da._render_type_mermaid(
        [],
        ["f downstream types conflict: ['int', '', 'str']"],
    )
    assert "int" in mermaid and "str" in mermaid


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_scope_normalization_and_timeout_cleanup_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._analysis_deadline_scope::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_scope_normalization_and_timeout_cleanup_edges(tmp_path: Path) -> None:
    da = _load()
    assert da._normalize_transparent_decorators(["  a,b  "]) == {"a", "b"}

    # _analysis_deadline_scope with default tick_limit path.
    args = argparse.Namespace(
        analysis_timeout_ticks=10,
        analysis_timeout_tick_ns=1_000_000,
        analysis_tick_limit=None,
    )
    with da._analysis_deadline_scope(args):
        da.check_deadline()

    # analyze_paths timeout cleanup should still flush best-effort emitters.
    timed_out = False
    try:
        with da.deadline_scope(da.Deadline.from_timeout_ticks(1, 1)):
            with da.deadline_clock_scope(da.GasMeter(limit=1)):
                da.analyze_paths(
                    [tmp_path / "missing.py"],
                    forest=da.Forest(),
                    recursive=True,
                    type_audit=False,
                    type_audit_report=False,
                    type_audit_max=10,
                    include_constant_smells=False,
                    include_unused_arg_smells=False,
                    include_bundle_forest=False,
                    include_lint_lines=False,
                    config=da.AuditConfig(project_root=tmp_path),
                )
    except da.TimeoutExceeded:
        timed_out = True
    assert timed_out is True


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_additional_branch_edges_scalar_helpers::dataflow_audit.py::gabion.analysis.dataflow_audit._invariant_term::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_transparent_decorators::dataflow_audit.py::gabion.analysis.dataflow_audit._parameter_default_map::dataflow_audit.py::gabion.analysis.dataflow_audit._parse_lint_location::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_additional_branch_edges_scalar_helpers() -> None:
    da = _load()
    # _invariant_term outer conditional false path.
    assert da._invariant_term(ast.parse("42").body[0].value, {"x"}) is None

    # _parameter_default_map re-enters default loop after first element.
    fn = ast.parse("def f(a=1, b=2, c=3):\n    return a\n").body[0]
    mapping = da._parameter_default_map(fn)
    assert set(mapping) == {"a", "b", "c"}

    # _parse_lint_location range marker branch with no numeric range match.
    parsed = da._parse_lint_location("a.py:1:2:-x trailing")
    assert parsed is not None

    # _normalize_transparent_decorators non-iterable branch.
    assert da._normalize_transparent_decorators(123) is None


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_additional_branch_edges_fingerprint_and_rewrite::dataflow_audit.py::gabion.analysis.dataflow_audit._branch_reachability_under_env::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_additional_branch_edges_fingerprint_and_rewrite() -> None:
    da = _load()
    path = Path("pkg/mod.py")
    groups = {path: {"f": [{"x"}]}}
    annots = {path: {"f": {"x": "int"}}}
    registry = da.PrimeRegistry()
    fp = da.bundle_fingerprint_dimensional(["int"], registry, None)

    # _compute_fingerprint_warnings: names present path skips warning.
    assert (
        da._compute_fingerprint_warnings(
            groups,
            annots,
            registry=registry,
            index={fp: {"Ctx"}},
        )
        == []
    )

    # _compute_fingerprint_provenance: branch where index is empty.
    assert da._compute_fingerprint_provenance(
        groups,
        annots,
        registry=registry,
        index={},
    )

    # _summarize_fingerprint_provenance: key/type branch.
    lines = da._summarize_fingerprint_provenance(
        [
            {
                "path": "pkg/mod.py",
                "function": "f",
                "bundle": ["x"],
                "base_keys": ["int"],
                "ctor_keys": [],
                "glossary_matches": [],
            }
        ]
    )
    assert any("base=" in line for line in lines)

    # _compute_fingerprint_rewrite_plans branch when coherence entry missing.
    plans = da._compute_fingerprint_rewrite_plans(
        [
            {
                "path": "pkg/mod.py",
                "function": "f",
                "bundle": ["x"],
                "glossary_matches": ["A", "B"],
                "base_keys": ["int"],
                "ctor_keys": [],
                "remainder": {"base": 0, "ctor": 0},
            }
        ],
        coherence=[],
        synth_version="synth@1",
    )
    assert plans

    # verify_rewrite_plan verification non-dict branch.
    plan = dict(plans[0])
    plan["verification"] = []
    assert "accepted" in da.verify_rewrite_plan(plan, post_provenance=plan["site"] and [])

    # _branch_reachability_under_env: node not in body/orelse branch.
    tree = ast.parse("if cond:\n    x = 1\nelse:\n    y = 2\n")
    parent = da.ParentAnnotator()
    parent.visit(tree)
    if_node = tree.body[0]
    assert (
        da._branch_reachability_under_env(if_node.test, parent.parents, {"cond": True})
        is da._EvalDecision.UNKNOWN
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_additional_branch_edges_flow_and_render::dataflow_audit.py::gabion.analysis.dataflow_audit._analysis_deadline_scope::dataflow_audit.py::gabion.analysis.dataflow_audit._analysis_index_resolved_call_edges_by_caller::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_bundle_evidence_lines::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_call_resolution_obligations_from_forest::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_deadline_local_info::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_additional_branch_edges_flow_and_render(tmp_path: Path) -> None:
    da = _load()
    # _DeadlineFunctionCollector elif non-name/non-attribute branch.
    fn = ast.parse("def f(deadline):\n    (lambda x: x)(deadline)\n").body[0]
    collector = da._DeadlineFunctionCollector(fn, {"deadline"})
    collector.visit(fn)
    assert collector.check_params == set()

    # _collect_deadline_local_info unknown alias source branch.
    assignments = [
        (
            [ast.Name(id="x", ctx=ast.Store())],
            ast.Name(id="unknown", ctx=ast.Load()),
            (0, 0, 0, 1),
        )
    ]
    info = da._collect_deadline_local_info(assignments, {"deadline"})
    assert "x" not in info.alias_to_param

    # _collect_call_resolution_obligations_from_forest non-list span branch.
    forest = da.Forest()
    suite = forest.add_suite_site("a.py", "pkg.f", "call", span=(1, 0, 1, 1))
    forest.nodes[suite].meta["span"] = (1, 0, 1, 1)
    forest.add_alt("CallResolutionObligation", (suite,), evidence={"callee": "c"})
    with pytest.raises(NeverThrown):
        da._collect_call_resolution_obligations_from_forest(forest)

    # _analysis_index_resolved_call_edges_by_caller require_transparent false path.
    index = da.AnalysisIndex(
        by_name={},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_call_edges=(),
    )
    assert (
        da._analysis_index_resolved_call_edges_by_caller(
            index,
            project_root=None,
            require_transparent=False,
        )
        == {}
    )

    # _collect_bundle_evidence_lines branch when first component yields evidence.
    site_a = forest.add_site("a.py", "f")
    bundle_a = forest.add_paramset(["x", "y"])
    forest.add_alt("SignatureBundle", (site_a, bundle_a))
    site_b = forest.add_site("b.py", "g")
    bundle_b = forest.add_paramset(["u", "v"])
    forest.add_alt("SignatureBundle", (site_b, bundle_b))
    groups = {Path("a.py"): {"f": [{"x", "y"}]}, Path("b.py"): {"g": [{"u", "v"}]}}
    bundle_sites = {
        Path("a.py"): {
            "f": [[{"span": [0, 0, 0, 1], "callee": "t", "params": ["x"], "slots": ["x"]}]]
        },
        Path("b.py"): {
            "g": [[{"span": [0, 0, 0, 1], "callee": "t", "params": ["u"], "slots": ["u"]}]]
        },
    }
    assert da._collect_bundle_evidence_lines(
        forest=forest,
        groups_by_path=groups,
        bundle_sites_by_path=bundle_sites,
    )

    # _emit_report type_suggestions branch.
    report = da.ReportCarrier(
        forest=forest,
        type_suggestions=["a.py:f.p can tighten to int"],
        type_ambiguities=[],
    )
    markdown, _ = da._emit_report({}, 1, report=report)
    assert "Type tightening candidates:" in markdown

    # _analysis_deadline_scope exceptional exit path.
    args = argparse.Namespace(
        analysis_timeout_ticks=10,
        analysis_timeout_tick_ns=1_000,
        analysis_tick_limit=5,
    )
    with pytest.raises(RuntimeError):
        with da._analysis_deadline_scope(args):
            raise RuntimeError("boom")


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_additional_branch_edges_reporting_and_exports::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_runtime_obligations::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_additional_branch_edges_reporting_and_exports(tmp_path: Path) -> None:
    da = _load()
    # _summarize_runtime_obligations with empty detail branch.
    lines = da._summarize_runtime_obligations(
        [{"status": "S", "contract": "c", "kind": "k", "detail": ""}]
    )
    assert lines and "detail=" not in lines[0]

    # _collect_module_exports assignment/augassign value branches.
    tree = ast.parse(
        "__all__ = ['a']\n"
        "__all__ += ['b']\n"
        "a = 1\n"
        "b = 2\n"
    )
    exports, _ = da._collect_module_exports(tree, module_name="pkg.mod", import_map={})
    assert {"a", "b"} <= exports

    # _emit_report type_suggestions disabled branch.
    report = da.ReportCarrier(
        forest=da.Forest(),
        type_suggestions=[],
        type_ambiguities=["a.py:f.p downstream types conflict: ['int']"],
    )
    markdown, _ = da._emit_report({}, 1, report=report)
    assert "Type ambiguities" in markdown


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_additional_branch_edges_structure_materialization::dataflow_audit.py::gabion.analysis.dataflow_audit._materialize_structured_suite_sites_for_tree::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_additional_branch_edges_structure_materialization(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text("def f(x):\n    return x\n", encoding="utf-8")
    forest = da.Forest()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    da._materialize_structured_suite_sites_for_tree(
        forest=forest,
        path=path,
        tree=tree,
        project_root=tmp_path,
    )

    # Function with no lineno => function_span is None branch.
    synthetic_fn = ast.FunctionDef(
        name="g",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass()],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    synthetic_tree = ast.Module(body=[synthetic_fn], type_ignores=[])
    da._materialize_structured_suite_sites_for_tree(
        forest=forest,
        path=path,
        tree=synthetic_tree,
        project_root=tmp_path,
    )
    assert any(node.kind == "SuiteSite" for node in forest.nodes.values())

    groups = {path: {"f": [{"x", "y"}]}}
    da._populate_bundle_forest(
        forest,
        groups_by_path=groups,
        file_paths=[path],
        project_root=tmp_path,
        include_all_sites=False,
        parse_failure_witnesses=[],
    )
    assert forest.alts


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_additional_branch_edges_class_and_call_resolution::dataflow_audit.py::gabion.analysis.dataflow_audit._accumulate_class_index_for_tree::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_additional_branch_edges_class_and_call_resolution(tmp_path: Path) -> None:
    da = _load()
    # _accumulate_class_index_for_tree base_name false path.
    class_index: dict[str, da.ClassInfo] = {}
    tree = ast.parse("class C(42):\n    pass\n")
    da._accumulate_class_index_for_tree(
        class_index,
        Path("mod.py"),
        tree,
        project_root=Path("."),
    )
    assert class_index

    # _resolve_class_candidates resolved_head branch.
    symbol_table = da.SymbolTable(imports={("pkg.mod", "pkg"): "pkg"}, internal_roots={"pkg"})
    candidates = da._resolve_class_candidates(
        "pkg.Base",
        module="pkg.mod",
        symbol_table=symbol_table,
        class_index={"pkg.Base": da.ClassInfo("pkg.Base", "pkg", [], set())},
    )
    assert "pkg.Base" in candidates

    # _resolve_method_in_hierarchy resolved recursion branch.
    by_qual = {
        "mod.Base.m": da.FunctionInfo(
            name="m",
            qual="mod.Base.m",
            path=tmp_path / "mod.py",
            params=[],
            annots={},
            calls=[],
            unused_params=set(),
            class_name="Base",
        )
    }
    resolved = da._resolve_method_in_hierarchy(
        "mod.Child",
        "m",
        class_index={
            "mod.Base": da.ClassInfo("mod.Base", "mod", [], {"m"}),
            "mod.Child": da.ClassInfo("mod.Child", "mod", ["Base"], set()),
        },
        by_qual=by_qual,
        symbol_table=da.SymbolTable(),
        seen=set(),
    )
    assert resolved is not None

    # _resolve_callee candidate-in-by_qual branch.
    caller = da.FunctionInfo(
        name="caller",
        qual="mod.Caller.caller",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name="Caller",
    )
    target = da.FunctionInfo(
        name="m",
        qual="mod.Caller.m",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name="Caller",
    )
    assert da._resolve_callee(
        "self.m",
        caller,
        by_name={"m": [target]},
        by_qual={target.qual: target},
        symbol_table=da.SymbolTable(),
        project_root=tmp_path,
        class_index={"mod.Caller": da.ClassInfo("mod.Caller", "mod", [], {"m"})},
    ) is target


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_additional_branch_edges_flow_and_registry::dataflow_audit.py::gabion.analysis.dataflow_audit._constant_smells_from_details::dataflow_audit.py::gabion.analysis.dataflow_audit._dataclass_registry_for_tree::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_type_flow::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_config_fields::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::dataflow_audit.py::gabion.analysis.dataflow_audit._paramset_key::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_additional_branch_edges_flow_and_registry(tmp_path: Path) -> None:
    da = _load()
    # _infer_type_flow changed-loop branch.
    callee = da.FunctionInfo(
        name="callee",
        qual="pkg.mod.callee",
        path=tmp_path / "mod.py",
        params=["p"],
        annots={"p": "int"},
        calls=[],
        unused_params=set(),
    )
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.mod.caller",
        path=tmp_path / "mod.py",
        params=["p"],
        annots={"p": "Any"},
        calls=[],
        unused_params=set(),
    )
    edge = da._ResolvedCallEdge(
        caller=caller,
        callee=callee,
        call=da.CallArgs(
            callee="callee",
            pos_map={"0": "p"},
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
    )
    index = da.AnalysisIndex(
        by_name={"caller": [caller], "callee": [callee]},
        by_qual={caller.qual: caller, callee.qual: callee},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_transparent_edges_by_caller={caller.qual: (edge,)},
    )
    _inferred, suggestions, _ambig, _ev = da._infer_type_flow(
        [tmp_path / "mod.py"],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=[],
        analysis_index=index,
    )
    assert suggestions

    # _constant_smells_from_details site suffix false path.
    smells = da._constant_smells_from_details(
        [
            da.ConstantFlowDetail(
                path=tmp_path / "mod.py",
                qual="q",
                name="f",
                param="p",
                value="1",
                count=1,
                sites=(),
            )
        ]
    )
    assert "(e.g." not in smells[0]

    # _iter_config_fields / _dataclass_registry_for_tree non-name assign target branches.
    config_tree = ast.parse("class C:\n    x, y = (1, 2)\n")
    assert da._iter_config_fields(
        Path("m.py"),
        tree=config_tree,
        parse_failure_witnesses=[],
    ) == {}
    dataclass_tree = ast.parse(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class D:\n"
        "    a, b = (1, 2)\n"
    )
    assert da._dataclass_registry_for_tree(
        Path("m.py"),
        dataclass_tree,
        project_root=Path("."),
    ) == {}

    # _iter_dataclass_call_bundles candidate branch.
    mod = tmp_path / "mod.py"
    mod.write_text(
        "def make(alias):\n"
        "    alias.Item(1)\n",
        encoding="utf-8",
    )
    symbol_table = da.SymbolTable(
        imports={("mod", "alias"): "pkg"},
        internal_roots={"pkg"},
        external_filter=True,
        star_imports={"mod": {"pkg"}},
    )
    assert da._iter_dataclass_call_bundles(
        mod,
        project_root=tmp_path,
        symbol_table=symbol_table,
        dataclass_registry={"pkg.Item": ["x", "y"]},
        parse_failure_witnesses=[],
    ) == set()

    # _paramset_key non-list params metadata branch.
    forest = da.Forest()
    paramset = forest.add_paramset(["x", "y"])
    forest.nodes[paramset].meta["params"] = "x,y"
    assert da._paramset_key(forest, paramset) == tuple(str(p) for p in paramset.key)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_additional_branch_edges_rendering_variants::dataflow_audit.py::gabion.analysis.dataflow_audit._render_type_mermaid::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::dataflow_audit.py::gabion.analysis.dataflow_audit.render_refactor_plan::dataflow_audit.py::gabion.analysis.dataflow_audit.render_reuse_lemma_stubs::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_additional_branch_edges_rendering_variants() -> None:
    da = _load()
    # compute_structure_reuse basic branch.
    snapshot = {
        "files": [
            {"path": "a.py", "functions": [{"name": "f", "bundles": [["x", "y"]]}]},
            {"path": "b.py", "functions": [{"name": "g", "bundles": [["x", "y"]]}]},
        ]
    }
    reuse = da.compute_structure_reuse(snapshot)
    assert reuse["suggested_lemmas"]

    # render_reuse_lemma_stubs value-none branch.
    stubs = da.render_reuse_lemma_stubs(
        {"suggested_lemmas": [{"kind": "bundle", "suggested_name": "lemma", "count": 1}]}
    )
    assert "\"plans\"" in stubs

    # render_refactor_plan empty order branch.
    plan_text = da.render_refactor_plan({"bundles": [{"bundle": ["x"], "order": [], "cycles": []}]})
    assert "Bundle: x" in plan_text

    # _render_type_mermaid bracket-strip branch.
    mermaid = da._render_type_mermaid([], ["f downstream types conflict: ['int', 'str']"])
    assert "int" in mermaid and "str" in mermaid


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_branch_shifted_lint_and_projection_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._build_module_artifacts::dataflow_audit.py::gabion.analysis.dataflow_audit._deadline_lint_lines::dataflow_audit.py::gabion.analysis.dataflow_audit._lint_lines_from_call_ambiguities::dataflow_audit.py::gabion.analysis.dataflow_audit._span_line_col::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_branch_shifted_lint_and_projection_edges(tmp_path: Path) -> None:
    da = _load()
    # _span_line_col helper + _deadline_lint_lines invalid span shape.
    assert da._span_line_col([1, 2, 3, 4]) == (2, 3)
    assert da._span_line_col([1, 2, 3]) == (None, None)
    lint_lines = da._deadline_lint_lines(
        [
            {
                "site": {"path": "a.py"},
                "span": {"bad": 1},
                "status": "UNKNOWN",
                "kind": "k",
                "detail": "d",
            }
        ]
    )
    assert lint_lines and "a.py:1:1" in lint_lines[0]

    # _lint_lines_from_call_ambiguities invalid span shape branch.
    ambiguity_lines = da._lint_lines_from_call_ambiguities(
        [
            {
                "site": {"path": "b.py", "span": (0, 0, 0, 0)},
                "candidate_count": "x",
                "kind": "ambiguous",
            }
        ]
    )
    assert ambiguity_lines and "b.py:1:1" in ambiguity_lines[0]

    # _build_module_artifacts parse-cache hit branch for duplicate paths.
    source = tmp_path / "m.py"
    source.write_text("x = 1\n", encoding="utf-8")
    parse_calls = 0

    def _parse(path: Path) -> ast.Module:
        nonlocal parse_calls
        parse_calls += 1
        return ast.parse(path.read_text(encoding="utf-8"))

    artifacts = da._build_module_artifacts(
        [source, source],
        specs=(
            da._ModuleArtifactSpec[list[str], tuple[str, ...]](
                artifact_id="mods",
                stage="scan",
                init=list,
                fold=lambda acc, path, tree: acc.append(
                    f"{path.name}:{len(getattr(tree, 'body', []))}"
                ),
                finish=tuple,
            ),
        ),
        parse_failure_witnesses=[],
        parse_module=_parse,
    )
    assert parse_calls == 1
    assert artifacts == (("m.py:1", "m.py:1"),)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_branch_shifted_rewrite_and_resolution_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._bind_call_args::dataflow_audit.py::gabion.analysis.dataflow_audit._classify_deadline_expr::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::dataflow_audit.py::gabion.analysis.dataflow_audit._deadline_loop_forwarded_params::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_branch_shifted_rewrite_and_resolution_edges(tmp_path: Path) -> None:
    da = _load()
    # verify_rewrite_plan non-dict verification payload (normalized fallback).
    plan = {
        "plan_id": "p",
        "site": {"path": "a.py", "function": "f", "bundle": ["x"]},
        "pre": {"base_keys": ["int"], "ctor_keys": [], "remainder": {"base": 0, "ctor": 0}},
        "rewrite": {"parameters": {"candidates": ["Ctx"]}},
        "post_expectation": {"match_strata": "exact"},
        "verification": [1],
    }
    post = [
        {
            "path": "a.py",
            "function": "f",
            "bundle": ["x"],
            "base_keys": ["int"],
            "ctor_keys": [],
            "glossary_matches": ["Ctx"],
            "remainder": {"base": 0, "ctor": 0},
        }
    ]
    verified = da.verify_rewrite_plan(plan, post_provenance=post)
    assert verified["accepted"] in {True, False}
    assert verified["predicate_results"]

    # _bind_call_args unknown keyword when callee has no **kwargs.
    call = ast.parse("f(x=1)").body[0].value
    assert isinstance(call, ast.Call)
    callee = da.FunctionInfo(
        name="f",
        qual="pkg.f",
        path=tmp_path / "m.py",
        params=["a"],
        annots={},
        calls=[],
        unused_params=set(),
        positional_params=("a",),
        kwonly_params=(),
        vararg=None,
        kwarg=None,
    )
    assert da._bind_call_args(call, callee, strictness="high") == {}

    # _classify_deadline_expr name unmatched in alias/origin maps.
    info = da._classify_deadline_expr(
        ast.Name(id="mystery", ctx=ast.Load()),
        alias_to_param={},
        origin_vars=set(),
    )
    assert info.kind == "unknown"

    # _deadline_loop_forwarded_params false branch for param not in caller set.
    forwarded = da._deadline_loop_forwarded_params(
        qual="pkg.caller",
        loop_fact=da._DeadlineLoopFacts(
            span=None,
            kind="loop",
            call_spans={(1, 0, 1, 1)},
        ),
        deadline_params={"pkg.caller": {"deadline"}, "pkg.callee": {"deadline"}},
        call_infos={
            "pkg.caller": [
                (
                    da.CallArgs(
                        callee="callee",
                        pos_map={},
                        kw_map={},
                        const_pos={},
                        const_kw={},
                        non_const_pos=set(),
                        non_const_kw=set(),
                        star_pos=[],
                        star_kw=[],
                        is_test=False,
                        span=(1, 0, 1, 1),
                    ),
                    da.FunctionInfo(
                        name="callee",
                        qual="pkg.callee",
                        path=tmp_path / "m.py",
                        params=["deadline"],
                        annots={},
                        calls=[],
                        unused_params=set(),
                    ),
                    {"deadline": da._DeadlineArgInfo(kind="param", param="other")},
                )
            ]
        },
    )
    assert forwarded == set()

    # _resolve_class_candidates dotted-head unresolved branch.
    candidates = da._resolve_class_candidates(
        "alias.Type",
        module="pkg.mod",
        symbol_table=da.SymbolTable(
            imports={("pkg.mod", "alias"): "external.alias"},
            internal_roots={"pkg"},
            external_filter=True,
        ),
        class_index={
            "pkg.mod.alias.Type": da.ClassInfo("pkg.mod.alias.Type", "pkg.mod", [], set())
        },
    )
    assert "pkg.mod.alias.Type" in candidates

    # _resolve_method_in_hierarchy recursion returns None branch.
    assert (
        da._resolve_method_in_hierarchy(
            "pkg.Child",
            "m",
            class_index={
                "pkg.Child": da.ClassInfo("pkg.Child", "pkg", ["Base"], set()),
                "pkg.Base": da.ClassInfo("pkg.Base", "pkg", [], set()),
            },
            by_qual={},
            symbol_table=da.SymbolTable(),
            seen=set(),
        )
        is None
    )

    # _resolve_callee self.method candidate missing in by_qual branch.
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.Mod.caller",
        path=tmp_path / "m.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name="Mod",
    )
    assert (
        da._resolve_callee(
            "self.missing",
            caller,
            by_name={},
            by_qual={},
            symbol_table=da.SymbolTable(),
            project_root=tmp_path,
            class_index={"pkg.Mod": da.ClassInfo("pkg.Mod", "pkg", [], set())},
        )
        is None
    )

    # _compute_fingerprint_warnings missing-glossary warning branch.
    registry = da.PrimeRegistry()
    bundle_fp = da.bundle_fingerprint_dimensional(["int"], registry, None)
    other_fp = da.bundle_fingerprint_dimensional(["str"], registry, None)
    warnings = da._compute_fingerprint_warnings(
        {Path("mod.py"): {"f": [{"x"}]}},
        {Path("mod.py"): {"f": {"x": "int"}}},
        registry=registry,
        index={other_fp: {"Ctx"}},
    )
    assert warnings and "missing glossary match" in warnings[0]


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_branch_shifted_flow_and_obligation_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_unused_arg_flow_indexed::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_knob_param_names::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_type_flow::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_branch_shifted_flow_and_obligation_edges(tmp_path: Path) -> None:
    da = _load()
    mod = tmp_path / "mod.py"
    mod.write_text(
        "def f(flag):\n"
        "    if False:\n"
        "        raise ValueError(flag)\n",
        encoding="utf-8",
    )
    path_value = da._normalize_snapshot_path(mod, tmp_path)
    obligations = da._collect_exception_obligations(
        [mod],
        project_root=tmp_path,
        ignore_params=set(),
        handledness_witnesses=[],
        deadness_witnesses=[
            {
                "path": path_value,
                "function": "f",
                "bundle": ["flag"],
                "environment": {"flag": "False"},
                "deadness_id": "dead:f",
            }
        ],
        never_exceptions=set(),
    )
    assert obligations and obligations[0]["status"] in {"UNKNOWN", "DEAD"}

    logical_path = Path("pkg/core.py")

    # _compute_knob_param_names const(None) branch.
    callee = da.FunctionInfo(
        name="callee",
        qual="pkg.callee",
        path=logical_path,
        params=["p"],
        annots={},
        calls=[],
        unused_params=set(),
        defaults={"p"},
    )
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.caller",
        path=logical_path,
        params=["x"],
        annots={},
        calls=[],
        unused_params=set(),
    )
    edge = da._ResolvedCallEdge(
        caller=caller,
        callee=callee,
        call=da.CallArgs(
            callee="callee",
            pos_map={},
            kw_map={},
            const_pos={"0": None},  # type: ignore[dict-item]
            const_kw={},
            non_const_pos=set(),
            non_const_kw=set(),
            star_pos=[],
            star_kw=[],
            is_test=False,
            span=None,
        ),
    )
    caller.calls.append(edge.call)
    index = da.AnalysisIndex(
        by_name={"caller": [caller], "callee": [callee]},
        by_qual={caller.qual: caller, callee.qual: callee},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_transparent_edges_by_caller={caller.qual: (edge,)},
    )
    assert (
        da._compute_knob_param_names(
            by_name=index.by_name,
            by_qual=index.by_qual,
            symbol_table=index.symbol_table,
            project_root=tmp_path,
            class_index=index.class_index,
            strictness="high",
            analysis_index=index,
        )
        == set()
    )

    # _collect_constant_flow_details const event with countable=False branch.
    def _iter_events(
        _edge: da._ResolvedCallEdge, *, strictness: str, include_variadics_in_low_star: bool
    ) -> list[da._ResolvedEdgeParamEvent]:
        del strictness, include_variadics_in_low_star
        return [da._ResolvedEdgeParamEvent(kind="const", param="p", value="1", countable=False)]

    def _reduce(
        _index: da.AnalysisIndex,
        *,
        project_root: Path | None,
        require_transparent: bool,
        spec: da._ResolvedEdgeReducerSpec[da._ConstantFlowFoldAccumulator, da._ConstantFlowFoldAccumulator],
    ) -> da._ConstantFlowFoldAccumulator:
        del project_root, require_transparent
        acc = spec.init()
        spec.fold(acc, edge)
        return spec.finish(acc)

    details = da._collect_constant_flow_details(
        [mod],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=[],
        analysis_index=index,
        iter_resolved_edge_param_events_fn=_iter_events,
        reduce_resolved_call_edges_fn=_reduce,
    )
    assert details and details[0].count == 0

    # _infer_type_flow no-op update branch (existing inferred annotation matches downstream).
    callee_any = da.FunctionInfo(
        name="callee_any",
        qual="pkg.callee_any",
        path=logical_path,
        params=["p"],
        annots={"p": "Any"},
        calls=[],
        unused_params=set(),
        transparent=True,
    )
    caller_any_call = da.CallArgs(
        callee="callee_any",
        pos_map={"0": "p"},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=(0, 0, 0, 1),
    )
    caller_any = da.FunctionInfo(
        name="caller_any",
        qual="pkg.caller_any",
        path=logical_path,
        params=["p"],
        annots={"p": "Any"},
        calls=[],
        unused_params=set(),
    )
    index_any = da.AnalysisIndex(
        by_name={"caller_any": [caller_any], "callee_any": [callee_any]},
        by_qual={caller_any.qual: caller_any, callee_any.qual: callee_any},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_transparent_edges_by_caller={
            caller_any.qual: (
                da._ResolvedCallEdge(
                    caller=caller_any,
                    callee=callee_any,
                    call=caller_any_call,
                ),
            )
        },
    )
    inferred_any, suggestions_any, _amb_any, _ev_any = da._infer_type_flow(
        [logical_path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=[],
        analysis_index=index_any,
    )
    assert inferred_any[caller_any.qual]["p"] == "Any"
    assert any("caller_any.p" in entry for entry in suggestions_any)

    # _analyze_unused_arg_flow_indexed call-span missing + non-const unused negative branches.
    logical_path = Path("pkg/mod.py")
    unused_call = da.CallArgs(
        callee="target",
        pos_map={"0": "x"},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos={"1"},
        non_const_kw={"v"},
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=None,
    )
    callee_unused = da.FunctionInfo(
        name="target",
        qual="pkg.target",
        path=logical_path,
        params=["u", "v"],
        annots={},
        calls=[],
        unused_params={"u"},
        transparent=True,
    )
    caller_unused = da.FunctionInfo(
        name="caller_unused",
        qual="pkg.caller_unused",
        path=logical_path,
        params=["x"],
        annots={},
        calls=[unused_call],
        unused_params=set(),
    )
    unused_edge = da._ResolvedCallEdge(
        caller=caller_unused,
        callee=callee_unused,
        call=unused_call,
    )
    unused_index = da.AnalysisIndex(
        by_name={"caller_unused": [caller_unused], "target": [callee_unused]},
        by_qual={caller_unused.qual: caller_unused, callee_unused.qual: callee_unused},
        symbol_table=da.SymbolTable(),
        class_index={},
        resolved_transparent_edges_by_caller={caller_unused.qual: (unused_edge,)},
    )
    smells = da._analyze_unused_arg_flow_indexed(
        da._IndexedPassContext(
            paths=[logical_path],
            project_root=tmp_path,
            ignore_params=set(),
            strictness="high",
            external_filter=True,
            transparent_decorators=None,
            parse_failure_witnesses=[],
            analysis_index=unused_index,
        )
    )
    assert any(":caller_unused passes param x" in entry for entry in smells)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_branch_shifted_exports_refactor_and_scope_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._analysis_deadline_scope::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_config_bundles::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::dataflow_audit.py::gabion.analysis.dataflow_audit._render_type_mermaid::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::dataflow_audit.py::gabion.analysis.dataflow_audit.build_refactor_plan::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_branch_shifted_exports_refactor_and_scope_edges(tmp_path: Path) -> None:
    da = _load()
    # _collect_module_exports __all__ value parse miss branches.
    tree = ast.parse(
        "__all__: list[str] = 0\n"
        "__all__ += 0\n"
        "def keep():\n"
        "    return 1\n"
    )
    exports, export_map = da._collect_module_exports(tree, module_name="pkg.mod", import_map={})
    assert "keep" in exports
    assert export_map.get("keep") == "pkg.mod.keep"

    # _collect_config_bundles assign-target non-name branch.
    config_mod = tmp_path / "cfg.py"
    config_mod.write_text(
        "class AppConfig:\n"
        "    x, y = (1, 2)\n",
        encoding="utf-8",
    )
    assert (
        da._collect_config_bundles(
            [config_mod],
            parse_failure_witnesses=[],
        )
        == {}
    )

    # _iter_dataclass_call_bundles star-import candidate miss branch.
    call_mod = tmp_path / "calls.py"
    call_mod.write_text("def f(alias):\n    alias.Item(1)\n", encoding="utf-8")
    bundles = da._iter_dataclass_call_bundles(
        call_mod,
        project_root=tmp_path,
        symbol_table=da.SymbolTable(
            imports={("calls", "alias"): "pkg"},
            internal_roots={"pkg"},
            external_filter=True,
            star_imports={"calls": {"pkg"}},
            module_exports={"pkg": {"alias"}},
        ),
        dataclass_registry={},
        parse_failure_witnesses=[],
    )
    assert bundles == set()

    # build_refactor_plan info-miss branch.
    ref_mod = tmp_path / "ref.py"
    ref_mod.write_text("def f(x):\n    return x\n", encoding="utf-8")
    plan = da.build_refactor_plan(
        {ref_mod: {"missing": [{"x"}]}},
        [ref_mod],
        config=da.AuditConfig(project_root=tmp_path),
    )
    assert any("No bundle components" in warning for warning in plan["warnings"])

    # _render_type_mermaid rhs without bracket wrapper.
    mermaid = da._render_type_mermaid([], ["f downstream types conflict: int, str"])
    assert "int" in mermaid and "str" in mermaid

    # analyze_paths timeout before collection-progress callback is defined.
    timed_out = False
    try:
        with da.deadline_scope(da.Deadline.from_timeout_ticks(10, 1)):
            with da.deadline_clock_scope(da.GasMeter(limit=2)):
                da.analyze_paths(
                    [tmp_path / "missing.py"],
                    forest=da.Forest(),
                    recursive=True,
                    type_audit=False,
                    type_audit_report=False,
                    type_audit_max=10,
                    include_constant_smells=False,
                    include_unused_arg_smells=False,
                    include_bundle_forest=False,
                    include_lint_lines=False,
                    config=da.AuditConfig(project_root=tmp_path),
                    collection_resume={"format_version": 0},
                )
    except da.TimeoutExceeded:
        timed_out = True
    assert timed_out is True

    # _analysis_deadline_scope with explicit tick-limit normal path.
    args = argparse.Namespace(
        analysis_timeout_ticks=10,
        analysis_timeout_tick_ns=1_000_000,
        analysis_tick_limit=3,
    )
    with da._analysis_deadline_scope(args):
        da.check_deadline()


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_write_output_helpers_cover_stdout_and_file::dataflow_audit.py::gabion.analysis.dataflow_audit._write_json_or_stdout::dataflow_audit.py::gabion.analysis.dataflow_audit._write_text_or_stdout::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_write_output_helpers_cover_stdout_and_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    da = _load()
    text_path = tmp_path / "out.txt"
    json_path = tmp_path / "out.json"

    da._write_text_or_stdout("-", "hello")
    assert capsys.readouterr().out == "hello\n"

    da._write_text_or_stdout(str(text_path), "world")
    assert text_path.read_text(encoding="utf-8") == "world"

    da._write_json_or_stdout("-", {"a": 1})
    stdout = capsys.readouterr().out
    assert '"a": 1' in stdout

    da._write_json_or_stdout(str(json_path), {"b": 2})
    assert '"b": 2' in json_path.read_text(encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_has_followup_actions_variants::dataflow_audit.py::gabion.analysis.dataflow_audit._has_followup_actions::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
@pytest.mark.parametrize(
    (
        "kwargs",
        "include_type_audit",
        "include_tree",
        "include_metrics",
        "include_decision",
        "expected",
    ),
    [
        ({}, True, False, False, False, False),
        ({"type_audit": True}, True, False, False, False, True),
        ({"type_audit": True}, False, False, False, False, False),
        ({}, True, True, False, False, True),
        ({}, True, False, True, False, True),
        ({}, True, False, False, True, True),
    ],
)
def test_has_followup_actions_variants(
    kwargs: dict[str, object],
    include_type_audit: bool,
    include_tree: bool,
    include_metrics: bool,
    include_decision: bool,
    expected: bool,
) -> None:
    da = _load()
    payload: dict[str, object] = {
        "type_audit": False,
        "synthesis_plan": None,
        "synthesis_report": False,
        "synthesis_protocols": None,
        "refactor_plan": False,
        "refactor_plan_json": None,
    }
    payload.update(kwargs)
    args = argparse.Namespace(**payload)
    assert (
        da._has_followup_actions(
            args,
            include_type_audit=include_type_audit,
            include_structure_tree=include_tree,
            include_structure_metrics=include_metrics,
            include_decision_snapshot=include_decision,
        )
        is expected
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_emit_sidecar_outputs_dispatches_expected_paths::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_sidecar_outputs::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_emit_sidecar_outputs_dispatches_expected_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    da = _load()
    analysis = da.AnalysisResult(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        type_suggestions=[],
        type_ambiguities=[],
        type_callsite_evidence=[],
        constant_smells=[],
        unused_arg_smells=[],
        forest=da.Forest(),
        lint_lines=["L1"],
        deadness_witnesses=[{"kind": "dead"}],
        coherence_witnesses=[{"kind": "coh"}],
        rewrite_plans=[{"kind": "plan"}],
        exception_obligations=[{"kind": "exc"}],
        handledness_witnesses=[{"kind": "handled"}],
        fingerprint_synth_registry={"registry": 1},
        fingerprint_provenance=[{"kind": "prov"}],
    )
    synth_path = tmp_path / "synth.json"
    prov_path = tmp_path / "prov.json"

    args = argparse.Namespace(
        lint=True,
        fingerprint_synth_json=str(synth_path),
        fingerprint_provenance_json=str(prov_path),
    )
    da._emit_sidecar_outputs(
        args=args,
        analysis=analysis,
        fingerprint_deadness_json="-",
        fingerprint_coherence_json=None,
        fingerprint_rewrite_plans_json=None,
        fingerprint_exception_obligations_json=None,
        fingerprint_handledness_json=None,
    )
    out = capsys.readouterr().out
    assert "L1" in out
    assert '"registry": 1' in synth_path.read_text(encoding="utf-8")
    assert '"kind": "prov"' in prov_path.read_text(encoding="utf-8")
    assert '"kind": "dead"' in out

    # require_content gate: synth/provenance are skipped when payloads are empty.
    empty_analysis = da.AnalysisResult(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        type_suggestions=[],
        type_ambiguities=[],
        type_callsite_evidence=[],
        constant_smells=[],
        unused_arg_smells=[],
        forest=da.Forest(),
    )
    synth_skip = tmp_path / "synth_skip.json"
    prov_skip = tmp_path / "prov_skip.json"
    args_skip = argparse.Namespace(
        lint=False,
        fingerprint_synth_json=str(synth_skip),
        fingerprint_provenance_json=str(prov_skip),
    )
    da._emit_sidecar_outputs(
        args=args_skip,
        analysis=empty_analysis,
        fingerprint_deadness_json=None,
        fingerprint_coherence_json=None,
        fingerprint_rewrite_plans_json=None,
        fingerprint_exception_obligations_json=None,
        fingerprint_handledness_json=None,
    )
    assert not synth_skip.exists()
    assert not prov_skip.exists()

# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_materialize_call_candidates_distinguishes_dynamic_unresolved::dataflow_audit.py::gabion.analysis.dataflow_audit._materialize_call_candidates::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_materialize_call_candidates_distinguishes_dynamic_unresolved() -> None:
    da = _load()
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.mod.caller",
        path=Path("pkg/mod.py"),
        params=[],
        annots={},
        calls=[
            da.CallArgs(callee="target", pos_map={}, kw_map={}, const_pos={}, const_kw={}, non_const_pos=set(), non_const_kw=set(), star_pos=[], star_kw=[], is_test=False, span=(1, 0, 1, 6)),
            da.CallArgs(callee="missing", pos_map={}, kw_map={}, const_pos={}, const_kw={}, non_const_pos=set(), non_const_kw=set(), star_pos=[], star_kw=[], is_test=False, span=(2, 0, 2, 7)),
            da.CallArgs(callee="getattr(handler, name)", pos_map={}, kw_map={}, const_pos={}, const_kw={}, non_const_pos=set(), non_const_kw=set(), star_pos=[], star_kw=[], is_test=False, span=(3, 0, 3, 20)),
            da.CallArgs(callee="maybe", pos_map={}, kw_map={}, const_pos={}, const_kw={}, non_const_pos=set(), non_const_kw=set(), star_pos=[], star_kw=[], is_test=False, span=(4, 0, 4, 5)),
        ],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    target = da.FunctionInfo(
        name="target",
        qual="pkg.mod.target",
        path=Path("pkg/mod.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    maybe = da.FunctionInfo(
        name="maybe",
        qual="pkg.mod.maybe",
        path=Path("pkg/mod.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )

    outcomes = {
        "target": da._CalleeResolutionOutcome("resolved", "resolved", "target", (target,)),
        "missing": da._CalleeResolutionOutcome("unresolved_internal", "unresolved_internal", "missing", (target,)),
        "getattr(handler, name)": da._CalleeResolutionOutcome("unresolved_dynamic", "unresolved_dynamic", "getattr(handler, name)", ()),
        "maybe": da._CalleeResolutionOutcome("ambiguous", "local_resolution", "maybe", (maybe,)),
    }

    forest = da.Forest()
    da._materialize_call_candidates(
        forest=forest,
        by_name={"caller": [caller], "target": [target], "maybe": [maybe]},
        by_qual={caller.qual: caller, target.qual: target, maybe.qual: maybe},
        symbol_table=da.SymbolTable(),
        project_root=Path("."),
        class_index={},
        resolve_callee_outcome_fn=lambda callee, *_args, **_kwargs: outcomes[callee],
    )

    obligation_kinds = {
        str(alt.evidence.get("kind", ""))
        for alt in forest.alts
        if alt.kind == "CallResolutionObligation"
    }
    assert "unresolved_internal_callee" in obligation_kinds
    assert "unresolved_dynamic_callee" in obligation_kinds
    assert len([alt for alt in forest.alts if alt.kind == "CallCandidate"]) == 2


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_call_resolution_obligation_evidence_returns_empty_on_mismatch::dataflow_audit.py::gabion.analysis.dataflow_audit._call_resolution_obligation_evidence::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_call_resolution_obligation_evidence_returns_empty_on_mismatch() -> None:
    da = _load()
    forest = da.Forest()
    suite = da.NodeId("SuiteSite", ("pkg/mod.py", "pkg.mod.caller"))
    forest.add_alt(
        "CallResolutionObligation",
        (suite,),
        evidence={"callee": "target", "kind": "unresolved_internal_callee"},
    )
    assert (
        da._call_resolution_obligation_evidence(
            forest,
            suite_id=suite,
            callee_key="other",
        )
        == {}
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_call_resolution_obligation_evidence_returns_matching_alt_after_skips::dataflow_audit.py::gabion.analysis.dataflow_audit._call_resolution_obligation_evidence::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_call_resolution_obligation_evidence_returns_matching_alt_after_skips() -> None:
    da = _load()
    forest = da.Forest()
    suite_target = da.NodeId("SuiteSite", ("pkg/mod.py", "pkg.mod.target"))
    suite_other = da.NodeId("SuiteSite", ("pkg/mod.py", "pkg.mod.other"))
    candidate_target = da.NodeId("SuiteSite", ("pkg/mod.py", "pkg.mod.callee"))
    forest.add_alt("CallCandidate", (suite_target, candidate_target), evidence={})
    forest.add_alt(
        "CallResolutionObligation",
        (suite_other,),
        evidence={"callee": "target", "kind": "other_suite"},
    )
    forest.add_alt(
        "CallResolutionObligation",
        (suite_target,),
        evidence={"callee": "target", "kind": "matching_suite"},
    )

    evidence = da._call_resolution_obligation_evidence(
        forest,
        suite_id=suite_target,
        callee_key="target",
    )
    assert evidence.get("kind") == "matching_suite"


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_collect_call_resolution_obligation_details_uses_first_matching_evidence_kind::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_call_resolution_obligation_details_from_forest::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_collect_call_resolution_obligation_details_uses_first_matching_evidence_kind() -> None:
    da = _load()
    forest = da.Forest()
    suite = forest.add_suite_site(
        "pkg/mod.py",
        "pkg.mod.caller",
        "call",
        span=(1, 0, 1, 5),
    )
    forest.add_alt(
        "CallResolutionObligation",
        (suite,),
        evidence={"callee": "target", "kind": "first_kind"},
    )
    forest.add_alt(
        "CallResolutionObligation",
        (suite,),
        evidence={"callee": "target", "kind": "second_kind"},
    )

    details = da._collect_call_resolution_obligation_details_from_forest(forest)
    assert len(details) == 1
    assert details[0][4] == "first_kind"


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_collect_call_resolution_obligation_details_ignores_empty_callee_index_entries::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_call_resolution_obligation_details_from_forest::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_collect_call_resolution_obligation_details_ignores_empty_callee_index_entries() -> None:
    da = _load()
    forest = da.Forest()
    suite = forest.add_suite_site(
        "pkg/mod.py",
        "pkg.mod.caller",
        "call",
        span=(1, 0, 1, 5),
    )
    forest.add_alt(
        "CallResolutionObligation",
        (suite,),
        evidence={"callee": "", "kind": "ignored_empty"},
    )
    forest.add_alt(
        "CallResolutionObligation",
        (suite,),
        evidence={"callee": "target", "kind": "resolved_kind"},
    )

    details = da._collect_call_resolution_obligation_details_from_forest(forest)
    assert len(details) == 1
    assert details[0][3] == "target"
    assert details[0][4] == "resolved_kind"

# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_fingerprint_rewrite_plans_emit_extended_kinds_with_proof_payloads::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_coherence::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans
def test_fingerprint_rewrite_plans_emit_extended_kinds_with_proof_payloads() -> None:
    provenance_entries = [
        {
            "path": "a.py",
            "function": "f",
            "bundle": ["a", "b"],
            "provenance_id": "prov:a.py:f:a,b",
            "base_keys": ["int", "str"],
            "ctor_keys": ["CtorA"],
            "remainder": {"base": 1, "ctor": 1},
            "glossary_matches": ["ctx_a", "ctx_b"],
        }
    ]
    coherence = da._compute_fingerprint_coherence(provenance_entries, synth_version="synth@1")
    plans = da._compute_fingerprint_rewrite_plans(
        provenance_entries,
        coherence,
        synth_version="synth@1",
    )

    kinds = {plan["rewrite"]["kind"] for plan in plans}
    assert kinds == {
        "BUNDLE_ALIGN",
        "CTOR_NORMALIZE",
        "SURFACE_CANONICALIZE",
        "AMBIENT_REWRITE",
    }

    by_kind = {plan["rewrite"]["kind"]: plan for plan in plans}
    ctor_plan = by_kind["CTOR_NORMALIZE"]
    assert ctor_plan["post_expectation"]["ctor_normalized"] is True
    assert any(p.get("kind") == "ctor_coherence" for p in ctor_plan["verification"]["predicates"])

    surface_plan = by_kind["SURFACE_CANONICALIZE"]
    assert surface_plan["post_expectation"]["surface_canonicalized"] is True
    assert any(p.get("kind") == "match_strata" for p in surface_plan["verification"]["predicates"])

    ambient_plan = by_kind["AMBIENT_REWRITE"]
    assert ambient_plan["post_expectation"]["ambient_normalized"] is True
    assert any(p.get("kind") == "remainder_non_regression" for p in ambient_plan["verification"]["predicates"])


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_missing_helper_branches_never_and_exception_annotations::dataflow_audit.py::gabion.analysis.dataflow_audit._annotation_exception_candidates::dataflow_audit.py::gabion.analysis.dataflow_audit._callee_key::dataflow_audit.py::gabion.analysis.dataflow_audit._enclosing_function_node::dataflow_audit.py::gabion.analysis.dataflow_audit._handler_is_broad::dataflow_audit.py::gabion.analysis.dataflow_audit._is_never_call::dataflow_audit.py::gabion.analysis.dataflow_audit._is_never_marker_raise::dataflow_audit.py::gabion.analysis.dataflow_audit._never_sort_key::dataflow_audit.py::gabion.analysis.dataflow_audit._refine_exception_name_from_annotations::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_missing_helper_branches_never_and_exception_annotations() -> None:
    da = _load()

    lambda_call = ast.parse("(lambda x: x)(1)").body[0].value
    assert da._is_never_call(lambda_call) is False
    assert da._is_never_marker_raise("never", "ValueError", {"TypeError"}) is False

    key = da._never_sort_key(
        {
            "status": "VIOLATION",
            "site": {"path": "a.py", "function": "f"},
            "span": ["bad", object(), 0, 1],
        }
    )
    assert key[3] == -1
    assert key[4] == -1

    assert da._enclosing_function_node(ast.parse("x = 1").body[0], {}) is None
    assert da._annotation_exception_candidates("List[") == ()
    assert "ValueError" in da._annotation_exception_candidates("typing.ValueError")

    exception_name, source, candidates = da._refine_exception_name_from_annotations(
        ast.parse("err").body[0].value,
        param_annotations={"err": "ValueError"},
    )
    assert exception_name == "ValueError"
    assert source == "PARAM_ANNOTATION"
    assert candidates == ("ValueError",)

    handler = ast.parse("try:\n    pass\nexcept:\n    pass\n").body[0].handlers[0]
    assert da._handler_is_broad(handler) is True
    assert da._callee_key("") == ""


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_missing_collect_handledness_and_exception_obligation_branches::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_handledness_witnesses::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_missing_collect_handledness_and_exception_obligation_branches(tmp_path: Path) -> None:
    da = _load()
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n", encoding="utf-8")
    mod = tmp_path / "mod.py"
    mod.write_text("raise SystemExit(1)\n", encoding="utf-8")

    witnesses = da._collect_handledness_witnesses(
        [bad, mod],
        project_root=tmp_path,
        ignore_params=set(),
    )
    assert any(entry.get("handledness_reason_code") == "SYSTEM_EXIT_CONVERT" for entry in witnesses)
    assert any(entry.get("site", {}).get("function") == "<module>" for entry in witnesses)

    obligations = da._collect_exception_obligations(
        [bad, mod],
        project_root=tmp_path,
        ignore_params=set(),
    )
    assert obligations
    assert any(entry.get("site", {}).get("function") == "<module>" for entry in obligations)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_missing_never_invariant_branches_and_keyword_reason::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::dataflow_audit.py::gabion.analysis.dataflow_audit._never_reason::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_missing_never_invariant_branches_and_keyword_reason(tmp_path: Path) -> None:
    da = _load()

    class _FalseyEnv(dict[str, str]):
        def __bool__(self) -> bool:
            return False

    broken = tmp_path / "broken.py"
    broken.write_text("def bad(:\n", encoding="utf-8")
    mod = tmp_path / "never_mod.py"
    mod.write_text(
        "def f(flag, extra, mystery):\n"
        "    if flag and extra:\n"
        "        gabion.never(reason='kw')\n"
        "    if mystery:\n"
        "        gabion.never('plain')\n"
        "\n"
        "gabion.never(reason='module')\n",
        encoding="utf-8",
    )

    deadness = [
        {
            "path": "never_mod.py",
            "function": "f",
            "bundle": ["flag"],
            "environment": _FalseyEnv({"flag": "False"}),
            "deadness_id": "dead:f:flag",
        }
    ]
    entries = da._collect_never_invariants(
        [broken, mod],
        project_root=tmp_path,
        ignore_params=set(),
        forest=da.Forest(),
        deadness_witnesses=deadness,
    )

    assert any(entry.get("site", {}).get("function") == "<module>" for entry in entries)
    assert any(entry.get("reason") == "kw" for entry in entries)
    assert any(
        entry.get("status") == "PROVEN_UNREACHABLE"
        and entry.get("environment_ref") == {"flag": False}
        for entry in entries
    )
    assert any(
        entry.get("status") == "OBLIGATION" and entry.get("undecidable_reason")
        for entry in entries
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_missing_deadline_obligation_control_flow_branches::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_deadline_obligations::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_missing_deadline_obligation_control_flow_branches(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "module.py"
    path.write_text("def marker():\n    return 1\n", encoding="utf-8")

    def _info(qual: str, *, params: list[str], annots: dict[str, str | None]) -> da.FunctionInfo:
        return da.FunctionInfo(
            name=qual.split(".")[-1],
            qual=qual,
            path=path,
            params=params,
            annots=annots,
            calls=[],
            unused_params=set(),
            function_span=(1, 0, 1, 6),
            param_spans={"deadline": (1, 0, 1, 8)} if "deadline" in params else {},
        )

    info_q1 = _info("pkg.q1", params=[], annots={})
    info_q2 = _info("pkg.q2", params=[], annots={})
    info_q3 = _info("pkg.q3", params=["deadline"], annots={"deadline": "Deadline"})
    index = da.AnalysisIndex(
        by_name={
            "q1": [info_q1],
            "q2": [info_q2],
            "q3": [info_q3],
        },
        by_qual={
            info_q1.qual: info_q1,
            info_q2.qual: info_q2,
            info_q3.qual: info_q3,
        },
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    local = da._DeadlineLocalInfo(origin_vars=set(), origin_spans={}, alias_to_param={})
    facts_q1 = da._DeadlineFunctionFacts(
        path=path,
        qual=info_q1.qual,
        span=(1, 0, 1, 1),
        loop=True,
        check_params=set(),
        ambient_check=False,
        loop_sites=[da._DeadlineLoopFacts(span=(1, 0, 1, 1), kind="loop", ambient_check=False)],
        local_info=local,
    )
    facts_q2 = da._DeadlineFunctionFacts(
        path=path,
        qual=info_q2.qual,
        span=(1, 0, 1, 1),
        loop=False,
        check_params=set(),
        ambient_check=True,
        loop_sites=[],
        local_info=local,
    )
    facts_q3 = da._DeadlineFunctionFacts(
        path=path,
        qual=info_q3.qual,
        span=(1, 0, 1, 1),
        loop=False,
        check_params=set(),
        ambient_check=True,
        loop_sites=[],
        local_info=local,
    )
    recursive_nodes = {
        da._function_suite_id(da._function_suite_key(path.name, info_q1.qual)),
        da._function_suite_id(da._function_suite_key(path.name, info_q2.qual)),
        da._function_suite_id(da._function_suite_key(path.name, info_q3.qual)),
    }
    obligations = da._collect_deadline_obligations(
        [path],
        project_root=tmp_path,
        config=da.AuditConfig(project_root=tmp_path, deadline_roots={"pkg.root"}),
        forest=da.Forest(),
        parse_failure_witnesses=[],
        analysis_index=index,
        extra_facts_by_qual={
            info_q1.qual: facts_q1,
            info_q2.qual: facts_q2,
            info_q3.qual: facts_q3,
            "pkg.none": None,  # type: ignore[dict-item]
        },
        materialize_call_candidates_fn=lambda **_kwargs: None,
        collect_call_nodes_by_path_fn=lambda *_args, **_kwargs: {},
        collect_deadline_function_facts_fn=lambda *_args, **_kwargs: {},
        collect_call_edges_from_forest_fn=lambda *_args, **_kwargs: {},
        collect_call_resolution_obligations_from_forest_fn=lambda *_args, **_kwargs: [],
        reachable_from_roots_fn=lambda *_args, **_kwargs: set(),
        collect_recursive_nodes_fn=lambda *_args, **_kwargs: recursive_nodes,
        resolve_callee_outcome_fn=lambda *_args, **_kwargs: da._CalleeResolutionOutcome(
            "unresolved_internal",
            "unresolved_internal",
            "",
            (),
        ),
    )
    assert any(
        entry.get("site", {}).get("function") == info_q1.qual
        and entry.get("kind") == "missing_carrier"
        for entry in obligations
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_missing_summary_warning_and_lint_branches::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_transitive_callers::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_protocol_lint_lines::dataflow_audit.py::gabion.analysis.dataflow_audit._exception_protocol_warnings::dataflow_audit.py::gabion.analysis.dataflow_audit._extract_smell_sample::dataflow_audit.py::gabion.analysis.dataflow_audit._forbid_adhoc_bundle_discovery::dataflow_audit.py::gabion.analysis.dataflow_audit._never_invariant_lint_lines::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_exception_obligations::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_missing_summary_warning_and_lint_branches() -> None:
    da = _load()

    assert da._summarize_exception_obligations([]) == []
    summarized = da._summarize_exception_obligations(
        [
            {"site": {"path": "a.py", "function": "f", "bundle": []}, "status": "UNKNOWN"},
            {"site": {"path": "a.py", "function": "g", "bundle": []}, "status": "UNKNOWN"},
        ],
        max_entries=1,
    )
    assert any(line.startswith("... 1 more") for line in summarized)

    assert da._summarize_never_invariants([]) == []
    never_lines = da._summarize_never_invariants(
        [
            {
                "status": "OBLIGATION",
                "site": {"path": "a.py", "function": "f"},
                "reason": "r",
                "undecidable_reason": "depends on params: x",
            },
            {
                "status": "OBLIGATION",
                "site": {"path": "a.py", "function": "g"},
                "span": [0, "x", 0, 1],
                "reason": "",
            },
        ],
        max_entries=1,
    )
    assert any("a.py:f" in line for line in never_lines)
    assert any("why=depends on params: x" in line for line in never_lines)
    assert any(line.startswith("... 1 more") for line in never_lines)

    warnings = da._exception_protocol_warnings(
        [{"protocol": "never", "status": "DEAD", "site": {"path": "a.py", "function": "f"}}]
    )
    assert warnings == []

    info_a = da.FunctionInfo(
        name="a",
        qual="pkg.a",
        path=Path("pkg/a.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
    )
    info_b = da.FunctionInfo(
        name="b",
        qual="pkg.b",
        path=Path("pkg/b.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
    )
    transitive = da._collect_transitive_callers(
        {"pkg.a": {"pkg.b"}, "pkg.b": {"pkg.b"}},
        {"pkg.a": info_a, "pkg.b": info_b},
    )
    assert "pkg.b" in transitive["pkg.a"]

    assert da._extract_smell_sample("smell (e.g. path.py:1:2: sample)") is not None

    lint_lines = da._exception_protocol_lint_lines(
        [
            {"protocol": "other"},
            {"protocol": "never", "status": "DEAD"},
            {"protocol": "never", "status": "UNKNOWN", "exception_path_id": "bad"},
        ]
    )
    assert lint_lines == []

    never_lint = da._never_invariant_lint_lines(
        [
            {"status": "VIOLATION", "span": "bad"},
            {
                "status": "VIOLATION",
                "site": {"path": "a.py"},
                "span": [0, 0, 0, 1],
                "witness_ref": "w:1",
            },
            {
                "status": "OBLIGATION",
                "site": {"path": "a.py"},
                "span": [0, 0, 0, 1],
                "undecidable_reason": "depends on params: p",
            },
        ]
    )
    assert any("witness=w:1" in line for line in never_lint)
    assert any("why=depends on params: p" in line for line in never_lint)

    with env_scope({"GABION_FORBID_ADHOC_BUNDLES": "1"}):
        with pytest.raises(AssertionError):
            da._forbid_adhoc_bundle_discovery("coverage")


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_missing_forest_population_and_handledness_summary_branches::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_handledness_witnesses::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_missing_forest_population_and_handledness_summary_branches(tmp_path: Path) -> None:
    da = _load()
    test_path = tmp_path / "tests" / "test_mod.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("def f(x):\n    return x\n", encoding="utf-8")

    info = da.FunctionInfo(
        name="f",
        qual="tests.test_mod.f",
        path=test_path,
        params=["x"],
        annots={},
        calls=[],
        unused_params=set(),
    )
    index = da.AnalysisIndex(
        by_name={"f": [info]},
        by_qual={info.qual: info},
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    forest = da.Forest()
    da._populate_bundle_forest(
        forest,
        groups_by_path={test_path: {"f": [{"x"}]}},
        file_paths=[test_path],
        project_root=tmp_path,
        include_all_sites=True,
        parse_failure_witnesses=[],
        analysis_index=index,
    )
    assert not any(node.meta.get("qual") == info.qual for node in forest.nodes.values())

    assert da._summarize_handledness_witnesses([]) == []
    summarized = da._summarize_handledness_witnesses(
        [
            {"site": {"path": "a.py", "function": "f", "bundle": []}, "result": "HANDLED"},
            {"site": {"path": "a.py", "function": "g", "bundle": []}, "result": "HANDLED"},
        ],
        max_entries=1,
    )
    assert any(line.startswith("... 1 more") for line in summarized)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_missing_analyze_file_internal_resolution_branches::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_missing_analyze_file_internal_resolution_branches(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "class C:\n"
        "    def m(self):\n"
        "        return 3\n"
        "\n"
        "def caller_global():\n"
        "    m()\n"
        "\n"
        "def outer():\n"
        "    class A:\n"
        "        def m(self):\n"
            "            return 1\n"
        "    class B:\n"
        "        def m(self):\n"
        "            return 2\n"
        "    def caller():\n"
        "        m()\n"
        "    caller()\n",
        encoding="utf-8",
    )
    groups, spans, sites = da._analyze_file_internal(path, config=None)
    assert isinstance(groups, dict)
    assert isinstance(spans, dict)
    assert isinstance(sites, dict)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_missing_resolution_and_knob_index_branches::dataflow_audit.py::gabion.analysis.dataflow_audit._base_identifier::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_closure_lambda_factories::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_knob_param_names::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_class_candidates::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_method_in_hierarchy::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_missing_resolution_and_knob_index_branches(tmp_path: Path) -> None:
    da = _load()

    malformed_attr = ast.Attribute(
        value=ast.Name(id="obj", ctx=ast.Load()),
        attr=1,  # type: ignore[arg-type]
        ctx=ast.Load(),
    )
    assert da._base_identifier(malformed_attr) is None
    assert da._resolve_class_candidates("", module="pkg.mod", symbol_table=None, class_index={}) == []
    assert (
        da._resolve_method_in_hierarchy(
            "pkg.C",
            "m",
            class_index={},
            by_qual={},
            symbol_table=None,
            seen={"pkg.C"},
        )
        is None
    )
    assert (
        da._resolve_method_in_hierarchy(
            "pkg.Missing",
            "m",
            class_index={},
            by_qual={},
            symbol_table=None,
            seen=set(),
        )
        is None
    )

    tree = ast.parse(
        "def outer():\n"
        "    def factory():\n"
        "        make = lambda value: value\n"
        "        return make\n"
    )
    lambda_node = tree.body[0].body[0].body[0].value
    lambda_span = da._node_span(lambda_node)
    assert lambda_span is not None
    parent = da.ParentAnnotator()
    parent.visit(tree)
    factories = da._collect_closure_lambda_factories(
        tree,
        module="mod",
        parent_map=parent.parents,
        lambda_qual_by_span={lambda_span: "mod.outer.factory.<lambda>:1:1"},
    )
    assert any(key.endswith("factory") for key in factories)

    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.mod.caller",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        lexical_scope=(),
    )
    target_a = da.FunctionInfo(
        name="target",
        qual="pkg.mod.target_a",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        lexical_scope=(),
    )
    target_b = da.FunctionInfo(
        name="target",
        qual="pkg.mod.target_b",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        lexical_scope=(),
    )
    seen_ambiguities: list[tuple[str, str]] = []
    assert (
        da._resolve_callee(
            "",
            caller,
            by_name={"target": [target_a, target_b]},
            by_qual={target_a.qual: target_a, target_b.qual: target_b},
            symbol_table=da.SymbolTable(),
            project_root=tmp_path,
            class_index={},
        )
        is None
    )
    assert (
        da._resolve_callee(
            "target",
            caller,
            by_name={"target": [target_a, target_b]},
            by_qual={target_a.qual: target_a, target_b.qual: target_b},
            symbol_table=da.SymbolTable(),
            project_root=tmp_path,
            class_index={},
            ambiguity_sink=lambda _caller, _call, _candidates, source, key: seen_ambiguities.append(
                (source, key)
            ),
        )
        is None
    )
    assert ("local_resolution", "target") in seen_ambiguities

    assert (
        da._compute_knob_param_names(
            by_name={},
            by_qual={},
            symbol_table=da.SymbolTable(),
            project_root=tmp_path,
            class_index={},
            strictness="high",
            analysis_index=None,
        )
        == set()
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_missing_dataclass_projection_and_rendering_branches::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_projection_from_forest::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::dataflow_audit.py::gabion.analysis.dataflow_audit._render_component_callsite_evidence::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_missing_dataclass_projection_and_rendering_branches(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def run(alias):\n"
        "    Qualified(1, 2)\n"
        "    Plain(1, 2)\n"
        "    alias.Attr(1, 2)\n"
        "    Overflow(1, 2)\n",
        encoding="utf-8",
    )
    parse_failures: list[dict[str, object]] = []
    symbol_table = da.SymbolTable(
        imports={},
        internal_roots={"pkg", "mod"},
        external_filter=True,
        star_imports={"mod": {"pkgstar"}},
        module_exports={"pkgstar": {"alias"}},
        module_export_map={"pkgstar": {"alias": "pkg"}},
    )
    bundles = da._iter_dataclass_call_bundles(
        path,
        project_root=tmp_path,
        symbol_table=symbol_table,
        dataclass_registry={
            "mod.Qualified": ["x", "y"],
            "Plain": ["x", "y"],
            "pkg.Attr": ["x", "y"],
            "mod.Overflow": ["x"],
        },
        parse_failure_witnesses=parse_failures,
    )
    assert ("x", "y") in bundles
    assert parse_failures

    forest = da.Forest()
    paramset = forest.add_paramset(["x"])
    forest.add_alt("SignatureBundle", (paramset,))
    missing_site = da.NodeId("FunctionSite", ("missing.py", "missing.qual"))
    forest.add_alt("SignatureBundle", (missing_site, paramset))
    projection = da._bundle_projection_from_forest(forest, file_paths=[path])
    assert isinstance(projection, da.BundleProjection)

    with pytest.raises(RuntimeError):
        da._emit_dot(object())  # type: ignore[arg-type]

    fn_id = da.NodeId("FunctionSite", ("a.py", "pkg.f"))
    bundle_id = da.NodeId("ParamSet", ("x",))
    lines = da._render_component_callsite_evidence(
        component=[fn_id, bundle_id],
        nodes={
            fn_id: {"kind": "fn", "path": "a.py", "qual": "pkg.f", "label": "a.py:pkg.f"},
            bundle_id: {"kind": "bundle", "label": "x"},
        },
        bundle_map={bundle_id: ("x",)},
        bundle_counts={("x",): 1},
        adj={fn_id: {bundle_id}, bundle_id: {fn_id}},
        documented_by_path={},
        declared_global=set(),
        bundle_site_index={},
        root=tmp_path,
        path_lookup={"a.py": tmp_path / "a.py"},
    )
    assert lines == []

    assert da._infer_root({path: {"run": [{"x"}]}}) == path
    assert da._infer_root({}) == Path(".")


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_missing_resume_and_plan_branches::dataflow_audit.py::gabion.analysis.dataflow_audit._analysis_index_resume_variants::dataflow_audit.py::gabion.analysis.dataflow_audit._deserialize_function_info_for_resume::dataflow_audit.py::gabion.analysis.dataflow_audit._load_baseline::dataflow_audit.py::gabion.analysis.dataflow_audit._render_type_mermaid::dataflow_audit.py::gabion.analysis.dataflow_audit._with_analysis_index_resume_variants::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::dataflow_audit.py::gabion.analysis.dataflow_audit.build_refactor_plan::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_missing_resume_and_plan_branches(tmp_path: Path) -> None:
    da = _load()

    allowed = {"mod.py": tmp_path / "mod.py"}
    info = da._deserialize_function_info_for_resume(
        {
            "name": "f",
            "qual": "mod.f",
            "path": "mod.py",
            "params": ["x"],
            "annots": {},
            "calls": [],
            "unused_params": [],
            "decision_surface_reasons": {1: ["skip"], "x": ["r", 1]},
        },
        allowed_paths=allowed,
    )
    assert info is not None
    assert info.decision_surface_reasons == {"x": {"r"}}

    variants = da._analysis_index_resume_variants(
        {
            da._ANALYSIS_INDEX_RESUME_VARIANTS_KEY: {
                1: {"format_version": 1},
                "bad_format": {"format_version": 2},
            }
        }
    )
    assert variants == {}

    payload = {"index_cache_identity": ""}
    assert da._with_analysis_index_resume_variants(payload=payload, previous_payload=None) == payload

    max_variants = da._ANALYSIS_INDEX_RESUME_MAX_VARIANTS
    previous_payload = {
        da._ANALYSIS_INDEX_RESUME_VARIANTS_KEY: {
            f"id_{idx}": {"format_version": 1, "index_cache_identity": f"id_{idx}"}
            for idx in range(max_variants + 2)
        }
    }
    trimmed = da._with_analysis_index_resume_variants(
        payload={"index_cache_identity": ""},
        previous_payload=previous_payload,
    )
    assert len(trimmed[da._ANALYSIS_INDEX_RESUME_VARIANTS_KEY]) == max_variants

    module = tmp_path / "mod.py"
    module.write_text(
        "def dec(flag):\n"
        "    if flag:\n"
        "        return 1\n"
        "    return 0\n"
        "\n"
        "def caller(flag):\n"
        "    return dec(flag)\n"
        "\n"
        "def val(v):\n"
        "    return (v == 1) * 2\n"
        "\n"
        "@opaque\n"
        "def hidden(flag):\n"
        "    return flag\n"
        "\n"
        "def uses_hidden(flag):\n"
        "    missing(flag)\n"
        "    hidden(flag)\n",
        encoding="utf-8",
    )
    groups = {
        module: {
            "dec": [{"flag"}],
            "caller": [{"flag"}],
            "val": [set()],
            "hidden": [{"flag"}],
            "uses_hidden": [{"flag"}],
        }
    }
    synthesis = da.build_synthesis_plan(
        groups,
        project_root=tmp_path,
        min_bundle_size=0,
        allow_singletons=True,
        merge_overlap_threshold=0.0,
    )
    assert synthesis["protocols"]
    assert any(
        "tier-2:decision-bundle-elevation" in protocol.get("evidence", [])
        and "value_decision_surface" in protocol.get("evidence", [])
        for protocol in synthesis["protocols"]
    )
    empty_bundle_synthesis = da.build_synthesis_plan(
        {module: {"uses_hidden": [set()]}},
        project_root=tmp_path,
        min_bundle_size=0,
        allow_singletons=True,
        merge_overlap_threshold=0.0,
    )
    assert isinstance(empty_bundle_synthesis["protocols"], list)
    bare_module = tmp_path / "bare.py"
    bare_module.write_text("def noop(x):\n    return x\n", encoding="utf-8")
    bare_synthesis = da.build_synthesis_plan(
        {bare_module: {"noop": [set(), {"x"}]}},
        project_root=tmp_path,
        min_bundle_size=0,
        allow_singletons=True,
        merge_overlap_threshold=0.5,
    )
    assert isinstance(bare_synthesis["protocols"], list)

    no_paths = da.build_refactor_plan(
        groups,
        [],
        config=da.AuditConfig(project_root=tmp_path),
    )
    assert no_paths["warnings"] == ["No files available for refactor plan."]

    refactor = da.build_refactor_plan(
        groups,
        [module],
        config=da.AuditConfig(
            project_root=tmp_path,
            transparent_decorators={"safe_transparent"},
        ),
    )
    assert isinstance(refactor["bundles"], list)

    mermaid = da._render_type_mermaid(
        [],
        ["skip me", "f downstream types conflict: [int, , str]"],
    )
    assert "int" in mermaid
    assert "str" in mermaid

    baseline_dir = tmp_path / "baseline_dir"
    baseline_dir.mkdir()
    assert da._load_baseline(baseline_dir) == set()

    analysis = da.analyze_paths(
        [module],
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
    assert module in analysis.groups_by_path


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_additional_parse_contract_and_decision_surface_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._decision_surface_form_entries::dataflow_audit.py::gabion.analysis.dataflow_audit._decision_surface_alt_evidence::dataflow_audit.py::gabion.analysis.dataflow_audit._imported_helper_targets::dataflow_audit.py::gabion.analysis.dataflow_audit._parse_witness_contract_violations::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_repo_module_path::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_additional_parse_contract_and_decision_surface_edges(tmp_path: Path) -> None:
    da = _load()

    fn = ast.parse(
        "def decision(flag):\n"
        "    return [item for item in [1, 2] if flag]\n"
    ).body[0]
    entries = da._decision_surface_form_entries(fn)
    assert any(kind == "comprehension_guard" for kind, _ in entries)

    class _Spec:
        def alt_evidence(self, boundary: str, descriptor: str) -> dict[str, object]:
            return {
                "boundary": boundary,
                "meta": {"descriptor": descriptor},
                "extra": "x",
            }

    alt_payload = da._decision_surface_alt_evidence(
        spec=_Spec(),
        boundary="boundary",
        descriptor="descriptor",
        params=["flag"],
        caller_count=0,
        reason_summary="heuristic",
    )
    assert alt_payload["meta"] == {"descriptor": "descriptor"}
    assert alt_payload["extra"] == "x"

    class _SpecNoMeta:
        def alt_evidence(self, boundary: str, descriptor: str) -> dict[str, object]:
            return {"boundary": boundary, "descriptor": descriptor}

    alt_payload_no_meta = da._decision_surface_alt_evidence(
        spec=_SpecNoMeta(),
        boundary="boundary",
        descriptor="descriptor",
        params=["flag"],
        caller_count=1,
        reason_summary="heuristic",
    )
    assert "meta" not in alt_payload_no_meta

    imported_targets = da._imported_helper_targets(
        ast.parse("from . import parse_failure_witnesses as helper\n")
    )
    assert imported_targets == {}

    assert da._resolve_repo_module_path("example.module") is None
    assert da._resolve_repo_module_path("gabion.analysis") is not None
    assert da._resolve_repo_module_path("gabion.__definitely_missing__") is None

    bad_contract = tmp_path / "bad_contract.py"
    bad_contract.write_text("def broken(:\n", encoding="utf-8")
    parse_error_violations = da._parse_witness_contract_violations(
        source_path=bad_contract,
        target_helpers=frozenset(["parse_failure_witnesses"]),
    )
    assert any("parse_error: SyntaxError" in line for line in parse_error_violations)

    imported_source = (
        "from gabion.analysis.timeout_context import TimeoutExceeded as helper\n"
    )
    import_parse_error_violations = da._parse_witness_contract_violations(
        source=imported_source,
        target_helpers=frozenset(["helper"]),
        module_function_map_fn=lambda _path: (_ for _ in ()).throw(
            SyntaxError("synthetic import parse error")
        ),
    )
    assert any(
        "import_parse_error: SyntaxError" in line
        for line in import_parse_error_violations
    )

    import_missing_symbol_violations = da._parse_witness_contract_violations(
        source=imported_source,
        target_helpers=frozenset(["helper"]),
        module_function_map_fn=lambda _path: {},
    )
    assert any(
        "missing helper definition" in line for line in import_missing_symbol_violations
    )

    unresolved_import_violations = da._parse_witness_contract_violations(
        source="from external.mod import helper\n",
        target_helpers=frozenset(["helper"]),
    )
    assert any("missing helper definition" in line for line in unresolved_import_violations)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_additional_dataflow_helper_branch_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._analysis_index_resume_variants::dataflow_audit.py::gabion.analysis.dataflow_audit._analysis_index_stage_cache::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_pattern_instances::dataflow_audit.py::gabion.analysis.dataflow_audit._bundle_site_index::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_lambda_function_infos::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_module_exports::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_synthesis_tiers_and_merge::dataflow_audit.py::gabion.analysis.dataflow_audit._deserialize_function_info_for_resume::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_synthesis_field_types::dataflow_audit.py::gabion.analysis.dataflow_audit._paramset_key::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee_outcome::dataflow_audit.py::gabion.analysis.dataflow_audit._build_analysis_index::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_additional_dataflow_helper_branch_edges(tmp_path: Path) -> None:
    da = _load()

    single_occurrence_instances = da._bundle_pattern_instances(
        groups_by_path={Path("pkg.py"): {"pkg.fn": [set(["a", "b"])]}}
    )
    assert len(single_occurrence_instances) == 1
    assert "near_miss" in single_occurrence_instances[0].suggestion

    export_tree = ast.parse(
        "from pkg.mod import imported\n"
        "__all__ = ['imported', 'unknown_export']\n"
        "__all__ += ['extra_export']\n"
        "not_all += ['skip']\n"
        "local_name = 1\n"
    )
    export_names, export_map = da._collect_module_exports(
        export_tree,
        module_name="pkg.sample",
        import_map={"imported": "pkg.mod.imported"},
    )
    assert "imported" in export_names
    assert export_map["imported"] == "pkg.mod.imported"
    assert "unknown_export" in export_names
    assert "unknown_export" not in export_map

    lambda_tree = ast.parse(
        "def outer():\n"
        "    return (lambda value: value)\n"
    )
    parent_annotator = da.ParentAnnotator()
    parent_annotator.visit(lambda_tree)
    lambda_infos = da._collect_lambda_function_infos(
        lambda_tree,
        path=tmp_path / "lambda_mod.py",
        module="pkg.lambda_mod",
        parent_map=parent_annotator.parents,
        ignore_params=None,
    )
    assert lambda_infos
    assert ".outer." in lambda_infos[0].qual

    top_level_tree = ast.parse("value = (lambda x: x)\n")
    top_level_parent_annotator = da.ParentAnnotator()
    top_level_parent_annotator.visit(top_level_tree)
    top_level_lambda_infos = da._collect_lambda_function_infos(
        top_level_tree,
        path=tmp_path / "top_level.py",
        module="pkg.top_level",
        parent_map=top_level_parent_annotator.parents,
        ignore_params=None,
    )
    assert top_level_lambda_infos
    assert ".outer." not in top_level_lambda_infos[0].qual

    closure_tree = ast.parse(
        "def outer():\n"
        "    annotation_only: int\n"
        "    return annotation_only\n"
    )
    closure_parent_annotator = da.ParentAnnotator()
    closure_parent_annotator.visit(closure_tree)
    closure_factories = da._collect_closure_lambda_factories(
        closure_tree,
        module="pkg.closure_mod",
        parent_map=closure_parent_annotator.parents,
        lambda_qual_by_span={},
    )
    assert closure_factories == {}

    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.caller",
        path=tmp_path / "caller.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        local_lambda_bindings={"local_lambda": ("pkg.local_lambda",)},
    )
    unresolved = da._resolve_callee_outcome(
        "unknown",
        caller,
        by_name={},
        by_qual={},
    )
    assert unresolved.status.startswith("unresolved")

    unresolved_with_explicit_bindings = da._resolve_callee_outcome(
        "unknown",
        caller,
        by_name={},
        by_qual={},
        local_lambda_bindings={},
    )
    assert unresolved_with_explicit_bindings.status.startswith("unresolved")

    forest = da.Forest()
    missing_paramset = da.NodeId("ParamSet", ("x", "y"))
    assert da._paramset_key(forest, missing_paramset) == ("x", "y")

    bundle_site_index = da._bundle_site_index(
        groups_by_path={
            Path("mod.py"): {"mod.fn": [set(["x", "y"]), set(["x", "z"])]}
        },
        bundle_sites_by_path={
            Path("mod.py"): {"mod.fn": [[{"line": 1, "col": 0}]]}
        },
    )
    assert bundle_site_index[("mod.py", "mod.fn", ("x", "y"))] == [
        [{"line": 1, "col": 0}]
    ]

    allowed_paths = {"mod.py": tmp_path / "mod.py"}
    function_info = da._deserialize_function_info_for_resume(
        {
            "name": "f",
            "qual": "mod.f",
            "path": "mod.py",
            "params": ["x"],
            "annots": {},
            "calls": [],
            "unused_params": [],
            "decision_surface_reasons": {"x": ["guarded"]},
            "decision_params": [],
            "value_decision_params": [],
            "value_decision_reasons": [],
            "scope": [],
            "lexical_scope": [],
            "positional_params": [],
            "kwonly_params": [],
            "param_spans": {},
        },
        allowed_paths=allowed_paths,
    )
    assert function_info is not None
    assert function_info.decision_surface_reasons == {"x": {"guarded"}}

    function_info_empty_reasons = da._deserialize_function_info_for_resume(
        {
            "name": "f",
            "qual": "mod.f",
            "path": "mod.py",
            "params": ["x"],
            "annots": {},
            "calls": [],
            "unused_params": [],
            "decision_surface_reasons": {"x": []},
            "decision_params": [],
            "value_decision_params": [],
            "value_decision_reasons": [],
            "scope": [],
            "lexical_scope": [],
            "positional_params": [],
            "kwonly_params": [],
            "param_spans": {},
        },
        allowed_paths=allowed_paths,
    )
    assert function_info_empty_reasons is not None
    assert function_info_empty_reasons.decision_surface_reasons == {}

    variants = da._analysis_index_resume_variants(
        {
            da._ANALYSIS_INDEX_RESUME_VARIANTS_KEY: {
                "id-1": {"format_version": 1, "value": "ok"}
            }
        }
    )
    assert variants["id-1"]["value"] == "ok"

    assert (
        da._analysis_index_resume_variants(
            {da._ANALYSIS_INDEX_RESUME_VARIANTS_KEY: []}
        )
        == {}
    )

    stage_path = tmp_path / "missing_stage.py"
    analysis_index = da.AnalysisIndex(
        by_name={},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
        index_cache_identity="index",
        projection_cache_identity="projection",
    )
    stage_result = da._analysis_index_stage_cache(
        analysis_index,
        [stage_path],
        spec=da._StageCacheSpec(
            stage=da._ParseModuleStage.FUNCTION_INDEX,
            cache_key="cache-key",
            build=lambda tree, _path: len(tree.body),
        ),
        parse_failure_witnesses=[],
        module_trees_fn=lambda *_args, **_kwargs: {
            stage_path: ast.parse("x = 1\n")
        },
    )
    assert stage_result[stage_path] == 1

    source_path = tmp_path / "source.py"
    source_path.write_text("def f(x):\n    return x\n", encoding="utf-8")
    index_snapshots: list[dict[str, object]] = []
    built_index = da._build_analysis_index(
        [source_path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        parse_failure_witnesses=[],
        on_progress=index_snapshots.append,
    )
    assert built_index.by_qual
    assert index_snapshots

    tiers, merged_evidence, _naming_context, _synth_config, bundle_fields = (
        da._compute_synthesis_tiers_and_merge(
            counts={},
            bundle_evidence={frozenset(["x"]): {"dataflow"}},
            root=tmp_path,
            max_tier=3,
            min_bundle_size=2,
            allow_singletons=False,
            merge_overlap_threshold=0.5,
        )
    )
    assert tiers == {}
    assert merged_evidence == {frozenset(["x"]): {"dataflow"}}
    assert bundle_fields == set()

    empty_context = da._SynthesisPlanContext(
        audit_config=da.AuditConfig(project_root=tmp_path),
        root=tmp_path,
        signature_meta={},
        path_list=[],
        parse_failure_witnesses=[],
        analysis_index=da.AnalysisIndex(
            by_name={},
            by_qual={},
            symbol_table=da.SymbolTable(),
            class_index={},
        ),
        by_name={},
        by_qual={},
        symbol_table=da.SymbolTable(),
        class_index={},
        transitive_callers={},
    )
    field_types, warnings = da._infer_synthesis_field_types(
        bundle_fields=set(),
        context=empty_context,
    )
    assert field_types == {}
    assert warnings == []


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_coverage_gaps.py::test_additional_exception_and_never_branch_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._annotation_exception_candidates::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_handledness_witnesses::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::dataflow_audit.py::gabion.analysis.dataflow_audit.verify_rewrite_plan::test_dataflow_audit_coverage_gaps.py::tests.test_dataflow_audit_coverage_gaps._load
def test_additional_exception_and_never_branch_edges(tmp_path: Path) -> None:
    da = _load()

    assert da._annotation_exception_candidates("pkg.CustomError") == ()

    module_path = tmp_path / "exceptions.py"
    module_path.write_text(
        "def handled(flag):\n"
        "    try:\n"
        "        if flag:\n"
        "            raise ValueError('boom')\n"
        "    except TypeError:\n"
        "        return 1\n"
        "    return 0\n"
        "\n"
        "def mark():\n"
        "    never()\n",
        encoding="utf-8",
    )

    handledness = da._collect_handledness_witnesses(
        [module_path],
        project_root=tmp_path,
        ignore_params=set(),
    )
    assert handledness
    assert any(
        str(entry.get("type_refinement_opportunity", ""))
        for entry in handledness
    )

    mutated_handledness: list[dict[str, object]] = []
    for entry in handledness:
        mutated = {str(key): entry[key] for key in entry}
        mutated["exception_type_candidates"] = "not-a-list"
        mutated_handledness.append(mutated)

    obligations = da._collect_exception_obligations(
        [module_path],
        project_root=tmp_path,
        ignore_params=set(),
        handledness_witnesses=mutated_handledness,
    )
    assert obligations
    remainder_values = [entry.get("remainder", {}) for entry in obligations]
    assert all(
        not (
            isinstance(remainder, dict)
            and "exception_type_candidates" in remainder
            and remainder["exception_type_candidates"] == "not-a-list"
        )
        for remainder in remainder_values
    )

    forest = da.Forest()
    never_entries = da._collect_never_invariants(
        [module_path],
        project_root=tmp_path,
        ignore_params=set(),
        forest=forest,
    )
    assert never_entries
    summary_lines = da._summarize_never_invariants(
        never_entries,
        include_proven_unreachable=True,
    )
    assert any("why=no witness env available" in line for line in summary_lines)

    explicit_obligation_summary = da._summarize_never_invariants(
        [
            {
                "status": "OBLIGATION",
                "site": {"path": "mod.py", "function": "mark", "bundle": []},
                "never_id": "never:mod.py:mark:1:4",
                "reason": "",
                "span": [0, 0, 0, 1],
            }
        ],
        include_proven_unreachable=True,
    )
    assert any("why=no witness env available" in line for line in explicit_obligation_summary)

    verification = da.verify_rewrite_plan(
        {
            "plan_id": "abstained-plan",
            "site": {"path": "mod.py", "function": "f", "bundle": []},
            "status": "ABSTAINED",
            "abstention": {"reason": "preconditions unsatisfied"},
        },
        post_provenance=[],
    )
    assert verification["accepted"] is False
    assert any(
        "abstention reason: preconditions unsatisfied" in issue
        for issue in verification["issues"]
    )

    verification_without_reason = da.verify_rewrite_plan(
        {
            "plan_id": "abstained-plan-2",
            "site": {"path": "mod.py", "function": "f", "bundle": []},
            "status": "ABSTAINED",
            "abstention": {},
        },
        post_provenance=[],
    )
    assert verification_without_reason["accepted"] is False

    dynamic_module_path = tmp_path / "dynamic_exceptions.py"
    dynamic_module_path.write_text(
        "def handled_dynamic(exc):\n"
        "    try:\n"
        "        raise exc\n"
        "    except TypeError:\n"
        "        return 1\n"
        "    return 0\n",
        encoding="utf-8",
    )
    dynamic_handledness = da._collect_handledness_witnesses(
        [dynamic_module_path],
        project_root=tmp_path,
        ignore_params=set(),
    )
    assert dynamic_handledness


# gabion:evidence E:function_site::tests/test_dataflow_audit_coverage_gaps.py::test_tail_branch_and_line_edges_for_eval_and_registry
def test_tail_branch_and_line_edges_for_eval_and_registry(tmp_path: Path) -> None:
    da = _load()

    table = da.SymbolTable(external_filter=False)
    table.star_imports["pkg.mod"] = {"pkg.star"}
    table.module_exports["pkg.star"] = {"Foo"}
    table.module_export_map["pkg.star"] = {}
    assert table.resolve_star("pkg.mod", "Foo") == "pkg.star.Foo"

    assert da._invariant_term(ast.Name(id="x", ctx=ast.Load()), {"y"}) is None
    assert (
        da._invariant_term(
            ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[ast.Name(id="x", ctx=ast.Load())],
                keywords=[],
            ),
            {"y"},
        )
        is None
    )
    assert (
        da._invariant_term(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="obj", ctx=ast.Load()),
                    attr="len",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id="x", ctx=ast.Load())],
                keywords=[],
            ),
            {"x"},
        )
        is None
    )
    assert (
        da._invariant_term(
            ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[ast.Name(id="x", ctx=ast.Load()), ast.Name(id="y", ctx=ast.Load())],
                keywords=[],
            ),
            {"x"},
        )
        is None
    )

    verification = da.verify_rewrite_plan(
        {
            "plan_id": "tail-predicate-paths",
            "site": {"path": "m.py", "function": "f", "bundle": []},
            "status": "READY",
            "rewrite": {"kind": "annotate_return", "parameters": {}},
            "verification": {"predicates": [{"kind": "base_conservation"}, {}]},
        },
        post_provenance=[],
    )
    assert verification["accepted"] is False

    foreign_stmt = ast.parse("x = 1").body[0]
    unrelated_block = ast.parse("y = 2").body
    assert da._node_in_block(foreign_stmt, unrelated_block) is False
    try_tree = ast.parse(
        "try:\n"
        "    x = 1\n"
        "except Exception:\n"
        "    pass\n"
        "y = 2\n"
    )
    try_parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(try_tree):
        for child in ast.iter_child_nodes(parent):
            try_parents[child] = parent
    trailing_assign = try_tree.body[-1]
    assert da._find_handling_try(trailing_assign, try_parents) is None

    assert da._unary_numeric_outcome(
        ast.UnaryOp(op=ast.USub(), operand=ast.Name(id="missing", ctx=ast.Load())),
        {},
    ).is_unknown()
    assert da._eval_value_expr(
        ast.UnaryOp(op=ast.Not(), operand=ast.Constant(value=True)),
        {},
    ).is_unknown()
    assert da._eval_bool_expr(
        ast.UnaryOp(op=ast.UAdd(), operand=ast.Constant(value=1)),
        {},
    ).is_unknown()
    assert da._eval_bool_expr(
        ast.BoolOp(
            op=ast.Add(),  # type: ignore[arg-type]
            values=[ast.Constant(value=True), ast.Constant(value=False)],
        ),
        {},
    ).is_unknown()
    assert da._eval_bool_expr(
        ast.Compare(
            left=ast.Constant(value=1),
            ops=[ast.Eq(), ast.Eq()],
            comparators=[ast.Constant(value=1), ast.Constant(value=2)],
        ),
        {},
    ).is_unknown()
    assert da._eval_bool_expr(
        ast.Compare(
            left=ast.Constant(value=1),
            ops=[ast.Is()],
            comparators=[ast.Constant(value=1)],
        ),
        {},
    ).is_unknown()

    never_call = ast.Call(
        func=ast.Name(id="never", ctx=ast.Load()),
        args=[ast.Name(id="reason_var", ctx=ast.Load())],
        keywords=[ast.keyword(arg="reason", value=ast.Name(id="fallback", ctx=ast.Load()))],
    )
    assert da._never_reason(never_call) is None

    unknown_path = tmp_path / "unknown_never.py"
    unknown_path.write_text(
        "def f(flag):\n"
        "    if flag:\n"
        "        never('reason')\n",
        encoding="utf-8",
    )
    never_entries = da._collect_never_invariants(
        [unknown_path],
        project_root=tmp_path,
        ignore_params=set(),
        forest=da.Forest(),
        deadness_witnesses=[
            {
                "path": "unknown_never.py",
                "function": "f",
                "bundle": ["other"],
                "environment": {"other": "0"},
                "deadness_id": "dead:unknown_never.py:f:1",
            }
        ],
    )
    assert any(
        "depends on params" in str(entry.get("undecidable_reason", ""))
        for entry in never_entries
    )

    assert (
        da._const_repr(
            ast.UnaryOp(op=ast.USub(), operand=ast.Name(id="x", ctx=ast.Load()))
        )
        is None
    )

    lambda_tree = ast.parse(
        "class Holder:\n"
        "    def f(self):\n"
        "        self.inner.cb = lambda: 1\n"
    )
    parent_map: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(lambda_tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent
    lambda_infos = da._collect_lambda_function_infos(
        lambda_tree,
        path=Path("m.py"),
        module="m",
        parent_map=parent_map,
        ignore_params=set(),
    )
    lambda_bindings = da._collect_lambda_bindings_by_caller(
        lambda_tree,
        module="m",
        parent_map=parent_map,
        lambda_infos=lambda_infos,
    )
    assert lambda_bindings == {}

    registry_tree = ast.parse(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class D:\n"
        "    x[0]: int = 1\n"
    )
    assert (
        da._dataclass_registry_for_tree(
            Path("m.py"),
            registry_tree,
            project_root=Path("."),
        )
        == {}
    )
