from __future__ import annotations

import ast
from pathlib import Path
import textwrap

import pytest

from gabion.analysis.timeout_context import (
    Deadline,
    TimeoutContext,
    deadline_scope,
    pack_call_stack,
)
from gabion.exceptions import NeverThrown
from gabion.order_contract import ordered_or_sorted

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import dataflow_audit as da

    return da

def _deadline_obligations(tmp_path: Path, source: str, roots: set[str]) -> list[dict]:
    result = _deadline_analysis(tmp_path, source, roots)
    return result.deadline_obligations



def _deadline_analysis(tmp_path: Path, source: str, roots: set[str]):
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(textwrap.dedent(source), encoding="utf-8")
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=True,
        strictness="high",
        deadline_roots=set(roots),
    )
    return da.analyze_paths(
        [target],
        forest=da.Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_deadness_witnesses=False,
        include_coherence_witnesses=False,
        include_rewrite_plans=False,
        include_exception_obligations=False,
        include_handledness_witnesses=False,
        include_never_invariants=False,
        include_decision_surfaces=False,
        include_value_decision_surfaces=False,
        include_invariant_propositions=False,
        include_lint_lines=False,
        include_ambiguities=False,
        include_bundle_forest=False,
        include_deadline_obligations=True,
        config=config,
    )
def _call(da, *, callee: str, is_test: bool = False, span: tuple[int, int, int, int] | None = None):
    return da.CallArgs(
        callee=callee,
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=is_test,
        span=span,
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_collect_call_ambiguities_indexed_preserves_duplicate_observations::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_call_ambiguities_indexed::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._call::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_collect_call_ambiguities_indexed_preserves_duplicate_observations(
) -> None:
    da = _load()
    call = _call(da, callee="target", span=(0, 0, 0, 1))
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.caller",
        path=Path("pkg/mod.py"),
        params=[],
        annots={},
        calls=[call],
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
    analysis_index = da.AnalysisIndex(
        by_name={"caller": [caller]},
        by_qual={caller.qual: caller, candidate.qual: candidate},
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    context = da._IndexedPassContext(
        paths=[Path("pkg/mod.py")],
        project_root=None,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        parse_failure_witnesses=[],
        analysis_index=analysis_index,
    )

    def _fake_resolve(*_args, **kwargs):
        ambiguity_sink = kwargs.get("ambiguity_sink")
        if callable(ambiguity_sink):
            ambiguity_sink(caller, call, [candidate], "local_resolution", "target")
            ambiguity_sink(caller, call, [candidate], "local_resolution", "target")
        return None

    ambiguities = da._collect_call_ambiguities_indexed(
        context,
        resolve_callee_fn=_fake_resolve,
    )
    assert len(ambiguities) == 2
    assert all(entry.kind == "local_resolution_ambiguous" for entry in ambiguities)
    assert all(entry.callee_key == "target" for entry in ambiguities)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_iter_dataclass_call_bundles_dynamic_starred_records_unresolved::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_dataclass_call_bundles::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_iter_dataclass_call_bundles_dynamic_starred_records_unresolved(tmp_path: Path) -> None:
    da = _load()
    mod = tmp_path / "mod.py"
    mod.write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass

            @dataclass
            class Bundle:
                a: int
                b: int

            vals = [1, 2]
            kws = {"a": 1, "b": 2}

            def build(dynamic_vals, dynamic_kwargs):
                Bundle(*vals)
                Bundle(**kws)
                Bundle(*[1, 2, 3])
                Bundle(**{1: 2})
                Bundle(c=3)
                Bundle(*dynamic_vals)
                Bundle(**dynamic_kwargs)
                Bundle(**{**kws})
            """
        ).strip()
        + "\n"
    )
    witnesses: list[dict[str, object]] = []
    bundles = da._iter_dataclass_call_bundles(
        mod,
        project_root=tmp_path,
        parse_failure_witnesses=witnesses,
    )
    assert bundles == set()
    unresolved = [
        entry
        for entry in witnesses
        if entry.get("error_type") == "UnresolvedStarredArgument"
    ]
    assert len(unresolved) == 7
    assert any(
        "positional_arity_overflow" in str(entry.get("error", ""))
        for entry in unresolved
    )
    assert any(
        "non-string literal key in ** dict" in str(entry.get("error", ""))
        for entry in unresolved
    )

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest
def test_populate_bundle_forest_empty_groups(tmp_path: Path) -> None:
    da = _load()
    forest = da.Forest()
    da._populate_bundle_forest(
        forest,
        groups_by_path={},
        file_paths=[],
        project_root=tmp_path,
        include_all_sites=True,
        ignore_params=set(),
        strictness="high",
        transparent_decorators=set(),
        parse_failure_witnesses=[],
    )
    assert forest.nodes == {}


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_populate_bundle_forest_progress_callback_emits_vector_snapshots::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_populate_bundle_forest_progress_callback_emits_vector_snapshots() -> None:
    da = _load()
    forest = da.Forest()
    groups_by_path: dict[Path, dict[str, list[set[str]]]] = {}
    for index in range(130):
        groups_by_path[Path(f"pkg/mod_{index}.py")] = {
            f"pkg.mod_{index}.fn": [set(["a", "b"])]
        }
    snapshots: list[dict[str, object]] = []

    da._populate_bundle_forest(
        forest,
        groups_by_path=groups_by_path,
        file_paths=[],
        project_root=Path("."),
        include_all_sites=False,
        ignore_params=set(),
        strictness="high",
        transparent_decorators=set(),
        parse_failure_witnesses=[],
        on_progress=snapshots.append,
    )

    assert snapshots
    first = snapshots[0]
    assert first.get("marker") == "start"
    assert first.get("primary_unit") == "forest_mutable_steps"
    done_series = [int(snapshot.get("primary_done", 0) or 0) for snapshot in snapshots]
    total_series = [int(snapshot.get("primary_total", 0) or 0) for snapshot in snapshots]
    assert done_series == ordered_or_sorted(
        done_series,
        source="test_populate_bundle_forest_progress_emits_monotonic_vector.done_series",
    )
    assert total_series == ordered_or_sorted(
        total_series,
        source="test_populate_bundle_forest_progress_emits_monotonic_vector.total_series",
    )
    assert done_series[-1] == total_series[-1]
    assert done_series[-1] >= 130


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_populate_bundle_forest_progress_callback_supports_legacy_no_arg_handler::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_populate_bundle_forest_progress_callback_supports_legacy_no_arg_handler() -> None:
    da = _load()
    forest = da.Forest()
    groups_by_path: dict[Path, dict[str, list[set[str]]]] = {}
    for index in range(130):
        groups_by_path[Path(f"pkg/mod_{index}.py")] = {
            f"pkg.mod_{index}.fn": [set(["a", "b"])]
        }
    calls = {"count": 0}

    def _legacy_no_arg_handler() -> None:
        calls["count"] += 1

    da._populate_bundle_forest(
        forest,
        groups_by_path=groups_by_path,
        file_paths=[],
        project_root=Path("."),
        include_all_sites=False,
        ignore_params=set(),
        strictness="high",
        transparent_decorators=set(),
        parse_failure_witnesses=[],
        on_progress=_legacy_no_arg_handler,
    )

    assert calls["count"] >= 2


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_analyze_paths_forest_progress_emits_intermediate_work_markers::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.deadline_scope
def test_analyze_paths_forest_progress_emits_intermediate_work_markers(
    tmp_path: Path,
) -> None:
    da = _load()
    module_paths: list[Path] = []
    for index in range(70):
        module_path = tmp_path / f"mod_{index:03d}.py"
        module_path.write_text(
            textwrap.dedent(
                f"""
                def callee_{index}(value):
                    return value

                def root_{index}(value):
                    return callee_{index}(value)
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        module_paths.append(module_path)

    forest_progress: list[tuple[int, int]] = []

    def _phase_progress(
        phase: str,
        _groups_by_path: dict[Path, dict[str, list[set[str]]]],
        _report: da.ReportCarrier,
        work_done: int,
        work_total: int,
    ) -> None:
        if phase == "forest":
            forest_progress.append((work_done, work_total))

    with deadline_scope(Deadline.from_timeout_ms(20_000)):
        da.analyze_paths(
            module_paths,
            forest=da.Forest(),
            recursive=True,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=0,
            include_constant_smells=False,
            include_unused_arg_smells=False,
            include_deadness_witnesses=False,
            include_coherence_witnesses=False,
            include_rewrite_plans=False,
            include_exception_obligations=False,
            include_handledness_witnesses=False,
            include_never_invariants=False,
            include_wl_refinement=False,
            include_decision_surfaces=False,
            include_value_decision_surfaces=False,
            include_invariant_propositions=False,
            include_lint_lines=False,
            include_ambiguities=False,
            include_bundle_forest=True,
            include_deadline_obligations=False,
            config=da.AuditConfig(
                project_root=tmp_path,
                exclude_dirs=set(),
                ignore_params=set(),
                external_filter=True,
                strictness="high",
            ),
            file_paths_override=module_paths,
            on_phase_progress=_phase_progress,
        )

    assert forest_progress
    work_values = [work_done for work_done, _ in forest_progress]
    min_work = min(work_values)
    max_work = max(work_values)
    assert min_work < max_work
    assert any(min_work < work_done < max_work for work_done in work_values)

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings
def test_compute_fingerprint_warnings_missing_annotations(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    groups_by_path = {target: {"caller": [set(["a", "b"])]}}
    annotations_by_path = {target: {"caller": {"a": "int"}}}
    warnings = da._compute_fingerprint_warnings(
        groups_by_path,
        annotations_by_path,
        registry=da.PrimeRegistry(),
        index={object(): set()},
    )
    assert warnings
    assert "missing type annotations" in warnings[0]


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_emit_report_fingerprint_warnings_are_non_blocking::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_emit_report_fingerprint_warnings_are_non_blocking() -> None:
    da = _load()
    report, violations = da._emit_report(
        {},
        10,
        report=da.ReportCarrier(
            forest=da.Forest(),
            fingerprint_warnings=["example fingerprint warning"],
        ),
        parse_witness_contract_violations_fn=lambda: [],
    )
    assert "Fingerprint warnings:" in report
    assert "example fingerprint warning" in report
    assert violations == []


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_known_violation_lines_dedupes_duplicates::dataflow_audit.py::gabion.analysis.dataflow_audit._known_violation_lines::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_known_violation_lines_dedupes_duplicates() -> None:
    da = _load()
    lines = da._known_violation_lines(
        da.ReportCarrier(
            forest=da.Forest(),
            decision_warnings=["dup", "dup"],
        )
    )
    assert lines == ["dup"]

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths
def test_analyze_paths_deadline_includes_forest_spec(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text("def callee(x):\n    return x\n", encoding="utf-8")
    with deadline_scope(Deadline.from_timeout_ms(10_000)):
        result = da.analyze_paths(
            [target],
            forest=da.Forest(),
            recursive=True,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=0,
            include_constant_smells=False,
            include_unused_arg_smells=False,
            include_bundle_forest=True,
            config=da.AuditConfig(
                project_root=tmp_path,
                exclude_dirs=set(),
                ignore_params=set(),
                external_filter=True,
                strictness="high",
            ),
        )
    assert result.forest is not None

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_deadline_obligations
def test_deadline_missing_carrier_for_loop(tmp_path: Path) -> None:
    obligations = _deadline_obligations(
        tmp_path,
        """
        def loop():
            for _ in range(1):
                pass
        """,
        roots={"mod.root"},
    )
    assert any(entry.get("kind") == "missing_carrier" for entry in obligations)

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_deadline_obligations
def test_deadline_none_arg_violation(tmp_path: Path) -> None:
    obligations = _deadline_obligations(
        tmp_path,
        """
        def callee(deadline: Deadline):
            return 1

        def root():
            callee(None)
        """,
        roots={"mod.root"},
    )
    assert any(entry.get("kind") == "none_arg" for entry in obligations)

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_deadline_obligations
def test_deadline_origin_not_allowlisted(tmp_path: Path) -> None:
    obligations = _deadline_obligations(
        tmp_path,
        """
        def callee(deadline: Deadline):
            return 1

        def helper():
            deadline = Deadline.from_timeout_ms(1_000)
            callee(deadline)
        """,
        roots={"mod.root"},
    )
    assert any(entry.get("kind") == "origin_not_allowlisted" for entry in obligations)

# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_collect_call_edges_filters_test_and_unresolved::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_call_edges::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._call::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_collect_call_edges_filters_test_and_unresolved() -> None:
    da = _load()
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.caller",
        path=Path("pkg/mod.py"),
        params=[],
        annots={},
        calls=[
            _call(da, callee="ignored", is_test=True),
            _call(da, callee="none"),
            _call(da, callee="target"),
        ],
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
    by_name = {"caller": [caller], "target": [candidate]}
    by_qual = {caller.qual: caller, candidate.qual: candidate}

    def _resolve(callee_key: str, *_args, **_kwargs):
        if callee_key == "none":
            return da._CalleeResolutionOutcome(
                status="unresolved",
                phase="none",
                callee_key=callee_key,
                candidates=(),
            )
        return da._CalleeResolutionOutcome(
            status="resolved",
            phase="resolved",
            callee_key=callee_key,
            candidates=(candidate,),
        )

    edges = da._collect_call_edges(
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=da.SymbolTable(),
        project_root=Path("."),
        class_index={},
        resolve_callee_outcome_fn=_resolve,
    )
    assert edges == {"pkg.caller": {"pkg.target"}}

# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_collect_call_edges_and_obligations_from_forest::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_call_edges_from_forest::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_call_resolution_obligations_from_forest::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_unresolved_call_sites_from_forest::dataflow_audit.py::gabion.analysis.dataflow_audit._function_suite_id::dataflow_audit.py::gabion.analysis.dataflow_audit._function_suite_key::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_collect_call_edges_and_obligations_from_forest() -> None:
    da = _load()
    forest = da.Forest()
    call_suite = forest.add_suite_site(
        "pkg/mod.py",
        "pkg.caller",
        "call",
        span=(1, 1, 1, 4),
    )
    non_call_suite = forest.add_suite_site(
        "pkg/mod.py",
        "pkg.caller",
        "function_body",
        span=(2, 1, 2, 4),
    )
    target_fn = forest.add_site("pkg/target.py", "pkg.target")
    junk = forest.add_node("Other", ("x",), {"x": 1})
    forest.add_alt("CallCandidate", (call_suite,))
    forest.add_alt("CallCandidate", (call_suite, junk))
    forest.add_alt("CallCandidate", (call_suite, target_fn))
    forest.add_alt("CallResolutionObligation", ())
    forest.add_alt("CallResolutionObligation", (non_call_suite,), evidence={"callee": "target"})
    forest.add_alt("CallResolutionObligation", (call_suite,), evidence={})
    forest.add_alt("CallResolutionObligation", (call_suite,), evidence={"callee": "target"})
    forest.add_alt(
        "CallResolutionObligation",
        (call_suite,),
        evidence={"callee": "target", "source": "duplicate"},
    )
    by_name = {
        "target": [
            da.FunctionInfo(
                name="target",
                qual="pkg.target",
                path=Path("tests/test_mod.py"),
                params=[],
                annots={},
                calls=[],
                unused_params=set(),
                function_span=(0, 0, 0, 1),
            ),
            da.FunctionInfo(
                name="target",
                qual="pkg.target",
                path=Path("target.py"),
                params=[],
                annots={},
                calls=[],
                unused_params=set(),
                function_span=(0, 0, 0, 1),
            ),
        ]
    }
    edges = da._collect_call_edges_from_forest(forest, by_name=by_name)
    caller_suite_id = da._function_suite_id(da._function_suite_key("pkg/mod.py", "pkg.caller"))
    assert caller_suite_id in edges
    assert edges[caller_suite_id]

    obligations = da._collect_call_resolution_obligations_from_forest(forest)
    assert obligations
    assert obligations[0][2] == (1, 1, 1, 4)
    unresolved = da._collect_unresolved_call_sites_from_forest(forest)
    assert unresolved
    assert unresolved[0][:2] == ("pkg/mod.py", "pkg.caller")

# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_collect_unresolved_call_sites_filters_non_suite_ids::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_unresolved_call_sites_from_forest::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_collect_unresolved_call_sites_filters_non_suite_ids() -> None:
    da = _load()
    caller_bad_kind = da.NodeId("FunctionSite", ("p", "q"))
    caller_missing_parts = da.NodeId("SuiteSite", ("p",))
    caller_missing_path = da.NodeId("SuiteSite", ("", "q"))
    caller_good = da.NodeId("SuiteSite", ("p", "q", "call"))
    suite_id = da.NodeId("SuiteSite", ("p", "q", "call"))
    out = da._collect_unresolved_call_sites_from_forest(
        da.Forest(),
        collect_call_resolution_obligations_from_forest_fn=lambda _forest: [
            (caller_bad_kind, suite_id, None, "a"),
            (caller_missing_parts, suite_id, None, "b"),
            (caller_missing_path, suite_id, None, "c"),
            (caller_good, suite_id, (1, 2, 3, 4), "d"),
        ],
    )
    assert out == [("p", "q", (1, 2, 3, 4), "d")]

# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_suite_identity_helpers_raise_on_missing_identity::dataflow_audit.py::gabion.analysis.dataflow_audit._node_to_function_suite_id::dataflow_audit.py::gabion.analysis.dataflow_audit._suite_caller_function_id::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_suite_identity_helpers_raise_on_missing_identity() -> None:
    da = _load()
    with pytest.raises(NeverThrown):
        da._suite_caller_function_id(
            da.Node(
                node_id=da.NodeId("SuiteSite", ("x",)),
                meta={"suite_kind": "call"},
            )
        )

    forest = da.Forest()
    missing_fn = da.NodeId("FunctionSite", ("x",))
    forest.nodes[missing_fn] = da.Node(node_id=missing_fn, meta={"path": "", "qual": "x"})
    with pytest.raises(NeverThrown):
        da._node_to_function_suite_id(forest, missing_fn)

    missing_suite = da.NodeId("SuiteSite", ("x",))
    forest.nodes[missing_suite] = da.Node(
        node_id=missing_suite,
        meta={"suite_kind": "function", "path": "a.py", "qual": ""},
    )
    with pytest.raises(NeverThrown):
        da._node_to_function_suite_id(forest, missing_suite)

# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_collect_call_resolution_obligations_requires_span::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_call_resolution_obligations_from_forest::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_collect_call_resolution_obligations_requires_span() -> None:
    da = _load()
    forest = da.Forest()
    call_suite = forest.add_suite_site(
        "pkg/mod.py",
        "pkg.caller",
        "call",
        span=(1, 1, 1, 4),
    )
    bad_suite = da.NodeId("SuiteSite", call_suite.key)
    forest.nodes[bad_suite] = da.Node(
        node_id=bad_suite,
        meta={
            "suite_kind": "call",
            "path": "pkg/mod.py",
            "qual": "pkg.caller",
            "span": [1, "x", 1, 4],
        },
    )
    forest.add_alt("CallResolutionObligation", (bad_suite,), evidence={"callee": "target"})
    with pytest.raises(NeverThrown):
        da._collect_call_resolution_obligations_from_forest(forest)

# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_materialize_call_candidates_covers_obligation_and_external_paths::dataflow_audit.py::gabion.analysis.dataflow_audit._materialize_call_candidates::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._call::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_materialize_call_candidates_covers_obligation_and_external_paths() -> None:
    da = _load()
    forest = da.Forest()
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.caller",
        path=Path("pkg/mod.py"),
        params=[],
        annots={},
        calls=[
            _call(da, callee="skip-test", is_test=True),
            _call(da, callee="external", span=None),
            _call(da, callee="internal", span=(1, 1, 1, 4)),
            _call(da, callee="internal", span=(1, 1, 1, 4)),
        ],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    by_name = {"caller": [caller]}
    by_qual = {caller.qual: caller}

    def _resolve(callee_key: str, *_args, **_kwargs):
        if callee_key == "external":
            return da._CalleeResolutionOutcome(
                status="unresolved_external",
                phase="external",
                callee_key=callee_key,
                candidates=(),
            )
        return da._CalleeResolutionOutcome(
            status="unresolved_internal",
            phase="internal",
            callee_key=callee_key,
            candidates=(),
        )

    da._materialize_call_candidates(
        forest=forest,
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=da.SymbolTable(),
        project_root=Path("."),
        class_index={},
        resolve_callee_outcome_fn=_resolve,
    )
    obligations = [
        alt
        for alt in forest.alts
        if alt.kind == "CallResolutionObligation"
    ]
    assert len(obligations) == 1

# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_sorted_graph_nodes_and_reachable_from_roots_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._reachable_from_roots::dataflow_audit.py::gabion.analysis.dataflow_audit._sorted_graph_nodes::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_sorted_graph_nodes_and_reachable_from_roots_edges() -> None:
    da = _load()
    mixed = da._sorted_graph_nodes({1, "2"})  # type: ignore[arg-type]
    assert mixed
    graph = {"a": {"b", "c"}, "b": {"d"}, "c": {"d"}, "d": set()}
    reachable = da._reachable_from_roots(graph, {"a", "a"})
    assert reachable == {"a", "b", "c", "d"}

# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_collect_deadline_obligations_call_resolution_filter_edges::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_deadline_obligations::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.deadline_scope
def test_collect_deadline_obligations_call_resolution_filter_edges(
    tmp_path: Path,
) -> None:
    da = _load()
    forest = da.Forest()
    missing_suite = da.NodeId("SuiteSite", ("pkg/mod.py", "pkg.root", "call"))
    non_call_suite = forest.add_suite_site(
        "pkg/mod.py",
        "pkg.root",
        "function",
        span=(1, 1, 1, 2),
    )
    forest.add_alt(
        "CallCandidate",
        (missing_suite, da.NodeId("FunctionSite", ("pkg/mod.py", "pkg.target"))),
    )
    forest.add_alt(
        "CallCandidate",
        (non_call_suite, da.NodeId("FunctionSite", ("pkg/mod.py", "pkg.target"))),
    )

    root_info = da.FunctionInfo(
        name="root",
        qual="pkg.root",
        path=Path("pkg/mod.py"),
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        function_span=(1, 1, 1, 2),
    )
    index = da.AnalysisIndex(
        by_name={"root": [root_info]},
        by_qual={root_info.qual: root_info},
        symbol_table=da.SymbolTable(),
        class_index={},
    )
    caller_not_reachable = da.NodeId("SuiteSite", ("pkg/mod.py", "pkg.root", "function"))
    caller_bad_kind = da.NodeId("FileSite", ("pkg/mod.py",))
    caller_empty = da.NodeId("SuiteSite", ("pkg/mod.py", "", "function"))
    caller_exempt = da.NodeId(
        "SuiteSite",
        ("pkg/mod.py", "gabion.analysis.timeout_context.helper", "function"),
    )
    caller_missing = da.NodeId("SuiteSite", ("pkg/mod.py", "pkg.missing", "function"))

    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=True,
        strictness="high",
        deadline_roots={"pkg.root"},
    )
    with deadline_scope(Deadline.from_timeout_ms(10_000)):
        obligations = da._collect_deadline_obligations(
            [tmp_path / "mod.py"],
            project_root=tmp_path,
            config=config,
            forest=forest,
            parse_failure_witnesses=[],
            analysis_index=index,
            materialize_call_candidates_fn=lambda **_kwargs: None,
            collect_call_nodes_by_path_fn=lambda *_args, **_kwargs: {},
            collect_deadline_function_facts_fn=lambda *_args, **_kwargs: {},
            collect_call_edges_from_forest_fn=lambda *_args, **_kwargs: {},
            collect_call_resolution_obligations_from_forest_fn=lambda _forest: [
                (caller_not_reachable, non_call_suite, (1, 1, 1, 2), "x"),
                (caller_bad_kind, non_call_suite, (1, 1, 1, 2), "x"),
                (caller_empty, non_call_suite, (1, 1, 1, 2), "x"),
                (caller_exempt, non_call_suite, (1, 1, 1, 2), "x"),
                (caller_missing, non_call_suite, (1, 1, 1, 2), "x"),
            ],
            reachable_from_roots_fn=lambda *_args, **_kwargs: {
                caller_bad_kind,
                caller_empty,
                caller_exempt,
                caller_missing,
            },
            collect_recursive_nodes_fn=lambda _edges: {
                da.NodeId("FileSite", ("x",)),
                da.NodeId("SuiteSite", ("pkg/mod.py", "", "function")),
                da.NodeId(
                    "SuiteSite",
                    ("pkg/mod.py", "gabion.analysis.timeout_context.rec", "function"),
                ),
            },
        )
    assert obligations == []

# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_analyze_paths_timeout_flushes_phase_emitters_best_effort::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.deadline_scope::timeout_context.py::gabion.analysis.timeout_context.pack_call_stack
def test_analyze_paths_timeout_flushes_phase_emitters_best_effort(
    tmp_path: Path,
) -> None:
    da = _load()
    module_path = tmp_path / "mod.py"
    module_path.write_text("def f(x):\n    return x\n", encoding="utf-8")
    callback_state = {"raised": False, "phases": []}

    def _phase_progress(
        phase: str,
        _groups_by_path: dict[Path, dict[str, list[set[str]]]],
        _report: da.ReportCarrier,
        _work_done: int,
        _work_total: int,
    ) -> None:
        callback_state["phases"].append(phase)
        if phase == "post" or callback_state["raised"]:
            callback_state["raised"] = True
            raise da.TimeoutExceeded(
                TimeoutContext(
                    call_stack=pack_call_stack([{"path": str(module_path), "qual": "mod.f"}])
                )
            )

    with pytest.raises(da.TimeoutExceeded):
        with deadline_scope(Deadline.from_timeout_ms(10_000)):
            da.analyze_paths(
                [module_path],
                forest=da.Forest(),
                recursive=True,
                type_audit=False,
                type_audit_report=False,
                type_audit_max=0,
                include_constant_smells=False,
                include_unused_arg_smells=False,
                include_deadness_witnesses=False,
                include_coherence_witnesses=False,
                include_rewrite_plans=False,
                include_exception_obligations=False,
                include_handledness_witnesses=False,
                include_never_invariants=False,
                include_wl_refinement=False,
                include_decision_surfaces=False,
                include_value_decision_surfaces=False,
                include_invariant_propositions=False,
                include_lint_lines=False,
                include_ambiguities=False,
                include_bundle_forest=False,
                include_deadline_obligations=False,
                config=da.AuditConfig(project_root=tmp_path),
                file_paths_override=[module_path],
                on_phase_progress=_phase_progress,
            )
    # The except-path flush invokes all phase emitters after the timeout.
    assert "forest" in callback_state["phases"]
    assert "edge" in callback_state["phases"]
    assert "post" in callback_state["phases"]


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_analyze_paths_phase_progress_emits_initial_edge_and_post_checkpoints::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load::timeout_context.py::gabion.analysis.timeout_context.Deadline.from_timeout_ms::timeout_context.py::gabion.analysis.timeout_context.deadline_scope
def test_analyze_paths_phase_progress_emits_initial_edge_and_post_checkpoints(
    tmp_path: Path,
) -> None:
    da = _load()
    module_path = tmp_path / "mod.py"
    module_path.write_text(
        textwrap.dedent(
            """
            from gabion.analysis.timeout_context import Deadline, check_deadline

            def root(deadline: Deadline):
                check_deadline(deadline)
                return deadline
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    phase_progress: dict[str, list[tuple[int, int]]] = {}
    post_markers: list[str] = []

    def _phase_progress(
        phase: str,
        _groups_by_path: dict[Path, dict[str, list[set[str]]]],
        _report: da.ReportCarrier,
        work_done: int,
        work_total: int,
    ) -> None:
        phase_progress.setdefault(phase, []).append((work_done, work_total))
        if phase == "post":
            post_markers.append(str(_report.progress_marker or ""))

    with deadline_scope(Deadline.from_timeout_ms(10_000)):
        da.analyze_paths(
            [module_path],
            forest=da.Forest(),
            recursive=True,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=0,
            include_constant_smells=True,
            include_unused_arg_smells=False,
            include_deadness_witnesses=False,
            include_coherence_witnesses=False,
            include_rewrite_plans=False,
            include_exception_obligations=False,
            include_handledness_witnesses=False,
            include_never_invariants=False,
            include_wl_refinement=False,
            include_decision_surfaces=False,
            include_value_decision_surfaces=False,
            include_invariant_propositions=False,
            include_lint_lines=False,
            include_ambiguities=False,
            include_bundle_forest=True,
            include_deadline_obligations=True,
            config=da.AuditConfig(
                project_root=tmp_path,
                exclude_dirs=set(),
                ignore_params=set(),
                external_filter=True,
                strictness="high",
                deadline_roots={"mod.root"},
            ),
            file_paths_override=[module_path],
            on_phase_progress=_phase_progress,
        )

    edge_progress = phase_progress.get("edge", [])
    post_progress = phase_progress.get("post", [])
    assert edge_progress
    assert post_progress
    assert edge_progress[0][0] == 0
    assert edge_progress[0][1] >= 1
    assert any(work_done > 0 for work_done, _ in edge_progress)
    assert post_progress[0][0] == 0
    assert post_progress[0][1] >= 1
    assert any(work_done > 0 for work_done, _ in post_progress)
    assert any(marker.startswith("deadline_obligations:") for marker in post_markers)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_materialize_call_candidates_emits_dynamic_obligation_kind::dataflow_audit.py::gabion.analysis.dataflow_audit._materialize_call_candidates::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._call::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._load
def test_materialize_call_candidates_emits_dynamic_obligation_kind() -> None:
    da = _load()
    forest = da.Forest()
    caller = da.FunctionInfo(
        name="caller",
        qual="pkg.caller",
        path=Path("pkg/mod.py"),
        params=[],
        annots={},
        calls=[_call(da, callee="getattr(svc, name)", span=(1, 1, 1, 8))],
        unused_params=set(),
        function_span=(0, 0, 0, 1),
    )
    da._materialize_call_candidates(
        forest=forest,
        by_name={"caller": [caller]},
        by_qual={caller.qual: caller},
        symbol_table=da.SymbolTable(),
        project_root=Path("."),
        class_index={},
        resolve_callee_outcome_fn=lambda *_args, **_kwargs: da._CalleeResolutionOutcome(
            status="unresolved_dynamic",
            phase="unresolved_dynamic",
            callee_key="getattr(svc, name)",
            candidates=(),
        ),
    )
    obligations = [alt for alt in forest.alts if alt.kind == "CallResolutionObligation"]
    assert obligations
    assert obligations[0].evidence.get("kind") in {
        "unresolved_dynamic_callee",
        "unresolved_dynamic_dispatch",
    }


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_deadline_nested_recursion_loop_attributes_inner_only::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._deadline_obligations
def test_deadline_nested_recursion_loop_attributes_inner_only(tmp_path: Path) -> None:
    obligations = _deadline_obligations(
        tmp_path,
        """
        def root(deadline: Deadline):
            for _ in range(1):
                for _ in range(1):
                    check_deadline(deadline)
                    root(deadline)
        """,
        roots={"mod.root"},
    )
    loop_obligations = [
        entry
        for entry in obligations
        if entry.get("kind") == "unchecked_deadline"
        and str(entry.get("detail", "")).startswith("deadline carrier not checked or forwarded")
    ]
    assert len(loop_obligations) == 1
    site = loop_obligations[0].get("site", {})
    assert isinstance(site, dict)
    assert site.get("suite_kind") == "loop"
    assert loop_obligations[0].get("span") == [2, 4, 5, 26]


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_deadline_suite_identity_stable_across_runs::order_contract.py::gabion.order_contract.ordered_or_sorted::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._deadline_analysis
def test_deadline_suite_identity_stable_across_runs(tmp_path: Path) -> None:
    source = """
    def root(deadline: Deadline):
        for _ in range(1):
            check_deadline(deadline)
    """
    first = _deadline_analysis(tmp_path, source, roots={"mod.root"})
    second = _deadline_analysis(tmp_path, source, roots={"mod.root"})
    first_ids = ordered_or_sorted(
        [
            str(entry.get("site", {}).get("suite_id", ""))
            for entry in first.deadline_obligations
        ],
        source="test_deadline_suite_identity_stable_across_runs.first_ids",
    )
    second_ids = ordered_or_sorted(
        [
            str(entry.get("site", {}).get("suite_id", ""))
            for entry in second.deadline_obligations
        ],
        source="test_deadline_suite_identity_stable_across_runs.second_ids",
    )
    assert first_ids == second_ids
    assert all(suite_id.startswith("suite:") for suite_id in first_ids)


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_edges.py::test_deadline_obligations_emit_suite_metadata_from_forest::test_dataflow_audit_edges.py::tests.test_dataflow_audit_edges._deadline_analysis
def test_deadline_obligations_emit_suite_metadata_from_forest(tmp_path: Path) -> None:
    analysis = _deadline_analysis(
        tmp_path,
        """
        def root(deadline: Deadline):
            target(None)

        def target(deadline: Deadline):
            return None
        """,
        roots={"mod.root"},
    )
    assert analysis.forest is not None
    by_identity = {
        str(node.meta.get("suite_id", "")): node
        for node in analysis.forest.nodes.values()
        if node.node_id.kind == "SuiteSite"
    }
    for entry in analysis.deadline_obligations:
        site = entry.get("site", {})
        assert isinstance(site, dict)
        suite_id = str(site.get("suite_id", ""))
        assert suite_id in by_identity
        suite_kind = str(site.get("suite_kind", ""))
        assert suite_kind == str(by_identity[suite_id].meta.get("suite_kind", ""))
