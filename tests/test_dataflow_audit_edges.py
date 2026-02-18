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

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import dataflow_audit as da

    return da

def _deadline_obligations(tmp_path: Path, source: str, roots: set[str]) -> list[dict]:
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
    result = da.analyze_paths(
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
    return result.deadline_obligations

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

def test_sorted_graph_nodes_and_reachable_from_roots_edges() -> None:
    da = _load()
    mixed = da._sorted_graph_nodes({1, "2"})  # type: ignore[arg-type]
    assert mixed
    graph = {"a": {"b", "c"}, "b": {"d"}, "c": {"d"}, "d": set()}
    reachable = da._reachable_from_roots(graph, {"a", "a"})
    assert reachable == {"a", "b", "c", "d"}

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
