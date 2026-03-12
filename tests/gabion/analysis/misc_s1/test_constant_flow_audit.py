from __future__ import annotations

from pathlib import Path
from tests.path_helpers import REPO_ROOT
from gabion.analysis.aspf.aspf import Forest
from gabion.analysis.foundation.json_types import ParseFailureWitnesses

def _load():
    repo_root = REPO_ROOT
    from gabion.analysis import (
        AuditConfig, analyze_paths)
    from gabion.analysis.dataflow.engine.dataflow_post_phase_analyses import (
        analyze_constant_flow_repo,
        analyze_deadness_flow_repo,
    )

    return AuditConfig, analyze_constant_flow_repo, analyze_deadness_flow_repo, analyze_paths

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_paths::config E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_never_invariants::forest E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._analyze_file_internal::stale_f18eab6be1db
# gabion:behavior primary=desired
def test_constant_flow_smells_and_star_paths(tmp_path: Path) -> None:
    AuditConfig, _, _, analyze_paths = _load()
    code = (
        "def target(a, b, c):\n"
        "    return a\n"
        "\n"
        "def caller(x, *args, **kwargs):\n"
        "    target(1, b=2, c=3)\n"
        "    target(*args, **kwargs)\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(code)
    config = AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="low",
        transparent_decorators=None,
    )
    analysis = analyze_paths(
        forest=Forest(),
        paths=[path],
        recursive=False,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=5,
        include_constant_smells=True,
        include_unused_arg_smells=False,
        config=config,
    )
    assert isinstance(analysis.constant_smells, list)

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::stale_88ad20dbfd21
# gabion:behavior primary=desired
def test_constant_flow_detects_constant_kw_and_ignores_non_const(tmp_path: Path) -> None:
    _, analyze_constant_flow_repo, _, _ = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a, b):\n"
        "    return a\n"
        "\n"
        "def caller(x):\n"
        "    return callee(a=1, b=x)\n"
    )
    smells = analyze_constant_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert any("callee.a only observed constant 1" in smell for smell in smells)

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::stale_760cf9afd6d3
# gabion:behavior primary=desired
def test_constant_flow_skips_test_paths(tmp_path: Path) -> None:
    _, analyze_constant_flow_repo, _, _ = _load()
    path = tmp_path / "tests" / "test_mod.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "def callee(a):\n"
        "    return a\n"
        "\n"
        "def caller():\n"
        "    return callee(1)\n"
    )
    smells = analyze_constant_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert smells == []

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::stale_afbbd83c86af
# gabion:behavior primary=desired
def test_constant_flow_low_strictness_star_handling(tmp_path: Path) -> None:
    _, analyze_constant_flow_repo, _, _ = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a, b):\n"
        "    return a\n"
        "\n"
        "def caller(*args, **kwargs):\n"
        "    return callee(*args, **kwargs)\n"
    )
    smells = analyze_constant_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="low",
        external_filter=True,
    )
    assert smells == []

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::stale_702a5402d7c9
# gabion:behavior primary=desired
def test_constant_flow_ignores_extra_pos_args(tmp_path: Path) -> None:
    _, analyze_constant_flow_repo, _, _ = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a):\n"
        "    return a\n"
        "\n"
        "def caller(x, y):\n"
        "    return callee(x, y)\n"
    )
    smells = analyze_constant_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert smells == []

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::stale_dd40e1a0f833
# gabion:behavior primary=desired
def test_constant_flow_tracks_non_const_kw(tmp_path: Path) -> None:
    _, analyze_constant_flow_repo, _, _ = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a, b):\n"
        "    return b\n"
        "\n"
        "def caller(x):\n"
        "    return callee(a=1, b=x + 1)\n"
    )
    smells = analyze_constant_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert any("callee.a only observed constant 1" in smell for smell in smells)

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::stale_c0b13d2d52eb
# gabion:behavior primary=desired
def test_constant_flow_skips_multi_value_constants(tmp_path: Path) -> None:
    _, analyze_constant_flow_repo, _, _ = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a):\n"
        "    return a\n"
        "\n"
        "def caller():\n"
        "    callee(1)\n"
        "    callee(2)\n"
    )
    smells = analyze_constant_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert smells == []

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._normalize_snapshot_path::root E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_constant_flow_details::stale_170827243d29
# gabion:behavior primary=desired
def test_deadness_witnesses_from_constant_flow(tmp_path: Path) -> None:
    _, _, analyze_deadness_flow_repo, _ = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a):\n"
        "    return a\n"
        "\n"
        "def caller():\n"
        "    return callee(1)\n"
    )
    witnesses = analyze_deadness_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert witnesses
    entry = witnesses[0]
    assert entry["path"].endswith("mod.py")
    assert entry["function"] == "callee"
    assert entry["bundle"] == ["a"]
    assert entry["environment"] == {"a": "1"}
    assert entry["result"] == "UNREACHABLE"

# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._format_call_site
# gabion:behavior primary=verboten facets=missing
def test_format_call_site_handles_missing_span(tmp_path: Path) -> None:
    repo_root = REPO_ROOT
    from gabion.analysis.dataflow.engine.dataflow_contracts import CallArgs, FunctionInfo
    from gabion.analysis.dataflow.engine.dataflow_post_phase_analyses import (
        _format_call_site,
    )

    caller = FunctionInfo(
        name="caller",
        qual="pkg.mod.caller",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        scope=("pkg", "mod"),
        function_span=(0, 0, 0, 1),
    )
    call = CallArgs(
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
        span=None,
    )
    assert _format_call_site(caller, call) == "mod.py:pkg.mod.caller"

    call_with_span = CallArgs(
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
        span=(4, 5, 4, 6),
    )
    assert _format_call_site(caller, call_with_span) == "mod.py:5:6:pkg.mod.caller"


# gabion:evidence E:function_site::dataflow_post_phase_analyses.py::gabion.analysis.dataflow_post_phase_analyses._collect_constant_flow_details
# gabion:behavior primary=desired
def test_collect_constant_flow_details_uses_injected_reduce_and_iter(
    tmp_path: Path,
) -> None:
    from gabion.analysis.dataflow.engine import dataflow_analysis_index as index_owner
    from gabion.analysis.dataflow.engine import dataflow_post_phase_analyses as post_phase

    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a):\n"
        "    return a\n"
        "\n"
        "def caller():\n"
        "    return callee(1)\n"
    )
    parse_failure_witnesses: ParseFailureWitnesses = []
    analysis_index = index_owner._build_analysis_index(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        parse_failure_witnesses=parse_failure_witnesses,
    )

    baseline = post_phase._collect_constant_flow_details(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=[],
        analysis_index=analysis_index,
    )

    seen: list[str] = []

    def _iter(*args, **kwargs):
        seen.append("iter")
        return post_phase._iter_resolved_edge_param_events(*args, **kwargs)

    def _reduce(*args, **kwargs):
        seen.append("reduce")
        return post_phase._reduce_resolved_call_edges(*args, **kwargs)

    injected = post_phase._collect_constant_flow_details(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        parse_failure_witnesses=[],
        analysis_index=analysis_index,
        iter_resolved_edge_param_events_fn=_iter,
        reduce_resolved_call_edges_fn=_reduce,
    )

    assert "iter" in seen
    assert "reduce" in seen
    assert [
        (detail.qual, detail.param, detail.value, detail.count)
        for detail in injected
    ] == [
        (detail.qual, detail.param, detail.value, detail.count)
        for detail in baseline
    ]


# gabion:evidence E:function_site::dataflow_lint_helpers.py::gabion.analysis.dataflow_lint_helpers._compute_lint_lines
# gabion:behavior primary=desired
def test_compute_lint_lines_uses_injected_projector() -> None:
    from gabion.analysis.dataflow.engine import dataflow_lint_helpers

    forest = Forest()
    seen: list[str] = []

    def _project(*, forest):
        seen.append("project")
        return [
            {
                "path": "mod.py",
                "line": 2,
                "col": 3,
                "code": "GABION_TEST",
                "message": "injected",
            }
        ]

    lines = dataflow_lint_helpers._compute_lint_lines(
        forest=forest,
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
        project_lint_rows_from_forest_fn=_project,
    )

    assert seen == ["project"]
    assert lines == ["mod.py:2:3: GABION_TEST injected"]


# gabion:evidence E:function_site::dataflow_lint_helpers.py::gabion.analysis.dataflow.engine.dataflow_lint_helpers._project_lint_rows_from_forest
# gabion:behavior primary=desired
def test_project_lint_rows_from_forest_uses_execution_ops(monkeypatch) -> None:
    from gabion.analysis.dataflow.engine import dataflow_lint_helpers

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        dataflow_lint_helpers,
        "_lint_findings_execution_ops",
        lambda: ("typed-lint-op",),
    )

    def _fake_apply_execution_ops(ops, relation):
        seen["ops"] = ops
        seen["relation"] = relation
        return relation

    monkeypatch.setattr(
        dataflow_lint_helpers,
        "apply_execution_ops",
        _fake_apply_execution_ops,
    )
    monkeypatch.setattr(
        dataflow_lint_helpers,
        "_materialize_projection_spec_rows",
        lambda **_kwargs: None,
    )

    projected = dataflow_lint_helpers._project_lint_rows_from_forest(
        forest=object(),
        relation_fn=lambda _forest: [
            {
                "path": "mod.py",
                "line": 2,
                "col": 3,
                "code": "GABION_TEST",
                "message": "typed",
                "sources": ["bundle_evidence"],
            }
        ],
    )

    assert seen["ops"] == ("typed-lint-op",)
    assert seen["relation"] == [
        {
            "path": "mod.py",
            "line": 2,
            "col": 3,
            "code": "GABION_TEST",
            "message": "typed",
            "sources": ["bundle_evidence"],
        }
    ]
    assert projected == seen["relation"]
