from __future__ import annotations

from pathlib import Path
from gabion.analysis.aspf import Forest

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis.dataflow_audit import (
        AuditConfig,
        analyze_constant_flow_repo,
        analyze_deadness_flow_repo,
        analyze_paths,
    )

    return AuditConfig, analyze_constant_flow_repo, analyze_deadness_flow_repo, analyze_paths

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::stale_f18eab6be1db
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

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::stale_88ad20dbfd21
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

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::stale_760cf9afd6d3
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

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::stale_afbbd83c86af
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

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::stale_702a5402d7c9
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

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::stale_dd40e1a0f833
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

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::stale_c0b13d2d52eb
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

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::stale_170827243d29
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

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._format_call_site
def test_format_call_site_handles_missing_span(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis.dataflow_audit import CallArgs, FunctionInfo, _format_call_site

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
