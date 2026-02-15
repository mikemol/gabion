from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import (
        AuditConfig,
        analyze_constant_flow_repo,
        build_synthesis_plan,
    )

    return AuditConfig, analyze_constant_flow_repo, build_synthesis_plan


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._merge_counts_by_knobs::knob_names E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::merge.py::gabion.synthesis.merge.merge_bundles::min_overlap E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_constant_flow_details::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_knob_param_names::strictness
def test_constant_flow_varargs_and_knob_branches(tmp_path: Path) -> None:
    AuditConfig, analyze_constant_flow_repo, build_synthesis_plan = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a: int, *args: str, **kwargs: float):\n"
        "    return a\n"
        "\n"
        "def caller(x, xs, kw):\n"
        "    callee(1, 2)\n"
        "    callee(1, x)\n"
        "    callee(1, x + 1)\n"
        "    callee(1, extra=1)\n"
        "    callee(1, extra=x)\n"
        "    callee(1, extra=x + 1)\n"
        "    callee(1, *xs)\n"
        "    callee(1, **kw)\n"
    )

    smells = analyze_constant_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="low",
        external_filter=False,
    )
    assert isinstance(smells, list)

    config = AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="low",
        transparent_decorators=None,
    )
    plan = build_synthesis_plan(
        {path: {}},
        project_root=tmp_path,
        max_tier=2,
        min_bundle_size=2,
        allow_singletons=False,
        config=config,
    )
    assert "warnings" in plan
