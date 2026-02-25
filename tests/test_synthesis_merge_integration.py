from __future__ import annotations

from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis.dataflow_audit import AuditConfig, build_synthesis_plan

    return AuditConfig, build_synthesis_plan

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._merge_counts_by_knobs::knob_names E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::merge.py::gabion.synthesis.merge.merge_bundles::min_overlap E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_knob_param_names::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::stale_623d82bb3512
def test_build_synthesis_plan_merges_overlapping_bundles(tmp_path: Path) -> None:
    AuditConfig, build_synthesis_plan = _load()
    groups_by_path = {
        tmp_path / "a.py": {"f": [{"a", "b", "c"}]},
        tmp_path / "b.py": {"g": [{"a", "b", "c", "d"}]},
    }
    plan = build_synthesis_plan(
        groups_by_path,
        project_root=tmp_path,
        max_tier=3,
        config=AuditConfig(project_root=tmp_path),
    )
    protocols = plan.get("protocols", [])
    bundles = {frozenset(entry.get("bundle", [])) for entry in protocols}
    assert frozenset({"a", "b", "c", "d"}) in bundles
    assert frozenset({"a", "b", "c"}) not in bundles
