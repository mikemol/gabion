from __future__ import annotations

from pathlib import Path
import textwrap

def _write(tmp_path: Path, rel: str, content: str) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip())
    return path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis.dataflow_audit import AuditConfig, build_synthesis_plan

    return AuditConfig, build_synthesis_plan

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._merge_counts_by_knobs::knob_names E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::merge.py::gabion.synthesis.merge.merge_bundles::min_overlap E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_knob_param_names::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::stale_b7f32d24c823
def test_synthesis_plan_applies_tiers_and_max_tier(tmp_path: Path) -> None:
    AuditConfig, build_synthesis_plan = _load()
    _write(
        tmp_path,
        "settings.py",
        """
        from dataclasses import dataclass

        @dataclass
        class AppConfig:
            a: int
            b: int
        """,
    )
    groups_by_path = {
        tmp_path / "mod.py": {
            "f": [{"a", "b"}],
            "g": [{"c", "d"}],
            "h": [{"c", "d"}],
            "i": [{"e", "f"}],
        }
    }
    plan = build_synthesis_plan(
        groups_by_path,
        project_root=tmp_path,
        max_tier=2,
        config=AuditConfig(project_root=tmp_path),
    )
    protocols = plan.get("protocols", [])
    tiers = {frozenset(p.get("bundle", [])): p.get("tier") for p in protocols}
    assert tiers.get(frozenset({"a", "b"})) == 1
    assert tiers.get(frozenset({"c", "d"})) == 2
    assert frozenset({"e", "f"}) not in tiers
