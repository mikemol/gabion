from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import AuditConfig, build_synthesis_plan

    return AuditConfig, build_synthesis_plan


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
    bundles = {tuple(sorted(entry.get("bundle", []))) for entry in protocols}
    assert ("a", "b", "c", "d") in bundles
    assert ("a", "b", "c") not in bundles
