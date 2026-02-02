from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip())
    return path


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import AuditConfig, build_synthesis_plan

    return AuditConfig, build_synthesis_plan


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
    tiers = {tuple(sorted(p.get("bundle", []))): p.get("tier") for p in protocols}
    assert tiers.get(("a", "b")) == 1
    assert tiers.get(("c", "d")) == 2
    assert ("e", "f") not in tiers
