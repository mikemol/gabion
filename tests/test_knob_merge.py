from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import AuditConfig, analyze_paths, build_synthesis_plan

    return AuditConfig, analyze_paths, build_synthesis_plan


def test_knob_merge_by_constant_params(tmp_path: Path) -> None:
    AuditConfig, analyze_paths, build_synthesis_plan = _load()
    source = textwrap.dedent(
        """
        def sink(a, b, mode):
            return a, b, mode

        def f(a, b, mode):
            return sink(a, b, mode)

        def g(a, b):
            return f(a, b, "fast")
        """
    )
    file_path = tmp_path / "mod.py"
    file_path.write_text(source.strip() + "\n")
    config = AuditConfig(project_root=tmp_path)
    groups_by_path = {
        file_path: {
            "f": [set(["a", "b", "mode"])],
            "g": [set(["a", "b"])],
        }
    }
    plan = build_synthesis_plan(
        groups_by_path,
        project_root=tmp_path,
        max_tier=3,
        merge_overlap_threshold=0.9,
        config=config,
    )
    bundles = [set(protocol["bundle"]) for protocol in plan.get("protocols", [])]
    assert {"a", "b", "mode"} in bundles
    assert {"a", "b"} not in bundles
