from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import analyze_paths, build_synthesis_plan, AuditConfig

    return analyze_paths, build_synthesis_plan, AuditConfig


def test_synthesis_plan_includes_control_context_evidence(tmp_path: Path) -> None:
    analyze_paths, build_synthesis_plan, AuditConfig = _load()
    source = textwrap.dedent(
        """
        def g(x, y):
            return x + y

        def f(a, b, c):
            if c:
                g(a, a)
            return g(b, b)
        """
    ).lstrip()
    file_path = tmp_path / "mod.py"
    file_path.write_text(source)
    config = AuditConfig(project_root=tmp_path)
    analysis = analyze_paths(
        [file_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        config=config,
    )
    plan = build_synthesis_plan(
        analysis.groups_by_path,
        project_root=tmp_path,
        max_tier=3,
        min_bundle_size=2,
        allow_singletons=False,
        config=config,
    )
    protocols = plan.get("protocols", [])
    assert protocols
    bundle_map = {
        tuple(sorted(p.get("bundle", []))): set(p.get("evidence", []))
        for p in protocols
    }
    evidence = bundle_map.get(("a", "b"))
    assert evidence is not None
    assert "dataflow" in evidence
    assert "control_context" in evidence
