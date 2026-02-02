from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import analyze_paths, build_refactor_plan
    from gabion.analysis.dataflow_audit import AuditConfig

    return analyze_paths, build_refactor_plan, AuditConfig


def test_refactor_plan_orders_callee_first(tmp_path: Path) -> None:
    analyze_paths, build_refactor_plan, AuditConfig = _load()
    source = textwrap.dedent(
        """
        def sink(x=None, y=None):
            return x, y

        def g(a, b):
            sink(x=a, y=a)
            sink(x=b, y=b)

        def f(a, b):
            return g(a, b)
        """
    )
    file_path = tmp_path / "mod.py"
    file_path.write_text(source.strip() + "\n")
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
    plan = build_refactor_plan(analysis.groups_by_path, [file_path], config=config)
    bundles = plan.get("bundles", [])
    assert bundles
    order = bundles[0].get("order", [])
    assert order == ["mod.g", "mod.f"]
