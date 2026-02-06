from __future__ import annotations

from pathlib import Path
import os
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def test_report_uses_forest_only_invariant(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text("def f(a, b):\n    return a\n")
    config = da.AuditConfig(project_root=tmp_path)
    analysis = da.analyze_paths(
        [path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_bundle_forest=True,
        config=config,
    )
    previous = os.environ.get("GABION_FORBID_ADHOC_BUNDLES")
    os.environ["GABION_FORBID_ADHOC_BUNDLES"] = "1"
    try:
        report, _ = da.render_report(
            analysis.groups_by_path,
            max_components=3,
            forest=analysis.forest,
        )
    finally:
        if previous is None:
            os.environ.pop("GABION_FORBID_ADHOC_BUNDLES", None)
        else:
            os.environ["GABION_FORBID_ADHOC_BUNDLES"] = previous
    assert "Dataflow grammar audit" in report
