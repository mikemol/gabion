from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import AuditConfig, analyze_paths, render_report

    return AuditConfig, analyze_paths, render_report


def test_report_includes_callsite_evidence_for_undocumented_bundle(tmp_path: Path) -> None:
    AuditConfig, analyze_paths, render_report = _load()
    (tmp_path / "mod.py").write_text(
        "def h(x):\n"
        "    return x\n"
        "\n"
        "def g(x, y):\n"
        "    h(x)\n"
        "    h(y)\n"
        "\n"
        "def f(a, b):\n"
        "    return g(a, b)\n"
    )
    config = AuditConfig(project_root=tmp_path, external_filter=False)
    analysis = analyze_paths(
        [tmp_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        config=config,
    )
    report, _ = render_report(
        analysis.groups_by_path,
        10,
        bundle_sites_by_path=analysis.bundle_sites_by_path,
    )
    assert "Callsite evidence (undocumented bundles):" in report
    assert "mod.py" in report
    assert "f ->" in report
    assert "forwards a, b" in report


def test_report_omits_callsite_evidence_for_documented_bundle(tmp_path: Path) -> None:
    AuditConfig, analyze_paths, render_report = _load()
    (tmp_path / "mod.py").write_text(
        "# dataflow-bundle: a, b\n"
        "def h(x):\n"
        "    return x\n"
        "\n"
        "def g(x, y):\n"
        "    h(x)\n"
        "    h(y)\n"
        "\n"
        "def f(a, b):\n"
        "    return g(a, b)\n"
    )
    config = AuditConfig(project_root=tmp_path, external_filter=False)
    analysis = analyze_paths(
        [tmp_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        config=config,
    )
    report, _ = render_report(
        analysis.groups_by_path,
        10,
        bundle_sites_by_path=analysis.bundle_sites_by_path,
    )
    assert "Callsite evidence (undocumented bundles):" in report
    assert "undocumented bundle: a, b" not in report
