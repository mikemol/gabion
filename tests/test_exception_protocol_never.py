from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_exception_protocol_never_violation(tmp_path: Path) -> None:
    da = _load()
    module_path = tmp_path / "mod.py"
    _write(
        module_path,
        "from gabion.exceptions import NeverRaise\n"
        "\n"
        "def f(a):\n"
        "    raise NeverRaise('nope')\n",
    )
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
        never_exceptions={"NeverRaise"},
    )
    analysis = da.analyze_paths(
        [tmp_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_exception_obligations=True,
        include_lint_lines=True,
        config=config,
    )
    obligations = analysis.exception_obligations
    assert any(entry.get("status") == "FORBIDDEN" for entry in obligations)
    assert any("GABION_EXC_NEVER" in line for line in analysis.lint_lines)
    report, violations = da._emit_report(
        analysis.groups_by_path,
        3,
        exception_obligations=obligations,
    )
    assert "Exception protocol violations" in report
    assert any("protocol=never" in line for line in violations)
