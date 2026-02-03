from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import AuditConfig, analyze_paths

    return AuditConfig, analyze_paths


def test_type_flow_suggestions_and_ambiguities(tmp_path: Path) -> None:
    AuditConfig, analyze_paths = _load()
    code = (
        "def callee_int(x: int):\n"
        "    return x\n"
        "\n"
        "def callee_str(x: str):\n"
        "    return x\n"
        "\n"
        "def caller(a):\n"
        "    return callee_int(a)\n"
        "\n"
        "def caller_conflict(b):\n"
        "    callee_int(b)\n"
        "    callee_str(b)\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(code)
    config = AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
        transparent_decorators=None,
    )
    analysis = analyze_paths(
        [path],
        recursive=False,
        type_audit=True,
        type_audit_report=True,
        type_audit_max=10,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        config=config,
    )
    assert any("caller.a" in entry for entry in analysis.type_suggestions)
    assert any("caller_conflict.b" in entry for entry in analysis.type_ambiguities)
