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
    from gabion.analysis.dataflow_audit import analyze_paths, compute_violations, AuditConfig

    return analyze_paths, compute_violations, AuditConfig


def test_config_dataclass_declares_bundle_outside_config_py(tmp_path: Path) -> None:
    analyze_paths, compute_violations, AuditConfig = _load()
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
    mod_path = _write(
        tmp_path,
        "mod.py",
        """
        def sink(x=None, y=None):
            return x, y

        def g(a, b):
            sink(x=a, y=a)
            sink(x=b, y=b)

        def h(c, d):
            sink(x=c, y=c)
            sink(x=d, y=d)
        """,
    )
    config = AuditConfig(project_root=tmp_path)
    analysis = analyze_paths(
        [tmp_path / "settings.py", mod_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_bundle_forest=True,
        config=config,
    )
    violations = compute_violations(
        analysis.groups_by_path,
        max_components=10,
        forest=analysis.forest,
    )
    joined = "\n".join(violations)
    assert "a, b" not in joined
    assert "c, d" in joined
