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
    from gabion.analysis.dataflow_audit import analyze_paths, AuditConfig

    return analyze_paths, AuditConfig


def test_local_definition_overrides_import(tmp_path: Path) -> None:
    analyze_paths, AuditConfig = _load()
    _write(
        tmp_path,
        "other.py",
        """
        def sink(x=None, y=None):
            return x, y
        """,
    )
    a_path = _write(
        tmp_path,
        "a.py",
        """
        from other import sink

        def target(x=None, y=None):
            return x, y

        def sink(p, q):
            target(x=p, y=p)
            target(x=q, y=q)

        def f(a, b):
            return sink(a, b)
        """,
    )
    config = AuditConfig(project_root=tmp_path)
    analysis = analyze_paths(
        [a_path, tmp_path / "other.py"],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        config=config,
    )
    groups = analysis.groups_by_path[a_path]
    assert "sink" in groups
    assert "f" in groups
    assert {"p", "q"} in groups["sink"]
    assert {"a", "b"} in groups["f"]
