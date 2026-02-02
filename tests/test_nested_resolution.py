from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import analyze_paths, AuditConfig

    return analyze_paths, AuditConfig


def test_nested_function_resolution(tmp_path: Path) -> None:
    analyze_paths, AuditConfig = _load()
    source = textwrap.dedent(
        """
        def sink(x=None, y=None):
            return x, y

        def outer(a, b):
            def inner(u, v):
                sink(x=u, y=u)
                sink(x=v, y=v)
            return inner(a, b)
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
    groups = analysis.groups_by_path[file_path]
    assert "outer.inner" in groups
    assert "outer" in groups
    assert {"u", "v"} in groups["outer.inner"]
    assert {"a", "b"} in groups["outer"]
