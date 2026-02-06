from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import analyze_paths, AuditConfig

    return analyze_paths, AuditConfig


# gabion:evidence E:bundle/alias_invariance
def test_direct_rename_preserves_bundle_identity(tmp_path: Path) -> None:
    analyze_paths, AuditConfig = _load()
    source = textwrap.dedent(
        """
        def sink(x=None, y=None):
            return x, y

        def f(a, b):
            x = a
            y = b
            sink(x=x, y=x)
            sink(x=y, y=y)
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
    bundles = analysis.groups_by_path[file_path].get("f", [])
    assert {"a", "b"} in bundles


# gabion:evidence E:bundle/alias_invariance
def test_chained_alias_preserves_bundle_identity(tmp_path: Path) -> None:
    analyze_paths, AuditConfig = _load()
    source = textwrap.dedent(
        """
        def sink(x=None, y=None):
            return x, y

        def f(a, b):
            x = a
            z = x
            y = b
            sink(x=z, y=z)
            sink(x=y, y=y)
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
    bundles = analysis.groups_by_path[file_path].get("f", [])
    assert {"a", "b"} in bundles
