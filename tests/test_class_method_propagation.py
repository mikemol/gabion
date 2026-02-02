from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import analyze_paths, AuditConfig

    return analyze_paths, AuditConfig


def test_method_call_propagates_bundle(tmp_path: Path) -> None:
    analyze_paths, AuditConfig = _load()
    source = textwrap.dedent(
        """
        def sink(x=None, y=None):
            return x, y

        class A:
            def g(self, a, b):
                sink(x=a, y=a)
                sink(x=b, y=b)

            def f(self, a, b):
                return self.g(a, b)
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
    assert "A.g" in groups
    assert "A.f" in groups
    assert {"a", "b"} in groups["A.g"]
    assert {"a", "b"} in groups["A.f"]


def test_inherited_method_call_propagates_bundle(tmp_path: Path) -> None:
    analyze_paths, AuditConfig = _load()
    source = textwrap.dedent(
        """
        def sink(x=None, y=None):
            return x, y

        class Base:
            def g(self, a, b):
                sink(x=a, y=a)
                sink(x=b, y=b)

        class Child(Base):
            def f(self, a, b):
                return self.g(a, b)
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
    assert "Base.g" in groups
    assert "Child.f" in groups
    assert {"a", "b"} in groups["Base.g"]
    assert {"a", "b"} in groups["Child.f"]
