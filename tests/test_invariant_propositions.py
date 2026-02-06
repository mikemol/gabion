from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import analyze_paths, AuditConfig

    return analyze_paths, AuditConfig


def _analyze(tmp_path: Path, source: str):
    analyze_paths, AuditConfig = _load()
    path = tmp_path / "mod.py"
    path.write_text(source)
    analysis = analyze_paths(
        [path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_invariant_propositions=True,
        config=AuditConfig(project_root=tmp_path),
    )
    return analysis


def test_invariant_extracts_len_equality(tmp_path: Path) -> None:
    source = textwrap.dedent(
        """
        def f(a, b):
            assert len(a) == len(b)
        """
    ).lstrip()
    analysis = _analyze(tmp_path, source)
    assert any(
        prop.form == "Equal" and prop.terms == ("a.length", "b.length")
        for prop in analysis.invariant_propositions
    )


def test_invariant_extracts_param_equality(tmp_path: Path) -> None:
    source = textwrap.dedent(
        """
        def f(a, b):
            assert a == b
        """
    ).lstrip()
    analysis = _analyze(tmp_path, source)
    assert any(
        prop.form == "Equal" and prop.terms == ("a", "b")
        for prop in analysis.invariant_propositions
    )


def test_invariant_ignores_non_param_asserts(tmp_path: Path) -> None:
    source = textwrap.dedent(
        """
        def f(a):
            x = [1]
            y = [2]
            assert len(x) == len(y)
        """
    ).lstrip()
    analysis = _analyze(tmp_path, source)
    assert analysis.invariant_propositions == []
