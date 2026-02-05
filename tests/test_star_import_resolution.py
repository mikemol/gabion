from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    return path


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import analyze_unused_arg_flow_repo

    return analyze_unused_arg_flow_repo


def test_star_import_resolution_disambiguates(tmp_path: Path) -> None:
    analyze_unused_arg_flow_repo = _load()
    _write(
        tmp_path,
        "a.py",
        """
        def helper(a, b):
            return b
        """,
    )
    _write(
        tmp_path,
        "c.py",
        """
        def helper(a, b):
            return a + b
        """,
    )
    b_path = _write(
        tmp_path,
        "b.py",
        """
        from a import *

        def caller(x, y):
            return helper(x, y)
        """,
    )
    smells = analyze_unused_arg_flow_repo(
        [tmp_path / "a.py", tmp_path / "b.py", tmp_path / "c.py"],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert any(
        smell.startswith("b.py:") and "passes param x to unused a.py:helper.a" in smell
        for smell in smells
    )


def test_star_import_resolves_reexport(tmp_path: Path) -> None:
    analyze_unused_arg_flow_repo = _load()
    _write(
        tmp_path,
        "a.py",
        """
        def helper(a, b):
            return b
        """,
    )
    _write(
        tmp_path,
        "b.py",
        """
        from a import helper
        __all__ = ["helper"]
        """,
    )
    _write(
        tmp_path,
        "c.py",
        """
        from b import *

        def caller(x, y):
            return helper(x, y)
        """,
    )
    smells = analyze_unused_arg_flow_repo(
        [tmp_path / "a.py", tmp_path / "b.py", tmp_path / "c.py"],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert any(
        smell.startswith("c.py:") and "passes param x to unused a.py:helper.a" in smell
        for smell in smells
    )
