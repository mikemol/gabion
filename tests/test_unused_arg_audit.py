from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    return path


def _load_analyzer():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import analyze_unused_arg_flow_repo

    return analyze_unused_arg_flow_repo


def test_unused_arg_smell_detected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "sample.py",
        """
        def callee(x, y):
            return y

        def caller(foo):
            return callee(foo, 1)
        """,
    )
    analyze_unused_arg_flow_repo = _load_analyzer()
    smells = analyze_unused_arg_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert (
        "sample.py:caller passes param foo to unused sample.py:callee.x"
        in smells
    )


def test_unused_arg_ignores_constants(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "sample.py",
        """
        def callee(x, y):
            return y

        def caller(foo):
            return callee(1, foo)
        """,
    )
    analyze_unused_arg_flow_repo = _load_analyzer()
    smells = analyze_unused_arg_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert not smells


def test_unused_arg_skips_test_calls(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "tests/test_sample.py",
        """
        def callee(x, y):
            return y

        def caller(foo):
            return callee(foo, 1)
        """,
    )
    analyze_unused_arg_flow_repo = _load_analyzer()
    smells = analyze_unused_arg_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert not smells


def test_unused_arg_non_const_pos_and_kw(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "sample.py",
        """
        def callee(x, y, z):
            return y

        def caller(a, b):
            return callee(a + 1, y=1, z=b + 2)
        """,
    )
    analyze_unused_arg_flow_repo = _load_analyzer()
    smells = analyze_unused_arg_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert any("non-constant arg at position 0" in smell for smell in smells)
    assert any("non-constant kw 'z'" in smell for smell in smells)


def test_unused_arg_star_kw_low_strictness(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "sample.py",
        """
        def callee(x, y):
            return y

        def caller(**kwargs):
            return callee(**kwargs)
        """,
    )
    analyze_unused_arg_flow_repo = _load_analyzer()
    smells = analyze_unused_arg_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="low",
        external_filter=True,
    )
    assert any("non-constant kw 'x'" in smell for smell in smells)


def test_unused_arg_ignores_extra_pos_args(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "sample.py",
        """
        def callee(x):
            return x

        def caller(a, b):
            return callee(a, b)
        """,
    )
    analyze_unused_arg_flow_repo = _load_analyzer()
    smells = analyze_unused_arg_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert smells == []


def test_unused_arg_edge_branches(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "sample.py",
        """
        def callee(x, y):
            return y

        def caller(a, b, c):
            callee(a, b, c)
            callee(a, b, c + 1)
            callee(x=a, y=b)
            callee(extra=a, x=b)
            callee(x=a, extra=b + 1)
        """,
    )
    analyze_unused_arg_flow_repo = _load_analyzer()
    smells = analyze_unused_arg_flow_repo(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
    )
    assert any("callee.x" in smell for smell in smells)
