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
