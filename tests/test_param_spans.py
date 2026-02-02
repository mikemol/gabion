from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import analyze_paths, AuditConfig

    return analyze_paths, AuditConfig


def test_param_spans_capture_signature_positions(tmp_path: Path) -> None:
    analyze_paths, AuditConfig = _load()
    source = textwrap.dedent(
        """
        def f(a, b):
            return a + b
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
    spans = analysis.param_spans_by_path[file_path]["f"]
    assert "a" in spans
    assert "b" in spans
    line = source.splitlines()[0]
    a_col = line.index("a")
    b_col = line.index("b")
    a_span = spans["a"]
    b_span = spans["b"]
    assert a_span[0] == 0
    assert b_span[0] == 0
    assert a_span[1] == a_col
    assert b_span[1] == b_col
    assert a_span[3] > a_span[1]
    assert b_span[3] > b_span[1]
