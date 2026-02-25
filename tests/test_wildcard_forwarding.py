from __future__ import annotations

from pathlib import Path
import textwrap

def _write(tmp_path: Path, rel: str, content: str) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    return path

def _load_analyzer():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis.dataflow_audit import analyze_unused_arg_flow_repo

    return analyze_unused_arg_flow_repo

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo._format::call E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::stale_49a04d196497
def test_unused_arg_flow_maps_star_args_low_strictness(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "sample.py",
        """
        def callee(a, b):
            return a

        def caller(*args):
            return callee(*args)
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
    assert any("non-constant arg at position 1" in smell for smell in smells)
