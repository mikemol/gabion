from __future__ import annotations

from pathlib import Path
import textwrap
from gabion.analysis.aspf import Forest

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis.dataflow_audit import analyze_paths, AuditConfig

    return analyze_paths, AuditConfig

def _analyze(tmp_path: Path, source: str):
    analyze_paths, AuditConfig = _load()
    path = tmp_path / "mod.py"
    path.write_text(source)
    analysis = analyze_paths(
        forest=Forest(),
        paths=[path],
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

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report
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

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report
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

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report
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
