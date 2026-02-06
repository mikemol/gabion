from __future__ import annotations

from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import (
        AuditConfig,
        InvariantProposition,
        analyze_paths,
    )

    return AuditConfig, InvariantProposition, analyze_paths


def test_invariant_emitters_are_applied(tmp_path: Path) -> None:
    AuditConfig, InvariantProposition, analyze_paths = _load()
    source = textwrap.dedent(
        """
        def f(a, b):
            return a + b
        """
    ).lstrip()
    path = tmp_path / "mod.py"
    path.write_text(source)

    def _emit(node):
        if node.name != "f":
            return []
        return [
            InvariantProposition(
                form="Equal",
                terms=("a", "b"),
                scope=None,
                source=None,
            )
        ]

    config = AuditConfig(project_root=tmp_path, invariant_emitters=(_emit,))
    analysis = analyze_paths(
        [path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_invariant_propositions=True,
        config=config,
    )
    assert any(
        prop.form == "Equal"
        and prop.terms == ("a", "b")
        and prop.source == "emitter"
        for prop in analysis.invariant_propositions
    )
