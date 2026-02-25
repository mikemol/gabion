from __future__ import annotations

from pathlib import Path
import textwrap
from gabion.analysis.aspf import Forest

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis.dataflow_audit import (
        AuditConfig,
        InvariantProposition,
        analyze_paths,
    )

    return AuditConfig, InvariantProposition, analyze_paths

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::stale_95308b76af50
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
        forest=Forest(),
        paths=[path],
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
