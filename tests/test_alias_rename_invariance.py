from __future__ import annotations

from pathlib import Path
import sys
import textwrap
from gabion.analysis.aspf import Forest


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import analyze_paths, AuditConfig

    return analyze_paths, AuditConfig


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_call_ambiguities::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path,include_all_sites E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_ambiguities,include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces
def test_direct_rename_preserves_bundle_identity(tmp_path: Path) -> None:
    analyze_paths, AuditConfig = _load()
    source = textwrap.dedent(
        """
        def sink(x=None, y=None):
            return x, y

        def f(a, b):
            x = a
            y = b
            sink(x=x, y=x)
            sink(x=y, y=y)
        """
    ).lstrip()
    file_path = tmp_path / "mod.py"
    file_path.write_text(source)
    config = AuditConfig(project_root=tmp_path)
    analysis = analyze_paths(
        forest=Forest(),
        paths=[file_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        config=config,
    )
    bundles = analysis.groups_by_path[file_path].get("f", [])
    assert {"a", "b"} in bundles


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_call_ambiguities::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path,include_all_sites E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_ambiguities,include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces
def test_chained_alias_preserves_bundle_identity(tmp_path: Path) -> None:
    analyze_paths, AuditConfig = _load()
    source = textwrap.dedent(
        """
        def sink(x=None, y=None):
            return x, y

        def f(a, b):
            x = a
            z = x
            y = b
            sink(x=z, y=z)
            sink(x=y, y=y)
        """
    ).lstrip()
    file_path = tmp_path / "mod.py"
    file_path.write_text(source)
    config = AuditConfig(project_root=tmp_path)
    analysis = analyze_paths(
        forest=Forest(),
        paths=[file_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        config=config,
    )
    bundles = analysis.groups_by_path[file_path].get("f", [])
    assert {"a", "b"} in bundles
