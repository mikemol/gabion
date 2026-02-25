from __future__ import annotations

from pathlib import Path
import textwrap
from gabion.analysis.aspf import Forest

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis.dataflow_audit import AuditConfig, analyze_paths, build_refactor_plan
    from gabion.refactor.engine import RefactorEngine
    from gabion.refactor.model import FieldSpec, RefactorRequest

    return AuditConfig, analyze_paths, build_refactor_plan, RefactorEngine, FieldSpec, RefactorRequest

def _apply_plan(plan) -> None:
    for edit in plan.edits:
        Path(edit.path).write_text(edit.replacement)

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::stale_7b7c10bc8b40
def test_refactor_idempotency(tmp_path: Path) -> None:
    (
        AuditConfig,
        analyze_paths,
        build_refactor_plan,
        RefactorEngine,
        FieldSpec,
        RefactorRequest,
    ) = _load()
    source = textwrap.dedent(
        """
        def sink(x=None, y=None):
            return x, y

        def g(a, b):
            sink(x=a, y=a)
            sink(x=b, y=b)

        def f(a, b):
            return g(a, b)
        """
    )
    file_path = tmp_path / "mod.py"
    file_path.write_text(source.strip() + "\n")
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
    plan = build_refactor_plan(analysis.groups_by_path, [file_path], config=config)
    bundles = plan.get("bundles", [])
    assert bundles
    bundle = bundles[0].get("bundle", [])
    assert bundle

    request = RefactorRequest(
        protocol_name="BundleProtocol",
        bundle=list(bundle),
        fields=[FieldSpec(name=name, type_hint="int") for name in bundle],
        target_path=str(file_path),
        target_functions=["g"],
        rationale="Idempotency test",
    )
    refactor_plan = RefactorEngine(project_root=tmp_path).plan_protocol_extraction(request)
    assert refactor_plan.edits
    _apply_plan(refactor_plan)

    analysis_after = analyze_paths(
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
    plan_after = build_refactor_plan(analysis_after.groups_by_path, [file_path], config=config)
    assert not plan_after.get("bundles", [])
