from __future__ import annotations

from pathlib import Path
from tests.path_helpers import REPO_ROOT
from gabion.analysis.aspf.aspf import Forest

def _load():
    repo_root = REPO_ROOT
    from gabion.analysis import AuditConfig, analyze_paths

    return AuditConfig, analyze_paths

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._iter_paths::config E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_never_invariants::forest E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.analyze_unused_arg_flow_repo::strictness E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._analyze_file_internal::stale_9c40c0905534
# gabion:behavior primary=desired
def test_type_flow_suggestions_and_ambiguities(tmp_path: Path) -> None:
    AuditConfig, analyze_paths = _load()
    code = (
        "def callee_int(x: int):\n"
        "    return x\n"
        "\n"
        "def callee_str(x: str):\n"
        "    return x\n"
        "\n"
        "def caller(a):\n"
        "    return callee_int(a)\n"
        "\n"
        "def caller_conflict(b):\n"
        "    callee_int(b)\n"
        "    callee_str(b)\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(code)
    config = AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
        transparent_decorators=None,
    )
    analysis = analyze_paths(
        forest=Forest(),
        paths=[path],
        recursive=False,
        type_audit=True,
        type_audit_report=True,
        type_audit_max=10,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        config=config,
    )
    assert any("caller.a" in entry for entry in analysis.type_suggestions)
    assert any("caller_conflict.b" in entry for entry in analysis.type_ambiguities)
