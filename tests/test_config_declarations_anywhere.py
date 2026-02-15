from __future__ import annotations

from pathlib import Path
import sys
import textwrap

from gabion.analysis.aspf import Forest


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip())
    return path


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import (
        AuditConfig,
        ReportCarrier,
        analyze_paths,
        compute_violations,
    )

    return analyze_paths, compute_violations, AuditConfig, ReportCarrier


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness
def test_config_dataclass_declares_bundle_outside_config_py(tmp_path: Path) -> None:
    analyze_paths, compute_violations, AuditConfig, ReportCarrier = _load()
    _write(
        tmp_path,
        "settings.py",
        """
        from dataclasses import dataclass

        @dataclass
        class AppConfig:
            a: int
            b: int
        """,
    )
    mod_path = _write(
        tmp_path,
        "mod.py",
        """
        def sink(x=None, y=None):
            return x, y

        def g(a, b):
            sink(x=a, y=a)
            sink(x=b, y=b)

        def h(c, d):
            sink(x=c, y=c)
            sink(x=d, y=d)
        """,
    )
    config = AuditConfig(project_root=tmp_path)
    analysis = analyze_paths(
        [tmp_path / "settings.py", mod_path],
        forest=Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_bundle_forest=True,
        config=config,
    )
    violations = compute_violations(
        analysis.groups_by_path,
        max_components=10,
        report=ReportCarrier(forest=analysis.forest),
    )
    joined = "\n".join(violations)
    assert "a, b" not in joined
    assert "c, d" in joined
