from __future__ import annotations

from pathlib import Path
import sys

from gabion.analysis.aspf import Forest


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import (
        AuditConfig,
        ReportCarrier,
        analyze_paths,
        build_refactor_plan,
        build_synthesis_plan,
        render_dot,
        render_protocol_stubs,
        render_refactor_plan,
        render_report,
    )

    return (
        AuditConfig,
        ReportCarrier,
        analyze_paths,
        build_refactor_plan,
        build_synthesis_plan,
        render_dot,
        render_protocol_stubs,
        render_refactor_plan,
        render_report,
    )


def _write_kitchen_sink(root: Path) -> None:
    pkg = root / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        "from .models import Config, UserConfig\n"
        "__all__ = [\"Config\", \"UserConfig\"]\n"
    )
    (pkg / "models.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from typing import Optional\n"
        "from .utils import helper\n"
        "\n"
        "@dataclass\n"
        "class Config:\n"
        "    name: str\n"
        "    count: int = 0\n"
        "\n"
        "@dataclass\n"
        "class UserConfig:\n"
        "    cfg: Config\n"
        "    tag: Optional[str] = None\n"
        "\n"
        "class Service:\n"
        "    def __init__(self, cfg: Config):\n"
        "        self.cfg = cfg\n"
        "\n"
        "    def run(self, a, b, *args, **kwargs):\n"
        "        return helper(a, b, *args, **kwargs)\n"
        "\n"
        "    @classmethod\n"
        "    def build(cls, cfg: Config):\n"
        "        return cls(cfg)\n"
        "\n"
        "    @staticmethod\n"
        "    def ping(x):\n"
        "        return x\n"
        "\n"
        "    @property\n"
        "    def value(self):\n"
        "        return self.cfg.count\n"
    )
    (pkg / "utils.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "def deco(fn):\n"
        "    def wrapper(*args, **kwargs):\n"
        "        return fn(*args, **kwargs)\n"
        "    return wrapper\n"
        "\n"
        "@deco\n"
        "def helper(a, b, *args, **kwargs):\n"
        "    # dataflow-bundle: a, b\n"
        "    c = a\n"
        "    d, e = (b, b)\n"
        "    data = {\"k\": c}\n"
        "    data[\"k\"] = d\n"
        "    return c, d\n"
        "\n"
        "def uses_kwargs(**kwargs):\n"
        "    return helper(**kwargs)\n"
    )
    (pkg / "calls.py").write_text(
        "from __future__ import annotations\n"
        "from .models import Config, Service\n"
        "\n"
        "def run(cfg: Config):\n"
        "    svc = Service.build(cfg)\n"
        "    return svc.run(cfg.name, cfg.count)\n"
    )
    (pkg / "nested.py").write_text(
        "def outer(x):\n"
        "    def inner(y):\n"
        "        return y\n"
        "    return inner(x)\n"
    )
    (pkg / "exports.py").write_text(
        "from .models import Config as Cfg, UserConfig\n"
        "__all__ = [\"Cfg\", \"UserConfig\"]\n"
    )
    (pkg / "star.py").write_text("from .exports import *\n")
    (root / "main.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from pkg.models import Config, UserConfig\n"
        "from pkg.calls import run\n"
        "\n"
        "@dataclass\n"
        "class LocalCfg:\n"
        "    value: int\n"
        "\n"
        "def entry():\n"
        "    cfg = Config(name=\"demo\", count=1)\n"
        "    user = UserConfig(cfg=cfg)\n"
        "    return run(cfg), user\n"
    )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee::by_qual,callee_key,caller,class_index,symbol_table E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_dot::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._build_function_index::ignore_params E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._merge_counts_by_knobs::knob_names E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::merge.py::gabion.synthesis.merge.merge_bundles::min_overlap E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_knob_param_names::strictness E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness
def test_kitchen_sink_analysis_outputs(tmp_path: Path) -> None:
    (
        AuditConfig,
        ReportCarrier,
        analyze_paths,
        build_refactor_plan,
        build_synthesis_plan,
        render_dot,
        render_protocol_stubs,
        render_refactor_plan,
        render_report,
    ) = _load()
    _write_kitchen_sink(tmp_path)
    config = AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
        transparent_decorators={"deco"},
    )
    analysis = analyze_paths(
        [tmp_path],
        forest=Forest(),
        recursive=True,
        type_audit=True,
        type_audit_report=True,
        type_audit_max=5,
        include_constant_smells=True,
        include_unused_arg_smells=True,
        include_bundle_forest=True,
        config=config,
    )
    report, violations = render_report(
        analysis.groups_by_path,
        5,
        report=ReportCarrier.from_analysis_result(analysis),
    )
    assert "Dataflow grammar" in report or "dataflow-grammar" in report
    assert isinstance(violations, list)

    dot = render_dot(analysis.forest)
    assert "digraph" in dot

    plan = build_synthesis_plan(
        analysis.groups_by_path,
        project_root=tmp_path,
        max_tier=2,
        min_bundle_size=1,
        allow_singletons=True,
        config=config,
    )
    stubs = render_protocol_stubs(plan, kind="dataclass")
    assert "class" in stubs

    refactor_plan = build_refactor_plan(
        analysis.groups_by_path,
        [tmp_path],
        config=config,
    )
    refactor_summary = render_refactor_plan(refactor_plan)
    assert "Refactoring plan" in refactor_summary
