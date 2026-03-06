# gabion:boundary_normalization_module
from __future__ import annotations

"""Facade compatibility module for legacy indexed-dataflow symbols."""

# Temporary boundary adapter retained for external import compatibility.
_BOUNDARY_ADAPTER_LIFECYCLE: dict[str, object] = {
    "actor": "codex",
    "rationale": "WS-5 hard-cut completed; retain facade alias surface while external importers migrate",
    "scope": "dataflow_facade.alias_surface",
    "start": "2026-03-05",
    "expiry": "WS-5 compatibility-shim retirement",
    "rollback_condition": "no external consumers require facade path aliases",
    "evidence_links": ["docs/ws5_decomposition_ledger.md"],
}

from gabion.analysis.dataflow.engine.dataflow_analysis_index import OptionalDecorators, OptionalParseFailures, OptionalProjectRoot, analyze_file
from gabion.analysis.dataflow.engine.dataflow_projection_materialization import CallAmbiguity
from gabion.analysis.dataflow.engine.dataflow_function_index_decision_support import (
    is_decision_surface,
)
from gabion.analysis.dataflow.engine.dataflow_ingested_analysis_support import (
    analyze_ingested_file,
)
from gabion.analysis.dataflow.engine.dataflow_post_phase_analyses import analyze_constant_flow_repo, analyze_deadness_flow_repo, analyze_decision_surfaces_repo, analyze_type_flow_repo_with_evidence, analyze_type_flow_repo_with_map, analyze_unused_arg_flow_repo, analyze_value_encoded_decisions_repo, generate_property_hook_manifest
from gabion.analysis.dataflow.engine.dataflow_contracts import (
    AnalysisResult,
    AuditConfig,
    CallArgs,
    ClassInfo,
    FunctionInfo,
    InvariantProposition,
    ParamUse,
    ReportCarrier,
    SymbolTable,
)
from gabion.analysis.dataflow.engine.dataflow_adapter_contract import (
    parse_adapter_capabilities,
)
from gabion.analysis.dataflow.engine.dataflow_fingerprint_helpers import verify_rewrite_plan, verify_rewrite_plans
from gabion.analysis.aspf.aspf import Alt, Forest, Node, NodeId
from gabion.analysis.core.visitors import ImportVisitor, ParentAnnotator, UseVisitor
from gabion.analysis.foundation.timeout_context import (
    Deadline,
    GasMeter,
    TimeoutExceeded,
    TimeoutTickCarrier,
    build_timeout_context_from_stack,
    check_deadline,
    deadline_clock_scope,
    deadline_loop_iter,
    deadline_scope,
    forest_scope,
    reset_forest,
    set_forest,
)
from gabion.analysis.projection.projection_registry import (
    DEADLINE_OBLIGATIONS_SUMMARY_SPEC,
    LINT_FINDINGS_SPEC,
    REPORT_SECTION_LINES_SPEC,
    WL_REFINEMENT_SPEC,
)
from gabion.analysis.dataflow.io.dataflow_reporting import render_report
from gabion.order_contract import sort_once
