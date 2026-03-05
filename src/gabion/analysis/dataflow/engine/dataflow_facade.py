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

from gabion.analysis.dataflow.engine.dataflow_deadline_contracts import *  # noqa: F401,F403
from gabion.analysis.dataflow.engine.dataflow_deadline_helpers import *  # noqa: F401,F403
from gabion.analysis.dataflow.engine.dataflow_analysis_index import *  # noqa: F401,F403
from gabion.analysis.dataflow.engine.dataflow_projection_materialization import *  # noqa: F401,F403
from gabion.analysis.dataflow.engine.dataflow_documented_bundles import (
    _iter_documented_bundles,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_decision_support import *  # noqa: F401,F403
from gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms import (
    _collect_recursive_functions,
    _collect_recursive_nodes,
    _reachable_from_roots,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_helpers import *  # noqa: F401,F403
from gabion.analysis.dataflow.engine.dataflow_ingest_helpers import (
    _collect_functions,
    _iter_paths,
)
from gabion.analysis.dataflow.engine.dataflow_ingested_analysis_support import *  # noqa: F401,F403
from gabion.analysis.dataflow.engine.dataflow_post_phase_analyses import *  # noqa: F401,F403
from gabion.analysis.dataflow.engine.dataflow_deadline_summary import (
    _summarize_deadline_obligations,
)
from gabion.analysis.dataflow.engine.dataflow_runtime_reporting import (
    _report_section_spec,
)
from gabion.analysis.dataflow.engine.dataflow_bundle_merge import (
    _merge_counts_by_knobs,
)
from gabion.analysis.dataflow.engine.dataflow_lambda_runtime_support import (
    _collect_lambda_bindings_by_caller,
    _collect_lambda_function_infos,
    _function_key,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_runtime_support import (
    _direct_lambda_callee_by_call_span,
    _materialize_direct_lambda_callees,
    _unused_params,
)
from gabion.analysis.dataflow.engine.dataflow_function_semantics import *  # noqa: F401,F403
from gabion.analysis.dataflow.engine.dataflow_resume_serialization import *  # noqa: F401,F403
from gabion.analysis.dataflow.engine.dataflow_contracts import *  # noqa: F401,F403
from gabion.analysis.dataflow.engine.dataflow_adapter_contract import (
    parse_adapter_capabilities,
)
from gabion.analysis.dataflow.engine.dataflow_fingerprint_helpers import *  # noqa: F401,F403
from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import *  # noqa: F401,F403
from gabion.analysis.dataflow.engine.dataflow_raw_runtime import (
    _resolve_synth_registry_path,
)

from gabion.analysis.dataflow.engine.dataflow_callee_resolution_support import (
    _callee_key,
    _resolve_method_in_hierarchy,
    _resolve_class_candidates,
)
from gabion.analysis.dataflow.engine.dataflow_local_class_hierarchy import (
    _collect_local_class_bases,
    _resolve_local_method_in_hierarchy,
)
from gabion.analysis.aspf.aspf import Alt, Forest, Node, NodeId
from gabion.analysis.core.visitors import ImportVisitor, ParentAnnotator, UseVisitor
from gabion.analysis.dataflow.engine.dataflow_lint_helpers import *  # noqa: F401,F403
# Preserve canonical owner identity for overlapping wildcard symbols.
from gabion.analysis.dataflow.engine.dataflow_analysis_index import (
    _build_symbol_table,
    _build_function_index,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_helpers import (
    _resolve_callee,
)
from gabion.analysis.dataflow.engine.dataflow_projection_materialization import (
    _lint_lines_from_call_ambiguities,
)
from gabion.analysis.dataflow.engine.dataflow_bundle_merge import (
    _merge_counts_by_knobs,
)
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
from gabion.analysis.indexed_scan.ast.expression_eval import EvalDecision as _EvalDecision
from gabion.analysis.projection.projection_registry import (
    DEADLINE_OBLIGATIONS_SUMMARY_SPEC,
    LINT_FINDINGS_SPEC,
    REPORT_SECTION_LINES_SPEC,
    WL_REFINEMENT_SPEC,
)
from gabion.analysis.dataflow.io.dataflow_projection_helpers import (
    _topologically_order_report_projection_specs,
)
from gabion.analysis.dataflow.io.dataflow_reporting import emit_report as _emit_report
from gabion.analysis.dataflow.io.dataflow_reporting_helpers import (
    render_mermaid_component as _render_mermaid_component,
)
from gabion.analysis.dataflow.io.dataflow_reporting import render_report
from gabion.order_contract import sort_once
