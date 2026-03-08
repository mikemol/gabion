from __future__ import annotations


from typing import Final

from gabion_governance import governance_audit_impl as _impl
from gabion_governance.consolidation_command import run_consolidation_cli
from gabion_governance.decision_tiers_command import run_decision_tiers_cli
from gabion_governance.docflow_command import run_docflow_cli
from gabion_governance.lint_summary_command import run_lint_summary_cli
from gabion_governance.sppf_graph_command import run_sppf_graph_cli
from gabion_governance.status_consistency_command import run_status_consistency_cli

# Temporary boundary adapter metadata for legacy direct module access.
# This adapter exists only to provide a stable import surface while core
# implementation remains in gabion_governance.governance_audit_impl.
BOUNDARY_ADAPTER_METADATA: Final[dict[str, object]] = {
    "actor": "codex",
    "rationale": "Preserve stable governance-audit API boundary at gabion.tooling.governance.",
    "scope": "module import surface gabion.tooling.governance.governance_audit",
    "start": "2026-03-04",
    "expiry": "2026-09-01",
    "rollback_condition": "All in-repo and external imports target gabion_governance.governance_audit_impl directly or a newly ratified canonical package path.",
    "evidence_links": [
        "tests/gabion/tooling/governance/test_governance_audit_adapter.py",
        "tests/gabion/tooling/docflow/test_docflow_violation_formatter.py",
    ],
}

CORE_GOVERNANCE_DOCS = _impl.CORE_GOVERNANCE_DOCS
GOVERNANCE_DOCS = _impl.GOVERNANCE_DOCS
DOCFLOW_AUDIT_INVARIANTS = _impl.DOCFLOW_AUDIT_INVARIANTS
NORMATIVE_LOOP_DOMAINS = _impl.NORMATIVE_LOOP_DOMAINS
STATUS_TRIPLET_OVERRIDE_MARKER = _impl.STATUS_TRIPLET_OVERRIDE_MARKER
_DEFAULT_AUDIT_GAS_LIMIT = _impl._DEFAULT_AUDIT_GAS_LIMIT

Doc = _impl.Doc
DocflowObligationResult = _impl.DocflowObligationResult
DocflowInvariant = _impl.DocflowInvariant

_audit_deadline_scope = _impl._audit_deadline_scope
_parse_frontmatter = _impl._parse_frontmatter
_agent_instruction_graph = _impl._agent_instruction_graph
_docflow_audit_context = _impl._docflow_audit_context
_load_docflow_docs = _impl._load_docflow_docs
_docflow_invariant_rows = _impl._docflow_invariant_rows
_evaluate_docflow_invariants = _impl._evaluate_docflow_invariants
_docflow_compliance_rows = _impl._docflow_compliance_rows
_sppf_sync_check = _impl._sppf_sync_check
_sppf_status_triplet_violations = _impl._sppf_status_triplet_violations
_emit_docflow_compliance = _impl._emit_docflow_compliance
_format_docflow_violation = _impl._format_docflow_violation
_make_invariant_spec = _impl._make_invariant_spec
_audit_gas_limit = _impl._audit_gas_limit
_extract_doc_body_notions_with_anchors = _impl._extract_doc_body_notions_with_anchors
_build_doc_implication_lattice = _impl._build_doc_implication_lattice
_compose_doc_dependency_matrices = _impl._compose_doc_dependency_matrices
_emit_docflow_implication_matrices = _impl._emit_docflow_implication_matrices

spec_from_dict = _impl.spec_from_dict

# Preserve canonical CLI command behavior at the governance adapter boundary.
run_docflow_cli = _impl.run_docflow_cli
run_sppf_graph_cli = _impl.run_sppf_graph_cli

__all__ = [
    "BOUNDARY_ADAPTER_METADATA",
    "CORE_GOVERNANCE_DOCS",
    "GOVERNANCE_DOCS",
    "DOCFLOW_AUDIT_INVARIANTS",
    "NORMATIVE_LOOP_DOMAINS",
    "STATUS_TRIPLET_OVERRIDE_MARKER",
    "_DEFAULT_AUDIT_GAS_LIMIT",
    "Doc",
    "DocflowObligationResult",
    "DocflowInvariant",
    "_agent_instruction_graph",
    "_audit_deadline_scope",
    "_audit_gas_limit",
    "_docflow_audit_context",
    "_docflow_compliance_rows",
    "_docflow_invariant_rows",
    "_emit_docflow_compliance",
    "_load_docflow_docs",
    "_evaluate_docflow_invariants",
    "_format_docflow_violation",
    "_make_invariant_spec",
    "_extract_doc_body_notions_with_anchors",
    "_build_doc_implication_lattice",
    "_compose_doc_dependency_matrices",
    "_emit_docflow_implication_matrices",
    "_parse_frontmatter",
    "_sppf_status_triplet_violations",
    "_sppf_sync_check",
    "run_consolidation_cli",
    "run_decision_tiers_cli",
    "run_docflow_cli",
    "run_lint_summary_cli",
    "run_sppf_graph_cli",
    "run_status_consistency_cli",
    "spec_from_dict",
]
