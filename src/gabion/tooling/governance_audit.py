from __future__ import annotations

# gabion:decision_protocol_module
# gabion:boundary_normalization_module

from typing import Any

from gabion_governance import governance_audit_impl as _impl

CORE_GOVERNANCE_DOCS = _impl.CORE_GOVERNANCE_DOCS
DOCFLOW_AUDIT_INVARIANTS = _impl.DOCFLOW_AUDIT_INVARIANTS
NORMATIVE_LOOP_DOMAINS = _impl.NORMATIVE_LOOP_DOMAINS
STATUS_TRIPLET_OVERRIDE_MARKER = _impl.STATUS_TRIPLET_OVERRIDE_MARKER
_DEFAULT_AUDIT_GAS_LIMIT = _impl._DEFAULT_AUDIT_GAS_LIMIT

Doc = _impl.Doc
DocflowObligationResult = _impl.DocflowObligationResult


def _audit_deadline_scope(*args: Any, **kwargs: Any) -> Any:
    return _impl._audit_deadline_scope(*args, **kwargs)


def _parse_frontmatter(*args: Any, **kwargs: Any) -> Any:
    return _impl._parse_frontmatter(*args, **kwargs)


def _agent_instruction_graph(*args: Any, **kwargs: Any) -> Any:
    return _impl._agent_instruction_graph(*args, **kwargs)


def _docflow_invariant_rows(*args: Any, **kwargs: Any) -> Any:
    return _impl._docflow_invariant_rows(*args, **kwargs)


def _evaluate_docflow_invariants(*args: Any, **kwargs: Any) -> Any:
    return _impl._evaluate_docflow_invariants(*args, **kwargs)


def _sppf_sync_check(*args: Any, **kwargs: Any) -> Any:
    return _impl._sppf_sync_check(*args, **kwargs)


def _sppf_status_triplet_violations(*args: Any, **kwargs: Any) -> Any:
    return _impl._sppf_status_triplet_violations(*args, **kwargs)


def _emit_docflow_compliance(*args: Any, **kwargs: Any) -> Any:
    return _impl._emit_docflow_compliance(*args, **kwargs)


def _audit_gas_limit(*args: Any, **kwargs: Any) -> Any:
    return _impl._audit_gas_limit(*args, **kwargs)


def run_docflow_cli(argv: list[str] | None = None) -> int:
    return _impl.run_docflow_cli(argv)


def run_sppf_graph_cli(argv: list[str] | None = None) -> int:
    return _impl.run_sppf_graph_cli(argv)


def run_status_consistency_cli(argv: list[str] | None = None) -> int:
    return _impl.run_status_consistency_cli(argv)


def run_decision_tiers_cli(argv: list[str] | None = None) -> int:
    return _impl.run_decision_tiers_cli(argv)


def run_consolidation_cli(argv: list[str] | None = None) -> int:
    return _impl.run_consolidation_cli(argv)


def run_lint_summary_cli(argv: list[str] | None = None) -> int:
    return _impl.run_lint_summary_cli(argv)


__all__ = [
    "CORE_GOVERNANCE_DOCS",
    "DOCFLOW_AUDIT_INVARIANTS",
    "NORMATIVE_LOOP_DOMAINS",
    "STATUS_TRIPLET_OVERRIDE_MARKER",
    "_DEFAULT_AUDIT_GAS_LIMIT",
    "Doc",
    "DocflowObligationResult",
    "_agent_instruction_graph",
    "_audit_deadline_scope",
    "_audit_gas_limit",
    "_docflow_invariant_rows",
    "_emit_docflow_compliance",
    "_evaluate_docflow_invariants",
    "_parse_frontmatter",
    "_sppf_status_triplet_violations",
    "_sppf_sync_check",
    "run_consolidation_cli",
    "run_decision_tiers_cli",
    "run_docflow_cli",
    "run_lint_summary_cli",
    "run_sppf_graph_cli",
    "run_status_consistency_cli",
]
