# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Deadline-summary compatibility owner during WS-5 migration."""

from gabion.analysis.dataflow.engine import dataflow_deadline_summary as _deadline_summary
from gabion.analysis.dataflow.engine.dataflow_deadline_summary import *  # noqa: F401,F403

# Temporary boundary adapter retained for external import compatibility.
_BOUNDARY_ADAPTER_LIFECYCLE: dict[str, object] = {
    "actor": "codex",
    "rationale": "WS-5 hard-cut completed; retain compatibility alias while external importers migrate",
    "scope": "dataflow_deadline_summary_owner.alias_surface",
    "start": "2026-03-05",
    "expiry": "WS-5 compatibility-shim retirement",
    "rollback_condition": "no external consumers require owner path aliases",
    "evidence_links": ["docs/ws5_decomposition_ledger.md"],
}

__all__ = list(getattr(_deadline_summary, "__all__", ()))
