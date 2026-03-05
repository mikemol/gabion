# gabion:boundary_normalization_module
from __future__ import annotations

"""Deadline-runtime compatibility owner during WS-5 migration."""

from gabion.analysis.dataflow.engine import dataflow_deadline_runtime as _deadline_runtime
from gabion.analysis.dataflow.engine.dataflow_deadline_runtime import *  # noqa: F401,F403

# Temporary boundary adapter retained for external import compatibility.
_BOUNDARY_ADAPTER_LIFECYCLE: dict[str, object] = {
    "actor": "codex",
    "rationale": "WS-5 hard-cut completed; retain compatibility alias while external importers migrate",
    "scope": "dataflow_deadline_runtime_owner.alias_surface",
    "start": "2026-03-05",
    "expiry": "WS-5 compatibility-shim retirement",
    "rollback_condition": "no external consumers require owner path aliases",
    "evidence_links": ["docs/ws5_decomposition_ledger.md"],
}

__all__ = list(getattr(_deadline_runtime, "__all__", ()))
