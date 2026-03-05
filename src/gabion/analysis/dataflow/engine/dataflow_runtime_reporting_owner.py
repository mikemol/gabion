# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Runtime-reporting compatibility owner during WS-5 migration."""

from gabion.analysis.dataflow.engine.dataflow_runtime_reporting import (
    ReportProjectionPhase,
    ReportProjectionSpec,
    _compute_violations,
    _report_section_spec,
)

__all__ = [
    "ReportProjectionPhase",
    "ReportProjectionSpec",
    "_compute_violations",
    "_report_section_spec",
]
