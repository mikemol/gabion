# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Deadline-summary compatibility owner during WS-5 migration."""

from gabion.analysis.dataflow.engine.dataflow_deadline_summary import (
    _summarize_deadline_obligations,
)

__all__ = ["_summarize_deadline_obligations"]
