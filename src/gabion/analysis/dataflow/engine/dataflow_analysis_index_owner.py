# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Analysis-index compatibility owner during WS-5 migration."""

from gabion.analysis.dataflow.engine import dataflow_analysis_index as _analysis_index
from gabion.analysis.dataflow.engine.dataflow_analysis_index import *  # noqa: F401,F403

__all__ = list(getattr(_analysis_index, "__all__", ()))
