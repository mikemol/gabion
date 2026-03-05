# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Canonical owner for legacy parse-failure runtime helpers."""

from gabion.analysis.dataflow.io.dataflow_parse_helpers import (
    _parse_module_tree_or_none as _parse_module_tree_runtime,
)


__all__ = ["_parse_module_tree_runtime"]
