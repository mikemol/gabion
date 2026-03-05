# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Canonical owner for legacy parse-failure runtime helpers."""

import ast
from pathlib import Path

from gabion.analysis.dataflow.engine.dataflow_parse_failures import (
    _PARSE_MODULE_ERROR_TYPES,
    _record_parse_failure_witness,
)
from gabion.analysis.foundation.json_types import JSONObject


def _parse_module_tree_runtime(
    path: Path,
    *,
    stage,
    parse_failure_witnesses: list[JSONObject],
):
    try:
        return ast.parse(path.read_text())
    except _PARSE_MODULE_ERROR_TYPES as exc:
        _record_parse_failure_witness(
            sink=parse_failure_witnesses,
            path=path,
            stage=stage,
            error=exc,
        )
        return None


__all__ = ["_parse_module_tree_runtime"]
