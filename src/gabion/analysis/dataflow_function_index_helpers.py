# gabion:boundary_normalization_module
from __future__ import annotations

"""Function-index helper boundary during runtime retirement."""


from pathlib import Path

from gabion.analysis.dataflow_contracts import FunctionInfo
from gabion.analysis.dataflow_indexed_file_scan import (
    _build_function_index as _build_function_index_runtime_adapter,
)
from gabion.analysis.json_types import JSONObject


def _build_function_index(
    paths: list[Path],
    project_root,
    ignore_params: set[str],
    strictness: str,
    transparent_decorators = None,
    *,
    parse_failure_witnesses: list[JSONObject],
) -> tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]]:
    return _build_function_index_runtime_adapter(
        paths,
        project_root,
        ignore_params,
        strictness,
        transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
    )

__all__ = ["_build_function_index"]
