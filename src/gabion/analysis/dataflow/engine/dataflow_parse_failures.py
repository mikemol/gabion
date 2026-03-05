# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from pathlib import Path

from gabion.analysis.foundation.json_types import JSONObject

_PARSE_MODULE_ERROR_TYPES = (
    OSError,
    UnicodeError,
    SyntaxError,
    ValueError,
    TypeError,
    MemoryError,
    RecursionError,
)


def _parse_failure_witness(
    *,
    path: Path,
    stage,
    error: Exception,
) -> JSONObject:
    stage_value = stage.value if hasattr(stage, "value") else stage
    return {
        "path": str(path),
        "stage": stage_value,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _record_parse_failure_witness(
    *,
    sink: list[JSONObject],
    path: Path,
    stage,
    error: Exception,
) -> None:
    sink.append(_parse_failure_witness(path=path, stage=stage, error=error))


def _parse_failure_sink(
    parse_failure_witnesses,
) -> list[JSONObject]:
    sink = parse_failure_witnesses
    if sink is None:
        sink = []
    return sink


__all__ = [
    "_PARSE_MODULE_ERROR_TYPES",
    "_parse_failure_sink",
    "_parse_failure_witness",
    "_record_parse_failure_witness",
]
