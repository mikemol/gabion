# gabion:grade_boundary kind=semantic_carrier_adapter name=projection_exec_ingress
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Final

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.projection.projection_exec import apply_execution_ops
from gabion.analysis.projection.projection_exec_plan import execution_ops_from_spec
from gabion.analysis.projection.projection_spec import ProjectionSpec
from gabion.json_types import JSONValue

PredicateRegistry = Mapping[
    str,
    Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool],
]
_EMPTY_PREDICATE_REGISTRY: Final[PredicateRegistry] = {}
_EMPTY_PARAMS_OVERRIDE: Final[Mapping[str, JSONValue]] = {}

BOUNDARY_ADAPTER_METADATA: Final[dict[str, object]] = {
    "actor": "codex",
    "rationale": (
        "Keep ProjectionSpec normalization, semantic-op erasure, and "
        "presentation shaping at execution ingress while projection_exec "
        "executes typed legacy runtime steps only."
    ),
    "scope": "projection_exec.ingress_projection_spec_normalization",
    "start": "2026-03-12",
    "expiry": "projection_exec semantic_carrier_adapter retirement",
    "rollback_condition": (
        "ProjectionSpec callers no longer require a row-runtime compatibility "
        "adapter."
    ),
    "evidence_links": [
        "src/gabion/analysis/projection/projection_exec.py",
        "docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc",
        "docs/audits/projection_semantic_fragment_ledger.md#projection_semantic_fragment_ledger",
    ],
}


def apply_spec(
    spec: ProjectionSpec,
    rows: Iterable[Mapping[str, JSONValue]],
    *,
    op_registry: PredicateRegistry = _EMPTY_PREDICATE_REGISTRY,
    params_override: Mapping[str, JSONValue] = _EMPTY_PARAMS_OVERRIDE,
) -> list[dict[str, JSONValue]]:
    check_deadline()
    runtime_params = _copy_json_mapping(spec.params)
    if params_override:
        runtime_params.update(
            {
                str(key): value
                for key, value in params_override.items()
            }
        )
    return apply_execution_ops(
        execution_ops_from_spec(spec),
        rows,
        op_registry=op_registry,
        runtime_params=runtime_params,
    )


def _copy_json_mapping(params: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    return {str(key): value for key, value in params.items()}
