from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gabion.schema import DataflowResponseEnvelopeDTO
from gabion.server_core.ingress_primitives import ExecuteCommandDeps
from gabion.server_core.primitive_contract_registry import ingress_stage_dependency_defaults


@dataclass(frozen=True)
class IngressStageDeps:
    normalize_dataflow_response_fn: Callable[[dict[str, object]], DataflowResponseEnvelopeDTO]
    materialize_execution_plan_fn: Callable[[dict[str, object]], object]
    default_execute_command_deps_fn: Callable[[], ExecuteCommandDeps]


def default_ingress_stage_deps() -> IngressStageDeps:
    defaults = ingress_stage_dependency_defaults()
    return IngressStageDeps(
        normalize_dataflow_response_fn=defaults["normalize_dataflow_response_fn"],
        materialize_execution_plan_fn=defaults["materialize_execution_plan_fn"],
        default_execute_command_deps_fn=defaults["default_execute_command_deps_fn"],
    )


__all__ = ["IngressStageDeps", "default_ingress_stage_deps"]
