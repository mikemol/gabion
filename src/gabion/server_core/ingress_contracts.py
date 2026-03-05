from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gabion.server_core.ingress_primitives import ExecuteCommandDeps
from gabion.server_core import command_orchestrator_primitives as legacy


@dataclass(frozen=True)
class IngressStageDeps:
    normalize_dataflow_response_fn: Callable[[dict[str, object]], dict[str, object]]
    materialize_execution_plan_fn: Callable[[dict[str, object]], object]
    default_execute_command_deps_fn: Callable[[], ExecuteCommandDeps]


def default_ingress_stage_deps() -> IngressStageDeps:
    return IngressStageDeps(
        normalize_dataflow_response_fn=legacy._normalize_dataflow_response,
        materialize_execution_plan_fn=legacy._materialize_execution_plan,
        default_execute_command_deps_fn=legacy._default_execute_command_deps,
    )


__all__ = ["IngressStageDeps", "default_ingress_stage_deps"]
