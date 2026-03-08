from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import singledispatch
from pathlib import Path
from typing import Any, Mapping

from gabion.invariants import never
from gabion.json_types import JSONObject


@dataclass(frozen=True)
class ExecutionPlanObligations:
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionPlanPolicyMetadata:
    deadline: dict[str, int] = field(default_factory=dict)
    baseline_mode: str = "none"
    docflow_mode: str = "disabled"


@dataclass(frozen=True)
class ExecutionPlan:
    requested_operations: list[str]
    inputs: dict[str, Any]
    derived_artifacts: list[str] = field(default_factory=list)
    obligations: ExecutionPlanObligations = field(default_factory=ExecutionPlanObligations)
    policy_metadata: ExecutionPlanPolicyMetadata = field(
        default_factory=ExecutionPlanPolicyMetadata
    )

    def as_json_dict(self) -> JSONObject:
        return {
            "requested_operations": [
                str(operation) for operation in self.requested_operations
            ],
            "inputs": _to_plain_json_value(self.inputs),
            "derived_artifacts": [str(path) for path in self.derived_artifacts],
            "obligations": {
                "preconditions": [
                    str(condition) for condition in self.obligations.preconditions
                ],
                "postconditions": [
                    str(condition) for condition in self.obligations.postconditions
                ],
            },
            "policy_metadata": {
                "deadline": {
                    str(key): int(value)
                    for key, value in self.policy_metadata.deadline.items()
                },
                "baseline_mode": str(self.policy_metadata.baseline_mode),
                "docflow_mode": str(self.policy_metadata.docflow_mode),
            },
        }


# gabion:decision_protocol
@singledispatch
def _to_plain_json_value(value: object) -> object:
    never("unregistered runtime type", value_type=type(value).__name__)


@_to_plain_json_value.register(dict)
def _(value: dict[object, object]) -> object:
    return {
        str(key): _to_plain_json_value(item)
        for key, item in value.items()
    }


@_to_plain_json_value.register(list)
def _(value: list[object]) -> object:
    return [_to_plain_json_value(item) for item in value]


@_to_plain_json_value.register(tuple)
def _(value: tuple[object, ...]) -> object:
    return [_to_plain_json_value(item) for item in value]


@_to_plain_json_value.register(set)
def _(value: set[object]) -> object:
    return [
        _to_plain_json_value(item)
        for item in sorted(value, key=repr)
    ]


@_to_plain_json_value.register(Path)
def _(value: Path) -> object:
    return str(value)


@_to_plain_json_value.register(type(None))
@_to_plain_json_value.register(str)
@_to_plain_json_value.register(int)
@_to_plain_json_value.register(float)
@_to_plain_json_value.register(bool)
def _(value: object) -> object:
    return value


def write_execution_plan_artifact(
    plan: ExecutionPlan,
    *,
    root: Path,
    rel_path: Path = Path("artifacts/out/execution_plan.json"),
) -> Path:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan.as_json_dict(), indent=2, sort_keys=False) + "\n")
    return target
