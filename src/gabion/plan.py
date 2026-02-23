from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
        payload = asdict(self)
        return {
            str(key): payload[key] for key in payload
        }


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
