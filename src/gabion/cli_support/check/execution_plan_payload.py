from __future__ import annotations

from dataclasses import dataclass

from gabion.json_types import JSONObject


@dataclass(frozen=True)
class ExecutionPlanRequestPayload:
    requested_operations: list[str]
    inputs: JSONObject
    derived_artifacts: list[str]
    obligations: dict[str, list[str]]
    policy_metadata: dict[str, object]

    def to_payload(self) -> JSONObject:
        return {
            "requested_operations": list(self.requested_operations),
            "inputs": dict(self.inputs),
            "derived_artifacts": list(self.derived_artifacts),
            "obligations": {
                "preconditions": list(self.obligations.get("preconditions") or []),
                "postconditions": list(self.obligations.get("postconditions") or []),
            },
            "policy_metadata": dict(self.policy_metadata),
        }
