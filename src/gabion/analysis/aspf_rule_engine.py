# gabion:ambiguity_boundary_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from gabion.policy_dsl import PolicyDecision, PolicyDomain, evaluate_policy


@dataclass(frozen=True)
class AspfOpportunityInput:
    two_cell: bool
    cofibration: bool

    def to_policy_payload(self) -> dict[str, object]:
        return {
            "witness": {
                "two_cell": self.two_cell,
                "cofibration": self.cofibration,
            }
        }


def classify_aspf_opportunity(payload: Mapping[str, object]) -> PolicyDecision:
    witness = payload.get("witness")
    two_cell = bool(witness.get("two_cell")) if isinstance(witness, Mapping) else False
    cofibration = bool(witness.get("cofibration")) if isinstance(witness, Mapping) else False
    projected = AspfOpportunityInput(two_cell=two_cell, cofibration=cofibration)
    return evaluate_policy(domain=PolicyDomain.ASPF_OPPORTUNITY, data=projected.to_policy_payload())
