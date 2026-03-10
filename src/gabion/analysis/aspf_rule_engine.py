# gabion:ambiguity_boundary_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from gabion.policy_dsl import PolicyDecision, PolicyDomain, evaluate_policy


_KIND_TO_STRUCTURE = {
    "materialize_load_observed": "one_cell",
    "materialize_load_fusion": "one_cell",
    "fungible_execution_path_substitution": "two_cell",
    "reusable_boundary_artifact": "two_cell",
    "cofibration_prime_embedding_reuse": "cofibration",
}


@dataclass(frozen=True)
class AspfOpportunityObservation:
    structure: str
    one_cell_count: int
    two_cell_count: int
    cofibration_count: int
    surface_count: int
    classification: str
    has_resume_load: bool
    has_resume_write: bool
    has_two_cell_witness: bool
    has_cofibration_witness: bool

    def to_policy_payload(self) -> dict[str, object]:
        return {
            "observation": {
                "structure": self.structure,
                "one_cell_count": self.one_cell_count,
                "two_cell_count": self.two_cell_count,
                "cofibration_count": self.cofibration_count,
                "surface_count": self.surface_count,
                "classification": self.classification,
                "has_resume_load": self.has_resume_load,
                "has_resume_write": self.has_resume_write,
            },
            "witness": {
                "two_cell": self.has_two_cell_witness,
                "cofibration": self.has_cofibration_witness,
            }
        }


def _int_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_observation(payload: Mapping[str, object]) -> AspfOpportunityObservation:
    raw_observation = payload.get("observation")
    observation = raw_observation if isinstance(raw_observation, Mapping) else payload
    raw_witness = payload.get("witness")
    witness = raw_witness if isinstance(raw_witness, Mapping) else {}

    kind = str(observation.get("kind", "")).strip()
    structure = str(observation.get("structure", "")).strip()
    if structure in ("",):
        structure = _KIND_TO_STRUCTURE.get(kind, "")

    one_cell_count = _int_value(observation.get("one_cell_count"))
    two_cell_count = _int_value(observation.get("two_cell_count"))
    cofibration_count = _int_value(observation.get("cofibration_count"))
    surface_count = _int_value(observation.get("surface_count"))
    classification = str(observation.get("classification", "")).strip()
    has_resume_load = bool(observation.get("has_resume_load"))
    has_resume_write = bool(observation.get("has_resume_write"))

    if kind == "materialize_load_observed":
        one_cell_count = max(one_cell_count, 1)
        has_resume_load = True
    if kind == "materialize_load_fusion":
        one_cell_count = max(one_cell_count, 2)
        has_resume_load = True
        has_resume_write = True
    if kind == "fungible_execution_path_substitution":
        two_cell_count = max(two_cell_count, 1)
        if classification in ("",):
            classification = "non_drift"
    if kind == "reusable_boundary_artifact":
        surface_count = max(surface_count, 2)
        one_cell_count = max(one_cell_count, surface_count)
    if kind == "cofibration_prime_embedding_reuse":
        cofibration_count = max(cofibration_count, 1)

    has_two_cell_witness = bool(witness.get("two_cell")) or two_cell_count > 0
    has_cofibration_witness = bool(witness.get("cofibration")) or cofibration_count > 0

    return AspfOpportunityObservation(
        structure=structure,
        one_cell_count=one_cell_count,
        two_cell_count=two_cell_count,
        cofibration_count=cofibration_count,
        surface_count=surface_count,
        classification=classification,
        has_resume_load=has_resume_load,
        has_resume_write=has_resume_write,
        has_two_cell_witness=has_two_cell_witness,
        has_cofibration_witness=has_cofibration_witness,
    )


def classify_aspf_opportunity(payload: Mapping[str, object]) -> PolicyDecision:
    projected = _coerce_observation(payload)
    return evaluate_policy(domain=PolicyDomain.ASPF_OPPORTUNITY, data=projected.to_policy_payload())
