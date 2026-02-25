"""Adapter from prime structural lab artifacts to ASPF witness objects."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from gabion.analysis.aspf_core import AspfOneCell, AspfTwoCellWitness, BasisZeroCell
from gabion.analysis.aspf_decision_surface import classify_drift_by_homotopy
from gabion.analysis.aspf_morphisms import (
    AspfPrimeBasis,
    DomainPrimeBasis,
    DomainToAspfCofibration,
    DomainToAspfCofibrationEntry,
)


@dataclass(frozen=True)
class LabPrimeArtifact:
    label: str
    prime: int


def convert_lab_artifacts(artifacts: list[LabPrimeArtifact]) -> dict[str, object]:
    entries = tuple(
        DomainToAspfCofibrationEntry(
            domain=DomainPrimeBasis(domain_key=f"lab:{item.label}", prime=item.prime),
            aspf=AspfPrimeBasis(aspf_key=f"aspf:{item.label}", prime=item.prime),
        )
        for item in artifacts
    )
    cofibration = DomainToAspfCofibration(entries=entries)
    cofibration.validate()

    start = BasisZeroCell("lab:start")
    end = BasisZeroCell("lab:end")
    left = AspfOneCell(start, end, "left_embedding", tuple(i.label for i in artifacts))
    right = AspfOneCell(start, end, "right_embedding", tuple(i.label for i in artifacts))
    witness = AspfTwoCellWitness(
        left=left,
        right=right,
        witness_id="lab:higher-path",
        reason="boundary-asymmetry-but-equivalent",
    )
    drift = classify_drift_by_homotopy(
        baseline_representative=left.representative,
        current_representative=right.representative,
        has_equivalence_witness=witness.is_compatible(),
    )
    return {
        "cofibration": cofibration.as_dict(),
        "higher_path_witness": {
            "witness_id": witness.witness_id,
            "compatible": witness.is_compatible(),
            "reason": witness.reason,
        },
        "boundary_asymmetry_scenario": {
            "left": left.as_dict(),
            "right": right.as_dict(),
            "drift_classification": drift,
        },
    }


def write_regression_artifact(output_path: Path) -> None:
    payload = convert_lab_artifacts(
        [LabPrimeArtifact("p2", 2), LabPrimeArtifact("p3", 3), LabPrimeArtifact("p5", 5)]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    artifact = root / "artifacts" / "aspf_regression.json"
    write_regression_artifact(artifact)
    print(str(artifact))
