# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .aspf_core import AspfTwoCellWitness


class RepresentativeSelectionMode(StrEnum):
    LEXICOGRAPHIC_MIN = "lexicographic_min"
    SHORTEST_PATH_THEN_LEXICOGRAPHIC = "shortest_path_then_lexicographic"


@dataclass(frozen=True)
class RepresentativeSelectionOptions:
    mode: RepresentativeSelectionMode
    candidates: tuple[str, ...]

    def validate(self) -> None:
        if not self.candidates:
            raise ValueError("Representative selection requires non-empty candidates")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("Representative candidates must be unique")


@dataclass(frozen=True)
class RepresentativeSelectionWitness:
    mode: RepresentativeSelectionMode
    selected: str
    candidates: tuple[str, ...]
    witness_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "selected": self.selected,
            "candidates": list(self.candidates),
            "witness_id": self.witness_id,
        }


def select_representative(options: RepresentativeSelectionOptions) -> RepresentativeSelectionWitness:
    options.validate()
    if options.mode is RepresentativeSelectionMode.LEXICOGRAPHIC_MIN:
        selected = min(options.candidates)
    elif options.mode is RepresentativeSelectionMode.SHORTEST_PATH_THEN_LEXICOGRAPHIC:
        selected = min(options.candidates, key=lambda value: (len(value), value))
    else:  # pragma: no cover - enum exhaustiveness
        raise ValueError(f"Unsupported representative selection mode: {options.mode}")
    return RepresentativeSelectionWitness(
        mode=options.mode,
        selected=selected,
        candidates=options.candidates,
        witness_id=f"rep:{options.mode.value}:{selected}",
    )


def classify_drift_by_homotopy(
    *,
    baseline_representative: str,
    current_representative: str,
    equivalence_witness: object = None,
    has_equivalence_witness: bool = False,
) -> str:
    if baseline_representative == current_representative:
        return "non_drift"
    if type(equivalence_witness) is AspfTwoCellWitness:
        if equivalence_witness.is_compatible() and equivalence_witness.links(
            baseline_representative=baseline_representative,
            current_representative=current_representative,
        ):
            return "non_drift"
    if bool(has_equivalence_witness):
        return "non_drift"
    return "drift"
