from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from gabion.analysis.aspf.aspf_core import AspfTwoCellWitness
from gabion.analysis.foundation.frozen_object_map import (
    ObjectEntry,
    make_object_map,
)
from gabion.invariants import never


class RepresentativeSelectionMode(StrEnum):
    LEXICOGRAPHIC_MIN = "lexicographic_min"
    SHORTEST_PATH_THEN_LEXICOGRAPHIC = "shortest_path_then_lexicographic"


@dataclass(frozen=True)
class RepresentativeSelectionWitnessId:
    mode: RepresentativeSelectionMode
    selected: str

    def as_dict(self) -> Mapping[str, WireValue]:
        return make_object_map(
            [
                ObjectEntry("mode", self.mode.value),
                ObjectEntry("selected", self.selected),
            ]
        )


@dataclass(frozen=True)
class RepresentativeSelectionOptions:
    mode: RepresentativeSelectionMode
    candidates: tuple[str, ...]

    def validate(self) -> None:
        _require(
            len(self.candidates) > 0,
            "Representative selection requires non-empty candidates",
        )
        _require(
            len(set(self.candidates)) == len(self.candidates),
            "Representative candidates must be unique",
        )


@dataclass(frozen=True)
class RepresentativeSelectionWitness:
    mode: RepresentativeSelectionMode
    selected: str
    candidates: tuple[str, ...]
    witness_id: RepresentativeSelectionWitnessId

    def as_dict(self) -> Mapping[str, WireValue]:
        return make_object_map(
            [
                ObjectEntry("mode", self.mode.value),
                ObjectEntry("selected", self.selected),
                ObjectEntry("candidates", list(self.candidates)),
                ObjectEntry("witness_id", self.witness_id.as_dict()),
            ]
        )


def select_representative(options: RepresentativeSelectionOptions) -> RepresentativeSelectionWitness:
    options.validate()
    selector = _mode_selector(options.mode)
    selected = selector(options.candidates)
    return RepresentativeSelectionWitness(
        mode=options.mode,
        selected=selected,
        candidates=options.candidates,
        witness_id=RepresentativeSelectionWitnessId(mode=options.mode, selected=selected),
    )


def classify_drift_by_homotopy(
    *,
    baseline_representative: str,
    current_representative: str,
    equivalence_witness: AspfTwoCellWitness = None,
    has_equivalence_witness: bool = False,
) -> str:
    non_drift = (
        baseline_representative == current_representative
        or _witness_links(
            equivalence_witness,
            baseline_representative=baseline_representative,
            current_representative=current_representative,
        )
        or has_equivalence_witness
    )
    return _DRIFT_LABEL_BY_NON_DRIFT[non_drift]


def _noop_validator(_: str) -> None:
    return None


def _raise_validation_error(message: str) -> None:
    raise ValueError(message)


_VALIDATION_HANDLERS: list[Callable[[str], None]] = [_noop_validator, _raise_validation_error]


def _require(condition: bool, message: str) -> None:
    _VALIDATION_HANDLERS[not condition](message)


def _raise_unsupported_mode(mode: RepresentativeSelectionMode) -> int:
    never("unsupported representative selection mode", mode=mode)


_MODE_RESOLVERS: list[Callable[[RepresentativeSelectionMode], int]] = [
    lambda mode: _SELECTION_MODES.index(mode),
    _raise_unsupported_mode,
]
_SELECTION_MODES: list[RepresentativeSelectionMode] = [
    RepresentativeSelectionMode.LEXICOGRAPHIC_MIN,
    RepresentativeSelectionMode.SHORTEST_PATH_THEN_LEXICOGRAPHIC,
]
_SELECTION_HANDLERS: list[Callable[[tuple[str, ...]], str]] = [
    lambda candidates: min(candidates),
    lambda candidates: min(candidates, key=lambda value: (len(value), value)),
]
_DRIFT_LABEL_BY_NON_DRIFT: list[str] = ["drift", "non_drift"]


def _mode_selector(mode: RepresentativeSelectionMode) -> Callable[[tuple[str, ...]], str]:
    resolver = _MODE_RESOLVERS[not (mode in _SELECTION_MODES)]
    mode_index = resolver(mode)
    return _SELECTION_HANDLERS[mode_index]


def _witness_links(
    equivalence_witness: AspfTwoCellWitness,
    *,
    baseline_representative: str,
    current_representative: str,
) -> bool:
    return type(equivalence_witness) is AspfTwoCellWitness and equivalence_witness.is_compatible() and equivalence_witness.links(
        baseline_representative=baseline_representative,
        current_representative=current_representative,
    )
