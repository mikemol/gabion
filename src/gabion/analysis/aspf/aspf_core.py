from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable
from typing import Protocol

from gabion.analysis.foundation.wire_types import WireValue
from gabion.analysis.foundation.frozen_object_map import (
    ObjectEntry,
    make_object_map,
)
from gabion.analysis.foundation.resume_codec import mapping_optional


class AspfZeroCell(Protocol):
    """Typed 0-cell contract in the ASPF basis."""

    label: str


@dataclass(frozen=True)
class BasisZeroCell:
    """Concrete 0-cell for ASPF path materialization."""

    label: str


@dataclass(frozen=True)
class AspfOneCell:
    """Typed path representative between 0-cells."""

    source: BasisZeroCell
    target: BasisZeroCell
    representative: str
    basis_path: tuple[str, ...]

    def as_dict(self) -> Mapping[str, WireValue]:
        return make_object_map(
            [
                ObjectEntry("source", self.source.label),
                ObjectEntry("target", self.target.label),
                ObjectEntry("representative", self.representative),
                ObjectEntry("basis_path", list(self.basis_path)),
            ]
        )


@dataclass(frozen=True)
class AspfTwoCellWitness:
    """Higher path witness showing equivalent 1-cell representatives."""

    left: AspfOneCell
    right: AspfOneCell
    witness_id: str
    reason: str

    def is_compatible(self) -> bool:
        return (
            self.left.source == self.right.source
            and self.left.target == self.right.target
            and self.left.basis_path == self.right.basis_path
        )

    def links(self, *, baseline_representative: str, current_representative: str) -> bool:
        left = self.left.representative
        right = self.right.representative
        return (left == baseline_representative and right == current_representative) or (
            left == current_representative and right == baseline_representative
        )

    def as_dict(self) -> Mapping[str, WireValue]:
        return make_object_map(
            [
                ObjectEntry("left", self.left.as_dict()),
                ObjectEntry("right", self.right.as_dict()),
                ObjectEntry("witness_id", self.witness_id),
                ObjectEntry("reason", self.reason),
            ]
        )


@dataclass(frozen=True)
class SuiteSiteEndpoint:
    kind: str
    key: tuple[str, ...]

    def as_dict(self) -> Mapping[str, WireValue]:
        return make_object_map(
            [
                ObjectEntry("kind", self.kind),
                ObjectEntry("key", list(self.key)),
            ]
        )


@dataclass(frozen=True)
class AspfCanonicalIdentityContract:
    identity_kind: str
    source: BasisZeroCell
    target: BasisZeroCell
    representative: str
    basis_path: tuple[str, ...]
    suite_site_source: SuiteSiteEndpoint
    suite_site_target: SuiteSiteEndpoint

    def as_dict(self) -> Mapping[str, WireValue]:
        return make_object_map(
            [
                ObjectEntry("identity_kind", self.identity_kind),
                ObjectEntry("source", self.source.label),
                ObjectEntry("target", self.target.label),
                ObjectEntry("representative", self.representative),
                ObjectEntry("basis_path", list(self.basis_path)),
                ObjectEntry(
                    "suite_site_endpoints",
                    make_object_map(
                        [
                            ObjectEntry("source", self.suite_site_source.as_dict()),
                            ObjectEntry("target", self.suite_site_target.as_dict()),
                        ]
                    ),
                ),
            ]
        )


@dataclass(frozen=True)
class _DecodeOneCellOutcome:
    valid: bool
    cell: AspfOneCell


type _OneCellPayload = Mapping[str, WireValue]


def parse_2cell_witness(payload: Mapping[str, WireValue]) -> AspfTwoCellWitness:
    left_payload = payload.get("left")
    right_payload = payload.get("right")
    witness_id_raw = payload.get("witness_id")
    reason_raw = payload.get("reason")

    def _decode_1cell(raw: _OneCellPayload) -> _DecodeOneCellOutcome:
        source = raw.get("source")
        target = raw.get("target")
        representative = raw.get("representative")
        basis_path = raw.get("basis_path")
        source_label = _coerce_str(source)
        target_label = _coerce_str(target)
        representative_label = _coerce_str(representative)
        basis_list = _LIST_COERCERS[_is_list(basis_path)](basis_path)
        basis_items = list(filter(_is_str, basis_list))
        basis_valid = _is_list(basis_path) and len(basis_items) == len(basis_list)
        decoded_cell = AspfOneCell(
            source=BasisZeroCell(source_label),
            target=BasisZeroCell(target_label),
            representative=representative_label,
            basis_path=(*basis_items,),
        )
        return _DecodeOneCellOutcome(
            valid=source_label != "" and target_label != "" and representative_label != "" and basis_valid,
            cell=decoded_cell,
        )

    left_outcome = _DecodeOneCellOutcome(
        valid=False,
        cell=AspfOneCell(
            source=BasisZeroCell(""),
            target=BasisZeroCell(""),
            representative="",
            basis_path=(),
        ),
    )
    right_outcome = left_outcome
    left_outcome = _decode_1cell(mapping_optional(left_payload) or _EMPTY_ONE_CELL_PAYLOAD)
    right_outcome = _decode_1cell(mapping_optional(right_payload) or _EMPTY_ONE_CELL_PAYLOAD)
    witness_fields_are_strings = _is_str(witness_id_raw) and _is_str(reason_raw)
    witness_is_valid = witness_fields_are_strings and left_outcome.valid and right_outcome.valid
    return _WITNESS_BUILDERS[witness_is_valid](
        left_outcome=left_outcome,
        right_outcome=right_outcome,
        witness_id=_coerce_str(witness_id_raw),
        reason=_coerce_str(reason_raw),
    )


def identity_1cell(cell: BasisZeroCell) -> AspfOneCell:
    return AspfOneCell(
        source=cell,
        target=cell,
        representative="id:%s" % cell.label,
        basis_path=(cell.label,),
    )


def compose_1cells(left: AspfOneCell, right: AspfOneCell) -> AspfOneCell:
    _require(
        left.target == right.source,
        "1-cell composition requires left.target == right.source",
    )
    stitched_basis = left.basis_path + right.basis_path[1:]
    return AspfOneCell(
        source=left.source,
        target=right.target,
        representative="%s;%s" % (left.representative, right.representative),
        basis_path=stitched_basis,
    )


def validate_2cell_compatibility(witness: AspfTwoCellWitness) -> None:
    _require(
        witness.is_compatible(),
        "2-cell witness must connect equivalent source/target/basis path",
    )


def _empty_text(_: WireValue) -> str:
    return ""


_TEXT_COERCERS: list[Callable[[WireValue], str]] = [_empty_text, lambda value: value]
_LIST_COERCERS: list[Callable[[WireValue], list[WireValue]]] = [
    lambda _: [],
    lambda value: value,
]
_EMPTY_ONE_CELL_PAYLOAD: Mapping[str, WireValue] = make_object_map([])


def _coerce_str(value: WireValue) -> str:
    return _TEXT_COERCERS[_is_str(value)](value)


def _is_str(value: WireValue) -> bool:
    return type(value) is str


def _is_list(value: WireValue) -> bool:
    return type(value) is list


def _build_witness(
    *,
    left_outcome: _DecodeOneCellOutcome,
    right_outcome: _DecodeOneCellOutcome,
    witness_id: str,
    reason: str,
) -> AspfTwoCellWitness:
    return AspfTwoCellWitness(
        left=left_outcome.cell,
        right=right_outcome.cell,
        witness_id=witness_id,
        reason=reason,
    )


def _reject_witness(
    *,
    left_outcome: _DecodeOneCellOutcome,
    right_outcome: _DecodeOneCellOutcome,
    witness_id: str,
    reason: str,
) -> AspfTwoCellWitness:
    _ = left_outcome
    _ = right_outcome
    _ = witness_id
    _ = reason
    return None


_WITNESS_BUILDERS: list[Callable[..., AspfTwoCellWitness]] = [
    _reject_witness,
    _build_witness,
]


def _noop_validator(_: str) -> None:
    return None


def _raise_validation_error(message: str) -> None:
    raise ValueError(message)


_VALIDATION_HANDLERS: list[Callable[[str], None]] = [_noop_validator, _raise_validation_error]


def _require(condition: bool, message: str) -> None:
    _VALIDATION_HANDLERS[not condition](message)
