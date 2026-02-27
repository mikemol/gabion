# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Protocol

from gabion.analysis.resume_codec import mapping_or_none


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

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source.label,
            "target": self.target.label,
            "representative": self.representative,
            "basis_path": list(self.basis_path),
        }


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
            and tuple(self.left.basis_path) == tuple(self.right.basis_path)
        )

    def links(self, *, baseline_representative: str, current_representative: str) -> bool:
        left = self.left.representative
        right = self.right.representative
        return (left, right) in {
            (baseline_representative, current_representative),
            (current_representative, baseline_representative),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
            "witness_id": self.witness_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SuiteSiteEndpoint:
    kind: str
    key: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "key": list(self.key)}


@dataclass(frozen=True)
class AspfCanonicalIdentityContract:
    identity_kind: str
    source: BasisZeroCell
    target: BasisZeroCell
    representative: str
    basis_path: tuple[str, ...]
    suite_site_source: SuiteSiteEndpoint
    suite_site_target: SuiteSiteEndpoint

    def as_dict(self) -> dict[str, object]:
        return {
            "identity_kind": self.identity_kind,
            "source": self.source.label,
            "target": self.target.label,
            "representative": self.representative,
            "basis_path": list(self.basis_path),
            "suite_site_endpoints": {
                "source": self.suite_site_source.as_dict(),
                "target": self.suite_site_target.as_dict(),
            },
        }


@dataclass(frozen=True)
class _DecodeOneCellOutcome:
    valid: bool
    cell: AspfOneCell


def parse_2cell_witness(payload: Mapping[str, object]) -> object:
    left_payload = payload.get("left")
    right_payload = payload.get("right")
    witness_id_raw = payload.get("witness_id")
    reason_raw = payload.get("reason")

    def _decode_1cell(raw: Mapping[str, object]) -> _DecodeOneCellOutcome:
        source = raw.get("source")
        target = raw.get("target")
        representative = raw.get("representative")
        basis_path = raw.get("basis_path")
        source_label = ""
        target_label = ""
        representative_label = ""
        basis_items: tuple[str, ...] = ()
        basis_valid = False

        match source:
            case str() as source_text:
                source_label = source_text
        match target:
            case str() as target_text:
                target_label = target_text
        match representative:
            case str() as representative_text:
                representative_label = representative_text
        match basis_path:
            case list() as basis_list:
                basis_items = tuple(str(item) for item in basis_list)
                basis_valid = True
        decoded_cell = AspfOneCell(
            source=BasisZeroCell(source_label),
            target=BasisZeroCell(target_label),
            representative=representative_label,
            basis_path=basis_items,
        )
        return _DecodeOneCellOutcome(
            valid=bool(
                source_label and target_label and representative_label and basis_valid
            ),
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
    left_outcome = _decode_1cell(mapping_or_none(left_payload) or {})
    right_outcome = _decode_1cell(mapping_or_none(right_payload) or {})

    match (witness_id_raw, reason_raw):
        case (str() as witness_id, str() as reason):
            if left_outcome.valid and right_outcome.valid:
                return AspfTwoCellWitness(
                    left=left_outcome.cell,
                    right=right_outcome.cell,
                    witness_id=witness_id,
                    reason=reason,
                )
            return None
        case _:
            return None


def identity_1cell(cell: BasisZeroCell) -> AspfOneCell:
    return AspfOneCell(
        source=cell,
        target=cell,
        representative=f"id:{cell.label}",
        basis_path=(cell.label,),
    )


def compose_1cells(left: AspfOneCell, right: AspfOneCell) -> AspfOneCell:
    if left.target != right.source:
        raise ValueError("1-cell composition requires left.target == right.source")
    stitched_basis = left.basis_path + right.basis_path[1:]
    return AspfOneCell(
        source=left.source,
        target=right.target,
        representative=f"{left.representative};{right.representative}",
        basis_path=stitched_basis,
    )


def validate_2cell_compatibility(witness: AspfTwoCellWitness) -> None:
    if not witness.is_compatible():
        raise ValueError("2-cell witness must connect equivalent source/target/basis path")
