# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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
