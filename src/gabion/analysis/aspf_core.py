# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
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


def parse_2cell_witness(payload: Mapping[str, object]) -> AspfTwoCellWitness | None:
    left_payload = payload.get("left")
    right_payload = payload.get("right")
    witness_id = payload.get("witness_id")
    reason = payload.get("reason")
    if not isinstance(left_payload, Mapping) or not isinstance(right_payload, Mapping):
        return None
    if not isinstance(witness_id, str) or not isinstance(reason, str):
        return None

    def _decode_1cell(raw: Mapping[str, object]) -> AspfOneCell | None:
        source = raw.get("source")
        target = raw.get("target")
        representative = raw.get("representative")
        basis_path = raw.get("basis_path")
        if not isinstance(source, str) or not isinstance(target, str) or not isinstance(representative, str):
            return None
        if not isinstance(basis_path, list) or not all(isinstance(item, str) for item in basis_path):
            return None
        return AspfOneCell(
            source=BasisZeroCell(source),
            target=BasisZeroCell(target),
            representative=representative,
            basis_path=tuple(basis_path),
        )

    left = _decode_1cell(left_payload)
    right = _decode_1cell(right_payload)
    if left is None or right is None:
        return None
    return AspfTwoCellWitness(left=left, right=right, witness_id=witness_id, reason=reason)


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
