from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import re

from gabion.analysis.aspf.aspf_lattice_algebra import canonical_structural_identity
from gabion.tooling.policy_substrate.site_identity import canonical_site_identity
from gabion.tooling.policy_substrate.identity_zone import (
    IdentityAtom,
    IdentityCarrier,
    IdentityDecomposition,
    IdentityDecompositionRelation,
    IdentityLocalInterner,
    IdentityZoneName,
)
from gabion.analysis.core.type_fingerprints import PrimeRegistry


class PolicyScannerIdentityNamespace(StrEnum):
    ITEM = "policy_scanner.item"
    DECOMPOSITION = "policy_scanner.decomposition"


class PolicyScannerDecompositionKind(StrEnum):
    CANONICAL = "canonical"
    SCANNER_KIND = "scanner_kind"
    RULE_ID = "rule_id"
    REL_PATH = "rel_path"
    REL_PATH_SEGMENT = "rel_path_segment"
    QUALNAME = "qualname"
    QUALNAME_SEGMENT = "qualname_segment"
    KIND = "kind"
    SITE_IDENTITY = "site_identity"
    STRUCTURAL_IDENTITY = "structural_identity"


class PolicyScannerDecompositionRelationKind(StrEnum):
    CANONICAL_OF = "canonical_of"
    ALTERNATE_OF = "alternate_of"
    EQUIVALENT_UNDER = "equivalent_under"
    DERIVED_FROM = "derived_from"


POLICY_SCANNER_ZONE = IdentityZoneName("policy_scanner")


_PrimeBackedIdentity = IdentityAtom[PolicyScannerIdentityNamespace]


PolicyScannerDecompositionIdentity = IdentityDecomposition[
    PolicyScannerIdentityNamespace,
    PolicyScannerDecompositionKind,
]


PolicyScannerDecompositionRelation = IdentityDecompositionRelation[
    PolicyScannerIdentityNamespace,
    PolicyScannerDecompositionKind,
    PolicyScannerDecompositionRelationKind,
]


@dataclass(frozen=True, order=True)
class PolicyScannerIdentity:
    canonical: _PrimeBackedIdentity
    scanner_kind: str = field(compare=False)
    rule_id: str = field(compare=False, default="")
    rel_path: str = field(compare=False, default="")
    qualname: str = field(compare=False, default="")
    line: int = field(compare=False, default=0)
    column: int = field(compare=False, default=0)
    kind: str = field(compare=False, default="")
    site_identity: str = field(compare=False, default="")
    structural_identity: str = field(compare=False, default="")
    label: str = field(compare=False, default="")
    decompositions: tuple[PolicyScannerDecompositionIdentity, ...] = field(
        default=(),
        compare=False,
    )
    relations: tuple[PolicyScannerDecompositionRelation, ...] = field(
        default=(),
        compare=False,
    )
    zone_name: IdentityZoneName = field(
        default=POLICY_SCANNER_ZONE,
        compare=False,
    )

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.label or self.canonical.token

    def as_carrier(
        self,
    ) -> IdentityCarrier[
        PolicyScannerIdentityNamespace,
        PolicyScannerDecompositionKind,
        PolicyScannerDecompositionRelationKind,
    ]:
        return IdentityCarrier(
            canonical=self.canonical,
            zone_name=self.zone_name,
            carrier_kind=self.scanner_kind,
            label=self.label or self.wire(),
            decompositions=self.decompositions,
            relations=self.relations,
            metadata={
                "rule_id": self.rule_id,
                "rel_path": self.rel_path,
                "qualname": self.qualname,
                "line": self.line,
                "column": self.column,
                "kind": self.kind,
                "site_identity": self.site_identity,
                "structural_identity": self.structural_identity,
            },
        )

    def provenance_payload(self) -> dict[str, object]:
        return {
            "zone_name": self.zone_name.value,
            "carrier_wire": self.wire(),
            "decomposition_wires": [item.wire() for item in self.decompositions],
        }

    def as_payload(self) -> dict[str, object]:
        return {
            "wire": self.wire(),
            "scanner_kind": self.scanner_kind,
            "rule_id": self.rule_id,
            "rel_path": self.rel_path,
            "qualname": self.qualname,
            "line": self.line,
            "column": self.column,
            "kind": self.kind,
            "site_identity": self.site_identity,
            "structural_identity": self.structural_identity,
            "label": self.label or self.wire(),
            "decompositions": [item.as_payload() for item in self.decompositions],
            "relations": [item.as_payload() for item in self.relations],
            "provenance": self.provenance_payload(),
        }


def canonical_policy_scanner_site_identity(
    *,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    scanner_kind: str,
    surface: str,
) -> str:
    return canonical_site_identity(
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        node_kind=scanner_kind,
        surface=surface,
    )


def canonical_policy_scanner_structural_identity(
    *,
    rel_path: str,
    qualname: str,
    structural_path: str,
    scanner_kind: str,
    surface: str,
) -> str:
    return canonical_structural_identity(
        rel_path=rel_path,
        qualname=qualname,
        structural_path=structural_path,
        node_kind=scanner_kind,
        surface=surface,
    )


@dataclass
class PolicyScannerIdentitySpace:
    registry: PrimeRegistry = field(default_factory=PrimeRegistry)
    _interner: IdentityLocalInterner[PolicyScannerIdentityNamespace] = field(
        init=False,
        repr=False,
    )
    _decomposition_cache: dict[
        _PrimeBackedIdentity,
        tuple[
            tuple[PolicyScannerDecompositionIdentity, ...],
            tuple[PolicyScannerDecompositionRelation, ...],
        ],
    ] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._interner = IdentityLocalInterner(registry=self.registry)

    @staticmethod
    def _segments(value: str) -> tuple[str, ...]:
        return IdentityLocalInterner.structural_segments(value)

    def _identity(
        self,
        *,
        namespace: PolicyScannerIdentityNamespace,
        token: str,
    ) -> _PrimeBackedIdentity:
        return self._interner.identity(namespace=namespace, token=token)

    def _decomposition_identity(
        self,
        *,
        origin: _PrimeBackedIdentity,
        decomposition_kind: PolicyScannerDecompositionKind,
        label: str,
        part_index: int = -1,
    ) -> PolicyScannerDecompositionIdentity:
        return self._interner.decomposition_identity(
            origin=origin,
            decomposition_namespace=PolicyScannerIdentityNamespace.DECOMPOSITION,
            decomposition_kind=decomposition_kind,
            label=label,
            part_index=part_index,
            canonical_kind=PolicyScannerDecompositionKind.CANONICAL,
        )

    def _decomposition_bundle(
        self,
        *,
        origin: _PrimeBackedIdentity,
        scanner_kind: str,
        rule_id: str,
        rel_path: str,
        qualname: str,
        kind: str,
        site_identity: str,
        structural_identity: str,
    ) -> tuple[
        tuple[PolicyScannerDecompositionIdentity, ...],
        tuple[PolicyScannerDecompositionRelation, ...],
    ]:
        cached = self._decomposition_cache.get(origin)
        if cached is not None:
            return cached
        canonical = self._decomposition_identity(
            origin=origin,
            decomposition_kind=PolicyScannerDecompositionKind.CANONICAL,
            label=origin.token,
        )
        scanner_kind_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=PolicyScannerDecompositionKind.SCANNER_KIND,
            label=scanner_kind,
        )
        rule_id_view = (
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=PolicyScannerDecompositionKind.RULE_ID,
                label=rule_id,
            )
            if rule_id
            else None
        )
        rel_path_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=PolicyScannerDecompositionKind.REL_PATH,
            label=rel_path,
        )
        rel_path_segments = tuple(
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=PolicyScannerDecompositionKind.REL_PATH_SEGMENT,
                label=segment,
                part_index=index,
            )
            for index, segment in enumerate(self._segments(rel_path))
        )
        qualname_view = (
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=PolicyScannerDecompositionKind.QUALNAME,
                label=qualname,
            )
            if qualname
            else None
        )
        qualname_segments = tuple(
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=PolicyScannerDecompositionKind.QUALNAME_SEGMENT,
                label=segment,
                part_index=index,
            )
            for index, segment in enumerate(self._segments(qualname))
        )
        kind_view = (
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=PolicyScannerDecompositionKind.KIND,
                label=kind,
            )
            if kind
            else None
        )
        site_identity_view = (
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=PolicyScannerDecompositionKind.SITE_IDENTITY,
                label=site_identity,
            )
            if site_identity
            else None
        )
        structural_identity_view = (
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=PolicyScannerDecompositionKind.STRUCTURAL_IDENTITY,
                label=structural_identity,
            )
            if structural_identity
            else None
        )
        decompositions = tuple(
            item
            for item in (
                canonical,
                scanner_kind_view,
                rule_id_view,
                rel_path_view,
                qualname_view,
                kind_view,
                site_identity_view,
                structural_identity_view,
                *rel_path_segments,
                *qualname_segments,
            )
            if item is not None
        )
        relations: list[PolicyScannerDecompositionRelation] = [
            PolicyScannerDecompositionRelation(
                relation_kind=PolicyScannerDecompositionRelationKind.CANONICAL_OF,
                source=canonical,
                target=item,
                rationale="scanner identity canonical decomposition view",
            )
            for item in decompositions
            if item is not canonical
        ]
        for item in (*rel_path_segments, *qualname_segments):
            relations.append(
                PolicyScannerDecompositionRelation(
                    relation_kind=PolicyScannerDecompositionRelationKind.DERIVED_FROM,
                    source=item,
                    target=canonical,
                    rationale="scanner identity segment derived from canonical item",
                )
            )
        bundle = (decompositions, tuple(relations))
        self._decomposition_cache[origin] = bundle
        return bundle

    def item_id(
        self,
        *,
        scanner_kind: str,
        rule_id: str,
        rel_path: str,
        qualname: str,
        line: int,
        column: int,
        kind: str,
        site_identity: str,
        structural_identity: str,
        label: str = "",
    ) -> PolicyScannerIdentity:
        canonical = self._identity(
            namespace=PolicyScannerIdentityNamespace.ITEM,
            token="::".join(
                (
                    scanner_kind.strip(),
                    rule_id.strip(),
                    rel_path.strip(),
                    qualname.strip(),
                    str(int(line)),
                    str(int(column)),
                    kind.strip(),
                    site_identity.strip(),
                    structural_identity.strip(),
                )
            ),
        )
        decompositions, relations = self._decomposition_bundle(
            origin=canonical,
            scanner_kind=scanner_kind.strip(),
            rule_id=rule_id.strip(),
            rel_path=rel_path.strip(),
            qualname=qualname.strip(),
            kind=kind.strip(),
            site_identity=site_identity.strip(),
            structural_identity=structural_identity.strip(),
        )
        return PolicyScannerIdentity(
            canonical=canonical,
            scanner_kind=scanner_kind.strip(),
            rule_id=rule_id.strip(),
            rel_path=rel_path.strip(),
            qualname=qualname.strip(),
            line=int(line),
            column=int(column),
            kind=kind.strip(),
            site_identity=site_identity.strip(),
            structural_identity=structural_identity.strip(),
            label=label.strip(),
            decompositions=decompositions,
            relations=relations,
        )

    def item_carrier(
        self,
        *,
        scanner_kind: str,
        rule_id: str,
        rel_path: str,
        qualname: str,
        line: int,
        column: int,
        kind: str,
        site_identity: str,
        structural_identity: str,
        label: str = "",
    ) -> IdentityCarrier[
        PolicyScannerIdentityNamespace,
        PolicyScannerDecompositionKind,
        PolicyScannerDecompositionRelationKind,
    ]:
        return self.item_id(
            scanner_kind=scanner_kind,
            rule_id=rule_id,
            rel_path=rel_path,
            qualname=qualname,
            line=line,
            column=column,
            kind=kind,
            site_identity=site_identity,
            structural_identity=structural_identity,
            label=label,
        ).as_carrier()


__all__ = [
    "POLICY_SCANNER_ZONE",
    "PolicyScannerDecompositionIdentity",
    "PolicyScannerDecompositionKind",
    "PolicyScannerDecompositionRelation",
    "PolicyScannerDecompositionRelationKind",
    "PolicyScannerIdentity",
    "PolicyScannerIdentityNamespace",
    "PolicyScannerIdentitySpace",
    "canonical_policy_scanner_site_identity",
    "canonical_policy_scanner_structural_identity",
]
