from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.tooling.policy_substrate.identity_zone import (
    IdentityAtom,
    IdentityCarrier,
    IdentityDecomposition,
    IdentityDecompositionRelation,
    IdentityLocalInterner,
    IdentityZoneName,
)


class HotspotQueueIdentityNamespace(StrEnum):
    ITEM = "hotspot_queue.item"
    DECOMPOSITION = "hotspot_queue.decomposition"


class HotspotQueueDecompositionKind(StrEnum):
    CANONICAL = "canonical"
    ITEM_KIND = "item_kind"
    PATH = "path"
    PATH_SEGMENT = "path_segment"


class HotspotQueueDecompositionRelationKind(StrEnum):
    CANONICAL_OF = "canonical_of"
    ALTERNATE_OF = "alternate_of"
    EQUIVALENT_UNDER = "equivalent_under"
    DERIVED_FROM = "derived_from"


HOTSPOT_QUEUE_ZONE = IdentityZoneName("hotspot_queue")

_HotspotAtom = IdentityAtom[HotspotQueueIdentityNamespace]
HotspotQueueDecompositionIdentity = IdentityDecomposition[
    HotspotQueueIdentityNamespace,
    HotspotQueueDecompositionKind,
]
HotspotQueueDecompositionRelation = IdentityDecompositionRelation[
    HotspotQueueIdentityNamespace,
    HotspotQueueDecompositionKind,
    HotspotQueueDecompositionRelationKind,
]


@dataclass(frozen=True, order=True)
class HotspotQueueIdentity:
    canonical: _HotspotAtom
    item_kind: str = field(compare=False)
    path: str = field(compare=False)
    label: str = field(compare=False, default="")
    decompositions: tuple[HotspotQueueDecompositionIdentity, ...] = field(
        default=(),
        compare=False,
    )
    relations: tuple[HotspotQueueDecompositionRelation, ...] = field(
        default=(),
        compare=False,
    )
    zone_name: IdentityZoneName = field(default=HOTSPOT_QUEUE_ZONE, compare=False)

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.label or self.path or self.wire()

    def as_carrier(
        self,
    ) -> IdentityCarrier[
        HotspotQueueIdentityNamespace,
        HotspotQueueDecompositionKind,
        HotspotQueueDecompositionRelationKind,
    ]:
        return IdentityCarrier(
            canonical=self.canonical,
            zone_name=self.zone_name,
            carrier_kind=self.item_kind,
            label=self.label or self.path or self.wire(),
            decompositions=self.decompositions,
            relations=self.relations,
            metadata={
                "item_kind": self.item_kind,
                "path": self.path,
            },
        )


@dataclass
class HotspotQueueIdentitySpace:
    registry: PrimeRegistry = field(default_factory=PrimeRegistry)
    _interner: IdentityLocalInterner[HotspotQueueIdentityNamespace] = field(
        init=False,
        repr=False,
    )
    _decomposition_cache: dict[
        _HotspotAtom,
        tuple[
            tuple[HotspotQueueDecompositionIdentity, ...],
            tuple[HotspotQueueDecompositionRelation, ...],
        ],
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._interner = IdentityLocalInterner(registry=self.registry)

    def _identity(self, *, namespace: HotspotQueueIdentityNamespace, token: str) -> _HotspotAtom:
        return self._interner.identity(namespace=namespace, token=token)

    def _decomposition_identity(
        self,
        *,
        origin: _HotspotAtom,
        decomposition_kind: HotspotQueueDecompositionKind,
        label: str,
        part_index: int = -1,
    ) -> HotspotQueueDecompositionIdentity:
        return self._interner.decomposition_identity(
            origin=origin,
            decomposition_namespace=HotspotQueueIdentityNamespace.DECOMPOSITION,
            decomposition_kind=decomposition_kind,
            label=label,
            part_index=part_index,
            canonical_kind=HotspotQueueDecompositionKind.CANONICAL,
        )

    def _decomposition_bundle(
        self,
        *,
        origin: _HotspotAtom,
        item_kind: str,
        path: str,
    ) -> tuple[
        tuple[HotspotQueueDecompositionIdentity, ...],
        tuple[HotspotQueueDecompositionRelation, ...],
    ]:
        cached = self._decomposition_cache.get(origin)
        if cached is not None:
            return cached
        canonical = self._decomposition_identity(
            origin=origin,
            decomposition_kind=HotspotQueueDecompositionKind.CANONICAL,
            label=origin.token,
        )
        item_kind_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=HotspotQueueDecompositionKind.ITEM_KIND,
            label=item_kind,
        )
        path_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=HotspotQueueDecompositionKind.PATH,
            label=path,
        )
        path_segments = tuple(
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=HotspotQueueDecompositionKind.PATH_SEGMENT,
                label=segment,
                part_index=index,
            )
            for index, segment in enumerate(
                IdentityLocalInterner.structural_segments(path)
            )
        )
        decompositions = (
            canonical,
            item_kind_view,
            path_view,
            *path_segments,
        )
        relations = (
            HotspotQueueDecompositionRelation(
                relation_kind=HotspotQueueDecompositionRelationKind.CANONICAL_OF,
                source=canonical,
                target=item_kind_view,
                rationale="item_kind_view",
            ),
            HotspotQueueDecompositionRelation(
                relation_kind=HotspotQueueDecompositionRelationKind.CANONICAL_OF,
                source=canonical,
                target=path_view,
                rationale="path_view",
            ),
            HotspotQueueDecompositionRelation(
                relation_kind=HotspotQueueDecompositionRelationKind.ALTERNATE_OF,
                source=item_kind_view,
                target=canonical,
                rationale="item_kind_view",
            ),
            HotspotQueueDecompositionRelation(
                relation_kind=HotspotQueueDecompositionRelationKind.ALTERNATE_OF,
                source=path_view,
                target=canonical,
                rationale="path_view",
            ),
            HotspotQueueDecompositionRelation(
                relation_kind=HotspotQueueDecompositionRelationKind.EQUIVALENT_UNDER,
                source=item_kind_view,
                target=canonical,
                rationale="item_kind",
            ),
            HotspotQueueDecompositionRelation(
                relation_kind=HotspotQueueDecompositionRelationKind.EQUIVALENT_UNDER,
                source=path_view,
                target=canonical,
                rationale="path",
            ),
            *(
                HotspotQueueDecompositionRelation(
                    relation_kind=HotspotQueueDecompositionRelationKind.DERIVED_FROM,
                    source=item,
                    target=path_view,
                    rationale="path_segment",
                )
                for item in path_segments
            ),
        )
        bundle = (decompositions, tuple(relations))
        self._decomposition_cache[origin] = bundle
        return bundle

    def item_id(
        self,
        *,
        item_kind: str,
        path: str,
        label: str = "",
    ) -> HotspotQueueIdentity:
        canonical = self._identity(
            namespace=HotspotQueueIdentityNamespace.ITEM,
            token=f"{item_kind}:{path}",
        )
        decompositions, relations = self._decomposition_bundle(
            origin=canonical,
            item_kind=item_kind,
            path=path,
        )
        return HotspotQueueIdentity(
            canonical=canonical,
            item_kind=item_kind,
            path=path,
            label=label,
            decompositions=decompositions,
            relations=relations,
        )


__all__ = [
    "HOTSPOT_QUEUE_ZONE",
    "HotspotQueueIdentity",
    "HotspotQueueIdentitySpace",
]
