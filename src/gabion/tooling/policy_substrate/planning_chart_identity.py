from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.tooling.policy_substrate.identity_zone import (
    HierarchicalIdentityGrammar,
    IdentityAtom,
    IdentityCarrier,
    IdentityDecomposition,
    IdentityDecompositionRelation,
    IdentityGrammarBundle,
    IdentityLocalInterner,
    IdentityZoneName,
)

if TYPE_CHECKING:
    from gabion.tooling.policy_substrate.planning_chart import (
        PlanningChartItem,
        PlanningChartSummary,
    )


class PlanningChartIdentityNamespace(StrEnum):
    ITEM = "planning_chart.item"
    ANCHOR = "planning_chart.anchor"
    DECOMPOSITION = "planning_chart.decomposition"


class PlanningChartDecompositionKind(StrEnum):
    CANONICAL = "canonical"
    PHASE_KIND = "phase_kind"
    ITEM_KIND = "item_kind"
    SOURCE_KIND = "source_kind"
    TRACKED_REF = "tracked_ref"
    TRACKED_REF_SEGMENT = "tracked_ref_segment"


class PlanningChartDecompositionRelationKind(StrEnum):
    CANONICAL_OF = "canonical_of"
    ALTERNATE_OF = "alternate_of"
    EQUIVALENT_UNDER = "equivalent_under"
    DERIVED_FROM = "derived_from"


PLANNING_CHART_ZONE = IdentityZoneName("planning_chart")
PLANNING_EXTERNAL_ANCHOR_ZONE = IdentityZoneName("planning_external_anchor")

_PlanningAtom = IdentityAtom[PlanningChartIdentityNamespace]
PlanningChartDecompositionIdentity = IdentityDecomposition[
    PlanningChartIdentityNamespace,
    PlanningChartDecompositionKind,
]
PlanningChartDecompositionRelation = IdentityDecompositionRelation[
    PlanningChartIdentityNamespace,
    PlanningChartDecompositionKind,
    PlanningChartDecompositionRelationKind,
]


@dataclass(frozen=True, order=True)
class PlanningChartIdentity:
    canonical: _PlanningAtom
    item_kind: str = field(compare=False)
    label: str = field(compare=False, default="")
    zone_name: IdentityZoneName = field(default=PLANNING_CHART_ZONE, compare=False)
    decompositions: tuple[PlanningChartDecompositionIdentity, ...] = field(
        default=(),
        compare=False,
    )
    relations: tuple[PlanningChartDecompositionRelation, ...] = field(
        default=(),
        compare=False,
    )
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def wire(self) -> str:
        return self.canonical.token

    def as_carrier(
        self,
    ) -> IdentityCarrier[
        PlanningChartIdentityNamespace,
        PlanningChartDecompositionKind,
        PlanningChartDecompositionRelationKind,
    ]:
        return IdentityCarrier(
            canonical=self.canonical,
            zone_name=self.zone_name,
            carrier_kind=self.item_kind,
            label=self.label or self.wire(),
            decompositions=self.decompositions,
            relations=self.relations,
            metadata=self.metadata,
        )


@dataclass
class PlanningChartIdentitySpace:
    registry: PrimeRegistry = field(default_factory=PrimeRegistry)
    _interner: IdentityLocalInterner[PlanningChartIdentityNamespace] = field(
        init=False,
        repr=False,
    )
    _decomposition_cache: dict[
        _PlanningAtom,
        tuple[
            tuple[PlanningChartDecompositionIdentity, ...],
            tuple[PlanningChartDecompositionRelation, ...],
        ],
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._interner = IdentityLocalInterner(registry=self.registry)

    def _identity(
        self,
        *,
        namespace: PlanningChartIdentityNamespace,
        token: str,
    ) -> _PlanningAtom:
        return self._interner.identity(namespace=namespace, token=token)

    def _decomposition_identity(
        self,
        *,
        origin: _PlanningAtom,
        decomposition_kind: PlanningChartDecompositionKind,
        label: str,
        part_index: int = -1,
    ) -> PlanningChartDecompositionIdentity:
        return self._interner.decomposition_identity(
            origin=origin,
            decomposition_namespace=PlanningChartIdentityNamespace.DECOMPOSITION,
            decomposition_kind=decomposition_kind,
            label=label,
            part_index=part_index,
            canonical_kind=PlanningChartDecompositionKind.CANONICAL,
        )

    def _decomposition_bundle(
        self,
        *,
        origin: _PlanningAtom,
        phase_kind: str,
        item_kind: str,
        source_kind: str,
        tracked_refs: tuple[str, ...],
    ) -> tuple[
        tuple[PlanningChartDecompositionIdentity, ...],
        tuple[PlanningChartDecompositionRelation, ...],
    ]:
        cached = self._decomposition_cache.get(origin)
        if cached is not None:
            return cached
        canonical = self._decomposition_identity(
            origin=origin,
            decomposition_kind=PlanningChartDecompositionKind.CANONICAL,
            label=origin.token,
        )
        phase_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=PlanningChartDecompositionKind.PHASE_KIND,
            label=phase_kind,
        )
        item_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=PlanningChartDecompositionKind.ITEM_KIND,
            label=item_kind,
        )
        source_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=PlanningChartDecompositionKind.SOURCE_KIND,
            label=source_kind,
        )
        tracked_views = tuple(
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=PlanningChartDecompositionKind.TRACKED_REF,
                label=tracked_ref,
            )
            for tracked_ref in tracked_refs
        )
        tracked_segments = tuple(
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=PlanningChartDecompositionKind.TRACKED_REF_SEGMENT,
                label=segment,
                part_index=index,
            )
            for tracked_ref in tracked_refs
            for index, segment in enumerate(
                IdentityLocalInterner.structural_segments(tracked_ref)
            )
        )
        decompositions = (
            canonical,
            phase_view,
            item_view,
            source_view,
            *tracked_views,
            *tracked_segments,
        )
        relations = (
            PlanningChartDecompositionRelation(
                relation_kind=PlanningChartDecompositionRelationKind.CANONICAL_OF,
                source=canonical,
                target=phase_view,
                rationale="phase_kind",
            ),
            PlanningChartDecompositionRelation(
                relation_kind=PlanningChartDecompositionRelationKind.CANONICAL_OF,
                source=canonical,
                target=item_view,
                rationale="item_kind",
            ),
            PlanningChartDecompositionRelation(
                relation_kind=PlanningChartDecompositionRelationKind.CANONICAL_OF,
                source=canonical,
                target=source_view,
                rationale="source_kind",
            ),
            PlanningChartDecompositionRelation(
                relation_kind=PlanningChartDecompositionRelationKind.ALTERNATE_OF,
                source=phase_view,
                target=canonical,
                rationale="phase_kind",
            ),
            PlanningChartDecompositionRelation(
                relation_kind=PlanningChartDecompositionRelationKind.ALTERNATE_OF,
                source=item_view,
                target=canonical,
                rationale="item_kind",
            ),
            PlanningChartDecompositionRelation(
                relation_kind=PlanningChartDecompositionRelationKind.ALTERNATE_OF,
                source=source_view,
                target=canonical,
                rationale="source_kind",
            ),
            *(
                PlanningChartDecompositionRelation(
                    relation_kind=PlanningChartDecompositionRelationKind.EQUIVALENT_UNDER,
                    source=tracked_view,
                    target=canonical,
                    rationale="tracked_ref",
                )
                for tracked_view in tracked_views
            ),
            *(
                PlanningChartDecompositionRelation(
                    relation_kind=PlanningChartDecompositionRelationKind.DERIVED_FROM,
                    source=segment_view,
                    target=tracked_view,
                    rationale="tracked_ref_segment",
                )
                for tracked_view in tracked_views
                for segment_view in tracked_segments
                if tracked_view.label and segment_view.label in tracked_view.label
            ),
        )
        bundle = (decompositions, tuple(relations))
        self._decomposition_cache[origin] = bundle
        return bundle

    def item_carrier(
        self,
        item: PlanningChartItem,
    ) -> IdentityCarrier[
        PlanningChartIdentityNamespace,
        PlanningChartDecompositionKind,
        PlanningChartDecompositionRelationKind,
    ]:
        tracked_refs = tuple((*item.tracked_node_ids, *item.tracked_object_ids))
        canonical = self._identity(
            namespace=PlanningChartIdentityNamespace.ITEM,
            token=item.item_id,
        )
        decompositions, relations = self._decomposition_bundle(
            origin=canonical,
            phase_kind=item.phase_kind,
            item_kind=item.item_kind,
            source_kind=item.source_kind,
            tracked_refs=tracked_refs,
        )
        return PlanningChartIdentity(
            canonical=canonical,
            item_kind=item.item_kind,
            label=item.title or item.item_id,
            zone_name=PLANNING_CHART_ZONE,
            decompositions=decompositions,
            relations=relations,
            metadata={
                "phase_kind": item.phase_kind,
                "source_kind": item.source_kind,
                "status_hint": item.status_hint,
                "selection_rank": item.selection_rank,
                "tracked_node_ids": item.tracked_node_ids,
                "tracked_object_ids": item.tracked_object_ids,
                "selected": item.selected,
            },
        ).as_carrier()

    def anchor_carrier(
        self,
        *,
        anchor_kind: str,
        anchor_ref: str,
    ) -> IdentityCarrier[
        PlanningChartIdentityNamespace,
        PlanningChartDecompositionKind,
        PlanningChartDecompositionRelationKind,
    ]:
        canonical = self._identity(
            namespace=PlanningChartIdentityNamespace.ANCHOR,
            token=f"{anchor_kind}:{anchor_ref}",
        )
        return IdentityCarrier(
            canonical=canonical,
            zone_name=PLANNING_EXTERNAL_ANCHOR_ZONE,
            carrier_kind=anchor_kind,
            label=anchor_ref,
            metadata={"anchor_kind": anchor_kind, "anchor_ref": anchor_ref},
        )


def build_planning_chart_identity_grammar(
    *,
    summary: PlanningChartSummary,
    resolved_carriers: Mapping[str, IdentityCarrier[object, object, object]] | None = None,
) -> IdentityGrammarBundle:
    grammar = HierarchicalIdentityGrammar()
    space = PlanningChartIdentitySpace()
    resolved = resolved_carriers or {}
    for phase in summary.phases:
        for item in phase.items:
            item_carrier = space.item_carrier(item)
            grammar.add_carrier(item_carrier)
            for tracked_node_id in item.tracked_node_ids:
                target_carrier = resolved.get(tracked_node_id)
                if target_carrier is not None:
                    grammar.add_carrier(target_carrier)
                    grammar.add_zone_morphism(
                        source_zone=PLANNING_CHART_ZONE.value,
                        target_zone=target_carrier.zone_name.value,
                        source_carrier_wire=item_carrier.wire(),
                        target_carrier_wire=target_carrier.wire(),
                        morphism_kind="tracks",
                        invertible=False,
                        retained_decomposition_kinds=("tracked_ref",),
                        erased_decomposition_kinds=(),
                    )
                    continue
                anchor = space.anchor_carrier(
                    anchor_kind="tracked_node",
                    anchor_ref=tracked_node_id,
                )
                grammar.add_carrier(anchor)
                grammar.add_zone_morphism(
                    source_zone=PLANNING_CHART_ZONE.value,
                    target_zone=PLANNING_EXTERNAL_ANCHOR_ZONE.value,
                    source_carrier_wire=item_carrier.wire(),
                    target_carrier_wire=anchor.wire(),
                    morphism_kind="derived_from",
                    invertible=False,
                    retained_decomposition_kinds=("tracked_ref",),
                    erased_decomposition_kinds=(),
                )
            for tracked_object_id in item.tracked_object_ids:
                target_carrier = resolved.get(tracked_object_id)
                if target_carrier is not None:
                    grammar.add_carrier(target_carrier)
                    grammar.add_zone_morphism(
                        source_zone=PLANNING_CHART_ZONE.value,
                        target_zone=target_carrier.zone_name.value,
                        source_carrier_wire=item_carrier.wire(),
                        target_carrier_wire=target_carrier.wire(),
                        morphism_kind="tracks",
                        invertible=False,
                        retained_decomposition_kinds=("tracked_ref",),
                        erased_decomposition_kinds=(),
                    )
                    continue
                anchor = space.anchor_carrier(
                    anchor_kind="tracked_object",
                    anchor_ref=tracked_object_id,
                )
                grammar.add_carrier(anchor)
                grammar.add_zone_morphism(
                    source_zone=PLANNING_CHART_ZONE.value,
                    target_zone=PLANNING_EXTERNAL_ANCHOR_ZONE.value,
                    source_carrier_wire=item_carrier.wire(),
                    target_carrier_wire=anchor.wire(),
                    morphism_kind="derived_from",
                    invertible=False,
                    retained_decomposition_kinds=("tracked_ref",),
                    erased_decomposition_kinds=(),
                )
    return grammar.bundle()


__all__ = [
    "PLANNING_CHART_ZONE",
    "PLANNING_EXTERNAL_ANCHOR_ZONE",
    "PlanningChartIdentitySpace",
    "build_planning_chart_identity_grammar",
]
