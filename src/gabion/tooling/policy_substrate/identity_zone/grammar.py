from __future__ import annotations

from dataclasses import dataclass, field

from gabion.tooling.policy_substrate.identity_zone.core import (
    IdentityCarrier,
    IdentityZoneName,
)
from gabion.tooling.policy_substrate.site_identity import stable_hash


def _dedupe_key(parts: tuple[object, ...]) -> str:
    return stable_hash(*parts)


@dataclass(frozen=True)
class IdentityKernelCongruence:
    kernel_congruence_id: str
    source_zone: str
    target_zone: str
    source_carrier_wire: str
    retained_decomposition_kinds: tuple[str, ...]
    erased_decomposition_kinds: tuple[str, ...]
    rationale: str = ""

    def as_payload(self) -> dict[str, object]:
        return {
            "kernel_congruence_id": self.kernel_congruence_id,
            "source_zone": self.source_zone,
            "target_zone": self.target_zone,
            "source_carrier_wire": self.source_carrier_wire,
            "retained_decomposition_kinds": list(self.retained_decomposition_kinds),
            "erased_decomposition_kinds": list(self.erased_decomposition_kinds),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class IdentityQuotientProjection:
    quotient_projection_id: str
    source_zone: str
    target_zone: str
    source_carrier_wire: str
    target_carrier_wire: str
    kernel_congruence_id: str
    rationale: str = ""

    def as_payload(self) -> dict[str, object]:
        return {
            "quotient_projection_id": self.quotient_projection_id,
            "source_zone": self.source_zone,
            "target_zone": self.target_zone,
            "source_carrier_wire": self.source_carrier_wire,
            "target_carrier_wire": self.target_carrier_wire,
            "kernel_congruence_id": self.kernel_congruence_id,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class IdentityReflectionFunctor:
    reflection_functor_id: str
    source_zone: str
    target_zone: str
    source_carrier_wire: str
    target_carrier_wire: str
    section_kind: str
    rationale: str = ""

    def as_payload(self) -> dict[str, object]:
        return {
            "reflection_functor_id": self.reflection_functor_id,
            "source_zone": self.source_zone,
            "target_zone": self.target_zone,
            "source_carrier_wire": self.source_carrier_wire,
            "target_carrier_wire": self.target_carrier_wire,
            "section_kind": self.section_kind,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class IdentityAdjointPair:
    adjoint_pair_id: str
    left_morphism_id: str
    right_morphism_id: str
    along_zone_boundary: str
    law_checks: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, object]:
        return {
            "adjoint_pair_id": self.adjoint_pair_id,
            "left_morphism_id": self.left_morphism_id,
            "right_morphism_id": self.right_morphism_id,
            "along_zone_boundary": self.along_zone_boundary,
            "law_checks": list(self.law_checks),
        }


@dataclass(frozen=True)
class IdentityFiberWitness:
    fiber_witness_id: str
    source_zone: str
    target_zone: str
    target_carrier_wire: str
    member_source_wires: tuple[str, ...]
    chosen_representative_wire: str = ""

    def as_payload(self) -> dict[str, object]:
        return {
            "fiber_witness_id": self.fiber_witness_id,
            "source_zone": self.source_zone,
            "target_zone": self.target_zone,
            "target_carrier_wire": self.target_carrier_wire,
            "member_source_wires": list(self.member_source_wires),
            "chosen_representative_wire": self.chosen_representative_wire,
        }


@dataclass(frozen=True)
class Identity2CellWitness:
    witness_id: str
    left_morphism_id: str
    right_morphism_id: str
    witness_kind: str
    rationale: str = ""

    def as_payload(self) -> dict[str, object]:
        return {
            "witness_id": self.witness_id,
            "left_morphism_id": self.left_morphism_id,
            "right_morphism_id": self.right_morphism_id,
            "witness_kind": self.witness_kind,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class IdentityZoneMorphism:
    morphism_id: str
    source_zone: str
    target_zone: str
    source_carrier_wire: str
    target_carrier_wire: str
    morphism_kind: str
    invertible: bool
    retained_decomposition_kinds: tuple[str, ...]
    erased_decomposition_kinds: tuple[str, ...]
    kernel_congruence_id: str = ""
    quotient_projection_id: str = ""
    reflection_functor_id: str = ""
    adjoint_pair_id: str = ""
    fiber_witness_ids: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, object]:
        return {
            "morphism_id": self.morphism_id,
            "source_zone": self.source_zone,
            "target_zone": self.target_zone,
            "source_carrier_wire": self.source_carrier_wire,
            "target_carrier_wire": self.target_carrier_wire,
            "morphism_kind": self.morphism_kind,
            "invertible": self.invertible,
            "retained_decomposition_kinds": list(self.retained_decomposition_kinds),
            "erased_decomposition_kinds": list(self.erased_decomposition_kinds),
            "kernel_congruence_id": self.kernel_congruence_id,
            "quotient_projection_id": self.quotient_projection_id,
            "reflection_functor_id": self.reflection_functor_id,
            "adjoint_pair_id": self.adjoint_pair_id,
            "fiber_witness_ids": list(self.fiber_witness_ids),
        }


@dataclass(frozen=True)
class IdentityGrammarBundle:
    zones: tuple[str, ...]
    carriers: tuple[IdentityCarrier[object, object, object], ...]
    morphisms: tuple[IdentityZoneMorphism, ...]
    kernel_congruences: tuple[IdentityKernelCongruence, ...]
    quotient_projections: tuple[IdentityQuotientProjection, ...]
    reflection_functors: tuple[IdentityReflectionFunctor, ...]
    adjoint_pairs: tuple[IdentityAdjointPair, ...]
    fibers: tuple[IdentityFiberWitness, ...]
    two_cells: tuple[Identity2CellWitness, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "zones": list(self.zones),
            "carriers": [item.as_payload() for item in self.carriers],
            "morphisms": [item.as_payload() for item in self.morphisms],
            "kernel_congruences": [
                item.as_payload() for item in self.kernel_congruences
            ],
            "quotient_projections": [
                item.as_payload() for item in self.quotient_projections
            ],
            "reflection_functors": [
                item.as_payload() for item in self.reflection_functors
            ],
            "adjoint_pairs": [item.as_payload() for item in self.adjoint_pairs],
            "fibers": [item.as_payload() for item in self.fibers],
            "two_cells": [item.as_payload() for item in self.two_cells],
        }


@dataclass
class HierarchicalIdentityGrammar:
    _zones: dict[str, IdentityZoneName] = field(default_factory=dict, init=False, repr=False)
    _carriers: dict[tuple[str, str], IdentityCarrier[object, object, object]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _morphisms: dict[str, IdentityZoneMorphism] = field(default_factory=dict, init=False, repr=False)
    _kernel_congruences: dict[str, IdentityKernelCongruence] = field(default_factory=dict, init=False, repr=False)
    _quotient_projections: dict[str, IdentityQuotientProjection] = field(default_factory=dict, init=False, repr=False)
    _reflection_functors: dict[str, IdentityReflectionFunctor] = field(default_factory=dict, init=False, repr=False)
    _adjoint_pairs: dict[str, IdentityAdjointPair] = field(default_factory=dict, init=False, repr=False)
    _fibers: dict[str, IdentityFiberWitness] = field(default_factory=dict, init=False, repr=False)
    _two_cells: dict[str, Identity2CellWitness] = field(default_factory=dict, init=False, repr=False)

    def add_zone(self, zone_name: IdentityZoneName | str) -> IdentityZoneName:
        zone = zone_name if isinstance(zone_name, IdentityZoneName) else IdentityZoneName(str(zone_name))
        self._zones[zone.value] = zone
        return zone

    def add_carrier(
        self,
        carrier: IdentityCarrier[object, object, object],
    ) -> IdentityCarrier[object, object, object]:
        self.add_zone(carrier.zone_name)
        key = (carrier.zone_name.value, carrier.wire())
        cached = self._carriers.get(key)
        if cached is not None:
            return cached
        self._carriers[key] = carrier
        return carrier

    def _remember_by_signature[T](self, store: dict[str, T], key_parts: tuple[object, ...], value_builder) -> T:
        signature = _dedupe_key(key_parts)
        cached = store.get(signature)
        if cached is not None:
            return cached
        value = value_builder(signature)
        store[signature] = value
        return value

    def add_kernel_congruence(
        self,
        *,
        source_zone: str,
        target_zone: str,
        source_carrier_wire: str,
        retained_decomposition_kinds: tuple[str, ...],
        erased_decomposition_kinds: tuple[str, ...],
        rationale: str = "",
    ) -> IdentityKernelCongruence:
        return self._remember_by_signature(
            self._kernel_congruences,
            (
                "kernel_congruence",
                source_zone,
                target_zone,
                source_carrier_wire,
                retained_decomposition_kinds,
                erased_decomposition_kinds,
                rationale,
            ),
            lambda signature: IdentityKernelCongruence(
                kernel_congruence_id=f"identity_kernel_congruence:{signature}",
                source_zone=source_zone,
                target_zone=target_zone,
                source_carrier_wire=source_carrier_wire,
                retained_decomposition_kinds=retained_decomposition_kinds,
                erased_decomposition_kinds=erased_decomposition_kinds,
                rationale=rationale,
            ),
        )

    def add_quotient_projection(
        self,
        *,
        source_zone: str,
        target_zone: str,
        source_carrier_wire: str,
        target_carrier_wire: str,
        kernel_congruence_id: str,
        rationale: str = "",
    ) -> IdentityQuotientProjection:
        return self._remember_by_signature(
            self._quotient_projections,
            (
                "quotient_projection",
                source_zone,
                target_zone,
                source_carrier_wire,
                target_carrier_wire,
                kernel_congruence_id,
                rationale,
            ),
            lambda signature: IdentityQuotientProjection(
                quotient_projection_id=f"identity_quotient_projection:{signature}",
                source_zone=source_zone,
                target_zone=target_zone,
                source_carrier_wire=source_carrier_wire,
                target_carrier_wire=target_carrier_wire,
                kernel_congruence_id=kernel_congruence_id,
                rationale=rationale,
            ),
        )

    def add_reflection_functor(
        self,
        *,
        source_zone: str,
        target_zone: str,
        source_carrier_wire: str,
        target_carrier_wire: str,
        section_kind: str,
        rationale: str = "",
    ) -> IdentityReflectionFunctor:
        return self._remember_by_signature(
            self._reflection_functors,
            (
                "reflection_functor",
                source_zone,
                target_zone,
                source_carrier_wire,
                target_carrier_wire,
                section_kind,
                rationale,
            ),
            lambda signature: IdentityReflectionFunctor(
                reflection_functor_id=f"identity_reflection_functor:{signature}",
                source_zone=source_zone,
                target_zone=target_zone,
                source_carrier_wire=source_carrier_wire,
                target_carrier_wire=target_carrier_wire,
                section_kind=section_kind,
                rationale=rationale,
            ),
        )

    def add_adjoint_pair(
        self,
        *,
        left_morphism_id: str,
        right_morphism_id: str,
        along_zone_boundary: str,
        law_checks: tuple[str, ...] = (),
    ) -> IdentityAdjointPair:
        return self._remember_by_signature(
            self._adjoint_pairs,
            (
                "adjoint_pair",
                left_morphism_id,
                right_morphism_id,
                along_zone_boundary,
                law_checks,
            ),
            lambda signature: IdentityAdjointPair(
                adjoint_pair_id=f"identity_adjoint_pair:{signature}",
                left_morphism_id=left_morphism_id,
                right_morphism_id=right_morphism_id,
                along_zone_boundary=along_zone_boundary,
                law_checks=law_checks,
            ),
        )

    def add_fiber_witness(
        self,
        *,
        source_zone: str,
        target_zone: str,
        target_carrier_wire: str,
        member_source_wires: tuple[str, ...],
        chosen_representative_wire: str = "",
    ) -> IdentityFiberWitness:
        ordered_members = tuple(sorted(set(member_source_wires)))
        return self._remember_by_signature(
            self._fibers,
            (
                "fiber_witness",
                source_zone,
                target_zone,
                target_carrier_wire,
                ordered_members,
                chosen_representative_wire,
            ),
            lambda signature: IdentityFiberWitness(
                fiber_witness_id=f"identity_fiber_witness:{signature}",
                source_zone=source_zone,
                target_zone=target_zone,
                target_carrier_wire=target_carrier_wire,
                member_source_wires=ordered_members,
                chosen_representative_wire=chosen_representative_wire,
            ),
        )

    def add_zone_morphism(
        self,
        *,
        source_zone: str,
        target_zone: str,
        source_carrier_wire: str,
        target_carrier_wire: str,
        morphism_kind: str,
        invertible: bool,
        retained_decomposition_kinds: tuple[str, ...],
        erased_decomposition_kinds: tuple[str, ...],
        kernel_congruence_id: str = "",
        quotient_projection_id: str = "",
        reflection_functor_id: str = "",
        adjoint_pair_id: str = "",
        fiber_witness_ids: tuple[str, ...] = (),
    ) -> IdentityZoneMorphism:
        return self._remember_by_signature(
            self._morphisms,
            (
                "zone_morphism",
                source_zone,
                target_zone,
                source_carrier_wire,
                target_carrier_wire,
                morphism_kind,
                invertible,
                retained_decomposition_kinds,
                erased_decomposition_kinds,
                kernel_congruence_id,
                quotient_projection_id,
                reflection_functor_id,
                adjoint_pair_id,
                fiber_witness_ids,
            ),
            lambda signature: IdentityZoneMorphism(
                morphism_id=f"identity_zone_morphism:{signature}",
                source_zone=source_zone,
                target_zone=target_zone,
                source_carrier_wire=source_carrier_wire,
                target_carrier_wire=target_carrier_wire,
                morphism_kind=morphism_kind,
                invertible=invertible,
                retained_decomposition_kinds=retained_decomposition_kinds,
                erased_decomposition_kinds=erased_decomposition_kinds,
                kernel_congruence_id=kernel_congruence_id,
                quotient_projection_id=quotient_projection_id,
                reflection_functor_id=reflection_functor_id,
                adjoint_pair_id=adjoint_pair_id,
                fiber_witness_ids=fiber_witness_ids,
            ),
        )

    def add_two_cell(
        self,
        *,
        left_morphism_id: str,
        right_morphism_id: str,
        witness_kind: str,
        rationale: str = "",
    ) -> Identity2CellWitness:
        return self._remember_by_signature(
            self._two_cells,
            ("two_cell", left_morphism_id, right_morphism_id, witness_kind, rationale),
            lambda signature: Identity2CellWitness(
                witness_id=f"identity_two_cell:{signature}",
                left_morphism_id=left_morphism_id,
                right_morphism_id=right_morphism_id,
                witness_kind=witness_kind,
                rationale=rationale,
            ),
        )

    def bundle(self) -> IdentityGrammarBundle:
        return IdentityGrammarBundle(
            zones=tuple(sorted(self._zones)),
            carriers=tuple(self._carriers.values()),
            morphisms=tuple(self._morphisms.values()),
            kernel_congruences=tuple(self._kernel_congruences.values()),
            quotient_projections=tuple(self._quotient_projections.values()),
            reflection_functors=tuple(self._reflection_functors.values()),
            adjoint_pairs=tuple(self._adjoint_pairs.values()),
            fibers=tuple(self._fibers.values()),
            two_cells=tuple(self._two_cells.values()),
        )


__all__ = [
    "HierarchicalIdentityGrammar",
    "Identity2CellWitness",
    "IdentityAdjointPair",
    "IdentityFiberWitness",
    "IdentityGrammarBundle",
    "IdentityKernelCongruence",
    "IdentityQuotientProjection",
    "IdentityReflectionFunctor",
    "IdentityZoneMorphism",
]
