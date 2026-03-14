from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
import re
from typing import Generic, TypeVar

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.timeout_context import (
    Deadline,
    deadline_clock_scope,
    deadline_scope,
)
from gabion.deadline_clock import MonotonicClock

NamespaceT = TypeVar("NamespaceT")
DecompositionKindT = TypeVar("DecompositionKindT")
RelationKindT = TypeVar("RelationKindT")


def identity_zone_text(value: object) -> str:
    match value:
        case IdentityZoneName():
            return value.value
        case _ if isinstance(getattr(value, "value", None), str):
            return str(getattr(value, "value"))
        case _:
            return str(value)


@dataclass(frozen=True, order=True)
class IdentityZoneName:
    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class IdentityAtom(Generic[NamespaceT]):
    atom_id: int
    namespace: NamespaceT = field(compare=False)
    token: str = field(compare=False)

    def wire(self) -> str:
        return self.token

    def __str__(self) -> str:
        return self.token


@dataclass(frozen=True, order=True)
class IdentityDecomposition(
    Generic[NamespaceT, DecompositionKindT],
):
    canonical: IdentityAtom[NamespaceT]
    decomposition_kind: DecompositionKindT = field(compare=False)
    origin: IdentityAtom[NamespaceT] = field(compare=False)
    label: str = field(compare=False, default="")
    part_index: int = field(compare=False, default=-1)

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.label or self.canonical.token

    def as_payload(self) -> dict[str, object]:
        return {
            "wire": self.canonical.token,
            "decomposition_kind": identity_zone_text(self.decomposition_kind),
            "origin_wire": self.origin.token,
            "origin_namespace": identity_zone_text(self.origin.namespace),
            "label": self.label or self.canonical.token,
            "part_index": self.part_index,
        }

    def to_payload(self) -> dict[str, object]:
        return self.as_payload()


@dataclass(frozen=True)
class IdentityDecompositionRelation(
    Generic[NamespaceT, DecompositionKindT, RelationKindT],
):
    relation_kind: RelationKindT
    source: IdentityDecomposition[NamespaceT, DecompositionKindT]
    target: IdentityDecomposition[NamespaceT, DecompositionKindT]
    rationale: str = ""

    def as_payload(self) -> dict[str, object]:
        return {
            "relation_kind": identity_zone_text(self.relation_kind),
            "source_wire": self.source.canonical.token,
            "target_wire": self.target.canonical.token,
            "source_kind": identity_zone_text(self.source.decomposition_kind),
            "target_kind": identity_zone_text(self.target.decomposition_kind),
            "rationale": self.rationale,
        }

    def to_payload(self) -> dict[str, object]:
        return self.as_payload()


@dataclass(frozen=True, order=True)
class IdentityCarrier(Generic[NamespaceT, DecompositionKindT, RelationKindT]):
    canonical: IdentityAtom[NamespaceT]
    zone_name: IdentityZoneName = field(compare=False)
    carrier_kind: str = field(compare=False, default="")
    label: str = field(compare=False, default="")
    decompositions: tuple[IdentityDecomposition[NamespaceT, DecompositionKindT], ...] = (
        field(default=(), compare=False)
    )
    relations: tuple[
        IdentityDecompositionRelation[NamespaceT, DecompositionKindT, RelationKindT],
        ...,
    ] = field(default=(), compare=False)
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.label or self.canonical.token

    def as_payload(self) -> dict[str, object]:
        return {
            "wire": self.wire(),
            "zone_name": self.zone_name.value,
            "carrier_kind": self.carrier_kind,
            "label": self.label or self.wire(),
            "decompositions": [item.as_payload() for item in self.decompositions],
            "relations": [item.as_payload() for item in self.relations],
            "metadata": dict(self.metadata),
        }

    def to_payload(self) -> dict[str, object]:
        return self.as_payload()


@dataclass
class IdentityLocalInterner(Generic[NamespaceT]):
    registry: PrimeRegistry = field(default_factory=PrimeRegistry)
    namespace_text: Callable[[NamespaceT], str] = identity_zone_text
    _adapter: PrimeIdentityAdapter = field(init=False, repr=False)
    _cache: dict[tuple[str, str], IdentityAtom[NamespaceT]] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        self._adapter = PrimeIdentityAdapter(registry=self.registry)

    @staticmethod
    def structural_segments(value: str) -> tuple[str, ...]:
        parts = [part for part in re.split(r"[:/._-]+", value.strip()) if part]
        seen: set[str] = set()
        ordered: list[str] = []
        for part in parts:
            if part in seen:
                continue
            seen.add(part)
            ordered.append(part)
        return tuple(ordered)

    def identity(self, *, namespace: NamespaceT, token: str) -> IdentityAtom[NamespaceT]:
        normalized = str(token).strip()
        namespace_key = self.namespace_text(namespace)
        cache_key = (namespace_key, normalized)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with ExitStack() as scope:
            scope.enter_context(deadline_clock_scope(MonotonicClock()))
            scope.enter_context(deadline_scope(Deadline.from_timeout_ms(60_000)))
            atom_id = self._adapter.get_or_assign(
                namespace=namespace_key,
                token=normalized,
            )
        created = IdentityAtom(
            atom_id=atom_id,
            namespace=namespace,
            token=normalized,
        )
        self._cache[cache_key] = created
        return created

    def decomposition_identity(
        self,
        *,
        origin: IdentityAtom[NamespaceT],
        decomposition_namespace: NamespaceT,
        decomposition_kind: DecompositionKindT,
        label: str,
        part_index: int = -1,
        canonical_kind: DecompositionKindT | None = None,
        token_builder: Callable[
            [IdentityAtom[NamespaceT], DecompositionKindT, str, int],
            str,
        ]
        | None = None,
    ) -> IdentityDecomposition[NamespaceT, DecompositionKindT]:
        if canonical_kind is not None and decomposition_kind == canonical_kind:
            return IdentityDecomposition(
                canonical=origin,
                decomposition_kind=decomposition_kind,
                origin=origin,
                label=origin.token,
                part_index=part_index,
            )
        normalized_label = label.strip()
        synthetic = self.identity(
            namespace=decomposition_namespace,
            token=(
                token_builder(origin, decomposition_kind, normalized_label, part_index)
                if token_builder is not None
                else "::".join(
                    (
                        self.namespace_text(origin.namespace),
                        origin.token,
                        identity_zone_text(decomposition_kind),
                        str(part_index),
                        normalized_label,
                    )
                )
            ),
        )
        return IdentityDecomposition(
            canonical=synthetic,
            decomposition_kind=decomposition_kind,
            origin=origin,
            label=normalized_label,
            part_index=part_index,
        )


__all__ = [
    "IdentityAtom",
    "IdentityCarrier",
    "IdentityDecomposition",
    "IdentityDecompositionRelation",
    "IdentityLocalInterner",
    "IdentityZoneName",
    "identity_zone_text",
]
