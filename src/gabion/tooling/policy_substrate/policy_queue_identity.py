from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from contextlib import ExitStack
from typing import TypeVar

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.timeout_context import (
    Deadline,
    deadline_clock_scope,
    deadline_scope,
)
from gabion.deadline_clock import MonotonicClock


class PolicyQueueIdentityNamespace(StrEnum):
    WORKSTREAM = "policy_queue.workstream"
    SUBQUEUE = "policy_queue.subqueue"
    TOUCHPOINT = "policy_queue.touchpoint"
    TOUCHSITE = "policy_queue.touchsite"
    SITE_REF = "policy_queue.site_ref"
    STRUCTURAL_REF = "policy_queue.structural_ref"
    ARTIFACT_NODE = "policy_queue.artifact_node"
    DECOMPOSITION = "policy_queue.decomposition"


class PolicyQueueDecompositionKind(StrEnum):
    CANONICAL = "canonical"
    NAMESPACE = "namespace"
    TOKEN = "token"
    NAMESPACE_SEGMENT = "namespace_segment"
    TOKEN_SEGMENT = "token_segment"


class PolicyQueueDecompositionRelationKind(StrEnum):
    CANONICAL_OF = "canonical_of"
    ALTERNATE_OF = "alternate_of"
    EQUIVALENT_UNDER = "equivalent_under"
    DERIVED_FROM = "derived_from"


@dataclass(frozen=True, order=True)
class _PrimeBackedIdentity:
    atom_id: int
    namespace: PolicyQueueIdentityNamespace = field(compare=False)
    token: str = field(compare=False)

    def wire(self) -> str:
        return self.token

    def __str__(self) -> str:
        return self.token


@dataclass(frozen=True, order=True)
class PolicyQueueDecompositionIdentity:
    canonical: _PrimeBackedIdentity
    decomposition_kind: PolicyQueueDecompositionKind = field(compare=False)
    origin: _PrimeBackedIdentity = field(compare=False)
    label: str = field(compare=False, default="")
    part_index: int = field(compare=False, default=-1)

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.label or self.canonical.token

    def as_payload(self) -> dict[str, object]:
        return {
            "wire": self.canonical.token,
            "decomposition_kind": self.decomposition_kind.value,
            "origin_wire": self.origin.token,
            "origin_namespace": self.origin.namespace.value,
            "label": self.label or self.canonical.token,
            "part_index": self.part_index,
        }


@dataclass(frozen=True)
class PolicyQueueDecompositionRelation:
    relation_kind: PolicyQueueDecompositionRelationKind
    source: PolicyQueueDecompositionIdentity
    target: PolicyQueueDecompositionIdentity
    rationale: str = ""

    def as_payload(self) -> dict[str, object]:
        return {
            "relation_kind": self.relation_kind.value,
            "source_wire": self.source.canonical.token,
            "target_wire": self.target.canonical.token,
            "source_kind": self.source.decomposition_kind.value,
            "target_kind": self.target.decomposition_kind.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, order=True)
class WorkstreamId:
    canonical: _PrimeBackedIdentity
    decompositions: tuple[PolicyQueueDecompositionIdentity, ...] = field(
        default=(),
        compare=False,
    )
    relations: tuple[PolicyQueueDecompositionRelation, ...] = field(
        default=(),
        compare=False,
    )

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.canonical.token


@dataclass(frozen=True, order=True)
class SubqueueId:
    canonical: _PrimeBackedIdentity
    decompositions: tuple[PolicyQueueDecompositionIdentity, ...] = field(
        default=(),
        compare=False,
    )
    relations: tuple[PolicyQueueDecompositionRelation, ...] = field(
        default=(),
        compare=False,
    )

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.canonical.token


@dataclass(frozen=True, order=True)
class TouchpointId:
    canonical: _PrimeBackedIdentity
    decompositions: tuple[PolicyQueueDecompositionIdentity, ...] = field(
        default=(),
        compare=False,
    )
    relations: tuple[PolicyQueueDecompositionRelation, ...] = field(
        default=(),
        compare=False,
    )

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.canonical.token


@dataclass(frozen=True, order=True)
class TouchsiteId:
    canonical: _PrimeBackedIdentity
    decompositions: tuple[PolicyQueueDecompositionIdentity, ...] = field(
        default=(),
        compare=False,
    )
    relations: tuple[PolicyQueueDecompositionRelation, ...] = field(
        default=(),
        compare=False,
    )

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.canonical.token


@dataclass(frozen=True, order=True)
class SiteReferenceId:
    canonical: _PrimeBackedIdentity
    decompositions: tuple[PolicyQueueDecompositionIdentity, ...] = field(
        default=(),
        compare=False,
    )
    relations: tuple[PolicyQueueDecompositionRelation, ...] = field(
        default=(),
        compare=False,
    )

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.canonical.token


@dataclass(frozen=True, order=True)
class StructuralReferenceId:
    canonical: _PrimeBackedIdentity
    decompositions: tuple[PolicyQueueDecompositionIdentity, ...] = field(
        default=(),
        compare=False,
    )
    relations: tuple[PolicyQueueDecompositionRelation, ...] = field(
        default=(),
        compare=False,
    )

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        return self.canonical.token


_IdentityCarrierT = TypeVar(
    "_IdentityCarrierT",
    WorkstreamId,
    SubqueueId,
    TouchpointId,
    TouchsiteId,
    SiteReferenceId,
    StructuralReferenceId,
)


@dataclass
class PolicyQueueIdentitySpace:
    registry: PrimeRegistry = field(default_factory=PrimeRegistry)
    _adapter: PrimeIdentityAdapter = field(init=False, repr=False)
    _cache: dict[
        tuple[PolicyQueueIdentityNamespace, str],
        _PrimeBackedIdentity,
    ] = field(init=False, repr=False, default_factory=dict)
    _decomposition_cache: dict[
        tuple[PolicyQueueIdentityNamespace, str],
        tuple[
            tuple[PolicyQueueDecompositionIdentity, ...],
            tuple[PolicyQueueDecompositionRelation, ...],
        ],
    ] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._adapter = PrimeIdentityAdapter(registry=self.registry)

    @staticmethod
    def _structural_segments(value: str) -> tuple[str, ...]:
        parts = [
            part
            for part in re.split(r"[:/._-]+", value.strip())
            if part
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for part in parts:
            if part in seen:
                continue
            seen.add(part)
            ordered.append(part)
        return tuple(ordered)

    def _identity(
        self,
        *,
        namespace: PolicyQueueIdentityNamespace,
        token: str,
    ) -> _PrimeBackedIdentity:
        normalized = str(token).strip()
        cache_key = (namespace, normalized)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with ExitStack() as scope:
            scope.enter_context(deadline_clock_scope(MonotonicClock()))
            scope.enter_context(deadline_scope(Deadline.from_timeout_ms(60_000)))
            atom_id = self._adapter.get_or_assign(
                namespace=str(namespace),
                token=normalized,
            )
        identity = _PrimeBackedIdentity(
            atom_id=atom_id,
            namespace=namespace,
            token=normalized,
        )
        self._cache[cache_key] = identity
        return identity

    def _decomposition_identity(
        self,
        *,
        origin: _PrimeBackedIdentity,
        decomposition_kind: PolicyQueueDecompositionKind,
        label: str,
        part_index: int = -1,
    ) -> PolicyQueueDecompositionIdentity:
        if decomposition_kind is PolicyQueueDecompositionKind.CANONICAL:
            return PolicyQueueDecompositionIdentity(
                canonical=origin,
                decomposition_kind=decomposition_kind,
                origin=origin,
                label=origin.token,
                part_index=part_index,
            )
        synthetic = self._identity(
            namespace=PolicyQueueIdentityNamespace.DECOMPOSITION,
            token="::".join(
                (
                    str(origin.namespace),
                    origin.token,
                    str(decomposition_kind),
                    str(part_index),
                    label.strip(),
                )
            ),
        )
        return PolicyQueueDecompositionIdentity(
            canonical=synthetic,
            decomposition_kind=decomposition_kind,
            origin=origin,
            label=label.strip(),
            part_index=part_index,
        )

    def _decomposition_bundle(
        self,
        *,
        origin: _PrimeBackedIdentity,
    ) -> tuple[
        tuple[PolicyQueueDecompositionIdentity, ...],
        tuple[PolicyQueueDecompositionRelation, ...],
    ]:
        cache_key = (origin.namespace, origin.token)
        cached = self._decomposition_cache.get(cache_key)
        if cached is not None:
            return cached

        canonical = self._decomposition_identity(
            origin=origin,
            decomposition_kind=PolicyQueueDecompositionKind.CANONICAL,
            label=origin.token,
        )
        namespace_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=PolicyQueueDecompositionKind.NAMESPACE,
            label=str(origin.namespace),
        )
        token_view = self._decomposition_identity(
            origin=origin,
            decomposition_kind=PolicyQueueDecompositionKind.TOKEN,
            label=origin.token,
        )
        namespace_segments = tuple(
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=PolicyQueueDecompositionKind.NAMESPACE_SEGMENT,
                label=segment,
                part_index=index,
            )
            for index, segment in enumerate(
                self._structural_segments(str(origin.namespace))
            )
        )
        token_segments = tuple(
            self._decomposition_identity(
                origin=origin,
                decomposition_kind=PolicyQueueDecompositionKind.TOKEN_SEGMENT,
                label=segment,
                part_index=index,
            )
            for index, segment in enumerate(self._structural_segments(origin.token))
        )
        decompositions = (
            canonical,
            namespace_view,
            token_view,
            *namespace_segments,
            *token_segments,
        )
        relations = (
            PolicyQueueDecompositionRelation(
                relation_kind=PolicyQueueDecompositionRelationKind.CANONICAL_OF,
                source=canonical,
                target=namespace_view,
                rationale="namespace_view",
            ),
            PolicyQueueDecompositionRelation(
                relation_kind=PolicyQueueDecompositionRelationKind.ALTERNATE_OF,
                source=namespace_view,
                target=canonical,
                rationale="namespace_view",
            ),
            PolicyQueueDecompositionRelation(
                relation_kind=PolicyQueueDecompositionRelationKind.EQUIVALENT_UNDER,
                source=namespace_view,
                target=canonical,
                rationale="origin_namespace",
            ),
            PolicyQueueDecompositionRelation(
                relation_kind=PolicyQueueDecompositionRelationKind.CANONICAL_OF,
                source=canonical,
                target=token_view,
                rationale="token_view",
            ),
            PolicyQueueDecompositionRelation(
                relation_kind=PolicyQueueDecompositionRelationKind.ALTERNATE_OF,
                source=token_view,
                target=canonical,
                rationale="token_view",
            ),
            PolicyQueueDecompositionRelation(
                relation_kind=PolicyQueueDecompositionRelationKind.EQUIVALENT_UNDER,
                source=token_view,
                target=canonical,
                rationale="origin_token",
            ),
            *(
                PolicyQueueDecompositionRelation(
                    relation_kind=PolicyQueueDecompositionRelationKind.DERIVED_FROM,
                    source=item,
                    target=namespace_view,
                    rationale="namespace_segment",
                )
                for item in namespace_segments
            ),
            *(
                PolicyQueueDecompositionRelation(
                    relation_kind=PolicyQueueDecompositionRelationKind.DERIVED_FROM,
                    source=item,
                    target=token_view,
                    rationale="token_segment",
                )
                for item in token_segments
            ),
        )
        bundle = (decompositions, tuple(relations))
        self._decomposition_cache[cache_key] = bundle
        return bundle

    def _wrap_identity(
        self,
        *,
        canonical: _PrimeBackedIdentity,
        constructor: type[_IdentityCarrierT],
    ) -> _IdentityCarrierT:
        decompositions, relations = self._decomposition_bundle(origin=canonical)
        return constructor(
            canonical=canonical,
            decompositions=decompositions,
            relations=relations,
        )

    def workstream_id(self, token: str) -> WorkstreamId:
        return self._wrap_identity(
            canonical=self._identity(
                namespace=PolicyQueueIdentityNamespace.WORKSTREAM,
                token=token,
            ),
            constructor=WorkstreamId,
        )

    def subqueue_id(self, token: str) -> SubqueueId:
        return self._wrap_identity(
            canonical=self._identity(
                namespace=PolicyQueueIdentityNamespace.SUBQUEUE,
                token=token,
            ),
            constructor=SubqueueId,
        )

    def touchpoint_id(self, token: str) -> TouchpointId:
        return self._wrap_identity(
            canonical=self._identity(
                namespace=PolicyQueueIdentityNamespace.TOUCHPOINT,
                token=token,
            ),
            constructor=TouchpointId,
        )

    def touchsite_id(self, token: str) -> TouchsiteId:
        return self._wrap_identity(
            canonical=self._identity(
                namespace=PolicyQueueIdentityNamespace.TOUCHSITE,
                token=token,
            ),
            constructor=TouchsiteId,
        )

    def site_ref_id(self, token: str) -> SiteReferenceId:
        return self._wrap_identity(
            canonical=self._identity(
                namespace=PolicyQueueIdentityNamespace.SITE_REF,
                token=token,
            ),
            constructor=SiteReferenceId,
        )

    def structural_ref_id(self, token: str) -> StructuralReferenceId:
        return self._wrap_identity(
            canonical=self._identity(
                namespace=PolicyQueueIdentityNamespace.STRUCTURAL_REF,
                token=token,
            ),
            constructor=StructuralReferenceId,
        )


def policy_queue_identity_view_payload(
    value: WorkstreamId
    | SubqueueId
    | TouchpointId
    | TouchsiteId
    | SiteReferenceId
    | StructuralReferenceId,
) -> dict[str, object]:
    return {
        "wire": value.wire(),
        "decompositions": [item.as_payload() for item in value.decompositions],
        "relations": [item.as_payload() for item in value.relations],
    }


def encode_policy_queue_identity(
    value: WorkstreamId
    | SubqueueId
    | TouchpointId
    | TouchsiteId
    | SiteReferenceId
    | StructuralReferenceId
    | PolicyQueueDecompositionIdentity
    | str,
) -> str:
    match value:
        case WorkstreamId() | SubqueueId() | TouchpointId() | TouchsiteId():
            return value.wire()
        case SiteReferenceId() | StructuralReferenceId():
            return value.wire()
        case PolicyQueueDecompositionIdentity():
            return value.wire()
        case str() as text:
            return text
        case _:
            return str(value)


__all__ = [
    "PolicyQueueIdentityNamespace",
    "PolicyQueueIdentitySpace",
    "PolicyQueueDecompositionIdentity",
    "PolicyQueueDecompositionKind",
    "PolicyQueueDecompositionRelation",
    "PolicyQueueDecompositionRelationKind",
    "SiteReferenceId",
    "StructuralReferenceId",
    "SubqueueId",
    "TouchpointId",
    "TouchsiteId",
    "WorkstreamId",
    "encode_policy_queue_identity",
    "policy_queue_identity_view_payload",
]
