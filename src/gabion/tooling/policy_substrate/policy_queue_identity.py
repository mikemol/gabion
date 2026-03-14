from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.tooling.policy_substrate.identity_zone import (
    IdentityAtom,
    IdentityDecomposition,
    IdentityDecompositionRelation,
    IdentityLocalInterner,
)


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


_PrimeBackedIdentity = IdentityAtom[PolicyQueueIdentityNamespace]


PolicyQueueDecompositionIdentity = IdentityDecomposition[
    PolicyQueueIdentityNamespace,
    PolicyQueueDecompositionKind,
]


PolicyQueueDecompositionRelation = IdentityDecompositionRelation[
    PolicyQueueIdentityNamespace,
    PolicyQueueDecompositionKind,
    PolicyQueueDecompositionRelationKind,
]


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


@dataclass(frozen=True, order=True)
class ArtifactNodeId:
    canonical: _PrimeBackedIdentity
    site_ref: SiteReferenceId = field(compare=False)
    structural_ref: StructuralReferenceId = field(compare=False)
    rel_path: str = field(compare=False, default="")
    qualname: str = field(compare=False, default="")
    line: int = field(compare=False, default=0)
    column: int = field(compare=False, default=0)
    decompositions: tuple[PolicyQueueDecompositionIdentity, ...] = field(
        default=(),
        compare=False,
    )
    relations: tuple[PolicyQueueDecompositionRelation, ...] = field(
        default=(),
        compare=False,
    )

    @property
    def site_identity(self) -> str:
        return self.site_ref.wire()

    @property
    def structural_identity(self) -> str:
        return self.structural_ref.wire()

    def wire(self) -> str:
        return self.canonical.token

    def __str__(self) -> str:
        if self.rel_path and self.qualname and self.line > 0:
            return f"{self.rel_path}:{self.line}::{self.qualname}"
        return self.canonical.token

    def as_payload(self) -> dict[str, object]:
        return {
            "wire": self.wire(),
            "site_identity": self.site_identity,
            "structural_identity": self.structural_identity,
            "rel_path": self.rel_path,
            "qualname": self.qualname,
            "line": self.line,
            "column": self.column,
        }


_IdentityCarrierT = TypeVar(
    "_IdentityCarrierT",
    WorkstreamId,
    SubqueueId,
    TouchpointId,
    TouchsiteId,
    SiteReferenceId,
    StructuralReferenceId,
    ArtifactNodeId,
)


@dataclass
class PolicyQueueIdentitySpace:
    registry: PrimeRegistry = field(default_factory=PrimeRegistry)
    _interner: IdentityLocalInterner[PolicyQueueIdentityNamespace] = field(
        init=False,
        repr=False,
    )
    _decomposition_cache: dict[
        tuple[PolicyQueueIdentityNamespace, str],
        tuple[
            tuple[PolicyQueueDecompositionIdentity, ...],
            tuple[PolicyQueueDecompositionRelation, ...],
        ],
    ] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._interner = IdentityLocalInterner(registry=self.registry)

    @staticmethod
    def _structural_segments(value: str) -> tuple[str, ...]:
        return IdentityLocalInterner.structural_segments(value)

    def _identity(
        self,
        *,
        namespace: PolicyQueueIdentityNamespace,
        token: str,
    ) -> _PrimeBackedIdentity:
        return self._interner.identity(namespace=namespace, token=token)

    def _decomposition_identity(
        self,
        *,
        origin: _PrimeBackedIdentity,
        decomposition_kind: PolicyQueueDecompositionKind,
        label: str,
        part_index: int = -1,
    ) -> PolicyQueueDecompositionIdentity:
        return self._interner.decomposition_identity(
            origin=origin,
            decomposition_namespace=PolicyQueueIdentityNamespace.DECOMPOSITION,
            decomposition_kind=decomposition_kind,
            label=label,
            part_index=part_index,
            canonical_kind=PolicyQueueDecompositionKind.CANONICAL,
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

    def artifact_node_id(
        self,
        *,
        site_identity: str,
        structural_identity: str,
        rel_path: str = "",
        qualname: str = "",
        line: int = 0,
        column: int = 0,
    ) -> ArtifactNodeId:
        normalized_site_identity = str(site_identity).strip()
        normalized_structural_identity = str(structural_identity).strip()
        if not normalized_site_identity or not normalized_structural_identity:
            raise ValueError(
                "artifact_node_id requires both site_identity and structural_identity"
            )
        canonical = self._identity(
            namespace=PolicyQueueIdentityNamespace.ARTIFACT_NODE,
            token="::".join(
                (
                    normalized_site_identity,
                    normalized_structural_identity,
                )
            ),
        )
        decompositions, relations = self._decomposition_bundle(origin=canonical)
        return ArtifactNodeId(
            canonical=canonical,
            site_ref=self.site_ref_id(normalized_site_identity),
            structural_ref=self.structural_ref_id(normalized_structural_identity),
            rel_path=str(rel_path).strip(),
            qualname=str(qualname).strip(),
            line=int(line),
            column=int(column),
            decompositions=decompositions,
            relations=relations,
        )


def policy_queue_identity_view_payload(
    value: WorkstreamId
    | SubqueueId
    | TouchpointId
    | TouchsiteId
    | SiteReferenceId
    | StructuralReferenceId
    | ArtifactNodeId,
) -> dict[str, object]:
    payload = {
        "wire": value.wire(),
        "decompositions": [item.as_payload() for item in value.decompositions],
        "relations": [item.as_payload() for item in value.relations],
    }
    if isinstance(value, ArtifactNodeId):
        payload.update(
            {
                "site_identity": value.site_identity,
                "structural_identity": value.structural_identity,
                "rel_path": value.rel_path,
                "qualname": value.qualname,
                "line": value.line,
                "column": value.column,
            }
        )
    return payload


def encode_policy_queue_identity(
    value: WorkstreamId
    | SubqueueId
    | TouchpointId
    | TouchsiteId
    | SiteReferenceId
    | StructuralReferenceId
    | ArtifactNodeId
    | PolicyQueueDecompositionIdentity
    | str,
) -> str:
    match value:
        case WorkstreamId() | SubqueueId() | TouchpointId() | TouchsiteId():
            return value.wire()
        case SiteReferenceId() | StructuralReferenceId() | ArtifactNodeId():
            return value.wire()
        case IdentityDecomposition():
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
    "ArtifactNodeId",
    "SiteReferenceId",
    "StructuralReferenceId",
    "SubqueueId",
    "TouchpointId",
    "TouchsiteId",
    "WorkstreamId",
    "encode_policy_queue_identity",
    "policy_queue_identity_view_payload",
]
