from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from contextlib import ExitStack

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


@dataclass(frozen=True, order=True)
class _PrimeBackedIdentity:
    atom_id: int
    namespace: PolicyQueueIdentityNamespace = field(compare=False)
    token: str = field(compare=False)

    def wire(self) -> str:
        return self.token


@dataclass(frozen=True, order=True)
class WorkstreamId:
    canonical: _PrimeBackedIdentity

    def wire(self) -> str:
        return self.canonical.wire()


@dataclass(frozen=True, order=True)
class SubqueueId:
    canonical: _PrimeBackedIdentity

    def wire(self) -> str:
        return self.canonical.wire()


@dataclass(frozen=True, order=True)
class TouchpointId:
    canonical: _PrimeBackedIdentity

    def wire(self) -> str:
        return self.canonical.wire()


@dataclass(frozen=True, order=True)
class TouchsiteId:
    canonical: _PrimeBackedIdentity

    def wire(self) -> str:
        return self.canonical.wire()


@dataclass(frozen=True, order=True)
class SiteReferenceId:
    canonical: _PrimeBackedIdentity

    def wire(self) -> str:
        return self.canonical.wire()


@dataclass(frozen=True, order=True)
class StructuralReferenceId:
    canonical: _PrimeBackedIdentity

    def wire(self) -> str:
        return self.canonical.wire()


@dataclass
class PolicyQueueIdentitySpace:
    registry: PrimeRegistry = field(default_factory=PrimeRegistry)
    _adapter: PrimeIdentityAdapter = field(init=False, repr=False)
    _cache: dict[
        tuple[PolicyQueueIdentityNamespace, str],
        _PrimeBackedIdentity,
    ] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._adapter = PrimeIdentityAdapter(registry=self.registry)

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

    def workstream_id(self, token: str) -> WorkstreamId:
        return WorkstreamId(
            canonical=self._identity(
                namespace=PolicyQueueIdentityNamespace.WORKSTREAM,
                token=token,
            )
        )

    def subqueue_id(self, token: str) -> SubqueueId:
        return SubqueueId(
            canonical=self._identity(
                namespace=PolicyQueueIdentityNamespace.SUBQUEUE,
                token=token,
            )
        )

    def touchpoint_id(self, token: str) -> TouchpointId:
        return TouchpointId(
            canonical=self._identity(
                namespace=PolicyQueueIdentityNamespace.TOUCHPOINT,
                token=token,
            )
        )

    def touchsite_id(self, token: str) -> TouchsiteId:
        return TouchsiteId(
            canonical=self._identity(
                namespace=PolicyQueueIdentityNamespace.TOUCHSITE,
                token=token,
            )
        )

    def site_ref_id(self, token: str) -> SiteReferenceId:
        return SiteReferenceId(
            canonical=self._identity(
                namespace=PolicyQueueIdentityNamespace.SITE_REF,
                token=token,
            )
        )

    def structural_ref_id(self, token: str) -> StructuralReferenceId:
        return StructuralReferenceId(
            canonical=self._identity(
                namespace=PolicyQueueIdentityNamespace.STRUCTURAL_REF,
                token=token,
            )
        )


def encode_policy_queue_identity(
    value: WorkstreamId
    | SubqueueId
    | TouchpointId
    | TouchsiteId
    | SiteReferenceId
    | StructuralReferenceId
    | str,
) -> str:
    match value:
        case WorkstreamId() | SubqueueId() | TouchpointId() | TouchsiteId():
            return value.wire()
        case SiteReferenceId() | StructuralReferenceId():
            return value.wire()
        case str() as text:
            return text
        case _:
            return str(value)


__all__ = [
    "PolicyQueueIdentityNamespace",
    "PolicyQueueIdentitySpace",
    "SiteReferenceId",
    "StructuralReferenceId",
    "SubqueueId",
    "TouchpointId",
    "TouchsiteId",
    "WorkstreamId",
    "encode_policy_queue_identity",
]
