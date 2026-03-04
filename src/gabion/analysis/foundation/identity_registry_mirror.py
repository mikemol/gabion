# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from gabion.analysis.core.identity_namespace import (
    SYNTH_NAMESPACE,
    TYPE_BASE_NAMESPACE,
    TYPE_CTOR_NAMESPACE,
    namespace_key,
)
from gabion.analysis.core.type_fingerprints import PrimeAssignmentEvent, PrimeRegistry
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.order_contract import OrderPolicy, sort_once

DEFAULT_MIRRORED_NAMESPACES: tuple[str, ...] = (
    TYPE_BASE_NAMESPACE,
    TYPE_CTOR_NAMESPACE,
    SYNTH_NAMESPACE,
)


@dataclass
class IdentityRegistryMirror:
    registry: PrimeRegistry
    identity_space: GlobalIdentitySpace
    allowed_namespaces: tuple[str, ...] = DEFAULT_MIRRORED_NAMESPACES
    _observer_id: int = 0
    _started: bool = False
    _allowed_namespace_lookup: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        check_deadline()
        normalized = tuple(
            sort_once(
                {
                    str(namespace).strip()
                    for namespace in self.allowed_namespaces
                    if str(namespace).strip()
                },
                source="IdentityRegistryMirror.__post_init__.allowed_namespaces",
                policy=OrderPolicy.SORT,
            )
        )
        self.allowed_namespaces = normalized
        self._allowed_namespace_lookup = set(normalized)

    def start(self) -> None:
        check_deadline()
        if not self._started:
            self._hydrate_existing_assignments()
            self._observer_id = int(
                self.registry.register_assignment_observer(
                    self._on_prime_assignment
                )
            )
            self._started = True

    def stop(self) -> None:
        observer_id = int(self._observer_id)
        self._observer_id = 0
        self._started = False
        if observer_id > 0:
            self.registry.unregister_assignment_observer(observer_id)

    def _hydrate_existing_assignments(self) -> None:
        check_deadline()
        for raw_key, atom_id in sort_once(
            self.registry.primes.items(),
            source="IdentityRegistryMirror._hydrate_existing_assignments.primes",
            policy=OrderPolicy.SORT,
        ):
            check_deadline()
            namespace, token = namespace_key(str(raw_key))
            if not self._namespace_allowed(namespace):
                continue
            self.identity_space.register_atom(
                namespace=namespace,
                token=token,
                atom_id=int(atom_id),
                record_allocation=False,
            )

    def _on_prime_assignment(self, event: PrimeAssignmentEvent) -> None:
        check_deadline()
        if not self._namespace_allowed(event.namespace):
            return
        self.identity_space.register_atom(
            namespace=event.namespace,
            token=event.token,
            atom_id=event.atom_id,
            record_allocation=True,
        )

    def _namespace_allowed(self, namespace: str) -> bool:
        check_deadline()
        return str(namespace) in self._allowed_namespace_lookup


def build_identity_registry_mirror(
    *,
    registry: PrimeRegistry,
    identity_space: GlobalIdentitySpace,
    allowed_namespaces: Iterable[str] = DEFAULT_MIRRORED_NAMESPACES,
) -> IdentityRegistryMirror:
    check_deadline()
    return IdentityRegistryMirror(
        registry=registry,
        identity_space=identity_space,
        allowed_namespaces=tuple(str(namespace) for namespace in allowed_namespaces),
    )


__all__ = [
    "DEFAULT_MIRRORED_NAMESPACES",
    "IdentityRegistryMirror",
    "build_identity_registry_mirror",
]
