# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from gabion.analysis.core.type_fingerprints import PrimeAssignmentEvent, PrimeRegistry
from gabion.analysis.foundation.identity_namespace_governance import (
    CANONICAL_MIRRORED_NAMESPACES,
    NamespaceRecord,
    apply_namespace_records_to_identity_space,
    iter_registry_namespace_records,
    normalize_namespaces,
)
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace
from gabion.analysis.foundation.timeout_context import check_deadline

DEFAULT_MIRRORED_NAMESPACES: tuple[str, ...] = CANONICAL_MIRRORED_NAMESPACES


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
            normalize_namespaces(
                self.allowed_namespaces,
                source="IdentityRegistryMirror.__post_init__.allowed_namespaces",
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
        records = iter_registry_namespace_records(
            primes=self.registry.primes,
            allowed_namespaces=self.allowed_namespaces,
        )
        tuple(
            apply_namespace_records_to_identity_space(
                identity_space=self.identity_space,
                records=records,
                record_allocation=False,
            )
        )

    def _on_prime_assignment(self, event: PrimeAssignmentEvent) -> None:
        check_deadline()
        namespace = str(event.namespace).strip()
        token = str(event.token).strip()
        if not namespace or not token:
            return
        if not self._namespace_allowed(namespace):
            return
        tuple(
            apply_namespace_records_to_identity_space(
                identity_space=self.identity_space,
                records=(
                    NamespaceRecord(
                        namespace=namespace,
                        token=token,
                        atom_id=int(event.atom_id),
                    ),
                ),
                record_allocation=True,
            )
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
        allowed_namespaces=tuple(map(str, allowed_namespaces)),
    )


__all__ = [
    "DEFAULT_MIRRORED_NAMESPACES",
    "IdentityRegistryMirror",
    "build_identity_registry_mirror",
]
