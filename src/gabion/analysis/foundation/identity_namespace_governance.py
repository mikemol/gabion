# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from gabion.analysis.core.identity_namespace import (
    SYNTH_NAMESPACE,
    TYPE_BASE_NAMESPACE,
    TYPE_CTOR_NAMESPACE,
    namespace_key,
)
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.order_contract import OrderPolicy, sort_once

CANONICAL_MIRRORED_NAMESPACES: tuple[str, ...] = (
    TYPE_BASE_NAMESPACE,
    TYPE_CTOR_NAMESPACE,
    SYNTH_NAMESPACE,
)
INTEGER_ANCHOR_NAMESPACE = "dataflow.progress.integer_anchor"


@dataclass(frozen=True)
class NamespaceRecord:
    namespace: str
    token: str
    atom_id: int


def normalize_namespaces(
    namespaces: Iterable[str],
    *,
    source: str,
) -> tuple[str, ...]:
    check_deadline()
    return tuple(
        sort_once(
            {
                str(namespace).strip()
                for namespace in namespaces
                if str(namespace).strip()
            },
            source=source,
            policy=OrderPolicy.SORT,
        )
    )


def iter_registry_namespace_records(
    *,
    primes: Mapping[str, int],
    allowed_namespaces: Iterable[str] = CANONICAL_MIRRORED_NAMESPACES,
) -> tuple[NamespaceRecord, ...]:
    check_deadline()
    normalized_allowed = normalize_namespaces(
        allowed_namespaces,
        source="iter_registry_namespace_records.allowed_namespaces",
    )
    allowed_namespace_lookup = set(normalized_allowed)
    records: list[NamespaceRecord] = []
    for raw_registry_key, atom_id in sort_once(
        primes.items(),
        source="iter_registry_namespace_records.primes",
        policy=OrderPolicy.SORT,
    ):
        check_deadline()
        namespace, token = namespace_key(str(raw_registry_key))
        namespace_text = str(namespace).strip()
        token_text = str(token).strip()
        if not namespace_text or not token_text:
            continue
        if namespace_text not in allowed_namespace_lookup:
            continue
        records.append(
            NamespaceRecord(
                namespace=namespace_text,
                token=token_text,
                atom_id=int(atom_id),
            )
        )
    return tuple(records)


def apply_namespace_records_to_identity_space(
    *,
    identity_space: GlobalIdentitySpace,
    records: Iterable[NamespaceRecord],
    record_allocation: bool,
) -> tuple[NamespaceRecord, ...]:
    check_deadline()
    normalized_records: list[NamespaceRecord] = []
    for record in records:
        check_deadline()
        namespace_text = str(record.namespace).strip()
        token_text = str(record.token).strip()
        if not namespace_text or not token_text:
            continue
        normalized_record = NamespaceRecord(
            namespace=namespace_text,
            token=token_text,
            atom_id=int(record.atom_id),
        )
        identity_space.register_atom(
            namespace=normalized_record.namespace,
            token=normalized_record.token,
            atom_id=normalized_record.atom_id,
            record_allocation=bool(record_allocation),
        )
        normalized_records.append(normalized_record)
    return tuple(normalized_records)


__all__ = [
    "CANONICAL_MIRRORED_NAMESPACES",
    "INTEGER_ANCHOR_NAMESPACE",
    "NamespaceRecord",
    "apply_namespace_records_to_identity_space",
    "iter_registry_namespace_records",
    "normalize_namespaces",
]
