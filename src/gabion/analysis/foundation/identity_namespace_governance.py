# gabion:decision_protocol_module
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
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
) -> Iterator[str]:
    check_deadline()
    normalized_namespaces = map(_normalize_namespace_text, namespaces)
    non_empty_namespaces = filter(bool, normalized_namespaces)
    return iter(
        sort_once(
            set(non_empty_namespaces),
            source=source,
            policy=OrderPolicy.SORT,
        )
    )


def iter_registry_namespace_records(
    *,
    primes: Mapping[str, int],
    allowed_namespaces: Iterable[str] = CANONICAL_MIRRORED_NAMESPACES,
) -> Iterator[NamespaceRecord]:
    check_deadline()
    allowed_namespace_lookup = set(
        normalize_namespaces(
            allowed_namespaces,
            source="iter_registry_namespace_records.allowed_namespaces",
        )
    )
    ordered_prime_items = sort_once(
        primes.items(),
        source="iter_registry_namespace_records.primes",
        policy=OrderPolicy.SORT,
    )
    normalized_records = map(_namespace_record_from_registry_item, ordered_prime_items)
    return filter(
        lambda record: _namespace_record_allowed(
            record=record,
            allowed_namespace_lookup=allowed_namespace_lookup,
        ),
        normalized_records,
    )


def apply_namespace_records_to_identity_space(
    *,
    identity_space: GlobalIdentitySpace,
    records: Iterable[NamespaceRecord],
    record_allocation: bool,
) -> Iterator[NamespaceRecord]:
    check_deadline()
    normalized_records = filter(
        _namespace_record_complete,
        map(_normalize_namespace_record, records),
    )
    for record in normalized_records:
        check_deadline()
        identity_space.register_atom(
            namespace=record.namespace,
            token=record.token,
            atom_id=record.atom_id,
            record_allocation=bool(record_allocation),
        )
        yield record


def _normalize_namespace_text(value: str) -> str:
    return str(value).strip()


def _namespace_record_from_registry_item(item: tuple[str, int]) -> NamespaceRecord:
    raw_registry_key, atom_id = item
    namespace, token = namespace_key(str(raw_registry_key))
    return NamespaceRecord(
        namespace=_normalize_namespace_text(str(namespace)),
        token=_normalize_namespace_text(str(token)),
        atom_id=int(atom_id),
    )


def _namespace_record_allowed(
    *,
    record: NamespaceRecord,
    allowed_namespace_lookup: set[str],
) -> bool:
    check_deadline()
    return _namespace_record_complete(record) and record.namespace in allowed_namespace_lookup


def _normalize_namespace_record(record: NamespaceRecord) -> NamespaceRecord:
    return NamespaceRecord(
        namespace=_normalize_namespace_text(record.namespace),
        token=_normalize_namespace_text(record.token),
        atom_id=int(record.atom_id),
    )


def _namespace_record_complete(record: NamespaceRecord) -> bool:
    return bool(record.namespace and record.token)


__all__ = [
    "CANONICAL_MIRRORED_NAMESPACES",
    "INTEGER_ANCHOR_NAMESPACE",
    "NamespaceRecord",
    "apply_namespace_records_to_identity_space",
    "iter_registry_namespace_records",
    "normalize_namespaces",
]
