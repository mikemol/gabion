from __future__ import annotations

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.identity_namespace_governance import (
    CANONICAL_MIRRORED_NAMESPACES,
    NamespaceRecord,
    apply_namespace_records_to_identity_space,
    iter_registry_namespace_records,
    normalize_namespaces,
)
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace


def _space_with_registry(registry: PrimeRegistry) -> GlobalIdentitySpace:
    return GlobalIdentitySpace(allocator=PrimeIdentityAdapter(registry=registry))


# gabion:behavior primary=desired
def test_normalize_namespaces_strips_dedupes_and_sorts() -> None:
    normalized = tuple(
        normalize_namespaces(
            (" type_ctor ", "", "synth", "type_base", "synth", " type_base"),
            source="test_identity_namespace_governance.normalize",
        )
    )
    assert normalized == ("synth", "type_base", "type_ctor")


# gabion:behavior primary=desired
def test_iter_registry_namespace_records_filters_and_orders_by_raw_key() -> None:
    records = tuple(
        iter_registry_namespace_records(
            primes={
                "synth:tail": 11,
                "int": 2,
                "site:FunctionSite": 13,
                "ctor:list": 3,
            },
            allowed_namespaces=CANONICAL_MIRRORED_NAMESPACES,
        )
    )
    assert records == (
        NamespaceRecord(namespace="type_ctor", token="list", atom_id=3),
        NamespaceRecord(namespace="type_base", token="int", atom_id=2),
        NamespaceRecord(namespace="synth", token="tail", atom_id=11),
    )


# gabion:behavior primary=desired
def test_apply_namespace_records_to_identity_space_controls_ledger_mode() -> None:
    space = _space_with_registry(PrimeRegistry())
    hydrated_records = (
        NamespaceRecord(namespace="type_base", token="int", atom_id=2),
        NamespaceRecord(namespace="type_ctor", token="list", atom_id=3),
    )
    applied_hydrated = tuple(
        apply_namespace_records_to_identity_space(
            identity_space=space,
            records=hydrated_records,
            record_allocation=False,
        )
    )
    assert applied_hydrated == hydrated_records
    assert tuple(space.allocation_records()) == ()
    assert space.token_for_atom(namespace="type_base", atom_id=2).token == "int"
    assert space.token_for_atom(namespace="type_ctor", atom_id=3).token == "list"

    applied_live = tuple(
        apply_namespace_records_to_identity_space(
            identity_space=space,
            records=(NamespaceRecord(namespace="synth", token="tail", atom_id=5),),
            record_allocation=True,
        )
    )
    assert applied_live == (
        NamespaceRecord(namespace="synth", token="tail", atom_id=5),
    )
    assert list(space.allocation_records_payload()) == [
        {
            "seq": 1,
            "namespace": "synth",
            "token": "tail",
            "atom_id": 5,
        }
    ]
