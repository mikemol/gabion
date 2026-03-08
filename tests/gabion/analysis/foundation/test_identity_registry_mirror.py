from __future__ import annotations

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.identity_namespace_governance import (
    CANONICAL_MIRRORED_NAMESPACES,
)
from gabion.analysis.foundation.identity_registry_mirror import (
    DEFAULT_MIRRORED_NAMESPACES,
    build_identity_registry_mirror,
)
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace


def _space_with_registry(
    registry: PrimeRegistry,
) -> GlobalIdentitySpace:
    return GlobalIdentitySpace(allocator=PrimeIdentityAdapter(registry=registry))


# gabion:behavior primary=desired
def test_identity_registry_mirror_hydrates_existing_allowed_namespaces_without_ledger_growth() -> None:
    registry = PrimeRegistry()
    int_atom = registry.get_or_assign("int")
    ctor_atom = registry.get_or_assign("ctor:list")
    _ = registry.get_or_assign("site:FunctionSite")
    space = _space_with_registry(registry)
    mirror = build_identity_registry_mirror(registry=registry, identity_space=space)

    mirror.start()
    try:
        assert space.allocation_records() == ()
        base_lookup = space.token_for_atom(namespace="type_base", atom_id=int_atom)
        ctor_lookup = space.token_for_atom(namespace="type_ctor", atom_id=ctor_atom)
        assert base_lookup.is_present is True
        assert base_lookup.token == "int"
        assert ctor_lookup.is_present is True
        assert ctor_lookup.token == "list"
    finally:
        mirror.stop()


# gabion:behavior primary=desired
def test_identity_registry_mirror_starts_and_stops_observer_lifecycle() -> None:
    registry = PrimeRegistry()
    space = _space_with_registry(registry)
    mirror = build_identity_registry_mirror(registry=registry, identity_space=space)

    mirror.start()
    registry.get_or_assign("int")
    records_after_start = space.allocation_records_payload()
    mirror.stop()
    registry.get_or_assign("str")
    records_after_stop = space.allocation_records_payload()

    assert len(records_after_start) == 1
    assert records_after_start == records_after_stop


# gabion:behavior primary=desired
def test_identity_registry_mirror_tracks_allowed_namespaces_and_ignores_disallowed() -> None:
    registry = PrimeRegistry()
    space = _space_with_registry(registry)
    mirror = build_identity_registry_mirror(registry=registry, identity_space=space)

    mirror.start()
    try:
        int_atom = registry.get_or_assign("int")
        _ = registry.get_or_assign("site:FunctionSite")
        ctor_atom = registry.get_or_assign("ctor:list")
        _ = registry.get_or_assign("synth:tail")
        _ = registry.get_or_assign("int")
    finally:
        mirror.stop()

    records = space.allocation_records_payload()
    assert [record["seq"] for record in records] == [1, 2, 3]
    assert {(record["namespace"], record["token"]) for record in records} == {
        ("type_base", "int"),
        ("type_ctor", "list"),
        ("synth", "tail"),
    }
    assert {(record["namespace"], record["atom_id"]) for record in records} == {
        ("type_base", int_atom),
        ("type_ctor", ctor_atom),
        ("synth", registry.get_or_assign("synth:tail")),
    }


# gabion:behavior primary=desired
def test_identity_registry_mirror_default_namespaces_are_governed() -> None:
    registry = PrimeRegistry()
    space = _space_with_registry(registry)
    mirror = build_identity_registry_mirror(
        registry=registry,
        identity_space=space,
        allowed_namespaces=(" synth ", "type_ctor", "type_base", "type_ctor"),
    )

    assert DEFAULT_MIRRORED_NAMESPACES == CANONICAL_MIRRORED_NAMESPACES
    assert mirror.allowed_namespaces == ("synth", "type_base", "type_ctor")
