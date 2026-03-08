from __future__ import annotations

import pytest

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.identity_space import (
    GlobalIdentitySpace,
    IdentityNamespace,
)


def _build_space() -> GlobalIdentitySpace:
    registry = PrimeRegistry()
    return GlobalIdentitySpace(allocator=PrimeIdentityAdapter(registry=registry))


# gabion:behavior primary=desired
def test_identity_space_interns_same_token_within_namespace() -> None:
    space = _build_space()
    first = space.intern_atom(namespace=IdentityNamespace.SYMBOL, token="MyNode")
    second = space.intern_atom(namespace=IdentityNamespace.SYMBOL, token="MyNode")
    assert first == second
    assert len(space.allocation_records()) == 1


# gabion:behavior primary=desired
def test_identity_space_distinguishes_same_token_across_namespaces() -> None:
    space = _build_space()
    symbol_atom = space.intern_atom(namespace=IdentityNamespace.SYMBOL, token="MyNode")
    feature_atom = space.intern_atom(namespace=IdentityNamespace.FEATURE, token="MyNode")
    assert symbol_atom != feature_atom
    assert len(space.allocation_records()) == 2


# gabion:behavior primary=desired
def test_identity_space_path_is_stable_and_order_sensitive() -> None:
    space = _build_space()
    first = space.intern_path(
        namespace=IdentityNamespace.SYMBOL,
        tokens=("MyClass", "build"),
    )
    second = space.intern_path(
        namespace=IdentityNamespace.SYMBOL,
        tokens=("MyClass", "build"),
    )
    reversed_path = space.intern_path(
        namespace=IdentityNamespace.SYMBOL,
        tokens=("build", "MyClass"),
    )

    assert first == second
    assert first != reversed_path


# gabion:behavior primary=desired
def test_identity_projection_tracks_order_vs_commutative_alias() -> None:
    space = _build_space()
    ordered_path = space.intern_path(
        namespace=IdentityNamespace.SYMBOL,
        tokens=("A", "B"),
    )
    reversed_path = space.intern_path(
        namespace=IdentityNamespace.SYMBOL,
        tokens=("B", "A"),
    )

    ordered_projection = space.project(path=ordered_path)
    reversed_projection = space.project(path=reversed_path)

    assert ordered_projection.basis_path != reversed_projection.basis_path
    assert ordered_projection.prime_product == reversed_projection.prime_product
    assert ordered_projection.witness["commutation_witness"] == {
        "carrier_relation": "ordered_basis_vs_commutative_scalar",
        "order_erased_by_prime_product": True,
        "order_preserved_by_basis_path": True,
    }


# gabion:behavior primary=desired
def test_identity_space_allocation_ledger_is_deterministic_and_replayable() -> None:
    first = _build_space()
    _ = first.intern_path(
        namespace=IdentityNamespace.SYMBOL,
        tokens=("ClassA", "method_x"),
    )
    _ = first.intern_path(
        namespace=IdentityNamespace.FEATURE,
        tokens=("call", "weight:5"),
    )
    first_records = first.allocation_records_payload()
    seed_payload = first.seed_payload()

    second = _build_space()
    second.load_seed_payload(seed_payload)
    _ = second.intern_path(
        namespace=IdentityNamespace.SYMBOL,
        tokens=("ClassA", "method_x"),
    )
    _ = second.intern_path(
        namespace=IdentityNamespace.FEATURE,
        tokens=("call", "weight:5"),
    )
    second_records = second.allocation_records_payload()

    assert first_records == second_records
    for record in second.allocation_records():
        lookup = second.token_for_atom(namespace=record.namespace, atom_id=record.atom_id)
        assert lookup.is_present
        assert lookup.token == record.token


# gabion:behavior primary=desired
def test_identity_space_register_atom_is_idempotent() -> None:
    space = _build_space()
    first = space.register_atom(
        namespace=IdentityNamespace.SYMBOL,
        token="MyNode",
        atom_id=2,
    )
    second = space.register_atom(
        namespace=IdentityNamespace.SYMBOL,
        token="MyNode",
        atom_id=2,
    )
    assert first == second == 2
    assert len(space.allocation_records()) == 1


# gabion:behavior primary=desired
def test_identity_space_register_atom_hydration_does_not_append_allocation_ledger() -> None:
    space = _build_space()
    atom_id = space.register_atom(
        namespace=IdentityNamespace.SYMBOL,
        token="Hydrated",
        atom_id=3,
        record_allocation=False,
    )
    assert atom_id == 3
    assert space.allocation_records() == ()
    lookup = space.token_for_atom(namespace=IdentityNamespace.SYMBOL, atom_id=3)
    assert lookup.is_present is True
    assert lookup.token == "Hydrated"


# gabion:behavior primary=desired
def test_identity_space_register_atom_rejects_token_atom_conflicts() -> None:
    space = _build_space()
    space.register_atom(
        namespace=IdentityNamespace.SYMBOL,
        token="A",
        atom_id=2,
    )
    with pytest.raises(ValueError):
        space.register_atom(
            namespace=IdentityNamespace.SYMBOL,
            token="A",
            atom_id=3,
        )
    with pytest.raises(ValueError):
        space.register_atom(
            namespace=IdentityNamespace.SYMBOL,
            token="B",
            atom_id=2,
        )
