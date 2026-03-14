from __future__ import annotations

from enum import StrEnum

from gabion.tooling.policy_substrate.identity_zone import (
    HierarchicalIdentityGrammar,
    IdentityCarrier,
    IdentityLocalInterner,
    IdentityZoneName,
)


class _Namespace(StrEnum):
    ITEM = "test.item"
    DECOMPOSITION = "test.decomposition"


class _DecompositionKind(StrEnum):
    CANONICAL = "canonical"
    PATH = "path"


def test_identity_local_interner_dedupes_atoms_without_rewriting() -> None:
    interner = IdentityLocalInterner[_Namespace]()

    first = interner.identity(namespace=_Namespace.ITEM, token="carrier:a")
    second = interner.identity(namespace=_Namespace.ITEM, token="carrier:a")

    assert first == second
    assert first.atom_id == second.atom_id
    assert first.wire() == "carrier:a"


def test_hierarchical_identity_grammar_dedupes_isomorphic_witnesses() -> None:
    interner = IdentityLocalInterner[_Namespace]()
    carrier = IdentityCarrier(
        canonical=interner.identity(namespace=_Namespace.ITEM, token="carrier:a"),
        zone_name=IdentityZoneName("scanner"),
        carrier_kind="violation",
        label="carrier:a",
    )
    grammar = HierarchicalIdentityGrammar()
    grammar.add_carrier(carrier)

    first = grammar.add_kernel_congruence(
        source_zone="scanner",
        target_zone="hotspot",
        source_carrier_wire=carrier.wire(),
        retained_decomposition_kinds=("path",),
        erased_decomposition_kinds=("qualname", "line"),
        rationale="file-face quotient",
    )
    second = grammar.add_kernel_congruence(
        source_zone="scanner",
        target_zone="hotspot",
        source_carrier_wire=carrier.wire(),
        retained_decomposition_kinds=("path",),
        erased_decomposition_kinds=("qualname", "line"),
        rationale="file-face quotient",
    )

    assert first == second
    assert len(grammar.bundle().kernel_congruences) == 1


def test_hierarchical_identity_grammar_does_not_mutate_origin_local_identity() -> None:
    interner = IdentityLocalInterner[_Namespace]()
    canonical = interner.identity(namespace=_Namespace.ITEM, token="carrier:a")
    carrier = IdentityCarrier(
        canonical=canonical,
        zone_name=IdentityZoneName("scanner"),
        carrier_kind="violation",
        label="carrier:a",
    )
    grammar = HierarchicalIdentityGrammar()
    grammar.add_carrier(carrier)
    grammar.add_fiber_witness(
        source_zone="scanner",
        target_zone="hotspot",
        target_carrier_wire="file:a",
        member_source_wires=(canonical.wire(),),
        chosen_representative_wire=canonical.wire(),
    )

    assert canonical.wire() == "carrier:a"
    assert canonical.namespace is _Namespace.ITEM
