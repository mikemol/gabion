from __future__ import annotations

from gabion.tooling.policy_substrate.policy_queue_identity import (
    PolicyQueueDecompositionKind,
    PolicyQueueDecompositionRelationKind,
    PolicyQueueIdentitySpace,
    policy_queue_identity_view_payload,
)


def test_policy_queue_identity_interns_structural_decompositions() -> None:
    identity_space = PolicyQueueIdentitySpace()

    touchpoint = identity_space.touchpoint_id("PSF-007-TP-005")
    touchpoint_again = identity_space.touchpoint_id("PSF-007-TP-005")

    assert touchpoint == touchpoint_again
    assert str(touchpoint) == "PSF-007-TP-005"
    assert touchpoint.canonical.atom_id == touchpoint_again.canonical.atom_id

    decomposition_kinds = {item.decomposition_kind for item in touchpoint.decompositions}
    assert decomposition_kinds == {
        PolicyQueueDecompositionKind.CANONICAL,
        PolicyQueueDecompositionKind.NAMESPACE,
        PolicyQueueDecompositionKind.TOKEN,
        PolicyQueueDecompositionKind.NAMESPACE_SEGMENT,
        PolicyQueueDecompositionKind.TOKEN_SEGMENT,
    }
    assert {
        item.relation_kind for item in touchpoint.relations
    } == {
        PolicyQueueDecompositionRelationKind.CANONICAL_OF,
        PolicyQueueDecompositionRelationKind.ALTERNATE_OF,
        PolicyQueueDecompositionRelationKind.EQUIVALENT_UNDER,
        PolicyQueueDecompositionRelationKind.DERIVED_FROM,
    }


def test_policy_queue_identity_view_payload_is_boundary_only() -> None:
    identity_space = PolicyQueueIdentitySpace()
    subqueue = identity_space.subqueue_id("PSF-007-SQ-001")

    payload = policy_queue_identity_view_payload(subqueue)

    assert payload["wire"] == "PSF-007-SQ-001"
    assert any(
        item["decomposition_kind"] == "token_segment"
        for item in payload["decompositions"]
    )
    assert any(
        item["relation_kind"] == "derived_from"
        for item in payload["relations"]
    )
