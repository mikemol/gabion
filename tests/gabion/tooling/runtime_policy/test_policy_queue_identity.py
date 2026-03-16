from __future__ import annotations

from gabion.tooling.policy_substrate.policy_queue_identity import (
    PolicyQueueDecompositionKind,
    PolicyQueueDecompositionRelationKind,
    PolicyQueueIdentitySpace,
    build_planner_queue_token,
    parse_planner_queue_token,
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


def test_policy_queue_identity_exposes_artifact_node_binding_carrier() -> None:
    identity_space = PolicyQueueIdentitySpace()
    artifact_node = identity_space.artifact_node_id(
        site_identity="site.decorated",
        structural_identity="struct.decorated",
        rel_path="src/gabion/sample.py",
        qualname="decorated",
        line=14,
        column=1,
    )

    payload = policy_queue_identity_view_payload(artifact_node)

    assert artifact_node.site_identity == "site.decorated"
    assert artifact_node.structural_identity == "struct.decorated"
    assert str(artifact_node) == "src/gabion/sample.py:14::decorated"
    assert payload["wire"] == "site.decorated::struct.decorated"
    assert payload["site_identity"] == "site.decorated"
    assert payload["structural_identity"] == "struct.decorated"
    assert payload["line"] == 14


def test_planner_queue_identity_is_deterministic_and_decodable() -> None:
    identity_space = PolicyQueueIdentitySpace()
    queue_id = identity_space.planner_queue_id(
        followup_family="coverage_gap",
        followup_class="code",
        selection_scope_kind="mixed_root_followup_family",
        selection_scope_id="coverage_gap:CSA-IDR,CSA-IGM",
        root_object_ids=("CSA-IGM", "CSA-IDR"),
    )
    queue_id_again = identity_space.queue_id(
        build_planner_queue_token(
            followup_family="coverage_gap",
            followup_class="code",
            selection_scope_kind="mixed_root_followup_family",
            selection_scope_id="coverage_gap:CSA-IDR,CSA-IGM",
            root_object_ids=("CSA-IDR", "CSA-IGM"),
        )
    )

    assert queue_id == queue_id_again
    assert queue_id.wire() == queue_id_again.wire()
    binding = parse_planner_queue_token(queue_id)
    assert binding.followup_family == "coverage_gap"
    assert binding.followup_class == "code"
    assert binding.selection_scope_kind == "mixed_root_followup_family"
    assert binding.selection_scope_id == "coverage_gap:CSA-IDR,CSA-IGM"
    assert binding.root_object_ids == ("CSA-IDR", "CSA-IGM")
    distinct_queue_id = identity_space.planner_queue_id(
        followup_family="coverage_gap",
        followup_class="code",
        selection_scope_kind="shared_root_workstream",
        selection_scope_id="CSA-IDR",
        root_object_ids=("CSA-IDR",),
    )
    assert distinct_queue_id != queue_id
