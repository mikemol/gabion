from __future__ import annotations

import warnings

import pytest

from gabion.analysis.foundation.marker_protocol import (
    MarkerGovernanceProfile,
    MarkerKind,
    MarkerLifecycleState,
    MarkerPayload,
    SemanticLinkKind,
    SemanticReference,
)
from gabion import invariants
from gabion.exceptions import NeverThrown
from gabion.runtime.policy_runtime import RuntimePolicyConfig, runtime_policy_scope


_CUSTOM_PROFILE = invariants.InvariantProfileConfig(
    never=invariants.InvariantRuntimeBehavior(throws=False, emits_warning=False, warning_limit=0),
    todo=invariants.InvariantRuntimeBehavior(throws=True, emits_warning=False, warning_limit=0),
    deprecated=invariants.InvariantRuntimeBehavior(throws=False, emits_warning=True, warning_limit=1),
)


def test_never_raises_never_thrown() -> None:
    with pytest.raises(NeverThrown):
        invariants.never({"summary": "boom", "control": "phase", "blocking_dependencies": ["dep:a"]}, flag=True)


def test_todo_does_not_raise() -> None:
    invariants.todo({"summary": "pending", "control": "emit", "blocking_dependencies": ["dep:b"]})


def test_deprecated_warns_without_raising_and_is_rate_limited() -> None:
    reasoning = {"summary": "legacy", "control": "codec", "blocking_dependencies": ["dep:c"]}
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        invariants.deprecated(reasoning)
        invariants.deprecated(reasoning)
    matches = [w for w in seen if issubclass(w.category, invariants.DeprecatedMarkerWarning)]
    assert len(matches) == 1


def test_require_not_none_non_strict() -> None:
    assert invariants.require_not_none(None, strict=False) is None
    assert invariants.require_not_none("ok", strict=False) == "ok"


def test_require_not_none_strict_raises() -> None:
    with pytest.raises(NeverThrown):
        invariants.require_not_none(None, strict=True)


def test_require_not_none_runtime_strict() -> None:
    with runtime_policy_scope(RuntimePolicyConfig(proof_mode_enabled=True)):
        with pytest.raises(NeverThrown):
            invariants.require_not_none(None)


def test_decision_and_boundary_markers_return_original_callable() -> None:
    def _sample() -> str:
        return "ok"

    assert invariants.decision_protocol(_sample) is _sample
    assert invariants.boundary_normalization(_sample) is _sample


def test_never_normalizes_marker_links_and_marker_payload_dict() -> None:
    with pytest.raises(NeverThrown) as exc_info:
        invariants.never(
            {"summary": "boom", "control": "taint", "blocking_dependencies": ["dep:z"]},
            links=[
                "bad",
                {"kind": "", "value": "skip"},
                {"kind": "object_id", "value": "taint_kind:control_ambiguity"},
            ],
        )
    assert exc_info.value.marker_payload.links == (
        SemanticReference(
            kind=SemanticLinkKind.OBJECT_ID,
            value="taint_kind:control_ambiguity",
        ),
    )

    custom_payload = MarkerPayload(
        marker_kind=MarkerKind.NEVER,
        reason="boom",
        owner="core",
        expiry="2099-01-01",
        lifecycle_state=MarkerLifecycleState.ACTIVE,
        links=[  # type: ignore[arg-type]
            SemanticReference(
                kind=SemanticLinkKind.OBJECT_ID,
                value="taint_kind:control_ambiguity",
            )
        ],
        reasoning=invariants.StructuredReasoning(
            summary="boom",
            control="control",
            blocking_dependencies=("dep",),
        ),
        env={},
    )
    payload = NeverThrown("boom", marker_payload=custom_payload).marker_payload_dict
    assert payload["links"] == [
        {
            "kind": "object_id",
            "value": "taint_kind:control_ambiguity",
        }
    ]


def test_marker_kind_resolution_is_profile_driven() -> None:
    with runtime_policy_scope(
        RuntimePolicyConfig(
            marker_governance_profile=MarkerGovernanceProfile.DEBT_LEDGER.value,
        )
    ):
        with pytest.raises(NeverThrown) as exc_info:
            invariants.never({"summary": "stop", "control": "scan", "blocking_dependencies": ["dep:a"]})
    assert exc_info.value.marker_payload.marker_kind is MarkerKind.NEVER


def test_invariant_factory_behavior_is_profile_configured() -> None:
    with runtime_policy_scope(
        RuntimePolicyConfig(
            marker_governance_profile=MarkerGovernanceProfile.GOVERNANCE.value,
            marker_invariant_profile=_CUSTOM_PROFILE,
        )
    ):
        # never is configured not to throw in this profile
        payload = invariants.never({"summary": "not-fatal", "control": "gate", "blocking_dependencies": ["dep"]})
        assert payload.reason == "not-fatal"

        # todo is configured to throw in this profile
        with pytest.raises(NeverThrown):
            invariants.todo({"summary": "fatal-todo", "control": "gate", "blocking_dependencies": ["dep"]})


def test_invariant_factory_entrypoint() -> None:
    payload = invariants.invariant_factory(
        "todo",
        {"summary": "factory", "control": "entry", "blocking_dependencies": ["dep:x"]},
    )
    assert payload.reasoning.control == "entry"
