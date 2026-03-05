from __future__ import annotations

import warnings

import pytest

from gabion.analysis.foundation.marker_protocol import (
    MarkerKind, MarkerLifecycleState, MarkerPayload, MarkerReasoning, SemanticLinkKind, SemanticReference)
from gabion import invariants
from gabion.exceptions import NeverThrown
from gabion.runtime.policy_runtime import RuntimePolicyConfig, runtime_policy_scope


# gabion:evidence E:function_site::invariants.py::gabion.invariants.never
def test_never_raises_never_thrown() -> None:
    with pytest.raises(NeverThrown):
        invariants.never("boom", flag=True)


# gabion:evidence E:call_footprint::tests/test_invariants.py::test_require_not_none_non_strict::invariants.py::gabion.invariants.require_not_none
def test_require_not_none_non_strict() -> None:
    assert invariants.require_not_none(None, strict=False) is None
    assert invariants.require_not_none("ok", strict=False) == "ok"


# gabion:evidence E:call_footprint::tests/test_invariants.py::test_require_not_none_strict_raises::invariants.py::gabion.invariants.require_not_none
def test_require_not_none_strict_raises() -> None:
    with pytest.raises(NeverThrown):
        invariants.require_not_none(None, strict=True)


# gabion:evidence E:call_footprint::tests/test_invariants.py::test_require_not_none_env_strict::env_helpers.py::tests.env_helpers.env_scope::invariants.py::gabion.invariants.require_not_none
def test_require_not_none_runtime_strict() -> None:
    with runtime_policy_scope(RuntimePolicyConfig(proof_mode_enabled=True)):
        with pytest.raises(NeverThrown):
            invariants.require_not_none(None)


# gabion:evidence E:function_site::invariants.py::gabion.invariants.decision_protocol
def test_decision_and_boundary_markers_return_original_callable() -> None:
    def _sample() -> str:
        return "ok"

    assert invariants.decision_protocol(_sample) is _sample
    assert invariants.boundary_normalization(_sample) is _sample


# gabion:evidence E:function_site::invariants.py::gabion.invariants.never
def test_never_reasoning_boundary_normalizer_sets_summary_and_dependencies() -> None:
    with pytest.raises(NeverThrown) as exc_info:
        invariants.never(
            reasoning={
                "summary": "  normalized summary  ",
                "control": "  decision branch  ",
                "blocking_dependencies": ["dep-b", "dep-a", " dep-b "],
            }
        )

    payload = exc_info.value.marker_payload
    assert payload.reason == "normalized summary"
    assert payload.reasoning.control == "decision branch"
    assert payload.reasoning.blocking_dependencies == ("dep-a", "dep-b")


# gabion:evidence E:function_site::invariants.py::gabion.invariants.never
def test_never_normalizes_marker_links_and_marker_payload_dict() -> None:
    with pytest.raises(NeverThrown) as exc_info:
        invariants.never(
            "boom",
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
        reasoning=MarkerReasoning(summary="boom", control="", blocking_dependencies=()),
        owner="core",
        expiry="2099-01-01",
        lifecycle_state=MarkerLifecycleState.ACTIVE,
        links=[  # type: ignore[arg-type]
            SemanticReference(
                kind=SemanticLinkKind.OBJECT_ID,
                value="taint_kind:control_ambiguity",
            )
        ],
        env={},
    )
    payload = NeverThrown("boom", marker_payload=custom_payload).marker_payload_dict
    assert payload["links"] == [
        {
            "kind": "object_id",
            "value": "taint_kind:control_ambiguity",
        }
    ]


# gabion:evidence E:function_site::invariants.py::gabion.invariants.never::invariants.py::gabion.invariants.todo::invariants.py::gabion.invariants.deprecated
def test_marker_behavior_profile_supports_no_throw_and_warn_matrix() -> None:
    config = RuntimePolicyConfig(
        marker_behavior_profile=invariants.MarkerBehaviorProfile(
            throw_never=False,
            throw_todo=False,
            throw_deprecated=True,
            warn_never=True,
            warn_todo=False,
            warn_deprecated=True,
            warning_cap=1,
        )
    )
    with runtime_policy_scope(config):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            assert invariants.never("reachable") is None
            assert invariants.todo("later") is None
            with pytest.raises(NeverThrown):
                invariants.deprecated("legacy")
    warning_messages = [str(item.message) for item in captured]
    assert warning_messages == ["never() marker reached: reachable", "deprecated() marker reached: legacy"]


# gabion:evidence E:function_site::invariants.py::gabion.invariants.todo
def test_marker_warning_cap_uses_normalized_reasoning_key_for_todo() -> None:
    config = RuntimePolicyConfig(
        marker_behavior_profile=invariants.MarkerBehaviorProfile(
            throw_todo=False,
            warn_todo=True,
            warning_cap=1,
        )
    )
    with runtime_policy_scope(config):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            assert invariants.todo("  same summary  ") is None
            assert invariants.todo(reasoning={"summary": "same summary"}) is None
            assert invariants.todo(reasoning={"summary": "different summary"}) is None
    assert [str(item.message) for item in captured] == [
        "todo() marker reached: same summary",
        "todo() marker reached: different summary",
    ]


# gabion:evidence E:function_site::invariants.py::gabion.invariants.never
def test_marker_warning_cap_saturates_for_never_identity() -> None:
    config = RuntimePolicyConfig(
        marker_behavior_profile=invariants.MarkerBehaviorProfile(
            throw_never=False,
            warn_never=True,
            warning_cap=2,
        )
    )
    with runtime_policy_scope(config):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            assert invariants.never("repeat") is None
            assert invariants.never("repeat") is None
            assert invariants.never("repeat") is None
    assert [str(item.message) for item in captured] == [
        "never() marker reached: repeat",
        "never() marker reached: repeat",
    ]
