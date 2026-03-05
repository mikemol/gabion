from __future__ import annotations

from gabion import invariants
from gabion.analysis.foundation.marker_protocol import (
    MarkerKindProfile,
    runtime_marker_kind_mapping_config,
)
from gabion.exceptions import NeverThrown
from gabion.order_contract import OrderPolicy, ordered_or_sorted
from gabion.runtime.policy_runtime import (
    RuntimePolicyConfig,
    apply_runtime_policy,
    runtime_policy_from_env,
    runtime_policy_scope,
)
from tests.env_helpers import env_scope


def test_runtime_policy_from_env_maps_order_and_proof_settings() -> None:
    with env_scope({"GABION_PROOF_MODE": "strict", "GABION_ORDER_POLICY": "enforce"}):
        config = runtime_policy_from_env()
    assert config.proof_mode_enabled is True
    assert config.order_policy is not None
    assert config.order_policy.value == "enforce"


def test_ambient_env_does_not_change_order_contract_without_adapter() -> None:
    with env_scope({"GABION_ORDER_POLICY": "enforce"}):
        assert ordered_or_sorted(["b", "a"], source="ambient-env") == ["a", "b"]


def test_runtime_policy_scope_applies_order_policy_from_env_config() -> None:
    with env_scope({"GABION_ORDER_POLICY": "enforce"}):
        config = runtime_policy_from_env()
    with runtime_policy_scope(config):
        try:
            ordered_or_sorted(["b", "a"], source="runtime-config")
        except NeverThrown:
            return
    raise AssertionError("expected enforce policy to reject unsorted input")


def test_ambient_env_does_not_change_proof_mode_without_adapter() -> None:
    with env_scope({"GABION_PROOF_MODE": "strict"}):
        assert invariants.require_not_none(None) is None


def test_runtime_policy_optional_order_policy_normalization_branches() -> None:
    with env_scope({"GABION_ORDER_POLICY": ""}):
        assert runtime_policy_from_env().order_policy is None

    with env_scope({"GABION_ORDER_POLICY": "off"}):
        assert runtime_policy_from_env().order_policy is OrderPolicy.SORT

    with env_scope({"GABION_ORDER_POLICY": "on"}):
        assert runtime_policy_from_env().order_policy is OrderPolicy.ENFORCE

    with env_scope({"GABION_ORDER_POLICY": "not-a-policy"}):
        assert runtime_policy_from_env().order_policy is None


def test_runtime_policy_from_env_parses_new_profile_fields() -> None:
    with env_scope(
        {
            "GABION_INVARIANT_RUNTIME_BEHAVIOR_PROFILE": "warn",
            "GABION_MARKER_KIND_PROFILE": "collapse_to_never",
        }
    ):
        config = runtime_policy_from_env()
    assert config.invariant_runtime_behavior_profile is invariants.InvariantRuntimeBehaviorProfile.WARN
    assert config.marker_kind_profile is MarkerKindProfile.COLLAPSE_TO_NEVER


def test_apply_runtime_policy_updates_invariant_and_marker_profiles() -> None:
    from gabion.analysis.foundation.marker_protocol import set_runtime_marker_kind_mapping_config
    from gabion.invariants import set_invariant_runtime_behavior_config

    base_invariant = invariants.invariant_runtime_behavior_config()
    base_marker = runtime_marker_kind_mapping_config()
    apply_runtime_policy(
        RuntimePolicyConfig(
            invariant_runtime_behavior_profile=invariants.InvariantRuntimeBehaviorProfile.WARN,
            marker_kind_profile=MarkerKindProfile.COLLAPSE_TO_NEVER,
        )
    )
    try:
        assert (
            invariants.invariant_runtime_behavior_config().profile
            is invariants.InvariantRuntimeBehaviorProfile.WARN
        )
        assert runtime_marker_kind_mapping_config().profile is MarkerKindProfile.COLLAPSE_TO_NEVER
    finally:
        set_invariant_runtime_behavior_config(base_invariant)
        set_runtime_marker_kind_mapping_config(base_marker)


def test_runtime_policy_scope_restores_nested_profile_state() -> None:
    baseline_invariant = invariants.invariant_runtime_behavior_config()
    baseline_marker = runtime_marker_kind_mapping_config()

    outer = RuntimePolicyConfig(
        invariant_runtime_behavior_profile=invariants.InvariantRuntimeBehaviorProfile.WARN,
        marker_kind_profile=MarkerKindProfile.COLLAPSE_TO_NEVER,
    )
    inner = RuntimePolicyConfig(
        invariant_runtime_behavior_profile=invariants.InvariantRuntimeBehaviorProfile.RATE,
        marker_kind_profile=MarkerKindProfile.NATIVE,
    )

    with runtime_policy_scope(outer):
        assert (
            invariants.invariant_runtime_behavior_config().profile
            is invariants.InvariantRuntimeBehaviorProfile.WARN
        )
        assert runtime_marker_kind_mapping_config().profile is MarkerKindProfile.COLLAPSE_TO_NEVER

        with runtime_policy_scope(inner):
            assert (
                invariants.invariant_runtime_behavior_config().profile
                is invariants.InvariantRuntimeBehaviorProfile.RATE
            )
            assert runtime_marker_kind_mapping_config().profile is MarkerKindProfile.NATIVE

        assert (
            invariants.invariant_runtime_behavior_config().profile
            is invariants.InvariantRuntimeBehaviorProfile.WARN
        )
        assert runtime_marker_kind_mapping_config().profile is MarkerKindProfile.COLLAPSE_TO_NEVER

    assert invariants.invariant_runtime_behavior_config() == baseline_invariant
    assert runtime_marker_kind_mapping_config() == baseline_marker
