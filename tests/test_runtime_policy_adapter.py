from __future__ import annotations

from gabion import invariants
from gabion.exceptions import NeverThrown
from gabion.order_contract import ordered_or_sorted
from gabion.runtime.policy_runtime import runtime_policy_from_env, runtime_policy_scope
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
