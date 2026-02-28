from __future__ import annotations

import pytest

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
