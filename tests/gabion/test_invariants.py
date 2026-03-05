from __future__ import annotations

import pytest

from gabion.analysis.foundation.marker_protocol import (
    MarkerKind, MarkerLifecycleState, MarkerPayload, SemanticLinkKind, SemanticReference)
from gabion import invariants
from gabion.exceptions import NeverThrown
from gabion.runtime.policy_runtime import RuntimePolicyConfig, runtime_policy_scope


@pytest.fixture(autouse=True)
def _reset_invariant_warning_counts() -> None:
    invariants.reset_invariant_warning_counts()


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


# gabion:evidence E:function_site::invariants.py::gabion.invariants.invariant_factory
@pytest.mark.parametrize(
    ("marker", "profile", "throws", "warns"),
    [
        ("never", "strict", True, False),
        ("never", "warn", True, False),
        ("never", "silent", True, False),
        ("todo", "strict", True, False),
        ("todo", "warn", False, True),
        ("todo", "silent", False, False),
        ("deprecated", "strict", True, False),
        ("deprecated", "warn", False, True),
        ("deprecated", "silent", False, False),
    ],
)
def test_invariant_factory_profile_behavior_matrix(
    marker: str,
    profile: str,
    throws: bool,
    warns: bool,
) -> None:
    kwargs = {"profile": profile}
    if warns:
        with pytest.warns(RuntimeWarning):
            invariants.invariant_factory(marker, "matrix", **kwargs)
    elif throws:
        with pytest.raises(NeverThrown):
            invariants.invariant_factory(marker, "matrix", **kwargs)
    else:
        invariants.invariant_factory(marker, "matrix", **kwargs)


# gabion:evidence E:function_site::invariants.py::gabion.invariants.invariant_factory
def test_invariant_factory_warning_rate_limit_is_deterministic() -> None:
    with pytest.warns(RuntimeWarning) as record:
        invariants.todo("later", profile="warn", owner="core", ticket="G-1")
        invariants.todo("later", profile="warn", owner="core", ticket="G-1")
        invariants.todo("later", profile="warn", owner="core", ticket="G-1")
    assert len(record) == 1

    with pytest.warns(RuntimeWarning) as next_record:
        invariants.todo("later", profile="warn", owner="core", ticket="G-2")
    assert len(next_record) == 1
