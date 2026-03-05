from __future__ import annotations

import warnings
from typing import cast

import pytest

from gabion.analysis.foundation.marker_protocol import (
    MarkerKind,
    MarkerLifecycleState,
    MarkerPayload,
    MarkerReasoning,
    SemanticLinkKind,
    SemanticReference,
)
from gabion import invariants
from gabion.exceptions import NeverThrown


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


# gabion:evidence E:call_footprint::tests/test_invariants.py::test_require_not_none_default_strict::invariants.py::gabion.invariants.require_not_none
def test_require_not_none_default_strict() -> None:
    with pytest.raises(NeverThrown):
        invariants.require_not_none(None)


# gabion:evidence E:function_site::invariants.py::gabion.invariants.decision_protocol
def test_decision_and_boundary_markers_return_original_callable() -> None:
    def _sample() -> str:
        return "ok"

    assert invariants.decision_protocol(_sample) is _sample
    assert invariants.boundary_normalization(_sample) is _sample


# gabion:evidence E:function_site::invariants.py::gabion.invariants.never


# gabion:evidence E:function_site::invariants.py::gabion.invariants.invariant_factory
def test_helper_functions_delegate_to_invariant_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object, dict[str, object]]] = []

    def _fake_factory(marker_kind: str, reasoning: object = "", **env: object) -> MarkerPayload:
        calls.append((marker_kind, reasoning, dict(env)))
        return cast(
            MarkerPayload,
            {"marker_kind": marker_kind, "reason": str(env.get("reason", ""))},
        )

    monkeypatch.setattr(invariants, "invariant_factory", _fake_factory)

    for helper, marker_kind in (
        (invariants.never, "never"),
        (invariants.todo, "todo"),
        (invariants.deprecated, "deprecated"),
    ):
        payload = helper("reason", owner="core")
        assert payload["marker_kind"] == marker_kind

    assert calls == [
        ("never", "", {"reason": "reason", "owner": "core"}),
        ("todo", "", {"reason": "reason", "owner": "core"}),
        ("deprecated", "", {"reason": "reason", "owner": "core"}),
    ]


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


def test_diagnostic_profile_returns_payload_and_emits_warning() -> None:
    with invariants.invariant_runtime_behavior_scope(
        invariants.InvariantRuntimeBehaviorConfig(profile=invariants.InvariantProfile.DIAGNOSTIC)
    ):
        with pytest.warns(invariants.InvariantMarkerWarning):
            never_payload = invariants.never("never diagnostic")
        assert never_payload.marker_kind is MarkerKind.NEVER

        with pytest.warns(invariants.InvariantMarkerWarning):
            todo_payload = invariants.todo("todo diagnostic")
        assert todo_payload.marker_kind is MarkerKind.TODO

        with pytest.warns(invariants.InvariantMarkerWarning):
            deprecated_payload = invariants.deprecated("deprecated diagnostic")
        assert deprecated_payload.marker_kind is MarkerKind.DEPRECATED


def test_diagnostic_profile_dedupes_repeated_warning_keys() -> None:
    with invariants.invariant_runtime_behavior_scope(
        invariants.InvariantRuntimeBehaviorConfig(profile=invariants.InvariantProfile.DIAGNOSTIC)
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            invariants.never(
                reasoning={
                    "summary": "same",
                    "control": "branch",
                    "blocking_dependencies": ["dep-a"],
                }
            )
            invariants.never(
                reasoning={
                    "summary": "same",
                    "control": "branch",
                    "blocking_dependencies": ["dep-a"],
                }
            )
            invariants.never(
                reasoning={
                    "summary": "different",
                    "control": "branch",
                    "blocking_dependencies": ["dep-a"],
                }
            )
        assert len(caught) == 2


def test_warning_key_changes_with_summary_control_and_dependencies() -> None:
    with invariants.invariant_runtime_behavior_scope(
        invariants.InvariantRuntimeBehaviorConfig(profile=invariants.InvariantProfile.DIAGNOSTIC)
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            invariants.never(
                reasoning={
                    "summary": "summary",
                    "control": "control-a",
                    "blocking_dependencies": ["dep-a"],
                }
            )
            invariants.never(
                reasoning={
                    "summary": "summary",
                    "control": "control-a",
                    "blocking_dependencies": ["dep-a"],
                }
            )
            invariants.never(
                reasoning={
                    "summary": "summary",
                    "control": "control-b",
                    "blocking_dependencies": ["dep-a"],
                }
            )
            invariants.never(
                reasoning={
                    "summary": "summary",
                    "control": "control-a",
                    "blocking_dependencies": ["dep-b"],
                }
            )
            invariants.never(
                reasoning={
                    "summary": "summary-2",
                    "control": "control-a",
                    "blocking_dependencies": ["dep-a"],
                }
            )
        assert len(caught) == 4


def test_warning_cap_suppresses_new_keys_after_limit() -> None:
    with invariants.invariant_runtime_behavior_scope(
        invariants.InvariantRuntimeBehaviorConfig(profile=invariants.InvariantProfile.SUNSET_GATE)
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for index in range(55):
                invariants.todo(reasoning={"summary": f"todo-{index}"})
            invariants.todo(reasoning={"summary": "todo-0"})
        assert len(caught) == 50


def test_invariant_runtime_behavior_scope_restores_warning_state() -> None:
    profile = invariants.InvariantRuntimeBehaviorConfig(profile=invariants.InvariantProfile.DIAGNOSTIC)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with invariants.invariant_runtime_behavior_scope(profile):
            invariants.never(reasoning={"summary": "same", "control": "branch"})
            with invariants.invariant_runtime_behavior_scope(profile):
                invariants.never(reasoning={"summary": "same", "control": "branch"})
            invariants.never(reasoning={"summary": "same", "control": "branch"})
    assert len(caught) == 2
