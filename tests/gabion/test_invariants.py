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
# gabion:behavior primary=verboten facets=never,raises
def test_never_raises_never_thrown() -> None:
    with pytest.raises(NeverThrown):
        invariants.never("boom", flag=True)


# gabion:evidence E:call_footprint::tests/test_invariants.py::test_require_not_none_non_strict::invariants.py::gabion.invariants.require_not_none
# gabion:behavior primary=verboten facets=none,strict
def test_require_not_none_non_strict() -> None:
    assert invariants.require_not_none(None, strict=False) is None
    assert invariants.require_not_none("ok", strict=False) == "ok"


# gabion:evidence E:call_footprint::tests/test_invariants.py::test_require_not_none_strict_raises::invariants.py::gabion.invariants.require_not_none
# gabion:behavior primary=verboten facets=none,raises,strict
def test_require_not_none_strict_raises() -> None:
    with pytest.raises(NeverThrown):
        invariants.require_not_none(None, strict=True)


# gabion:evidence E:call_footprint::tests/test_invariants.py::test_require_not_none_default_strict::invariants.py::gabion.invariants.require_not_none
# gabion:behavior primary=verboten facets=none,strict
def test_require_not_none_default_strict() -> None:
    with pytest.raises(NeverThrown):
        invariants.require_not_none(None)


# gabion:evidence E:function_site::invariants.py::gabion.invariants.decision_protocol
# gabion:behavior primary=desired
def test_decision_and_boundary_markers_return_original_callable() -> None:
    def _sample() -> str:
        return "ok"

    assert invariants.decision_protocol(_sample) is _sample
    assert invariants.boundary_normalization(_sample) is _sample


# gabion:behavior primary=desired
def test_invariant_decorator_attaches_marker_payload_metadata() -> None:
    @invariants.todo_decorator(
        "debt marker",
        owner="core",
        reasoning={
            "summary": "debt marker",
            "control": "refactor-plan",
            "blocking_dependencies": ["dep-b", "dep-a", "dep-b"],
        },
    )
    def _flagged_function() -> str:
        return "ok"

    @invariants.deprecated_decorator("legacy class marker", owner="runtime")
    class _FlaggedClass:
        pass

    function_payloads = invariants.invariant_decorations(_flagged_function)
    assert len(function_payloads) == 1
    function_payload = function_payloads[0]
    assert function_payload.marker_kind is MarkerKind.TODO
    assert function_payload.reasoning.control == "refactor-plan"
    assert function_payload.reasoning.blocking_dependencies == ("dep-a", "dep-b")

    class_payloads = invariants.invariant_decorations(_FlaggedClass)
    assert len(class_payloads) == 1
    assert class_payloads[0].marker_kind is MarkerKind.DEPRECATED
    assert _flagged_function() == "ok"


# gabion:behavior primary=desired
def test_invariant_decorator_stacks_and_emits_no_runtime_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @invariants.never_decorator("outer never")
        @invariants.todo_decorator("inner todo")
        def _stacked_function() -> str:
            return "stacked"

    payloads = invariants.invariant_decorations(_stacked_function)
    assert [payload.marker_kind for payload in payloads] == [
        MarkerKind.TODO,
        MarkerKind.NEVER,
    ]
    assert _stacked_function() == "stacked"
    assert caught == []


# gabion:behavior primary=verboten facets=never
def test_never_string_only_call_emits_deprecation_marker_before_never(
) -> None:
    deprecated_calls: list[tuple[str, dict[str, object]]] = []
    factory_calls: list[tuple[str, dict[str, object]]] = []

    def _fake_deprecated(reason: str = "", **env: object) -> MarkerPayload:
        deprecated_calls.append((reason, dict(env)))
        raise NeverThrown("deprecated marker path")

    def _fake_factory(marker_kind: str, reasoning: object = "", **env: object) -> MarkerPayload:
        _ = reasoning
        factory_calls.append((marker_kind, dict(env)))
        raise RuntimeError("never factory reached")

    with pytest.raises(RuntimeError, match="never factory reached"):
        invariants.never(
            "legacy reason",
            emit_legacy_never_string_reason_deprecation_fn=(
                lambda reason: invariants._emit_legacy_never_string_reason_deprecation(
                    reason,
                    deprecated_fn=_fake_deprecated,
                )
            ),
            invariant_factory_fn=_fake_factory,
        )

    assert len(deprecated_calls) == 1
    deprecation_reason, deprecation_env = deprecated_calls[0]
    assert "string-only API is deprecated" in deprecation_reason
    assert deprecation_env.get("legacy_api") == "never(reason: str)"
    assert len(factory_calls) == 1
    assert factory_calls[0][0] == "never"


# gabion:behavior primary=verboten facets=never
def test_never_with_structured_reasoning_skips_string_only_deprecation(
) -> None:
    deprecated_calls = 0

    def _unexpected_legacy_deprecation(_reason: str) -> None:
        nonlocal deprecated_calls
        deprecated_calls += 1

    with pytest.raises(NeverThrown):
        invariants.never(
            "legacy-style message still present",
            reasoning={"summary": "structured path"},
            emit_legacy_never_string_reason_deprecation_fn=_unexpected_legacy_deprecation,
        )

    assert deprecated_calls == 0


# gabion:behavior primary=verboten facets=never
def test_never_with_metadata_only_still_emits_string_only_deprecation_preflight(
) -> None:
    deprecated_calls: list[tuple[str, dict[str, object]]] = []

    def _fake_deprecated(reason: str = "", **env: object) -> MarkerPayload:
        deprecated_calls.append((reason, dict(env)))
        raise NeverThrown("deprecated marker path")

    with pytest.raises(NeverThrown):
        invariants.never(
            "legacy-style message still present",
            owner="core",
            emit_legacy_never_string_reason_deprecation_fn=(
                lambda reason: invariants._emit_legacy_never_string_reason_deprecation(
                    reason,
                    deprecated_fn=_fake_deprecated,
                )
            ),
        )

    assert len(deprecated_calls) == 1


# gabion:evidence E:function_site::invariants.py::gabion.invariants.never


# gabion:evidence E:function_site::invariants.py::gabion.invariants.invariant_factory
# gabion:behavior primary=desired
def test_helper_functions_delegate_to_invariant_factory(
) -> None:
    calls: list[tuple[str, object, dict[str, object]]] = []

    def _fake_factory(marker_kind: str, reasoning: object = "", **env: object) -> MarkerPayload:
        calls.append((marker_kind, reasoning, dict(env)))
        return cast(
            MarkerPayload,
            {"marker_kind": marker_kind, "reason": str(env.get("reason", ""))},
        )

    helper_calls = (
        (
            invariants.never,
            "never",
            {
                "reasoning": {"summary": "reason"},
                "owner": "core",
                "invariant_factory_fn": _fake_factory,
            },
        ),
        (
            invariants.todo,
            "todo",
            {
                "owner": "core",
                "invariant_factory_fn": _fake_factory,
            },
        ),
        (
            invariants.deprecated,
            "deprecated",
            {
                "owner": "core",
                "invariant_factory_fn": _fake_factory,
            },
        ),
    )
    for helper, marker_kind, kwargs in helper_calls:
        payload = helper("reason", **kwargs)
        assert payload["marker_kind"] == marker_kind

    assert calls == [
        (
            "never",
            {"summary": "reason"},
            {"reason": "reason", "owner": "core"},
        ),
        ("todo", "", {"reason": "reason", "owner": "core"}),
        ("deprecated", "", {"reason": "reason", "owner": "core"}),
    ]


# gabion:behavior primary=verboten facets=never
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
# gabion:behavior primary=verboten facets=never
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


# gabion:behavior primary=desired
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


# gabion:behavior primary=desired
@pytest.mark.parametrize(
    ("profile", "warns"),
    (
        (invariants.InvariantProfile.STRICT, False),
        (invariants.InvariantProfile.DIAGNOSTIC, True),
        (invariants.InvariantProfile.DEBT_GATE, True),
        (invariants.InvariantProfile.SUNSET_GATE, True),
    ),
)
def test_todo_is_non_throwing_in_all_profiles(
    profile: invariants.InvariantProfile,
    warns: bool,
) -> None:
    with invariants.invariant_runtime_behavior_scope(
        invariants.InvariantRuntimeBehaviorConfig(profile=profile)
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            payload = invariants.todo(
                reasoning={
                    "summary": f"todo-{profile.value}",
                    "control": "pr412.identity_contract.partial",
                    "blocking_dependencies": ["typed_contract_migration"],
                }
            )
    assert payload.marker_kind is MarkerKind.TODO
    assert bool(caught) is warns


# gabion:behavior primary=desired
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


# gabion:behavior primary=desired
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


# gabion:behavior primary=desired
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


# gabion:behavior primary=desired
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
