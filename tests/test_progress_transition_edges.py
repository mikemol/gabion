from __future__ import annotations

import pytest

import gabion.commands.progress_transition as progress_transition_module
from gabion.commands.progress_transition import (
    NormalizedProgressTransition,
    ProgressMarkerParts,
    ProgressNode,
    normalize_progress_transition_boundary,
    normalize_progress_transition_from_phase_progress,
    progress_transition_v1_payload,
    progress_transition_v2_payload,
    transition_event_kind_from_phase_progress,
    transition_marker_from_phase_progress,
    transition_primary_from_phase_progress,
    transition_reason_from_phase_progress,
    validate_progress_transition,
)


@pytest.mark.parametrize(
    ("phase_progress",),
    [
        (
            {
                "event_kind": "progress",
                "progress_transition_v1": {"event_kind": "progress"},
            },
        ),
        (
            {
                "phase": "post",
                "event_kind": "invalid",
                "progress_transition_v1": {
                    "phase": "post",
                    "event_kind": "invalid",
                },
            },
        ),
        (
            {
                "phase": "post",
                "event_kind": "progress",
                "progress_transition_v1": {
                    "phase": "post",
                    "event_kind": "progress",
                },
            },
        ),
    ],
)
# gabion:evidence E:call_footprint::tests/test_progress_transition_edges.py::test_normalize_progress_transition_from_phase_progress_rejects_invalid_shapes::progress_transition.py::gabion.commands.progress_transition.normalize_progress_transition_from_phase_progress
def test_normalize_progress_transition_from_phase_progress_rejects_invalid_shapes(
    phase_progress: dict[str, object],
) -> None:
    assert normalize_progress_transition_from_phase_progress(phase_progress) is None


@pytest.mark.parametrize(
    ("phase_progress", "expected_done", "expected_total"),
    [
        (
            {
                "phase": "post",
                "event_kind": "progress",
                "phase_progress_v2": {
                    "primary_unit": "post_tasks",
                    "primary_done": 9,
                    "primary_total": 5,
                },
                "progress_transition_v1": {
                    "phase": "post",
                    "event_kind": "progress",
                    "parent": {"unit": "post_tasks"},
                    "child": {"marker_text": "fingerprint:normalize"},
                },
            },
            5,
            5,
        ),
        (
            {
                "phase": "post",
                "event_kind": "progress",
                "progress_transition_v1": {
                    "phase": "post",
                    "event_kind": "progress",
                    "parent": {"unit": "post_tasks", "done": 9, "total": 6},
                    "child": {"marker_text": "fingerprint:normalize"},
                },
            },
            6,
            6,
        ),
    ],
)
# gabion:evidence E:call_footprint::tests/test_progress_transition_edges.py::test_normalize_progress_transition_from_phase_progress_clamps_done_to_total::progress_transition.py::gabion.commands.progress_transition.normalize_progress_transition_from_phase_progress
def test_normalize_progress_transition_from_phase_progress_clamps_done_to_total(
    phase_progress: dict[str, object],
    expected_done: int,
    expected_total: int,
) -> None:
    transition = normalize_progress_transition_from_phase_progress(phase_progress)
    assert transition is not None
    assert transition.primary_done == expected_done
    assert transition.primary_total == expected_total


# gabion:evidence E:call_footprint::tests/test_progress_transition_edges.py::test_normalize_progress_transition_from_phase_progress_falls_back_to_progress_marker::progress_transition.py::gabion.commands.progress_transition.normalize_progress_transition_from_phase_progress
def test_normalize_progress_transition_from_phase_progress_falls_back_to_progress_marker() -> None:
    transition = normalize_progress_transition_from_phase_progress(
        {
            "phase": "post",
            "event_kind": "progress",
            "progress_marker": "fingerprint:done",
            "work_done": 5,
            "work_total": 6,
            "progress_transition_v1": {
                "phase": "post",
                "event_kind": "progress",
                "parent": {"unit": "post_tasks", "done": 5, "total": 6},
                "child": {},
            },
        }
    )
    assert transition is not None
    assert transition.marker.marker_text == "fingerprint:done"


# gabion:evidence E:call_footprint::tests/test_progress_transition_edges.py::test_normalize_progress_transition_from_phase_progress_falls_back_to_progress_marker_when_child_is_not_mapping::progress_transition.py::gabion.commands.progress_transition.normalize_progress_transition_from_phase_progress
def test_normalize_progress_transition_from_phase_progress_falls_back_to_progress_marker_when_child_is_not_mapping() -> None:
    transition = normalize_progress_transition_from_phase_progress(
        {
            "phase": "post",
            "event_kind": "progress",
            "progress_marker": "fingerprint:warnings",
            "work_done": 5,
            "work_total": 6,
            "progress_transition_v1": {
                "phase": "post",
                "event_kind": "progress",
                "parent": {"unit": "post_tasks", "done": 5, "total": 6},
                "child": "invalid-child-shape",
            },
        }
    )
    assert transition is not None
    assert transition.marker.marker_text == "fingerprint:warnings"


# gabion:evidence E:call_footprint::tests/test_progress_transition_edges.py::test_transition_helper_accessors_return_none_when_transition_missing::progress_transition.py::gabion.commands.progress_transition.transition_event_kind_from_phase_progress
def test_transition_helper_accessors_return_none_when_transition_missing() -> None:
    phase_progress = {"phase": "post", "event_kind": "progress"}
    assert transition_marker_from_phase_progress(phase_progress) is None
    assert transition_primary_from_phase_progress(phase_progress) == ("", None, None)
    assert transition_event_kind_from_phase_progress(phase_progress) is None


# gabion:evidence E:call_footprint::tests/test_progress_transition_edges.py::test_transition_helper_accessors_return_transition_fields_when_present::progress_transition.py::gabion.commands.progress_transition.transition_marker_from_phase_progress
def test_transition_helper_accessors_return_transition_fields_when_present() -> None:
    phase_progress = {
        "phase": "post",
        "event_kind": "progress",
        "progress_transition_v1": {
            "phase": "post",
            "event_kind": "terminal",
            "parent": {"unit": "post_tasks", "done": 6, "total": 6},
            "child": {"marker_text": "complete"},
        },
    }
    assert transition_marker_from_phase_progress(phase_progress) == "complete"
    assert transition_primary_from_phase_progress(phase_progress) == ("post_tasks", 6, 6)
    assert transition_event_kind_from_phase_progress(phase_progress) == "terminal"


# gabion:evidence E:call_footprint::tests/test_progress_transition_edges.py::test_transition_reason_from_phase_progress_rejects_non_string_reason::progress_transition.py::gabion.commands.progress_transition.transition_reason_from_phase_progress
def test_transition_reason_from_phase_progress_rejects_non_string_reason() -> None:
    reason = transition_reason_from_phase_progress(
        {
            "progress_transition_v1": {
                "reason": 123,
            }
        }
    )
    assert reason is None


def test_transition_reason_from_phase_progress_returns_none_when_transition_missing() -> None:
    assert transition_reason_from_phase_progress({}) is None


# gabion:evidence E:call_footprint::tests/test_progress_transition_edges.py::test_validate_progress_transition_rejects_mutated_terminal_replay::progress_transition.py::gabion.commands.progress_transition.validate_progress_transition
def test_validate_progress_transition_rejects_mutated_terminal_replay() -> None:
    previous = normalize_progress_transition_boundary(
        phase="post",
        analysis_state="analysis_post_done",
        event_kind="terminal",
        primary_unit="post_tasks",
        primary_done=6,
        primary_total=6,
        progress_marker="complete",
    )
    current = normalize_progress_transition_boundary(
        phase="post",
        analysis_state="analysis_post_complete",
        event_kind="terminal",
        primary_unit="post_tasks",
        primary_done=6,
        primary_total=6,
        progress_marker="complete",
    )
    decision = validate_progress_transition(previous=previous, current=current)
    assert decision.valid is False
    assert decision.reason == "invalid_terminal_replay_mutated_state"


# gabion:evidence E:call_footprint::tests/test_progress_transition_edges.py::test_validate_progress_transition_allows_checkpoint_terminal_replay_as_parent_hold::progress_transition.py::gabion.commands.progress_transition.validate_progress_transition
def test_validate_progress_transition_allows_checkpoint_terminal_replay_as_parent_hold() -> None:
    previous = normalize_progress_transition_boundary(
        phase="post",
        analysis_state="analysis_post_done",
        event_kind="terminal",
        primary_unit="post_tasks",
        primary_done=6,
        primary_total=6,
        progress_marker="complete",
    )
    current = normalize_progress_transition_boundary(
        phase="post",
        analysis_state="analysis_post_done",
        event_kind="checkpoint",
        primary_unit="post_tasks",
        primary_done=6,
        primary_total=6,
        progress_marker="complete",
    )
    decision = validate_progress_transition(previous=previous, current=current)
    assert decision.valid is True
    assert decision.reason == "parent_held"


# gabion:evidence E:call_footprint::tests/test_progress_transition_edges.py::test_normalize_progress_transition_from_phase_progress_prefers_recursive_v2_payload::progress_transition.py::gabion.commands.progress_transition.normalize_progress_transition_from_phase_progress
def test_normalize_progress_transition_from_phase_progress_prefers_recursive_v2_payload() -> None:
    transition = normalize_progress_transition_from_phase_progress(
        {
            "phase": "post",
            "analysis_state": "analysis_post_in_progress",
            "event_kind": "progress",
            "work_done": 2,
            "work_total": 6,
            "progress_marker": "stale",
            "progress_transition_v2": {
                "phase": "post",
                "analysis_state": "analysis_post_in_progress",
                "event_kind": "progress",
                "root": {
                    "identity": "post_root",
                    "unit": "post_tasks",
                    "done": 5,
                    "total": 6,
                    "marker_text": "root",
                    "children": [
                        {
                            "identity": "fingerprint",
                            "unit": "post_tasks",
                            "done": 5,
                            "total": 6,
                            "marker_text": "fingerprint",
                            "children": [
                                {
                                    "identity": "fingerprint:normalize",
                                    "unit": "post_tasks",
                                    "done": 5,
                                    "total": 6,
                                    "marker_text": "fingerprint:normalize",
                                    "children": [],
                                }
                            ],
                        }
                    ],
                },
                "active_path": [
                    "post_root",
                    "fingerprint",
                    "fingerprint:normalize",
                ],
            },
            "progress_transition_v1": {
                "phase": "post",
                "event_kind": "terminal",
                "parent": {"unit": "post_tasks", "done": 6, "total": 6},
                "child": {"marker_text": "complete"},
            },
        }
    )
    assert transition is not None
    assert transition.primary_done == 5
    assert transition.primary_total == 6
    assert transition.marker.marker_text == "fingerprint:normalize"
    assert transition.active_path == (
        "post_root",
        "fingerprint",
        "fingerprint:normalize",
    )


@pytest.mark.parametrize(
    ("transition", "expected_reason"),
    [
        (
            NormalizedProgressTransition(
                phase="post",
                analysis_state="analysis_post_in_progress",
                event_kind="progress",
                root=ProgressNode(
                    identity="root",
                    unit="post_tasks",
                    done=5,
                    total=6,
                    marker=ProgressMarkerParts(
                        marker_text="fingerprint:normalize",
                        marker_family="fingerprint",
                        marker_step="normalize",
                    ),
                    children=(),
                ),
                active_path=("root", "missing"),
                terminal_complete=False,
            ),
            "invalid_active_path",
        ),
        (
            NormalizedProgressTransition(
                phase="post",
                analysis_state="analysis_post_in_progress",
                event_kind="progress",
                root=ProgressNode(
                    identity="root",
                    unit="post_tasks",
                    done=5,
                    total=6,
                    marker=ProgressMarkerParts(
                        marker_text="fingerprint",
                        marker_family="fingerprint",
                        marker_step="",
                    ),
                    children=(
                        ProgressNode(
                            identity="fingerprint",
                            unit="post_tasks",
                            done=5,
                            total=6,
                            marker=ProgressMarkerParts(
                                marker_text="fingerprint:normalize",
                                marker_family="fingerprint",
                                marker_step="normalize",
                            ),
                            children=(),
                        ),
                        ProgressNode(
                            identity="fingerprint",
                            unit="post_tasks",
                            done=5,
                            total=6,
                            marker=ProgressMarkerParts(
                                marker_text="fingerprint:warnings",
                                marker_family="fingerprint",
                                marker_step="warnings",
                            ),
                            children=(),
                        ),
                    ),
                ),
                active_path=("root", "fingerprint"),
                terminal_complete=False,
            ),
            "invalid_duplicate_sibling_identity",
        ),
    ],
)
# gabion:evidence E:call_footprint::tests/test_progress_transition_edges.py::test_validate_progress_transition_rejects_invalid_recursive_structures::progress_transition.py::gabion.commands.progress_transition.validate_progress_transition
def test_validate_progress_transition_rejects_invalid_recursive_structures(
    transition: NormalizedProgressTransition,
    expected_reason: str,
) -> None:
    decision = validate_progress_transition(previous=None, current=transition)
    assert decision.valid is False
    assert decision.reason == expected_reason


# gabion:evidence E:call_footprint::tests/test_progress_transition_edges.py::test_validate_progress_transition_rejects_node_regression::progress_transition.py::gabion.commands.progress_transition.validate_progress_transition
def test_validate_progress_transition_rejects_node_regression() -> None:
    previous = normalize_progress_transition_boundary(
        phase="post",
        analysis_state="analysis_post_in_progress",
        event_kind="progress",
        primary_unit="post_tasks",
        primary_done=5,
        primary_total=6,
        progress_marker="fingerprint:normalize",
    )
    current = normalize_progress_transition_boundary(
        phase="post",
        analysis_state="analysis_post_in_progress",
        event_kind="progress",
        primary_unit="post_tasks",
        primary_done=4,
        primary_total=6,
        progress_marker="fingerprint:warnings",
    )
    decision = validate_progress_transition(previous=previous, current=current)
    assert decision.valid is False
    assert decision.reason == "invalid_node_done_regressed"


# gabion:evidence E:call_footprint::tests/test_progress_transition_edges.py::test_progress_transition_payload_projection_includes_recursive_path::progress_transition.py::gabion.commands.progress_transition.progress_transition_v2_payload
def test_progress_transition_payload_projection_includes_recursive_path() -> None:
    transition = normalize_progress_transition_boundary(
        phase="post",
        analysis_state="analysis_post_in_progress",
        event_kind="progress",
        primary_unit="post_tasks",
        primary_done=6,
        primary_total=6,
        progress_marker="complete",
    )
    v2 = progress_transition_v2_payload(
        transition=transition,
        reason="terminal_transition",
        effective_event_kind="terminal",
    )
    v1 = progress_transition_v1_payload(
        transition=transition,
        reason="terminal_transition",
        effective_event_kind="terminal",
    )
    assert v2["active_path"] == list(transition.active_path)
    assert isinstance(v1["child"], dict)
    child = v1["child"]
    assert child.get("path") == list(transition.active_path)
    assert child.get("marker_text") == "complete"


def test_transition_reason_from_phase_progress_accepts_v2_reason() -> None:
    reason = transition_reason_from_phase_progress(
        {"progress_transition_v2": {"reason": "parent_held"}}
    )
    assert reason == "parent_held"


def test_normalized_progress_transition_active_node_falls_back_to_root() -> None:
    root = ProgressNode(
        identity="root",
        unit="post_tasks",
        done=1,
        total=2,
        marker=ProgressMarkerParts(
            marker_text="root",
            marker_family="root",
            marker_step="",
        ),
        children=(),
    )
    transition_empty = NormalizedProgressTransition(
        phase="post",
        analysis_state="analysis_post_in_progress",
        event_kind="progress",
        root=root,
        active_path=(),
        terminal_complete=False,
    )
    transition_wrong_root = NormalizedProgressTransition(
        phase="post",
        analysis_state="analysis_post_in_progress",
        event_kind="progress",
        root=root,
        active_path=("other",),
        terminal_complete=False,
    )
    assert transition_empty.active_node is root
    assert transition_wrong_root.active_node is root


@pytest.mark.parametrize(
    "phase_progress",
    [
        {
            "event_kind": "progress",
            "progress_transition_v2": {"event_kind": "progress", "root": {}},
        },
        {
            "phase": "post",
            "event_kind": "invalid",
            "work_done": 1,
            "work_total": 2,
            "progress_transition_v2": {
                "phase": "post",
                "event_kind": "invalid",
                "root": {},
            },
        },
        {
            "phase": "post",
            "event_kind": "progress",
            "progress_transition_v2": {
                "phase": "post",
                "event_kind": "progress",
                "root": {},
            },
        },
        {
            "phase": "post",
            "event_kind": "progress",
            "work_done": 1,
            "work_total": 2,
            "progress_transition_v2": {
                "phase": "post",
                "event_kind": "progress",
                "root": [],
            },
        },
        {
            "phase": "post",
            "event_kind": "progress",
            "work_done": 1,
            "work_total": 2,
            "progress_transition_v2": {
                "phase": "post",
                "event_kind": "progress",
                "root": {"children": [123]},
            },
        },
        {
            "phase": "post",
            "event_kind": "progress",
            "work_done": 1,
            "work_total": 2,
            "progress_transition_v2": {
                "phase": "post",
                "event_kind": "progress",
                "root": {
                    "identity": "root",
                    "children": [
                        {
                            "identity": "child",
                            "children": [123],
                        }
                    ],
                },
            },
        },
        {
            "phase": "post",
            "event_kind": "progress",
            "work_done": 1,
            "work_total": 2,
            "progress_transition_v2": {
                "phase": "post",
                "event_kind": "progress",
                "root": {
                    "identity": "root",
                    "done": 1,
                    "total": 2,
                    "marker_text": "root",
                },
                "active_path": ["other"],
            },
        },
    ],
)
def test_normalize_progress_transition_from_phase_progress_v2_rejects_invalid_shapes(
    phase_progress: dict[str, object],
) -> None:
    assert normalize_progress_transition_from_phase_progress(phase_progress) is None


@pytest.mark.parametrize(
    "active_path_payload",
    [
        "bad-shape",
        [],
        ["root", 1],
    ],
)
def test_normalize_progress_transition_from_phase_progress_v2_falls_back_active_path(
    active_path_payload: object,
) -> None:
    transition = normalize_progress_transition_from_phase_progress(
        {
            "phase": "post",
            "event_kind": "progress",
            "work_done": 1,
            "work_total": 2,
            "progress_transition_v2": {
                "phase": "post",
                "event_kind": "progress",
                "root": {
                    "identity": "root",
                    "done": 1,
                    "total": 2,
                    "marker": {
                        "marker_text": "fingerprint:normalize",
                        "marker_family": "fingerprint",
                        "marker_step": "normalize",
                    },
                    "children": [],
                },
                "active_path": active_path_payload,
            },
        }
    )
    assert transition is not None
    assert transition.active_path == ("root",)


def test_normalize_progress_transition_from_phase_progress_v2_applies_identity_fallback() -> None:
    transition = normalize_progress_transition_from_phase_progress(
        {
            "phase": "post",
            "event_kind": "progress",
            "work_done": 1,
            "work_total": 2,
            "progress_transition_v2": {
                "phase": "post",
                "event_kind": "progress",
                "root": {
                    "identity": "",
                    "done": 1,
                    "total": 2,
                    "marker": {
                        "marker_text": "fingerprint:normalize",
                    },
                    "children": [],
                },
            },
        }
    )
    assert transition is not None
    assert transition.root.identity == "__root__"


def test_normalize_progress_transition_from_phase_progress_v1_accepts_explicit_identities() -> None:
    transition = normalize_progress_transition_from_phase_progress(
        {
            "phase": "post",
            "event_kind": "progress",
            "work_done": 5,
            "work_total": 6,
            "progress_transition_v1": {
                "phase": "post",
                "event_kind": "progress",
                "parent": {
                    "identity": "parent-id",
                    "unit": "post_tasks",
                    "done": 5,
                    "total": 6,
                },
                "child": {
                    "identity": "child-id",
                    "marker_text": "fingerprint:normalize",
                },
            },
        }
    )
    assert transition is not None
    assert transition.root.identity == "parent-id"
    assert transition.active_node.identity == "child-id"


def test_normalize_progress_transition_from_phase_progress_v1_rejects_empty_marker() -> None:
    transition = normalize_progress_transition_from_phase_progress(
        {
            "phase": "post",
            "event_kind": "progress",
            "work_done": 5,
            "work_total": 6,
            "progress_transition_v1": {
                "phase": "post",
                "event_kind": "progress",
                "parent": {"unit": "post_tasks", "done": 5, "total": 6},
                "child": {"marker_text": ""},
            },
        }
    )
    assert transition is None


def test_normalize_progress_transition_boundary_clamps_done_to_total() -> None:
    transition = normalize_progress_transition_boundary(
        phase="post",
        analysis_state=None,
        event_kind="progress",
        primary_unit="post_tasks",
        primary_done=7,
        primary_total=3,
        progress_marker="fingerprint:normalize",
    )
    assert transition.primary_done == 3
    assert transition.primary_total == 3


def test_private_node_identity_from_marker_uses_marker_text_fallback() -> None:
    marker = ProgressMarkerParts(marker_text="standalone", marker_family="", marker_step="")
    identity = progress_transition_module._node_identity_from_marker(marker)
    assert identity == "standalone"


@pytest.mark.parametrize(
    ("node", "expected_reason"),
    [
        (
            ProgressNode(
                identity="",
                unit="post_tasks",
                done=1,
                total=2,
                marker=ProgressMarkerParts(
                    marker_text="x",
                    marker_family="x",
                    marker_step="",
                ),
                children=(),
            ),
            "invalid_empty_node_identity",
        ),
        (
            ProgressNode(
                identity="root",
                unit="post_tasks",
                done=3,
                total=2,
                marker=ProgressMarkerParts(
                    marker_text="x",
                    marker_family="x",
                    marker_step="",
                ),
                children=(),
            ),
            "invalid_node_done_exceeds_total",
        ),
        (
            ProgressNode(
                identity="root",
                unit="post_tasks",
                done=1,
                total=2,
                marker=ProgressMarkerParts(
                    marker_text="x",
                    marker_family="x",
                    marker_step="",
                ),
                children=(
                    ProgressNode(
                        identity="",
                        unit="post_tasks",
                        done=1,
                        total=2,
                        marker=ProgressMarkerParts(
                            marker_text="y",
                            marker_family="y",
                            marker_step="",
                        ),
                        children=(),
                    ),
                ),
            ),
            "invalid_empty_node_identity",
        ),
    ],
)
def test_private_validate_tree_structure_reports_expected_reason(
    node: ProgressNode,
    expected_reason: str,
) -> None:
    reason = progress_transition_module._validate_tree_structure(node)
    assert reason == expected_reason


@pytest.mark.parametrize(
    ("previous", "current", "expected_reason"),
    [
        (
            ProgressNode(
                identity="a",
                unit="u",
                done=1,
                total=2,
                marker=ProgressMarkerParts("m", "m", ""),
                children=(),
            ),
            ProgressNode(
                identity="b",
                unit="u",
                done=1,
                total=2,
                marker=ProgressMarkerParts("m", "m", ""),
                children=(),
            ),
            "invalid_node_identity_drift",
        ),
        (
            ProgressNode(
                identity="a",
                unit="u",
                done=1,
                total=3,
                marker=ProgressMarkerParts("m", "m", ""),
                children=(),
            ),
            ProgressNode(
                identity="a",
                unit="u",
                done=1,
                total=2,
                marker=ProgressMarkerParts("m", "m", ""),
                children=(),
            ),
            "invalid_node_total_regressed",
        ),
        (
            ProgressNode(
                identity="a",
                unit="u",
                done=1,
                total=2,
                marker=ProgressMarkerParts("m", "m", ""),
                children=(
                    ProgressNode(
                        identity="child",
                        unit="u",
                        done=1,
                        total=2,
                        marker=ProgressMarkerParts("m", "m", ""),
                        children=(),
                    ),
                ),
            ),
            ProgressNode(
                identity="a",
                unit="u",
                done=1,
                total=2,
                marker=ProgressMarkerParts("m", "m", ""),
                children=(
                    ProgressNode(
                        identity="child",
                        unit="u",
                        done=0,
                        total=2,
                        marker=ProgressMarkerParts("m", "m", ""),
                        children=(),
                    ),
                ),
            ),
            "invalid_node_done_regressed",
        ),
    ],
)
def test_private_validate_tree_progress_transition_reports_expected_reason(
    previous: ProgressNode,
    current: ProgressNode,
    expected_reason: str,
) -> None:
    reason = progress_transition_module._validate_tree_progress_transition(previous, current)
    assert reason == expected_reason


def test_validate_progress_transition_rejects_invalid_previous_shapes() -> None:
    current = normalize_progress_transition_boundary(
        phase="post",
        analysis_state="analysis_post_in_progress",
        event_kind="progress",
        primary_unit="post_tasks",
        primary_done=5,
        primary_total=6,
        progress_marker="fingerprint:normalize",
    )
    previous_invalid_tree = NormalizedProgressTransition(
        phase="post",
        analysis_state="analysis_post_in_progress",
        event_kind="progress",
        root=ProgressNode(
            identity="",
            unit="post_tasks",
            done=5,
            total=6,
            marker=ProgressMarkerParts("x", "x", ""),
            children=(),
        ),
        active_path=("root",),
        terminal_complete=False,
    )
    previous_invalid_path = NormalizedProgressTransition(
        phase="post",
        analysis_state="analysis_post_in_progress",
        event_kind="progress",
        root=ProgressNode(
            identity="root",
            unit="post_tasks",
            done=5,
            total=6,
            marker=ProgressMarkerParts("x", "x", ""),
            children=(),
        ),
        active_path=("other",),
        terminal_complete=False,
    )
    tree_decision = validate_progress_transition(
        previous=previous_invalid_tree,
        current=current,
    )
    path_decision = validate_progress_transition(
        previous=previous_invalid_path,
        current=current,
    )
    assert tree_decision.reason == "invalid_previous_transition_state"
    assert path_decision.reason == "invalid_previous_active_path"
