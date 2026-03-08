from __future__ import annotations

from gabion.commands import progress_contract


def _progress_transition_v2_payload(
    *,
    event_kind: str,
    done: int,
    total: int,
    marker: str,
    reason: str = "parent_held",
) -> dict[str, object]:
    return {
        "phase": "post",
        "analysis_state": "analysis_post_in_progress",
        "event_kind": event_kind,
        "reason": reason,
        "root": {
            "identity": "post_root",
            "unit": "post_tasks",
            "done": done,
            "total": total,
            "marker_text": "root",
            "children": [
                {
                    "identity": marker,
                    "unit": "post_tasks",
                    "done": done,
                    "total": total,
                    "marker_text": marker,
                    "children": [],
                }
            ],
        },
        "active_path": ["post_root", marker],
    }


def _canonical_progress_notification(
    payload: dict[str, object],
    *,
    adaptation_kind: str = "valid",
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "gabion/canonical_progress_event_v2",
        "format_version": 2,
        "adaptation_kind": adaptation_kind,
        "event": None,
        "adaptation_error": "",
        "identity_allocation_delta_v1": [],
        "rejected_progress_payload_v2": None,
    }
    if adaptation_kind == "valid":
        phase = str(payload.get("phase", "") or "")
        event_kind = str(payload.get("event_kind", "") or "")
        value["event"] = {
            "schema_version": 1,
            "sequence": 1,
            "run_id": "run:test",
            "source": "dataflow.progress",
            "phase": phase,
            "kind": event_kind,
            "identity_projection": {},
            "payload": dict(payload),
            "causal_refs": [],
            "event_id": "run:test:1",
        }
    else:
        value["adaptation_error"] = "canonical adaptation rejected"
        value["rejected_progress_payload_v2"] = dict(payload)
    return {
        "method": progress_contract.LSP_PROGRESS_NOTIFICATION_METHOD,
        "params": {
            "token": progress_contract.LSP_PROGRESS_TOKEN_V2,
            "value": value,
        },
    }


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_progress_dimensions_summary_keeps_zero_total_without_clamp::progress_contract.py::gabion.commands.progress_contract.phase_progress_dimensions_summary
# gabion:behavior primary=verboten facets=edge
def test_phase_progress_dimensions_summary_keeps_zero_total_without_clamp() -> None:
    summary = progress_contract.phase_progress_dimensions_summary(
        {
            "dimensions": {
                "zero": {"done": 5, "total": 0},
                "skip_bool": {"done": True, "total": 1},
            }
        }
    )
    assert summary == "zero=5/0"


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_timeline_row_primary_fallback_and_empty_primary_paths::progress_contract.py::gabion.commands.progress_contract.phase_timeline_row_from_phase_progress
# gabion:behavior primary=allowed_unwanted facets=fallback
def test_phase_timeline_row_primary_fallback_and_empty_primary_paths() -> None:
    row_with_zero_total = progress_contract.phase_timeline_row_from_phase_progress(
        {
            "phase": "collection",
            "work_done": 7,
            "work_total": 0,
        }
    )
    assert "| 7/0 |" in row_with_zero_total

    row_empty_primary = progress_contract.phase_timeline_row_from_phase_progress(
        {"phase": "collection"}
    )
    fields = [field.strip() for field in row_empty_primary.strip("|").split("|")]
    assert len(fields) == len(progress_contract.phase_timeline_header_columns())
    # The primary column remains empty when no primary values and no primary unit exist.
    assert fields[7] == ""


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_timeline_row_prefers_progress_transition_marker_payload::progress_contract.py::gabion.commands.progress_contract.phase_timeline_row_from_phase_progress
# gabion:behavior primary=verboten facets=edge
def test_phase_timeline_row_prefers_progress_transition_marker_payload() -> None:
    row = progress_contract.phase_timeline_row_from_phase_progress(
        {
            "phase": "post",
            "event_kind": "progress",
            "progress_marker": "stale-marker",
            "work_done": 5,
            "work_total": 6,
            "progress_transition_v2": _progress_transition_v2_payload(
                event_kind="progress",
                done=5,
                total=6,
                marker="fingerprint:warnings",
            ),
        }
    )
    assert "| fingerprint:warnings |" in row


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_timeline_row_prefers_progress_transition_parent_payload::progress_contract.py::gabion.commands.progress_contract.phase_timeline_row_from_phase_progress
# gabion:behavior primary=verboten facets=edge
def test_phase_timeline_row_prefers_progress_transition_parent_payload() -> None:
    row = progress_contract.phase_timeline_row_from_phase_progress(
        {
            "phase": "post",
            "event_kind": "terminal",
            "work_done": 1,
            "work_total": 9,
            "phase_progress_v2": {
                "primary_unit": "post_tasks",
                "primary_done": 1,
                "primary_total": 9,
            },
            "progress_transition_v2": _progress_transition_v2_payload(
                event_kind="terminal",
                done=6,
                total=6,
                marker="complete",
                reason="terminal_transition",
            ),
        }
    )
    assert "| 6/6 post_tasks |" in row


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_progress_signature_includes_transition_parent_and_reason::progress_contract.py::gabion.commands.progress_contract.phase_progress_signature
# gabion:behavior primary=verboten facets=edge
def test_phase_progress_signature_includes_transition_parent_and_reason() -> None:
    signature = progress_contract.phase_progress_signature(
        {
            "phase": "post",
            "event_kind": "terminal",
            "progress_marker": "stale",
            "work_done": 6,
            "work_total": 6,
            "progress_transition_v2": _progress_transition_v2_payload(
                event_kind="terminal",
                done=6,
                total=6,
                marker="complete",
                reason="terminal_transition",
            ),
        }
    )
    assert "complete" in signature
    assert "post_tasks" in signature
    assert "terminal_transition" in signature


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_progress_from_notification_prefers_transition_payload_for_marker_and_work::progress_contract.py::gabion.commands.progress_contract.phase_progress_from_progress_notification
# gabion:behavior primary=verboten facets=edge
def test_phase_progress_from_notification_prefers_transition_payload_for_marker_and_work() -> None:
    notification = _canonical_progress_notification(
        {
            "phase": "post",
            "event_kind": "progress",
            "work_done": 1,
            "work_total": 9,
            "progress_marker": "stale",
            "progress_transition_v2": _progress_transition_v2_payload(
                event_kind="terminal",
                done=6,
                total=6,
                marker="complete",
                reason="terminal_transition",
            ),
        }
    )
    phase_progress = progress_contract.phase_progress_from_progress_notification(
        notification
    )
    assert isinstance(phase_progress, dict)
    assert phase_progress["progress_marker"] == "complete"
    assert phase_progress["work_done"] == 6
    assert phase_progress["work_total"] == 6
    assert phase_progress["event_kind"] == "terminal"


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_progress_from_notification_prefers_canonical_v2_payload::progress_contract.py::gabion.commands.progress_contract.phase_progress_from_progress_notification
# gabion:behavior primary=verboten facets=edge
def test_phase_progress_from_notification_prefers_canonical_v2_payload() -> None:
    notification = _canonical_progress_notification(
        {
            "phase": "post",
            "event_kind": "terminal",
            "work_done": 6,
            "work_total": 6,
            "progress_marker": "complete",
        }
    )
    phase_progress = progress_contract.phase_progress_from_progress_notification(
        notification
    )
    assert isinstance(phase_progress, dict)
    assert phase_progress["phase"] == "post"
    assert phase_progress["event_kind"] == "terminal"
    assert phase_progress["progress_marker"] == "complete"
    assert phase_progress["work_done"] == 6
    assert phase_progress["work_total"] == 6


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_progress_from_notification_uses_v2_rejected_fallback_payload::progress_contract.py::gabion.commands.progress_contract.phase_progress_from_progress_notification
# gabion:behavior primary=allowed_unwanted facets=fallback
def test_phase_progress_from_notification_uses_v2_rejected_fallback_payload() -> None:
    fallback_payload = {
        "phase": "post",
        "event_kind": "progress",
        "work_done": 1,
        "work_total": 9,
        "progress_marker": "fingerprint:normalize",
        "progress_transition_v2": _progress_transition_v2_payload(
            event_kind="progress",
            done=1,
            total=9,
            marker="fingerprint:normalize",
        ),
    }
    notification = _canonical_progress_notification(
        fallback_payload,
        adaptation_kind="rejected",
    )
    phase_progress = progress_contract.phase_progress_from_progress_notification(
        notification
    )
    assert isinstance(phase_progress, dict)
    assert phase_progress["phase"] == "post"
    assert phase_progress["progress_marker"] == "fingerprint:normalize"
    assert phase_progress["work_done"] == 1
    assert phase_progress["work_total"] == 9


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_progress_from_notification_keeps_legacy_values_when_transition_not_normalizable::progress_contract.py::gabion.commands.progress_contract.phase_progress_from_progress_notification
# gabion:behavior primary=allowed_unwanted facets=legacy
def test_phase_progress_from_notification_keeps_legacy_values_when_transition_not_normalizable() -> None:
    notification = _canonical_progress_notification(
        {
            "phase": "post",
            "event_kind": "progress",
            "work_done": "invalid",
            "work_total": "invalid",
            "progress_marker": "fingerprint:normalize",
            "progress_transition_v2": {
                "parent": {"unit": "post_tasks"},
                "child": {"marker_text": "fingerprint:normalize"},
            },
        }
    )
    phase_progress = progress_contract.phase_progress_from_progress_notification(
        notification
    )
    assert isinstance(phase_progress, dict)
    assert phase_progress["progress_marker"] == "fingerprint:normalize"
    assert phase_progress["work_done"] is None
    assert phase_progress["work_total"] is None
    assert phase_progress["event_kind"] == "progress"


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_timeline_row_prefers_progress_transition_v2_payload::progress_contract.py::gabion.commands.progress_contract.phase_timeline_row_from_phase_progress
# gabion:behavior primary=verboten facets=edge
def test_phase_timeline_row_prefers_progress_transition_v2_payload() -> None:
    row = progress_contract.phase_timeline_row_from_phase_progress(
        {
            "phase": "post",
            "event_kind": "progress",
            "progress_marker": "stale-marker",
            "work_done": 2,
            "work_total": 9,
            "progress_transition_v2": {
                "phase": "post",
                "analysis_state": "analysis_post_in_progress",
                "event_kind": "progress",
                "reason": "parent_held",
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
        }
    )
    assert "| fingerprint:normalize |" in row
    assert "| 5/6 post_tasks |" in row
    assert "| post_root > fingerprint > fingerprint:normalize |" in row
    assert "| 5/6 post_tasks |" in row


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_progress_from_notification_prefers_transition_v2_event_kind::progress_contract.py::gabion.commands.progress_contract.phase_progress_from_progress_notification
# gabion:behavior primary=verboten facets=edge
def test_phase_progress_from_notification_prefers_transition_v2_event_kind() -> None:
    notification = _canonical_progress_notification(
        {
            "phase": "post",
            "event_kind": "progress",
            "work_done": 1,
            "work_total": 9,
            "progress_marker": "stale",
            "progress_transition_v2": _progress_transition_v2_payload(
                event_kind="terminal",
                done=6,
                total=6,
                marker="complete",
                reason="terminal_transition",
            ),
        }
    )
    phase_progress = progress_contract.phase_progress_from_progress_notification(
        notification
    )
    assert isinstance(phase_progress, dict)
    assert phase_progress["progress_marker"] == "complete"
    assert phase_progress["work_done"] == 6
    assert phase_progress["work_total"] == 6
    assert phase_progress["event_kind"] == "terminal"


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_timeline_row_emits_transition_detail_columns::progress_contract.py::gabion.commands.progress_contract.phase_timeline_row_from_phase_progress
# gabion:behavior primary=verboten facets=edge
def test_phase_timeline_row_emits_transition_detail_columns() -> None:
    row = progress_contract.phase_timeline_row_from_phase_progress(
        {
            "phase": "post",
            "event_kind": "terminal",
            "work_done": 6,
            "work_total": 6,
            "progress_transition_v2": {
                "reason": "terminal_transition",
                "event_kind": "terminal",
                "root": {
                    "identity": "post_root",
                    "unit": "post_tasks",
                    "done": 6,
                    "total": 6,
                    "marker_text": "root",
                    "children": [
                        {
                            "identity": "complete",
                            "unit": "post_tasks",
                            "done": 6,
                            "total": 6,
                            "marker_text": "complete",
                            "children": [],
                        }
                    ],
                },
                "active_path": ["post_root", "complete"],
            },
        }
    )
    fields = [field.strip() for field in row.strip("|").split("|")]
    columns = progress_contract.phase_timeline_header_columns()
    assert len(fields) == len(columns)
    by_name = {columns[idx]: fields[idx] for idx in range(len(columns))}
    assert by_name["progress_path"] == "post_root > complete"
    assert by_name["active_primary"] == "6/6 post_tasks"
    assert by_name["active_depth"] == "1"
    assert by_name["transition_reason"] == "terminal_transition"
    assert by_name["root_identity"] == "post_root"
    assert by_name["active_identity"] == "complete"
    assert by_name["marker_family"] == "complete"
    assert by_name["marker_step"] == ""
    assert by_name["active_children"] == "0"


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_timeline_row_emits_marker_decomposition_without_transition::progress_contract.py::gabion.commands.progress_contract.phase_timeline_row_from_phase_progress
# gabion:behavior primary=verboten facets=edge
def test_phase_timeline_row_emits_marker_decomposition_without_transition() -> None:
    row = progress_contract.phase_timeline_row_from_phase_progress(
        {
            "phase": "post",
            "progress_marker": "fingerprint:normalize",
            "event_kind": "progress",
        }
    )
    fields = [field.strip() for field in row.strip("|").split("|")]
    columns = progress_contract.phase_timeline_header_columns()
    by_name = {columns[idx]: fields[idx] for idx in range(len(columns))}
    assert by_name["marker_family"] == "fingerprint"
    assert by_name["marker_step"] == "normalize"
