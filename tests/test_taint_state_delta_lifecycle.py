from __future__ import annotations

import json

from gabion.analysis import taint_delta, taint_lifecycle, taint_state


def _marker_rows() -> list[dict[str, object]]:
    return [
        {
            "marker_kind": "todo",
            "marker_id": "todo:aaa",
            "marker_site_id": "never:a",
            "reason": "normalize adapter",
            "site": {"path": "a.py", "function": "f", "suite_id": "suite:a"},
            "links": [
                {"kind": "policy_id", "value": "NCI-LSP-FIRST"},
                {"kind": "object_id", "value": "justification_code:J1"},
                {"kind": "object_id", "value": "boundary_id:boundary-a"},
            ],
        },
        {
            "marker_kind": "todo",
            "marker_id": "todo:bbb",
            "marker_site_id": "never:b",
            "reason": "pending",
            "site": {"path": "b.py", "function": "g", "suite_id": "suite:b"},
            "links": [],
        },
    ]


def test_taint_state_roundtrip_and_summary() -> None:
    payload = taint_state.build_state_payload(
        marker_rows=_marker_rows(),
        boundary_registry=[
            {
                "boundary_id": "boundary-a",
                "suite_id": "suite:a",
                "allowed_taint_kinds": ["control_ambiguity"],
            }
        ],
        profile="contain",
    )
    parsed = taint_state.parse_state_payload(payload)
    assert parsed.profile.value == "contain"
    assert parsed.records
    assert parsed.witnesses
    assert parsed.summary["total"] == len(parsed.records)
    assert parsed.generated_by_spec_id


def test_taint_baseline_delta_and_markdown() -> None:
    contain_payload = taint_state.build_state_payload(
        marker_rows=_marker_rows(),
        boundary_registry=[
            {
                "boundary_id": "boundary-a",
                "suite_id": "suite:a",
                "allowed_taint_kinds": ["control_ambiguity"],
            }
        ],
        profile="contain",
    )
    observe_payload = taint_state.build_state_payload(
        marker_rows=_marker_rows(),
        boundary_registry=[],
        profile="observe",
    )
    baseline = taint_delta.parse_baseline_payload(
        taint_delta.build_baseline_payload(
            taint_state.parse_state_payload(contain_payload).records
        )
    )
    current = taint_delta.parse_baseline_payload(
        taint_delta.build_baseline_payload(
            taint_state.parse_state_payload(observe_payload).records
        )
    )
    delta_payload = taint_delta.build_delta_payload(
        baseline,
        current,
        baseline_path="baselines/taint_baseline.json",
    )
    markdown = taint_delta.render_markdown(delta_payload)
    assert "strict_unresolved" in markdown
    assert "generated_by_spec_id" in markdown
    assert delta_payload["summary"]["by_status"]["delta"]


def test_taint_lifecycle_readiness_promotion_and_demotion() -> None:
    state_payload = taint_state.build_state_payload(
        marker_rows=_marker_rows(),
        boundary_registry=[],
        profile="observe",
    )
    readiness = taint_lifecycle.build_readiness_payload(
        taint_state_payload=state_payload,
        current_profile="contain",
    )
    assert readiness["profiles"]
    promotion = taint_lifecycle.build_promotion_decision_payload(
        readiness_payload=readiness,
        current_profile="contain",
    )
    assert promotion["decision"] in {"hold", "promote", "demote"}
    demotion = taint_lifecycle.build_demotion_incidents_payload(
        taint_state_payload=state_payload,
        current_profile="enforce",
    )
    assert isinstance(demotion["incidents"], list)


def test_taint_state_delta_and_lifecycle_branch_edges(tmp_path) -> None:
    baseline_payload = taint_delta.build_baseline_payload(
        taint_state.parse_state_payload(
            taint_state.build_state_payload(
                marker_rows=_marker_rows(),
                boundary_registry=[],
                profile="observe",
            )
        ).records
    )
    baseline_payload_no_summary = dict(baseline_payload)
    baseline_payload_no_summary["summary"] = {}
    parsed = taint_delta.parse_baseline_payload(baseline_payload_no_summary)
    assert parsed.summary["total"] >= 1
    delta_payload = taint_delta.build_delta_payload(parsed, parsed)
    assert "baseline" not in delta_payload

    state_payload = taint_state.build_state_payload(
        marker_rows=_marker_rows(),
        boundary_registry=[],
        profile="observe",
    )
    state_payload_missing_summary = dict(state_payload)
    state_payload_missing_summary["summary"] = {}
    state_payload_missing_summary["taint_witnesses"] = [None]
    parsed_state = taint_state.parse_state_payload(state_payload_missing_summary)
    assert parsed_state.summary["total"] >= 1

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state_payload_missing_summary), encoding="utf-8")
    loaded_state = taint_state.load_state(str(state_path))
    assert loaded_state.records

    readiness_payload = {
        "profiles": [
            None,
            {"profile": "observe", "ready": True},
            {"profile": "contain", "ready": True},
        ]
    }
    promotion = taint_lifecycle.build_promotion_decision_payload(
        readiness_payload=readiness_payload,
        current_profile="observe",
    )
    assert promotion["decision"] == "promote"
    assert taint_lifecycle._evaluate_gate(
        gate_id="not-real",
        by_status={},
        state_payload={},
    ) is False
    assert taint_lifecycle._recovery_requirements("missing_witness") == [
        "emit_erasure_witness",
        "attach_policy_basis_and_justification_code",
    ]
    assert taint_lifecycle._recovery_requirements("expired_exemption") == [
        "renew_or_remove_exemption",
        "revalidate_expiry_window",
    ]
