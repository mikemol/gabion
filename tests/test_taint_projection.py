from __future__ import annotations

from datetime import date

from gabion.analysis import taint_projection


def test_parse_taint_boundary_registry_normalizes_rows() -> None:
    payload = {
        "boundaries": [
            {
                "boundary_id": "b-main",
                "suite_id": "suite:main",
                "allowed_taint_kinds": ["control", "type_ambiguity"],
                "owner": "core",
                "reason": "ingress",
                "expiry": "2999-01-01",
            },
            {"boundary_id": "", "suite_id": "skip"},
            "invalid",
        ]
    }
    boundaries = taint_projection.parse_taint_boundary_registry(payload)
    assert len(boundaries) == 1
    row = boundaries[0]
    assert row.boundary_id == "b-main"
    assert row.suite_id == "suite:main"
    assert tuple(kind.value for kind in row.allowed_taint_kinds) == (
        "control_ambiguity",
        "type_ambiguity",
    )
    assert row.is_expired(today=date(2025, 1, 1)) is False


def test_project_taint_ledgers_applies_witness_status_and_boundary_rules() -> None:
    marker_rows = [
        {
            "marker_kind": "todo",
            "marker_id": "todo:a",
            "marker_site_id": "never:path:a",
            "reason": "normalize branch",
            "owner": "core",
            "site": {"path": "a.py", "function": "f", "suite_id": "suite:a"},
            "links": [
                {"kind": "policy_id", "value": "NCI-LSP-FIRST"},
                {"kind": "object_id", "value": "justification_code:J1"},
                {"kind": "object_id", "value": "boundary_id:b-main"},
                {"kind": "object_id", "value": "taint_kind:control_ambiguity"},
            ],
        },
        {
            "marker_kind": "deprecated",
            "marker_id": "deprecated:b",
            "reason": "legacy shape",
            "expiry": "2000-01-01",
            "site": {"path": "b.py", "function": "g"},
            "links": [],
        },
    ]
    boundaries = taint_projection.parse_taint_boundary_registry(
        [
            {
                "boundary_id": "b-main",
                "suite_id": "suite:a",
                "allowed_taint_kinds": ["control_ambiguity"],
            }
        ]
    )

    observe_records, observe_witnesses = taint_projection.project_taint_ledgers(
        marker_rows=marker_rows,
        boundary_registry=boundaries,
        profile=taint_projection.TaintProfile.OBSERVE,
        today=date(2026, 1, 1),
    )
    assert len(observe_records) == 2
    assert len(observe_witnesses) == 1
    assert any(row["status"] == "resolved" for row in observe_records)
    assert any(row["status"] == "expired_exemption" for row in observe_records)

    contain_records, _ = taint_projection.project_taint_ledgers(
        marker_rows=marker_rows,
        boundary_registry=(),
        profile=taint_projection.TaintProfile.CONTAIN,
        today=date(2026, 1, 1),
    )
    assert any(row["status"] == "illegal_locus" for row in contain_records)

    summary = taint_projection.build_taint_summary(contain_records)
    assert summary["total"] == 2
    assert summary["strict_unresolved"] >= 1


def test_normalize_taint_profile_aliases_and_boundary_payloads() -> None:
    assert (
        taint_projection.normalize_taint_profile("strict-core")
        is taint_projection.TaintProfile.ENFORCE
    )
    assert (
        taint_projection.normalize_taint_profile("boundary")
        is taint_projection.TaintProfile.CONTAIN
    )
    boundaries = taint_projection.parse_taint_boundary_registry(
        {
            "boundaries": [
                {
                    "boundary_id": "b0",
                    "suite_id": "suite:z",
                    "allowed_taint_kinds": ["boolean"],
                }
            ]
        }
    )
    payloads = taint_projection.boundary_payloads(boundaries)
    assert payloads == [
        {
            "boundary_id": "b0",
            "suite_id": "suite:z",
            "allowed_taint_kinds": ["boolean_ambiguity"],
            "owner": "",
            "reason": "",
            "expiry": "",
        }
    ]
