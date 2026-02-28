from __future__ import annotations

from datetime import date

from gabion.analysis import taint_projection


# gabion:evidence E:function_site::taint_projection.py::gabion.analysis.taint_projection.parse_taint_boundary_registry
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


# gabion:evidence E:function_site::taint_projection.py::gabion.analysis.taint_projection.project_taint_ledgers
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
    assert all("diagnostic_codes" in row for row in observe_records)

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
    diagnostics = summary.get("diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics["records_with_diagnostics"] >= 1


# gabion:evidence E:function_site::taint_projection.py::gabion.analysis.taint_projection.project_taint_ledgers
def test_project_taint_ledgers_unknown_taint_tag_is_strict_blocking() -> None:
    records, _ = taint_projection.project_taint_ledgers(
        marker_rows=[
            {
                "marker_kind": "todo",
                "marker_id": "todo:unknown",
                "site": {"path": "sample.py", "function": "f", "suite_id": "suite:sample"},
                "links": [
                    {"kind": "policy_id", "value": "NCI-LSP-FIRST"},
                    {"kind": "object_id", "value": "justification_code:J1"},
                    {"kind": "object_id", "value": "taint_kind:not-a-kind"},
                    {"kind": "object_id", "value": "boundary_id:boundary-main"},
                ],
            }
        ],
        boundary_registry=taint_projection.parse_taint_boundary_registry(
            [
                {
                    "boundary_id": "boundary-main",
                    "suite_id": "suite:sample",
                    "allowed_taint_kinds": ["control_ambiguity"],
                }
            ]
        ),
        profile=taint_projection.TaintProfile.CONTAIN,
    )
    assert len(records) == 1
    assert records[0]["status"] == "unresolved"
    assert "unknown_taint_kind_tag" in records[0]["diagnostic_codes"]


# gabion:evidence E:function_site::taint_projection.py::gabion.analysis.taint_projection.project_taint_ledgers
def test_project_taint_ledgers_missing_witness_fields_are_diagnostic() -> None:
    records, _ = taint_projection.project_taint_ledgers(
        marker_rows=[
            {
                "marker_kind": "todo",
                "marker_id": "todo:missing",
                "site": {"path": "sample.py", "function": "f", "suite_id": "suite:sample"},
                "links": [
                    {"kind": "object_id", "value": "taint_kind:control_ambiguity"},
                    {"kind": "object_id", "value": "boundary_id:boundary-main"},
                ],
            }
        ],
        boundary_registry=taint_projection.parse_taint_boundary_registry(
            [
                {
                    "boundary_id": "boundary-main",
                    "suite_id": "suite:sample",
                    "allowed_taint_kinds": ["control_ambiguity"],
                }
            ]
        ),
        profile=taint_projection.TaintProfile.CONTAIN,
    )
    assert len(records) == 1
    assert records[0]["status"] == "missing_witness"
    assert "missing_witness_field:policy_basis" in records[0]["diagnostic_codes"]
    assert "missing_witness_field:justification_code" in records[0]["diagnostic_codes"]


# gabion:evidence E:function_site::taint_projection.py::gabion.analysis.taint_projection.project_taint_ledgers
def test_project_taint_ledgers_deterministic_ids_and_order_across_reruns() -> None:
    marker_rows = [
        {
            "marker_kind": "todo",
            "marker_id": "todo:1",
            "site": {"path": "a.py", "function": "f", "suite_id": "suite:a"},
            "links": [
                {"kind": "policy_id", "value": "NCI-LSP-FIRST"},
                {"kind": "object_id", "value": "justification_code:J1"},
                {"kind": "object_id", "value": "boundary_id:boundary-a"},
            ],
        },
        {
            "marker_kind": "deprecated",
            "marker_id": "deprecated:2",
            "site": {"path": "b.py", "function": "g", "suite_id": "suite:b"},
            "links": [],
        },
    ]
    boundaries = taint_projection.parse_taint_boundary_registry(
        [
            {
                "boundary_id": "boundary-a",
                "suite_id": "suite:a",
                "allowed_taint_kinds": ["control_ambiguity"],
            }
        ]
    )
    records_a, witnesses_a = taint_projection.project_taint_ledgers(
        marker_rows=marker_rows,
        boundary_registry=boundaries,
        profile="contain",
    )
    records_b, witnesses_b = taint_projection.project_taint_ledgers(
        marker_rows=marker_rows,
        boundary_registry=boundaries,
        profile="contain",
    )
    assert records_a == records_b
    assert witnesses_a == witnesses_b


# gabion:evidence E:function_site::taint_projection.py::gabion.analysis.taint_projection.normalize_taint_profile
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


# gabion:evidence E:function_site::taint_projection.py::gabion.analysis.taint_projection._date_from_iso
def test_taint_projection_helper_branches_for_links_and_dates() -> None:
    assert taint_projection._mapping_payload("not-mapping") == {}
    assert taint_projection._normalize_links(
        [{"kind": "", "value": "x"}, {"kind": "object_id", "value": ""}, {"kind": "object_id", "value": "k:v"}]
    ) == ({"kind": "object_id", "value": "k:v"},)
    tags = taint_projection._semantic_link_tags(
        (
            {"kind": "", "value": "ignored"},
            {"kind": "policy_id", "value": "NCI-A"},
            {"kind": "object_id", "value": "not-tagged"},
            {"kind": "object_id", "value": "boundary_id:"},
            {"kind": "object_id", "value": "boundary_id:b-main"},
        )
    )
    assert tags["policy_basis"] == "NCI-A"
    assert tags["boundary_id"] == "b-main"
    assert taint_projection.TaintBoundaryLocus(
        boundary_id="b",
        suite_id="s",
        allowed_taint_kinds=(),
        expiry="",
    ).is_expired(today=date(2026, 1, 1)) is False
    assert taint_projection._normalized_today("today") == date.today()
    assert (
        taint_projection._status_for_entry(
            profile=taint_projection.TaintProfile.CONTAIN,
            taint_kind=taint_projection.TaintKind.CONTROL_AMBIGUITY,
            witness=taint_projection.TaintErasureWitness(
                taint_kind=taint_projection.TaintKind.CONTROL_AMBIGUITY,
                source_suite_id="suite:a",
                target_suite_id="suite:a",
                eraser_id="e",
                input_shape="i",
                output_shape="o",
                policy_basis="NCI-A",
                justification_code="J1",
            ),
            boundary=None,
            expiry="",
            today=date(2026, 1, 1),
            diagnostic_codes=(),
        )
        is taint_projection.TaintStatus.ILLEGAL_LOCUS
    )


# gabion:evidence E:function_site::taint_projection.py::gabion.analysis.taint_projection.project_taint_ledgers
def test_project_taint_ledgers_observe_unknown_tag_and_boundary_suite_fallback() -> None:
    records, _ = taint_projection.project_taint_ledgers(
        marker_rows=[
            {
                "marker_kind": "todo",
                "marker_id": "todo:unknown-observe",
                "site": {"path": "x.py", "function": "f", "suite_id": "suite:x"},
                "links": [
                    {"kind": "policy_id", "value": "NCI-LSP-FIRST"},
                    {"kind": "object_id", "value": "justification_code:J1"},
                    {"kind": "object_id", "value": "taint_kind:invalid"},
                ],
            }
        ],
        boundary_registry=taint_projection.parse_taint_boundary_registry(
            [{"boundary_id": "b-suite", "suite_id": "suite:x", "allowed_taint_kinds": ["control_ambiguity"]}]
        ),
        profile=taint_projection.TaintProfile.OBSERVE,
    )
    assert records[0]["boundary_id"] == "b-suite"
    assert records[0]["status"] == "unresolved"


# gabion:evidence E:function_site::taint_projection.py::gabion.analysis.taint_projection.project_taint_ledgers
def test_project_taint_ledgers_boundary_strict_paths_and_summary_without_diagnostics() -> None:
    contain_records, _ = taint_projection.project_taint_ledgers(
        marker_rows=[
            {
                "marker_kind": "todo",
                "marker_id": "todo:strict",
                "site": {"path": "strict.py", "function": "f", "suite_id": "suite:strict"},
                "links": [
                    {"kind": "policy_id", "value": "NCI-LSP-FIRST"},
                    {"kind": "object_id", "value": "justification_code:J1"},
                    {"kind": "object_id", "value": "taint_kind:control_ambiguity"},
                ],
            }
        ],
        boundary_registry=(),
        profile=taint_projection.TaintProfile.CONTAIN,
    )
    assert contain_records[0]["status"] == "illegal_locus"

    forbidden_records, _ = taint_projection.project_taint_ledgers(
        marker_rows=[
            {
                "marker_kind": "todo",
                "marker_id": "todo:forbidden",
                "site": {"path": "forbidden.py", "function": "f", "suite_id": "suite:forbidden"},
                "links": [
                    {"kind": "policy_id", "value": "NCI-LSP-FIRST"},
                    {"kind": "object_id", "value": "justification_code:J1"},
                    {"kind": "object_id", "value": "taint_kind:control_ambiguity"},
                    {"kind": "object_id", "value": "boundary_id:b-forbidden"},
                ],
            }
        ],
        boundary_registry=taint_projection.parse_taint_boundary_registry(
            [{"boundary_id": "b-forbidden", "suite_id": "suite:forbidden", "allowed_taint_kinds": ["type_ambiguity"]}]
        ),
        profile=taint_projection.TaintProfile.CONTAIN,
    )
    assert forbidden_records[0]["status"] == "illegal_locus"
    assert "boundary_taint_kind_forbidden" in forbidden_records[0]["diagnostic_codes"]

    resolved_summary = taint_projection.build_taint_summary(
        [
            {
                "status": "resolved",
                "taint_kind": "control_ambiguity",
                "diagnostic_codes": [],
            }
        ]
    )
    assert "diagnostics" not in resolved_summary
