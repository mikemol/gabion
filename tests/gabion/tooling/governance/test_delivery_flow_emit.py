from __future__ import annotations

import json
from pathlib import Path

from scripts.governance import delivery_flow_emit


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_junit(path: Path, *, failures: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    testcases = []
    for test_id in failures:
        rel_path, _, raw_name = test_id.rpartition("::")
        testcases.extend(
            [
                f'    <testcase classname="{rel_path.replace("/", ".").removesuffix(".py")}" name="{raw_name or test_id}" file="{rel_path or "tests/test_sample.py"}" line="1">',
                '      <failure message="AssertionError">boom</failure>',
                "    </testcase>",
            ]
        )
    path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="utf-8"?>',
                "<testsuites>",
                f'  <testsuite name="pytest" tests="{max(1, len(failures))}" failures="{len(failures)}" errors="0">',
                *testcases,
                "  </testsuite>",
                "</testsuites>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _base_telemetry_history() -> dict[str, object]:
    return {
        "schema_version": 1,
        "runs": [
            {
                "run_id": "run-001",
                "generated_at_utc": "2026-03-19T12:00:00Z",
                "trend_window_runs": 5,
                "timings_seconds_by_step": {
                    "checks_wrapper": 10.0,
                    "full_pytest": 90.0,
                },
                "suite_red_state": True,
                "open_blocker_ids": [
                    "test:tests/gabion/tooling/test_ci.py::test_ci::test_lane",
                    "surface:local:checks",
                ],
                "loops": [
                    {
                        "loop_id": "docflow.contradictions",
                        "domain": "governance",
                        "violation_count": 1,
                        "trend_delta": 0,
                        "recurrence_rate": 0.4,
                        "false_positive_overrides": 0,
                        "time_to_correction_runs": None,
                    }
                ],
            },
            {
                "run_id": "run-002",
                "generated_at_utc": "2026-03-20T12:00:00Z",
                "trend_window_runs": 5,
                "timings_seconds_by_step": {
                    "checks_wrapper": 50.0,
                    "full_pytest": 110.0,
                },
                "suite_red_state": True,
                "open_blocker_ids": [
                    "test:tests/gabion/tooling/test_ci.py::test_ci::test_lane",
                    "surface:local:checks",
                ],
                "loops": [
                    {
                        "loop_id": "docflow.contradictions",
                        "domain": "governance",
                        "violation_count": 2,
                        "trend_delta": 1,
                        "recurrence_rate": 0.7,
                        "false_positive_overrides": 0,
                        "time_to_correction_runs": 2,
                    }
                ],
            },
        ],
    }


# gabion:behavior primary=desired
def test_emit_delivery_flow_summary_outputs_current_and_trend_fields(tmp_path: Path) -> None:
    junit = tmp_path / "artifacts/test_runs/junit.xml"
    local_ci = tmp_path / "artifacts/out/local_ci_repro_contract.json"
    observability = tmp_path / "artifacts/audit_reports/observability_violations.json"
    telemetry_history = tmp_path / "artifacts/out/governance_telemetry_history.json"
    summary_out = tmp_path / "artifacts/out/delivery_flow_summary.json"

    _write_junit(junit, failures=("tests/gabion/tooling/test_ci.py::test_lane",))
    _write_json(
        local_ci,
        {
            "surfaces": [
                {"surface_id": "workflow:checks", "status": "pass"},
                {"surface_id": "local:checks", "status": "fail"},
            ],
            "relations": [
                {"relation_id": "local->workflow", "status": "fail"},
            ],
        },
    )
    _write_json(
        observability,
        {
            "violations": [
                {
                    "label": "checks_wrapper",
                    "reason": "max_gap_meaningful_line_exceeded",
                }
            ]
        },
    )
    _write_json(telemetry_history, _base_telemetry_history())

    assert (
        delivery_flow_emit.main(
            [
                "--run-id",
                "run-002",
                "--junit",
                str(junit),
                "--local-ci-contract",
                str(local_ci),
                "--observability",
                str(observability),
                "--telemetry-history",
                str(telemetry_history),
                "--summary-out",
                str(summary_out),
            ]
        )
        == 0
    )

    payload = json.loads(summary_out.read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "delivery_flow_summary"
    assert payload["current"]["suite_red_state"] is True
    assert payload["current"]["failing_test_case_count"] == 1
    assert payload["current"]["local_ci_failed_surface_ids"] == ["local:checks"]
    assert payload["current"]["local_ci_failed_relation_ids"] == ["local->workflow"]
    assert payload["current"]["observability_violation_ids"] == ["checks_wrapper"]
    assert payload["current"]["severe_runtime_regression_current_band"] is True
    assert payload["trend"]["recurring_loop_ids"] == ["docflow.contradictions"]
    assert payload["trend"]["closure_lag_loop_ids"] == ["docflow.contradictions"]
    assert payload["trend"]["max_time_to_correction_runs"] == 2


# gabion:behavior primary=desired
def test_emit_delivery_flow_summary_derives_repeat_stalled_unstable_and_red_state_dwell(
    tmp_path: Path,
) -> None:
    junit = tmp_path / "artifacts/test_runs/junit.xml"
    local_ci = tmp_path / "artifacts/out/local_ci_repro_contract.json"
    telemetry_history = tmp_path / "artifacts/out/governance_telemetry_history.json"
    summary_out = tmp_path / "artifacts/out/delivery_flow_summary.json"

    _write_junit(junit, failures=("tests/gabion/tooling/test_ci.py::test_lane",))
    _write_json(
        local_ci,
        {
            "surfaces": [{"surface_id": "local:checks", "status": "fail"}],
            "relations": [],
        },
    )
    _write_json(
        telemetry_history,
        {
            "schema_version": 1,
            "runs": [
                {
                    "run_id": "run-001",
                    "generated_at_utc": "2026-03-18T12:00:00Z",
                    "trend_window_runs": 5,
                    "timings_seconds_by_step": {"checks_wrapper": 10.0, "full_pytest": 90.0},
                    "suite_red_state": True,
                    "open_blocker_ids": [
                        "test:tests/gabion/tooling/test_ci.py::test_ci::test_lane",
                        "surface:local:checks",
                    ],
                    "loops": [],
                },
                {
                    "run_id": "run-gap",
                    "generated_at_utc": "2026-03-19T12:00:00Z",
                    "trend_window_runs": 5,
                    "timings_seconds_by_step": {"checks_wrapper": 12.0, "full_pytest": 92.0},
                    "suite_red_state": False,
                    "open_blocker_ids": ["surface:local:checks"],
                    "loops": [],
                },
                {
                    "run_id": "run-just-before",
                    "generated_at_utc": "2026-03-20T11:30:00Z",
                    "trend_window_runs": 5,
                    "timings_seconds_by_step": {"checks_wrapper": 15.0, "full_pytest": 93.0},
                    "suite_red_state": True,
                    "open_blocker_ids": [
                        "test:tests/gabion/tooling/test_ci.py::test_ci::test_lane",
                        "surface:local:checks",
                    ],
                    "loops": [],
                },
                _base_telemetry_history()["runs"][1],
            ],
        },
    )

    assert (
        delivery_flow_emit.main(
            [
                "--run-id",
                "run-002",
                "--junit",
                str(junit),
                "--local-ci-contract",
                str(local_ci),
                "--telemetry-history",
                str(telemetry_history),
                "--summary-out",
                str(summary_out),
            ]
        )
        == 0
    )

    payload = json.loads(summary_out.read_text(encoding="utf-8"))
    repeat_ids = set(payload["current"]["repeat_blocker_ids"])
    assert "test:tests/gabion/tooling/test_ci.py::test_ci::test_lane" in repeat_ids
    assert payload["current"]["stalled_blocker_runs_by_id"][
        "test:tests/gabion/tooling/test_ci.py::test_ci::test_lane"
    ] == 2
    assert (
        "test:tests/gabion/tooling/test_ci.py::test_ci::test_lane"
        in payload["current"]["unstable_blocker_ids"]
    )
    assert payload["trend"]["red_state_dwell_runs"] == 2


# gabion:behavior primary=desired
def test_emit_delivery_flow_summary_bounds_history_window(tmp_path: Path) -> None:
    telemetry_history = tmp_path / "artifacts/out/governance_telemetry_history.json"
    summary_out = tmp_path / "artifacts/out/delivery_flow_summary.json"
    _write_json(
        telemetry_history,
        {
            "schema_version": 1,
            "runs": [
                {
                    "run_id": "run-000",
                    "generated_at_utc": "2026-03-18T11:00:00Z",
                    "trend_window_runs": 5,
                    "timings_seconds_by_step": {},
                    "suite_red_state": False,
                    "open_blocker_ids": [],
                    "loops": [],
                },
                {
                    "run_id": "run-001",
                    "generated_at_utc": "2026-03-19T11:00:00Z",
                    "trend_window_runs": 5,
                    "timings_seconds_by_step": {},
                    "suite_red_state": False,
                    "open_blocker_ids": [],
                    "loops": [],
                },
                {
                    "run_id": "run-002",
                    "generated_at_utc": "2026-03-20T11:00:00Z",
                    "trend_window_runs": 5,
                    "timings_seconds_by_step": {},
                    "suite_red_state": False,
                    "open_blocker_ids": [],
                    "loops": [],
                },
            ],
        },
    )

    assert (
        delivery_flow_emit.main(
            [
                "--run-id",
                "run-002",
                "--telemetry-history",
                str(telemetry_history),
                "--summary-out",
                str(summary_out),
                "--history-window-runs",
                "2",
            ]
        )
        == 0
    )

    payload = json.loads(summary_out.read_text(encoding="utf-8"))
    assert [row["run_id"] for row in payload["history"]] == ["run-001", "run-002"]
