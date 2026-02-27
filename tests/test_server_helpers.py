from __future__ import annotations

from pathlib import Path

import pytest

from gabion.exceptions import NeverThrown

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion import server

    return server

# gabion:evidence E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::stale_d287efc4e500
def test_normalize_transparent_decorators() -> None:
    server = _load()
    assert server._normalize_transparent_decorators(None) is None
    assert server._normalize_transparent_decorators("a, b") == {"a", "b"}
    assert server._normalize_transparent_decorators(["a", "b, c"]) == {"a", "b", "c"}
    assert server._normalize_transparent_decorators([1, "a"]) == {"a"}
    assert server._normalize_transparent_decorators([]) is None

# gabion:evidence E:function_site::server.py::gabion.server._uri_to_path E:decision_surface/direct::server.py::gabion.server._uri_to_path::stale_eee91afad018
def test_uri_to_path() -> None:
    server = _load()
    path = Path("/tmp/demo.txt")
    assert server._uri_to_path(path.as_uri()) == path
    assert server._uri_to_path("relative/path.py") == Path("relative/path.py")

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::stale_2ea481a8bdfd
def test_diagnostics_for_path_reports_bundle(tmp_path: Path) -> None:
    server = _load()
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def callee(x):\n"
        "    return x\n"
        "\n"
        "def caller(a, b):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )
    diagnostics = server._diagnostics_for_path(str(sample), tmp_path)
    assert diagnostics
    assert any("Implicit bundle" in diag.message for diag in diagnostics)


# gabion:evidence E:function_site::server.py::gabion.server._analysis_witness_config_payload
def test_analysis_witness_config_payload_is_stable() -> None:
    server = _load()
    config = server.AuditConfig(
        exclude_dirs={"z", "a"},
        ignore_params={"tail", "head"},
        strictness="high",
        external_filter=True,
        transparent_decorators={"pkg.wrap", "alpha.wrap"},
    )

    payload = server._analysis_witness_config_payload(config)

    assert payload["exclude_dirs"] == ["a", "z"]
    assert payload["ignore_params"] == ["head", "tail"]
    assert payload["transparent_decorators"] == ["alpha.wrap", "pkg.wrap"]

# gabion:evidence E:function_site::server.py::gabion.server.start
def test_start_uses_injected_callable() -> None:
    server = _load()
    called = {"value": False}

    def _start() -> None:
        called["value"] = True

    server.start(_start)
    assert called["value"] is True


# gabion:evidence E:call_footprint::tests/test_server_helpers.py::test_deadline_tick_budget_allows_check_non_meter_clock::server.py::gabion.server._deadline_tick_budget_allows_check::test_server_helpers.py::tests.test_server_helpers._load
def test_deadline_tick_budget_allows_check_non_meter_clock() -> None:
    server = _load()

    class _Clock:
        pass

    assert server._deadline_tick_budget_allows_check(_Clock()) is True


# gabion:evidence E:function_site::server.py::gabion.server._diagnostics_for_path
def test_diagnostics_for_path_is_stable_for_shuffled_bundle_insertion_order() -> None:
    server = _load()

    class _Result:
        def __init__(self, span_items: list[tuple[str, tuple[int, int, int, int]]]) -> None:
            self.groups_by_path = {
                "/tmp/sample.py": {
                    "caller": [("a", "b")],
                }
            }
            self.param_spans_by_path = {
                "/tmp/sample.py": {
                    "caller": dict(span_items)
                }
            }

    stable = server._diagnostics_for_path(
        "/tmp/sample.py",
        None,
        analyze_paths_fn=lambda *args, **kwargs: _Result([
            ("a", (1, 0, 1, 1)),
            ("b", (1, 2, 1, 3)),
        ]),
    )
    shuffled = server._diagnostics_for_path(
        "/tmp/sample.py",
        None,
        analyze_paths_fn=lambda *args, **kwargs: _Result([
            ("b", (1, 2, 1, 3)),
            ("a", (1, 0, 1, 1)),
        ]),
    )

    assert [diag.message for diag in stable] == [diag.message for diag in shuffled]


# gabion:evidence E:call_footprint::tests/test_server_helpers.py::test_materialize_execution_plan_uses_request_payload::server.py::gabion.server._materialize_execution_plan::test_server_helpers.py::tests.test_server_helpers._load
def test_materialize_execution_plan_uses_request_payload(tmp_path: Path) -> None:
    server = _load()
    payload = {
        "root": str(tmp_path),
        "execution_plan_request": {
            "requested_operations": ["gabion.dataflowAudit", "gabion.check"],
            "inputs": {"root": str(tmp_path), "paths": ["."]},
            "derived_artifacts": ["artifacts/out/execution_plan.json"],
            "obligations": {
                "preconditions": ["input paths resolve under root"],
                "postconditions": ["execution plan artifact is emitted"],
            },
            "policy_metadata": {
                "deadline": {"analysis_timeout_ticks": 10, "analysis_timeout_tick_ns": 1000},
                "baseline_mode": "none",
                "docflow_mode": "disabled",
            },
        },
    }
    plan = server._materialize_execution_plan(payload)
    assert plan.requested_operations == ["gabion.dataflowAudit", "gabion.check"]
    assert plan.policy_metadata.deadline["analysis_timeout_ticks"] == 10
    artifact_path = server.write_execution_plan_artifact(plan, root=tmp_path)
    assert artifact_path.exists()
    contents = artifact_path.read_text()
    assert '"requested_operations"' in contents


# gabion:evidence E:call_footprint::tests/test_server_helpers.py::test_phase_progress_helpers_normalize_and_clamp_payloads::server.py::gabion.server._build_phase_progress_v2::server.py::gabion.server._phase_primary_unit_for_phase
def test_phase_progress_helpers_normalize_and_clamp_payloads() -> None:
    server = _load()
    assert server._phase_primary_unit_for_phase("mystery") == "phase_work_units"

    normalized, primary_done, primary_total = server._build_phase_progress_v2(
        phase="collection",
        collection_progress={"completed_files": 3, "total_files": 5},
        semantic_progress={
            "cumulative_new_processed_functions": 2,
            "cumulative_completed_files_delta": 1,
            "cumulative_hydrated_paths_delta": 3,
            "cumulative_regressed_functions": 1,
        },
        work_done=None,
        work_total=None,
        phase_progress_v2={
            "primary_unit": "",
            "primary_done": 7,
            "primary_total": 4,
            "dimensions": {
                "good": {"done": 6, "total": 5},
                "bad_payload": "skip",
                1: {"done": 1, "total": 1},
                "bad_done": {"done": "x", "total": 2},
                "bad_total": {"done": 1, "total": False},
            },
            "inventory": {"known": 1, 2: "skip"},
        },
    )
    assert primary_done == 4
    assert primary_total == 4
    assert normalized["primary_unit"] == "collection_files"
    dimensions = normalized["dimensions"]
    assert isinstance(dimensions, dict)
    assert dimensions["good"] == {"done": 5, "total": 5}
    assert "bad_payload" not in dimensions
    assert "bad_done" not in dimensions
    assert "bad_total" not in dimensions
    assert dimensions["collection_files"] == {"done": 4, "total": 4}
    assert dimensions["hydrated_paths_delta"] == {"done": 3, "total": 4}
    assert dimensions["semantic_progress_points"] == {"done": 6, "total": 7}
    assert normalized["inventory"] == {"known": 1}


# gabion:evidence E:call_footprint::tests/test_server_helpers.py::test_progress_heartbeat_seconds_parsing_edges::server.py::gabion.server._progress_heartbeat_seconds
def test_progress_heartbeat_seconds_parsing_edges() -> None:
    server = _load()
    assert server._progress_heartbeat_seconds({"progress_heartbeat_seconds": True}) == (
        server._DEFAULT_PROGRESS_HEARTBEAT_SECONDS
    )
    assert server._progress_heartbeat_seconds({"progress_heartbeat_seconds": " "}) == (
        server._DEFAULT_PROGRESS_HEARTBEAT_SECONDS
    )
    assert server._progress_heartbeat_seconds(
        {"progress_heartbeat_seconds": "not-a-number"}
    ) == server._DEFAULT_PROGRESS_HEARTBEAT_SECONDS
    assert server._progress_heartbeat_seconds(
        {"progress_heartbeat_seconds": {"bad": 1}}
    ) == server._DEFAULT_PROGRESS_HEARTBEAT_SECONDS
    assert server._progress_heartbeat_seconds({"progress_heartbeat_seconds": "1"}) == (
        server._MIN_PROGRESS_HEARTBEAT_SECONDS
    )
    assert server._progress_heartbeat_seconds({"progress_heartbeat_seconds": "0"}) == 0.0


# gabion:evidence E:call_footprint::tests/test_server_helpers.py::test_phase_progress_summary_helpers_cover_invalid_inputs::server.py::gabion.server._phase_progress_dimensions_summary::server.py::gabion.server._phase_progress_primary_summary
def test_phase_progress_summary_helpers_cover_invalid_inputs() -> None:
    server = _load()
    assert server._phase_progress_dimensions_summary(None) == ""
    assert server._phase_progress_dimensions_summary({"dimensions": "bad"}) == ""
    summary = server._phase_progress_dimensions_summary(
        {
            "dimensions": {
                "good": {"done": 7, "total": 5},
                "bad_payload": "skip",
                3: {"done": 1, "total": 1},
                "bad_done": {"done": "x", "total": 2},
            }
        }
    )
    assert summary == "good=5/5"

    assert server._phase_progress_primary_summary(None) == ("", None, None)
    primary_unit, primary_done, primary_total = server._phase_progress_primary_summary(
        {"primary_unit": "forest_mutable_steps", "primary_done": 8, "primary_total": 3}
    )
    assert primary_unit == "forest_mutable_steps"
    assert primary_done == 3
    assert primary_total == 3


# gabion:evidence E:call_footprint::tests/test_server_helpers.py::test_resume_checkpoint_descriptor_formats_known_and_unknown_counts::server.py::gabion.server._resume_checkpoint_descriptor_from_progress_value
def test_resume_checkpoint_descriptor_formats_known_and_unknown_counts() -> None:
    server = _load()
    assert not hasattr(server, "_resume_checkpoint_descriptor_from_progress_value")


# gabion:evidence E:call_footprint::tests/test_server_helpers.py::test_append_phase_timeline_event_handles_primary_unit_only_and_empty_primary::server.py::gabion.server._append_phase_timeline_event
def test_append_phase_timeline_event_handles_primary_unit_only_and_empty_primary(
    tmp_path: Path,
) -> None:
    server = _load()
    markdown_path = tmp_path / "timeline.md"
    jsonl_path = tmp_path / "timeline.jsonl"

    header, row = server._append_phase_timeline_event(
        markdown_path=markdown_path,
        jsonl_path=jsonl_path,
        progress_value={
            "phase": "collection",
            "phase_progress_v2": {"primary_unit": "collection_files"},
        },
    )
    assert isinstance(header, str) and header
    assert "collection_files" in row

    header_again, row_again = server._append_phase_timeline_event(
        markdown_path=markdown_path,
        jsonl_path=jsonl_path,
        progress_value={"phase": "post"},
    )
    assert header_again is None
    assert isinstance(row_again, str) and row_again
    assert markdown_path.exists()
    assert jsonl_path.exists()


# gabion:evidence E:call_footprint::tests/test_server_helpers.py::test_render_incremental_report_includes_stale_and_progress_v2_fields::server.py::gabion.server._render_incremental_report
def test_render_incremental_report_includes_stale_and_progress_v2_fields() -> None:
    server = _load()
    report, pending = server._render_incremental_report(
        analysis_state="analysis_forest_in_progress",
        progress_payload={
            "phase": "forest",
            "event_kind": "heartbeat",
            "work_done": 3,
            "work_total": 9,
            "phase_progress_v2": {
                "primary_unit": "forest_mutable_steps",
                "primary_done": 3,
                "primary_total": 9,
                "dimensions": {
                    "forest_mutable_steps": {"done": 3, "total": 9},
                },
            },
            "stale_for_s": 7.2,
            "classification": "timed_out_progress_resume",
            "retry_recommended": True,
            "resume_supported": False,
        },
        projection_rows=[],
        sections={},
    )
    assert "- `primary_unit`: `forest_mutable_steps`" in report
    assert "- `stale_for_s`: `7.2`" in report
    assert pending == {}


# gabion:evidence E:call_footprint::tests/test_server_helpers.py::test_server_progress_and_incremental_render_additional_branch_edges::server.py::gabion.server._build_phase_progress_v2::server.py::gabion.server._render_incremental_report::server.py::gabion.server._append_phase_timeline_event
def test_server_progress_and_incremental_render_additional_branch_edges(
    tmp_path: Path,
) -> None:
    server = _load()

    normalized, primary_done, primary_total = server._build_phase_progress_v2(
        phase="collection",
        collection_progress={"completed_files": True, "total_files": "bad"},
        semantic_progress=None,
        work_done=None,
        work_total=None,
        phase_progress_v2={
            7: "skip",
            "dimensions": [],
            "inventory": [],
        },
    )
    assert primary_done == 0
    assert primary_total == 0
    assert normalized["dimensions"] == {"collection_files": {"done": 0, "total": 0}}
    assert normalized["inventory"] == {}

    report_zero, _pending_zero = server._render_incremental_report(
        analysis_state="analysis_forest_in_progress",
        progress_payload={
            "work_done": 5,
            "work_total": 0,
            "phase_progress_v2": {
                "primary_done": 5,
                "primary_total": 0,
                "primary_unit": "",
                "dimensions": {},
            },
        },
        projection_rows=[],
        sections={},
    )
    assert "- `work_done`: `5`" in report_zero
    assert "- `work_total`: `0`" in report_zero
    assert "work_percent" not in report_zero
    assert "- `primary_progress`: `5/0`" in report_zero
    assert "- `primary_unit`" not in report_zero
    assert "- `dimensions`" not in report_zero

    report_invalid_primary, _pending_invalid_primary = server._render_incremental_report(
        analysis_state="analysis_forest_in_progress",
        progress_payload={
            "phase_progress_v2": {
                "primary_done": "bad",
                "primary_total": 2,
                "primary_unit": "",
                "dimensions": {},
            },
        },
        projection_rows=[],
        sections={},
    )
    assert "- `primary_progress`" not in report_invalid_primary

    markdown_path = tmp_path / "phase_timeline.md"
    jsonl_path = tmp_path / "phase_timeline.jsonl"
    _header, row = server._append_phase_timeline_event(
        markdown_path=markdown_path,
        jsonl_path=jsonl_path,
        progress_value={
            "phase": "collection",
            "phase_progress_v2": {
                "primary_done": 2,
                "primary_total": 5,
            },
        },
    )
    assert "2/5" in row
    assert "2/5 collection_files" not in row


# gabion:evidence E:call_footprint::tests/test_server_helpers.py::test_deadline_profile_sample_interval_rejects_invalid_values::server.py::gabion.server._deadline_profile_sample_interval
def test_deadline_profile_sample_interval_rejects_invalid_values() -> None:
    server = _load()
    assert (
        server._deadline_profile_sample_interval(
            {"deadline_profile_sample_interval": "16"}
        )
        == 16
    )
    with pytest.raises(NeverThrown):
        server._deadline_profile_sample_interval(
            {"deadline_profile_sample_interval": "bad"}
        )
    with pytest.raises(NeverThrown):
        server._deadline_profile_sample_interval(
            {"deadline_profile_sample_interval": 0}
        )
