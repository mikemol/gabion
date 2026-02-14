from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from gabion import server
from gabion.exceptions import NeverThrown
from gabion.analysis import (
    ambiguity_delta,
    ambiguity_state,
    evidence_keys,
    test_annotation_drift,
    test_annotation_drift_delta,
    test_obsolescence,
    test_obsolescence_delta,
    test_obsolescence_state,
)


class _DummyWorkspace:
    def __init__(self, root_path: str) -> None:
        self.root_path = root_path


class _DummyServer:
    def __init__(self, root_path: str) -> None:
        self.workspace = _DummyWorkspace(root_path)


@dataclass
class _CommandResult:
    exit_code: int
    violations: int


_TIMEOUT_PAYLOAD = {
    "analysis_timeout_ticks": 1000,
    "analysis_timeout_tick_ns": 1_000_000,
}


def _with_timeout(payload: dict) -> dict:
    return {**_TIMEOUT_PAYLOAD, **payload}


def _write_bundle_module(path: Path) -> None:
    path.write_text(
        "def callee(x):\n"
        "    return x\n"
        "\n"
        "def caller(a, b):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )


def _write_type_conflict_module(path: Path) -> None:
    path.write_text(
        "def callee_int(x: int):\n"
        "    return x\n"
        "\n"
        "def callee_str(x: str):\n"
        "    return x\n"
        "\n"
        "def caller_conflict(b):\n"
        "    callee_int(b)\n"
        "    callee_str(b)\n"
    )


def _write_many_functions_module(path: Path, *, count: int = 400) -> None:
    lines: list[str] = []
    for index in range(count):
        lines.append(f"def fn_{index}(value):")
        lines.append("    return value")
        lines.append("")
    path.write_text("\n".join(lines))


def _artifact_out_dir(root: Path) -> Path:
    artifact_dir = root / "artifacts" / "out"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_dash_outputs(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "dot": "-",
                "synthesis_plan": "-",
                "synthesis_protocols": "-",
                "refactor_plan_json": "-",
            }
        ),
    )
    assert "dot" in result
    assert "synthesis_plan" in result
    assert "synthesis_protocols" in result
    assert "refactor_plan" in result
    assert result.get("analysis_state") == "succeeded"


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_invalid_synth_min_occurrences(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    config_path = tmp_path / "gabion.toml"
    config_path.write_text(
        "[fingerprints]\n"
        "user_context = [\"int\", \"str\"]\n"
        "synth_min_occurrences = \"bad\"\n"
    )
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "config": str(config_path),
            }
        ),
    )
    assert result.get("exit_code") == 0


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_reports_synthesis_error(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "synthesis_plan": "-",
                "synthesis_min_bundle_size": "bad",
            }
        ),
    )
    assert result.get("synthesis_errors")


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_ignores_invalid_timeout(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            _with_timeout(
                {
                    "root": str(tmp_path),
                    "paths": [str(module_path)],
                    "analysis_timeout_ticks": "nope",
                }
            ),
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_reports_timeout(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(tmp_path / "report.md"),
                "analysis_timeout_ticks": 1,
                "analysis_timeout_tick_ns": 1,
                "deadline_profile": True,
            }
        ),
    )
    assert result.get("exit_code") == 2
    assert result.get("timeout") is True
    assert str(result.get("analysis_state", "")).startswith("timed_out_")
    assert "timeout_context" in result
    timeout_context = result.get("timeout_context")
    assert isinstance(timeout_context, dict)
    assert "deadline_profile" in timeout_context
    progress = timeout_context.get("progress")
    assert isinstance(progress, dict)
    assert str(progress.get("classification", "")).startswith("timed_out_")


def test_execute_command_timeout_supports_in_progress_resume_checkpoint(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "many.py"
    _write_many_functions_module(module_path, count=800)
    checkpoint_path = tmp_path / "artifacts" / "audit_reports" / "resume.json"
    command_payload: dict[str, object] = {
        "root": str(tmp_path),
        "paths": [str(module_path)],
        "report": str(tmp_path / "report.md"),
        "allow_external": True,
        "analysis_timeout_ticks": 1,
        "analysis_timeout_tick_ns": 20_000_000,
        "deadline_profile": True,
        "resume_checkpoint": str(checkpoint_path),
    }
    defaults = server.dataflow_defaults(tmp_path, None)
    merged_payload = server.merge_payload(command_payload, defaults)
    config = server.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(merged_payload.get("exclude", [])),
        ignore_params=set(merged_payload.get("ignore_params", [])),
        external_filter=not bool(merged_payload.get("allow_external", False)),
        strictness=str(merged_payload.get("strictness", "high")),
        transparent_decorators=server._normalize_transparent_decorators(
            merged_payload.get("transparent_decorators")
        ),
    )
    file_paths = server.resolve_analysis_paths([module_path], config=config)
    witness = server._analysis_input_witness(
        root=tmp_path,
        file_paths=file_paths,
        recursive=True,
        include_invariant_propositions=True,
        config=config,
    )
    server._write_analysis_resume_checkpoint(
        path=checkpoint_path,
        input_witness=witness,
        input_manifest_digest=None,
        collection_resume={
            "format_version": 2,
            "completed_paths": [],
            "groups_by_path": {},
            "param_spans_by_path": {},
            "bundle_sites_by_path": {},
            "in_progress_scan_by_path": {
                str(module_path): {
                    "phase": "function_scan",
                }
            },
            "invariant_propositions": [],
        },
    )
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(ls, command_payload)
    assert result.get("timeout") is True
    timeout_context = result.get("timeout_context")
    assert isinstance(timeout_context, dict)
    progress = timeout_context.get("progress")
    assert isinstance(progress, dict)
    assert progress.get("resume_supported") is True
    resume = progress.get("resume")
    assert isinstance(resume, dict)
    token = resume.get("resume_token")
    assert isinstance(token, dict)
    assert token.get("phase") == "analysis_collection"
    assert token.get("in_progress_files") == 1


def test_execute_command_timeout_writes_partial_incremental_report(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "many.py"
    report_path = tmp_path / "report.md"
    phase_checkpoint_path = tmp_path / "report_phase_checkpoint.json"
    _write_many_functions_module(module_path, count=800)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
                {
                    "root": str(tmp_path),
                    "paths": [str(module_path)],
                    "report": str(report_path),
                    "resume_checkpoint": str(tmp_path / "resume.json"),
                    "analysis_timeout_ticks": 5_000,
                    "analysis_timeout_tick_ns": 1_000,
                }
            ),
        )
    assert result.get("timeout") is True
    assert report_path.exists()
    report_text = report_path.read_text()
    assert "Incremental Status" in report_text
    assert "PENDING (phase:" in report_text
    assert phase_checkpoint_path.exists()
    phase_payload = json.loads(phase_checkpoint_path.read_text())
    phases = phase_payload.get("phases")
    assert isinstance(phases, dict)
    collection_phase = phases.get("collection")
    progress = (result.get("timeout_context") or {}).get("progress")
    assert isinstance(progress, dict)
    obligations = progress.get("incremental_obligations")
    assert isinstance(obligations, list)
    assert any(
        isinstance(entry, dict)
        and entry.get("contract") == "progress_report_contract"
        and entry.get("status") == "SATISFIED"
        for entry in obligations
    )
    if isinstance(collection_phase, dict):
        assert (
            "Collection progress checkpoint (provisional)." in report_text
            or "Collection bootstrap checkpoint (provisional)." in report_text
        )
        assert "## Section `intro`\nPENDING" not in report_text
        if collection_phase.get("status") == "checkpointed":
            assert int(collection_phase.get("in_progress_files", 0)) >= 1
            assert int(collection_phase.get("total_files", 0)) >= 1
        else:
            assert collection_phase.get("status") == "bootstrap"
    else:
        assert any(
            isinstance(entry, dict)
            and entry.get("contract") == "resume_contract"
            and entry.get("kind") == "no_projection_progress"
            and entry.get("status") == "VIOLATION"
            for entry in obligations
        )


def test_execute_command_timeout_marks_stale_section_journal(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "many.py"
    report_path = tmp_path / "report.md"
    journal_path = tmp_path / "report_sections.json"
    _write_many_functions_module(module_path, count=800)
    journal_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "witness_digest": "stale",
                "sections": {
                    "components": {
                        "phase": "forest",
                        "deps": ["intro"],
                        "status": "resolved",
                        "lines": ["### Component 1", "stale"],
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(report_path),
                "analysis_timeout_ticks": 1,
                "analysis_timeout_tick_ns": 1,
            }
        ),
    )
    assert result.get("timeout") is True
    progress = (result.get("timeout_context") or {}).get("progress")
    assert isinstance(progress, dict)
    obligations = progress.get("incremental_obligations")
    assert isinstance(obligations, list)
    assert any(
        isinstance(entry, dict)
        and entry.get("contract") == "incremental_projection_contract"
        and entry.get("detail") in {"stale_input", "policy"}
        for entry in obligations
    )


def test_execute_command_writes_phase_checkpoint_when_incremental_enabled(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    report_path = tmp_path / "report.md"
    phase_checkpoint_path = tmp_path / "report_phase_checkpoint.json"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
                {
                    "root": str(tmp_path),
                    "paths": [str(module_path)],
                    "report": str(report_path),
                    "emit_timeout_progress_report": True,
                    "analysis_timeout_ticks": 5_000,
                }
            ),
        )
    assert result.get("analysis_state") == "succeeded"
    assert phase_checkpoint_path.exists()
    payload = json.loads(phase_checkpoint_path.read_text())
    phases = payload.get("phases")
    assert isinstance(phases, dict)
    assert "collection" in phases
    assert "forest" in phases
    assert "edge" in phases
    assert "post" in phases
    post_phase = phases["post"]
    assert isinstance(post_phase, dict)
    assert post_phase.get("status") == "final"


def test_incremental_obligations_require_restart_on_witness_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "resume.json"
    checkpoint_path.write_text("{}")
    obligations = server._incremental_progress_obligations(
        analysis_state="timed_out_progress_resume",
        progress_payload={
            "classification": "timed_out_progress_resume",
            "resume_supported": True,
        },
        resume_checkpoint_path=checkpoint_path,
        partial_report_written=True,
        report_requested=True,
        projection_rows=[
            {"section_id": "components", "phase": "forest", "deps": ["intro"]},
        ],
        sections={"components": ["resolved"]},
        pending_reasons={"intro": "stale_input"},
    )
    assert any(
        isinstance(entry, dict)
        and entry.get("contract") == "resume_contract"
        and entry.get("kind") == "restart_required_on_witness_mismatch"
        and entry.get("status") == "VIOLATION"
        for entry in obligations
    )
    assert any(
        isinstance(entry, dict)
        and entry.get("contract") == "resume_contract"
        and entry.get("kind") == "no_projection_progress"
        and entry.get("status") == "SATISFIED"
        for entry in obligations
    )


def test_incremental_obligations_flag_no_projection_progress() -> None:
    obligations = server._incremental_progress_obligations(
        analysis_state="timed_out_progress_resume",
        progress_payload={
            "classification": "timed_out_progress_resume",
            "resume_supported": True,
        },
        resume_checkpoint_path=None,
        partial_report_written=True,
        report_requested=True,
        projection_rows=[
            {"section_id": "intro", "phase": "collection", "deps": []},
            {"section_id": "components", "phase": "forest", "deps": ["intro"]},
        ],
        sections={},
        pending_reasons={"intro": "policy", "components": "missing_dep"},
    )
    assert any(
        isinstance(entry, dict)
        and entry.get("contract") == "resume_contract"
        and entry.get("kind") == "no_projection_progress"
        and entry.get("status") == "VIOLATION"
        for entry in obligations
    )


def test_incremental_obligations_require_substantive_progress_for_resume() -> None:
    obligations = server._incremental_progress_obligations(
        analysis_state="timed_out_progress_resume",
        progress_payload={
            "classification": "timed_out_progress_resume",
            "resume_supported": True,
            "semantic_progress": {
                "substantive_progress": False,
                "monotonic_progress": True,
            },
        },
        resume_checkpoint_path=None,
        partial_report_written=True,
        report_requested=True,
        projection_rows=[
            {"section_id": "intro", "phase": "collection", "deps": []},
        ],
        sections={"intro": ["Collection progress checkpoint (provisional)."]},
        pending_reasons={},
    )
    assert any(
        isinstance(entry, dict)
        and entry.get("contract") == "resume_contract"
        and entry.get("kind") == "substantive_progress_required"
        and entry.get("status") == "VIOLATION"
        for entry in obligations
    )


def test_incremental_obligations_flag_semantic_progress_regression() -> None:
    obligations = server._incremental_progress_obligations(
        analysis_state="timed_out_no_progress",
        progress_payload={
            "classification": "timed_out_no_progress",
            "resume_supported": False,
            "semantic_progress": {
                "substantive_progress": False,
                "monotonic_progress": False,
            },
        },
        resume_checkpoint_path=None,
        partial_report_written=True,
        report_requested=False,
        projection_rows=[],
        sections={},
        pending_reasons={},
    )
    assert any(
        isinstance(entry, dict)
        and entry.get("contract") == "resume_contract"
        and entry.get("kind") == "progress_monotonicity"
        and entry.get("status") == "VIOLATION"
        for entry in obligations
    )


def test_collection_progress_intro_lines_include_resume_counts() -> None:
    lines = server._collection_progress_intro_lines(
        collection_resume={
            "completed_paths": ["a.py"],
            "in_progress_scan_by_path": {"b.py": {"phase": "scan_pending"}},
            "semantic_progress": {
                "new_processed_functions_count": 3,
                "regressed_processed_functions_count": 0,
                "completed_files_delta": 1,
                "substantive_progress": True,
            },
        },
        total_files=3,
    )
    assert "Collection progress checkpoint (provisional)." in lines
    assert "- `completed_files`: `1`" in lines
    assert "- `in_progress_files`: `1`" in lines
    assert "- `remaining_files`: `2`" in lines
    assert "- `new_processed_functions`: `3`" in lines
    assert "- `substantive_progress`: `True`" in lines


def test_collection_progress_intro_lines_reject_path_order_regression() -> None:
    with pytest.raises(NeverThrown):
        server._collection_progress_intro_lines(
            collection_resume={
                "completed_paths": [],
                "in_progress_scan_by_path": {
                    "b.py": {"phase": "scan_pending"},
                    "a.py": {"phase": "scan_pending"},
                },
            },
            total_files=2,
        )


# gabion:evidence E:decision_surface/direct::server.py::gabion.server._analysis_input_witness::config,file_paths,include_invariant_propositions,recursive,root E:decision_surface/direct::server.py::gabion.server._load_analysis_resume_checkpoint::input_witness,path E:decision_surface/direct::server.py::gabion.server._write_analysis_resume_checkpoint::collection_resume,input_witness,path E:decision_surface/direct::server.py::gabion.server._execute_command_total::on_collection_progress
def test_execute_command_reuses_collection_checkpoint(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    command_payload = _with_timeout(
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "dot": "-",
            "resume_checkpoint": str(
                tmp_path / "artifacts" / "audit_reports" / "resume-checkpoint.json"
            ),
            "allow_external": True,
            "exclude": [],
            "ignore_params": [],
            "strictness": "high",
            "transparent_decorators": [],
        }
    )
    config = server.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
        transparent_decorators=set(),
    )
    file_paths = server.resolve_analysis_paths([module_path], config=config)
    snapshots: list[dict[str, object]] = []
    server.analyze_paths(
        paths=[module_path],
        forest=server.Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_bundle_forest=True,
        config=config,
        file_paths_override=file_paths,
        on_collection_progress=snapshots.append,
    )
    assert snapshots
    checkpoint_path = Path(str(command_payload["resume_checkpoint"]))
    witness = server._analysis_input_witness(
        root=tmp_path,
        file_paths=file_paths,
        recursive=True,
        include_invariant_propositions=False,
        config=config,
    )
    server._write_analysis_resume_checkpoint(
        path=checkpoint_path,
        input_witness=witness,
        input_manifest_digest=None,
        collection_resume=snapshots[-1],
    )
    result = server.execute_command(ls, command_payload)
    assert result.get("exit_code") == 0
    resume = result.get("analysis_resume")
    assert isinstance(resume, dict)
    assert resume.get("reused_files") == 1
    assert resume.get("total_files") == 1
    assert not checkpoint_path.exists()


def test_analysis_input_witness_interns_ast_normal_forms(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    config = server.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
        transparent_decorators=set(),
    )
    file_paths = server.resolve_analysis_paths([module_path], config=config)
    witness = server._analysis_input_witness(
        root=tmp_path,
        file_paths=file_paths,
        recursive=True,
        include_invariant_propositions=False,
        config=config,
    )
    assert witness.get("format_version") == 2
    assert isinstance(witness.get("witness_digest"), str)
    table = witness.get("ast_intern_table")
    assert isinstance(table, dict)
    files = witness.get("files")
    assert isinstance(files, list)
    assert files
    file_entry = files[0]
    assert isinstance(file_entry, dict)
    ast_ref = file_entry.get("ast_ref")
    assert isinstance(ast_ref, str)
    assert ast_ref in table


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_ignores_invalid_tick_ns(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "analysis_timeout_ticks": 1,
                "analysis_timeout_tick_ns": "nope",
            },
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_uses_timeout_ms(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "analysis_timeout_ms": 1000,
        },
    )
    assert result.get("exit_code") == 0


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_invalid_timeout_ms_ignored(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "analysis_timeout_ms": "nope",
            },
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_uses_timeout_seconds(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "analysis_timeout_seconds": "10",
        },
    )
    assert result.get("exit_code") == 0


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_invalid_timeout_seconds_ignored(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "analysis_timeout_seconds": "nope",
            },
        )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_fingerprint_outputs_and_decision_snapshot(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "def callee(x: int):\n"
        "    return x\n"
        "\n"
        "def caller_one(a: int, b: int):\n"
        "    callee(a)\n"
        "    callee(b)\n"
        "\n"
        "def caller_two(a: int, b: int):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )
    config_path = tmp_path / "gabion.toml"
    config_path.write_text(
        "[fingerprints]\n"
        "user_context = [\"int\"]\n"
        "synth_min_occurrences = 2\n"
        "\n"
        "[decision]\n"
        "tier2 = [\"a\"]\n"
    )
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "config": str(config_path),
            "fingerprint_synth_json": "-",
            "fingerprint_provenance_json": "-",
            "fingerprint_deadness_json": "-",
            "fingerprint_coherence_json": "-",
            "fingerprint_rewrite_plans_json": "-",
            "fingerprint_exception_obligations_json": "-",
            "fingerprint_handledness_json": "-",
            "decision_snapshot": "-",
        }),
    )
    assert "fingerprint_synth_registry" in result
    assert "fingerprint_provenance" in result
    assert "fingerprint_deadness" in result
    assert "fingerprint_coherence" in result
    assert "fingerprint_rewrite_plans" in result
    assert "fingerprint_exception_obligations" in result
    assert "fingerprint_handledness" in result
    assert "decision_snapshot" in result


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_writes_fingerprint_outputs(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "def callee(x: int):\n"
        "    return x\n"
        "\n"
        "def caller_one(a: int, b: int):\n"
        "    callee(a)\n"
        "    callee(b)\n"
        "\n"
        "def caller_two(a: int, b: int):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )
    synth_registry_path = tmp_path / "synth_registry.json"
    synth_registry_path.write_text(
        "{\"version\": \"synth@1\", \"min_occurrences\": 2, \"entries\": []}"
    )
    config_path = tmp_path / "gabion.toml"
    config_path.write_text(
        "[fingerprints]\n"
        "user_context = [\"int\"]\n"
        "synth_min_occurrences = 2\n"
        f"synth_registry_path = \"{synth_registry_path}\"\n"
    )
    synth_path = tmp_path / "fingerprints.json"
    provenance_path = tmp_path / "provenance.json"
    deadness_path = tmp_path / "deadness.json"
    coherence_path = tmp_path / "coherence.json"
    rewrite_plans_path = tmp_path / "rewrite_plans.json"
    exception_obligations_path = tmp_path / "exception_obligations.json"
    handledness_path = tmp_path / "handledness.json"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "config": str(config_path),
            "fingerprint_synth_json": str(synth_path),
            "fingerprint_provenance_json": str(provenance_path),
            "fingerprint_deadness_json": str(deadness_path),
            "fingerprint_coherence_json": str(coherence_path),
            "fingerprint_rewrite_plans_json": str(rewrite_plans_path),
            "fingerprint_exception_obligations_json": str(exception_obligations_path),
            "fingerprint_handledness_json": str(handledness_path),
        }),
    )
    assert result.get("exit_code") == 0
    assert synth_path.exists()
    assert provenance_path.exists()
    assert deadness_path.exists()
    assert coherence_path.exists()
    assert rewrite_plans_path.exists()
    assert exception_obligations_path.exists()
    assert handledness_path.exists()


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_writes_decision_snapshot(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    decision_path = tmp_path / "decision.json"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "decision_snapshot": str(decision_path),
        }),
    )
    assert result.get("exit_code") == 0
    assert decision_path.exists()


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_baseline_apply(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("preexisting\n")
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "baseline": str(baseline_path),
            "fail_on_violations": True,
        }),
    )
    assert result.get("baseline_written") is False
    assert _CommandResult(
        exit_code=result.get("exit_code", -1),
        violations=result.get("violations", -1),
    ) == _CommandResult(exit_code=1, violations=result.get("violations", -1))


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_fail_on_type_ambiguities(tmp_path: Path) -> None:
    module_path = tmp_path / "types.py"
    _write_type_conflict_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "fail_on_type_ambiguities": True,
        }),
    )
    assert result.get("exit_code") == 1
    assert result.get("type_ambiguities")


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_includes_lint_lines(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "lint": True,
        }),
    )
    assert "lint_lines" in result


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_report_baseline_write(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    report_path = tmp_path / "report.md"
    baseline_path = tmp_path / "baseline.txt"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "report": str(report_path),
            "baseline": str(baseline_path),
            "baseline_write": True,
            "fail_on_violations": True,
        }),
    )
    assert result.get("baseline_written") is True
    assert result.get("exit_code") == 0
    assert report_path.exists()
    assert baseline_path.exists()


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count E:decision_surface/direct::server.py::gabion.server.execute_structure_reuse::payload
def test_execute_structure_reuse_missing_snapshot() -> None:
    result = server.execute_structure_reuse(None, _with_timeout({}))
    assert result.get("exit_code") == 2


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count E:decision_surface/direct::server.py::gabion.server.execute_structure_reuse::payload
def test_execute_structure_reuse_payload_none() -> None:
    with pytest.raises(NeverThrown):
        server.execute_structure_reuse(None, None)


def test_execute_structure_reuse_payload_non_dict() -> None:
    with pytest.raises(NeverThrown):
        server.execute_structure_reuse(None, [])  # type: ignore[arg-type]


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count E:decision_surface/direct::server.py::gabion.server.execute_structure_reuse::payload
def test_execute_structure_reuse_invalid_snapshot(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    result = server.execute_structure_reuse(
        None, _with_timeout({"snapshot": str(bad)})
    )
    assert result.get("exit_code") == 2


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count E:decision_surface/direct::server.py::gabion.server.execute_structure_reuse::payload
def test_execute_structure_reuse_writes_lemma_stubs(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        "{\"format_version\": 1, \"root\": null, \"files\": []}"
    )
    result = server.execute_structure_reuse(
        None,
        _with_timeout(
            {"snapshot": str(snapshot_path), "lemma_stubs": "-", "min_count": "bad"}
        ),
    )
    assert result.get("exit_code") == 2
    assert "min_count must be an integer" in str(result.get("errors"))


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count E:decision_surface/direct::server.py::gabion.server.execute_structure_reuse::payload
def test_execute_structure_reuse_writes_lemma_stubs_file(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        "{\"format_version\": 1, \"root\": null, \"files\": []}"
    )
    lemma_path = tmp_path / "lemmas.py"
    result = server.execute_structure_reuse(
        None,
        _with_timeout({"snapshot": str(snapshot_path), "lemma_stubs": str(lemma_path)}),
    )
    assert result.get("exit_code") == 0
    assert lemma_path.exists()


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_decision_diff::payload
def test_execute_decision_diff_missing_paths() -> None:
    result = server.execute_decision_diff(None, _with_timeout({}))
    assert result.get("exit_code") == 2


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_decision_diff::payload
def test_execute_decision_diff_payload_none() -> None:
    with pytest.raises(NeverThrown):
        server.execute_decision_diff(None, None)


def test_execute_decision_diff_payload_non_dict() -> None:
    with pytest.raises(NeverThrown):
        server.execute_decision_diff(None, [])  # type: ignore[arg-type]


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_decision_diff::payload
def test_execute_decision_diff_invalid_snapshot(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    result = server.execute_decision_diff(
        None,
        _with_timeout({"baseline": str(bad), "current": str(bad)}),
    )
    assert result.get("exit_code") == 2


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_decision_diff::payload
def test_execute_decision_diff_valid_snapshot(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(
        "{\"format_version\": 1, \"root\": null, \"decision_surfaces\": [\"a\"], \"value_decision_surfaces\": []}"
    )
    current.write_text(
        "{\"format_version\": 1, \"root\": null, \"decision_surfaces\": [\"a\", \"b\"], \"value_decision_surfaces\": [\"x\"]}"
    )
    result = server.execute_decision_diff(
        None,
        _with_timeout({"baseline": str(baseline), "current": str(current)}),
    )
    assert result.get("exit_code") == 0
    diff = result.get("diff") or {}
    assert "decision_surfaces" in diff


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_structure_diff::payload
def test_execute_structure_diff_requires_timeout_payload() -> None:
    with pytest.raises(NeverThrown):
        server.execute_structure_diff(None, None)


def test_execute_structure_diff_rejects_non_dict_payload() -> None:
    with pytest.raises(NeverThrown):
        server.execute_structure_diff(None, [])  # type: ignore[arg-type]


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_report_appends_sections(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    report_path = tmp_path / "report.md"
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("legacy\n")
    refactor_json = tmp_path / "refactor.json"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "report": str(report_path),
            "baseline": str(baseline_path),
            "decision_snapshot": "-",
            "structure_tree": "-",
            "structure_metrics": "-",
            "dot": "-",
            "synthesis_report": True,
            "refactor_plan": True,
            "refactor_plan_json": str(refactor_json),
        }),
    )
    assert result.get("baseline_written") is False
    assert "decision_snapshot" in result
    assert "structure_tree" in result
    assert "structure_metrics" in result
    assert "dot" in result
    assert report_path.exists()
    report_text = report_path.read_text()
    assert "decision_surfaces" in report_text
    assert "Synthesis plan" in report_text
    assert "Refactoring plan" in report_text
    assert "Resumability obligations:" in report_text
    assert "Incremental report obligations:" in report_text
    assert refactor_json.exists()


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_emits_test_reports(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = _artifact_out_dir(tmp_path)
    artifact_dir = _artifact_out_dir(tmp_path)
    artifact_dir = _artifact_out_dir(tmp_path)
    artifact_dir = _artifact_out_dir(tmp_path)
    artifact_dir = _artifact_out_dir(tmp_path)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_test_evidence_suggestions": True,
            "emit_test_obsolescence": True,
        }),
    )
    assert (artifact_dir / "test_evidence_suggestions.json").exists()
    assert (artifact_dir / "test_obsolescence_report.json").exists()
    assert (out_dir / "test_evidence_suggestions.md").exists()
    assert (out_dir / "test_obsolescence_report.md").exists()
    assert "test_evidence_suggestions_summary" in result
    assert "test_obsolescence_summary" in result


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_emits_obsolescence_delta(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = _artifact_out_dir(tmp_path)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [
                    {"key": {"k": "paramset", "params": ["x"]}, "display": "E:paramset::x"}
                ],
                "status": "mapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    baseline_payload = test_obsolescence_delta.build_baseline_payload_from_paths(
        str(out_dir / "test_evidence.json"),
        str(out_dir / "evidence_risk_registry.json"),
    )
    baseline_path = test_obsolescence_delta.resolve_baseline_path(tmp_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    test_obsolescence_delta.write_baseline(str(baseline_path), baseline_payload)

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_test_obsolescence_delta": True,
        }),
    )
    assert (artifact_dir / "test_obsolescence_delta.json").exists()
    assert (out_dir / "test_obsolescence_delta.md").exists()
    assert "test_obsolescence_delta_summary" in result


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_obsolescence_delta_from_state(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = _artifact_out_dir(tmp_path)
    key = evidence_keys.make_paramset_key(["x"])
    ref = test_obsolescence.EvidenceRef(
        key=key,
        identity=evidence_keys.key_identity(key),
        display=evidence_keys.render_display(key),
        opaque=False,
    )
    evidence_by_test = {"tests/test_sample.py::test_alpha": [ref]}
    status_by_test = {"tests/test_sample.py::test_alpha": "mapped"}
    candidates, summary = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    state_payload = test_obsolescence_state.build_state_payload(
        evidence_by_test, status_by_test, candidates, summary
    )
    state_path = artifact_dir / "test_obsolescence_state.json"
    state_path.write_text(json.dumps(state_payload, indent=2, sort_keys=True) + "\n")

    baseline_payload = state_payload["baseline"]
    baseline_path = test_obsolescence_delta.resolve_baseline_path(tmp_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    test_obsolescence_delta.write_baseline(str(baseline_path), baseline_payload)

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_test_obsolescence_delta": True,
            "test_obsolescence_state": str(state_path),
        }),
    )
    assert (artifact_dir / "test_obsolescence_delta.json").exists()
    assert (out_dir / "test_obsolescence_delta.md").exists()
    assert "test_obsolescence_delta_summary" in result


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_rejects_missing_obsolescence_state(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            _with_timeout({
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_obsolescence_delta": True,
                "test_obsolescence_state": str(
                    tmp_path / "artifacts" / "out" / "missing.json"
                ),
            }),
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_obsolescence_state(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = _artifact_out_dir(tmp_path)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )
    ls = _DummyServer(str(tmp_path))
    server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_test_obsolescence_state": True,
        }),
    )
    assert (artifact_dir / "test_obsolescence_state.json").exists()


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_writes_obsolescence_baseline(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "write_test_obsolescence_baseline": True,
        }),
    )
    baseline_path = test_obsolescence_delta.resolve_baseline_path(tmp_path)
    assert baseline_path.exists()
    assert result.get("test_obsolescence_baseline_written") is True


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_annotation_drift_delta(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_sample.py"
    test_file.write_text(
        "# gabion:evidence E:function_site::sample.py::pkg.fn\n"
        "def test_alpha():\n"
        "    assert True\n"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = _artifact_out_dir(tmp_path)
    key = {"k": "function_site", "site": {"path": "sample.py", "qual": "pkg.fn"}}
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [{"key": key, "display": "E:function_site::sample.py::pkg.fn"}],
                "status": "mapped",
            }
        ],
        "evidence_index": [
            {
                "key": key,
                "display": "E:function_site::sample.py::pkg.fn",
                "tests": ["tests/test_sample.py::test_alpha"],
            }
        ],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    drift_payload = test_annotation_drift.build_annotation_drift_payload(
        [tests_dir],
        root=tmp_path,
        evidence_path=out_dir / "test_evidence.json",
    )
    baseline_payload = test_annotation_drift_delta.build_baseline_payload(
        drift_payload.get("summary", {})
    )
    baseline_path = test_annotation_drift_delta.resolve_baseline_path(tmp_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    test_annotation_drift_delta.write_baseline(str(baseline_path), baseline_payload)

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(tests_dir)],
            "emit_test_annotation_drift_delta": True,
        }),
    )
    assert (artifact_dir / "test_annotation_drift_delta.json").exists()
    assert (out_dir / "test_annotation_drift_delta.md").exists()
    assert "test_annotation_drift_delta_summary" in result


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_rejects_missing_annotation_drift_state(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            _with_timeout({
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_annotation_drift_delta": True,
                "test_annotation_drift_state": str(
                    tmp_path / "artifacts" / "out" / "missing.json"
                ),
            }),
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_rejects_invalid_annotation_drift_state(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    _artifact_out_dir(tmp_path)
    state_path = tmp_path / "artifacts" / "out" / "test_annotation_drift.json"
    state_path.write_text(json.dumps(["bad"]), encoding="utf-8")
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            _with_timeout({
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_annotation_drift_delta": True,
                "test_annotation_drift_state": str(state_path),
            }),
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_annotation_drift_delta_from_state(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = _artifact_out_dir(tmp_path)
    drift_payload = {
        "version": 1,
        "summary": {
            "legacy_ambiguous": 0,
            "legacy_tag": 0,
            "ok": 1,
            "orphaned": 0,
        },
        "entries": [],
        "generated_by_spec_id": "spec",
        "generated_by_spec": {},
    }
    state_path = artifact_dir / "test_annotation_drift.json"
    state_path.write_text(json.dumps(drift_payload, indent=2, sort_keys=True) + "\n")

    baseline_payload = test_annotation_drift_delta.build_baseline_payload(
        drift_payload["summary"]
    )
    baseline_path = test_annotation_drift_delta.resolve_baseline_path(tmp_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    test_annotation_drift_delta.write_baseline(str(baseline_path), baseline_payload)

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_test_annotation_drift_delta": True,
            "test_annotation_drift_state": str(state_path),
        }),
    )
    assert (artifact_dir / "test_annotation_drift_delta.json").exists()
    assert (out_dir / "test_annotation_drift_delta.md").exists()
    assert "test_annotation_drift_delta_summary" in result


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_emits_annotation_drift(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        "# gabion:evidence E:function_site::sample.py::pkg.fn\n"
        "def test_alpha():\n"
        "    assert True\n"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = _artifact_out_dir(tmp_path)
    key = {"k": "function_site", "site": {"path": "sample.py", "qual": "pkg.fn"}}
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [{"key": key, "display": "E:function_site::sample.py::pkg.fn"}],
                "status": "mapped",
            }
        ],
        "evidence_index": [
            {
                "key": key,
                "display": "E:function_site::sample.py::pkg.fn",
                "tests": ["tests/test_sample.py::test_alpha"],
            }
        ],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(tests_dir)],
            "emit_test_annotation_drift": True,
        }),
    )
    assert (artifact_dir / "test_annotation_drift.json").exists()
    assert (out_dir / "test_annotation_drift.md").exists()
    assert "test_annotation_drift_summary" in result


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_emits_call_clusters(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        "from sample import caller\n"
        "\n"
        "def test_alpha():\n"
        "    caller(1, 2)\n"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = _artifact_out_dir(tmp_path)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(tests_dir), str(module_path)],
            "emit_call_clusters": True,
        }),
    )
    assert (artifact_dir / "call_clusters.json").exists()
    assert (out_dir / "call_clusters.md").exists()
    assert "call_clusters_summary" in result


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_emits_call_cluster_consolidation(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        "from sample import caller\n"
        "\n"
        "def test_alpha():\n"
        "    caller(1, 2)\n"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = _artifact_out_dir(tmp_path)
    call_footprint = evidence_keys.make_call_footprint_key(
        path="tests/test_sample.py",
        qual="test_alpha",
        targets=[{"path": "sample.py", "qual": "caller"}],
    )
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [
                    {
                        "key": call_footprint,
                        "display": evidence_keys.render_display(call_footprint),
                    }
                ],
                "status": "mapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(tests_dir), str(module_path)],
            "emit_call_cluster_consolidation": True,
        }),
    )
    assert (artifact_dir / "call_cluster_consolidation.json").exists()
    assert (out_dir / "call_cluster_consolidation.md").exists()
    assert "call_cluster_consolidation_summary" in result


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_requires_annotation_drift_baseline(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        "# gabion:evidence E:function_site::sample.py::pkg.fn\n"
        "def test_alpha():\n"
        "    assert True\n"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = {"k": "function_site", "site": {"path": "sample.py", "qual": "pkg.fn"}}
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [],
        "evidence_index": [
            {
                "key": key,
                "display": "E:function_site::sample.py::pkg.fn",
                "tests": [],
            }
        ],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            _with_timeout({
                "root": str(tmp_path),
                "paths": [str(tests_dir)],
                "emit_test_annotation_drift_delta": True,
            }),
        )


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_writes_annotation_drift_baseline(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text("def test_alpha():\n    assert True\n")

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "write_test_annotation_drift_baseline": True,
        }),
    )
    baseline_path = test_annotation_drift_delta.resolve_baseline_path(tmp_path)
    assert baseline_path.exists()
    assert result.get("test_annotation_drift_baseline_written") is True


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_rejects_annotation_drift_flags(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            _with_timeout({
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_annotation_drift_delta": True,
                "write_test_annotation_drift_baseline": True,
            }),
        )


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_requires_ambiguity_baseline(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            _with_timeout({
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_ambiguity_delta": True,
            }),
        )


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_ambiguity_delta(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    baseline_payload = ambiguity_delta.build_baseline_payload([])
    baseline_path = ambiguity_delta.resolve_baseline_path(tmp_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    ambiguity_delta.write_baseline(str(baseline_path), baseline_payload)

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_ambiguity_delta": True,
        }),
    )
    out_dir = tmp_path / "out"
    artifact_dir = _artifact_out_dir(tmp_path)
    assert (artifact_dir / "ambiguity_delta.json").exists()
    assert (out_dir / "ambiguity_delta.md").exists()
    assert "ambiguity_delta_summary" in result


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_ambiguity_delta_from_state(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = _artifact_out_dir(tmp_path)
    witnesses = [
        {
            "kind": "local_resolution_ambiguous",
            "site": {
                "path": "sample.py",
                "function": "caller",
                "span": [1, 0, 1, 5],
            },
            "candidate_count": 2,
        }
    ]
    state_payload = ambiguity_state.build_state_payload(witnesses)
    state_path = artifact_dir / "ambiguity_state.json"
    state_path.write_text(json.dumps(state_payload, indent=2, sort_keys=True) + "\n")

    baseline_payload = ambiguity_delta.build_baseline_payload(witnesses)
    baseline_path = ambiguity_delta.resolve_baseline_path(tmp_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    ambiguity_delta.write_baseline(str(baseline_path), baseline_payload)

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_ambiguity_delta": True,
            "ambiguity_state": str(state_path),
        }),
    )
    assert (artifact_dir / "ambiguity_delta.json").exists()
    assert (out_dir / "ambiguity_delta.md").exists()
    assert "ambiguity_delta_summary" in result


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_rejects_missing_ambiguity_state(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            _with_timeout({
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_ambiguity_delta": True,
                "ambiguity_state": str(tmp_path / "artifacts" / "out" / "missing.json"),
            }),
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_ambiguity_state(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_ambiguity_state": True,
        }),
    )
    artifact_dir = _artifact_out_dir(tmp_path)
    assert (artifact_dir / "ambiguity_state.json").exists()


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_writes_ambiguity_baseline(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout({
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "write_ambiguity_baseline": True,
        }),
    )
    baseline_path = ambiguity_delta.resolve_baseline_path(tmp_path)
    assert baseline_path.exists()
    assert result.get("ambiguity_baseline_written") is True


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_rejects_ambiguity_flags(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            _with_timeout({
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_ambiguity_delta": True,
                "write_ambiguity_baseline": True,
            }),
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_rejects_ambiguity_state_conflict(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            _with_timeout({
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_ambiguity_state": True,
                "ambiguity_state": "artifacts/out/ambiguity_state.json",
            }),
        )

# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_rejects_conflicting_obsolescence_flags(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            _with_timeout({
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_obsolescence_delta": True,
                "write_test_obsolescence_baseline": True,
            }),
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_rejects_obsolescence_state_conflict(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            _with_timeout({
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_obsolescence_state": True,
                "test_obsolescence_state": "artifacts/out/test_obsolescence_state.json",
            }),
        )


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_requires_obsolescence_baseline(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(
            ls,
            _with_timeout({
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_obsolescence_delta": True,
            }),
        )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_defaults_payload(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(ls, None)


def test_execute_command_rejects_non_dict_payload(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(ls, [])  # type: ignore[arg-type]


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_refactor::ls,payload
def test_execute_refactor_valid_payload(tmp_path: Path) -> None:
    module_path = tmp_path / "target.py"
    module_path.write_text("def f(a, b):\n    return a + b\n")
    ls = _DummyServer(str(tmp_path))
    result = server.execute_refactor(
        ls,
        _with_timeout({
            "protocol_name": "ExampleProto",
            "bundle": ["a", "b"],
            "target_path": str(module_path),
            "target_functions": [],
        }),
    )
    assert result.get("errors") == []
    edits = result.get("edits", [])
    assert edits


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_refactor::ls,payload
def test_execute_refactor_invalid_payload(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_refactor(ls, _with_timeout({"protocol_name": 123}))
    assert result.get("errors")


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_refactor::ls,payload
def test_execute_refactor_payload_none(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_refactor(ls, None)


def test_execute_refactor_payload_non_dict(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_refactor(ls, [])  # type: ignore[arg-type]


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_synthesis::payload
def test_execute_synthesis_invalid_payload(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_synthesis(ls, _with_timeout({"bundles": "not-a-list"}))
    assert result.get("errors")


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_synthesis::payload
def test_execute_synthesis_records_bundle_tiers(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_synthesis(
        ls,
        _with_timeout({
            "bundles": [
                {"bundle": ["a", "b"], "tier": 2},
            ],
            "existing_names": [],
        }),
    )
    assert result.get("errors") == []
    assert "protocols" in result


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_synthesis::payload
def test_execute_synthesis_payload_none(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_synthesis(ls, None)


def test_execute_synthesis_payload_non_dict(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_synthesis(ls, [])  # type: ignore[arg-type]


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_synthesis::payload
def test_execute_synthesis_skips_empty_bundle(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_synthesis(
        ls,
        _with_timeout({
            "bundles": [{"bundle": [], "tier": 2}],
        }),
    )
    assert result.get("protocols") == []
