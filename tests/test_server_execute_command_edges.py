from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from gabion import server
from gabion.analysis.timeout_context import TimeoutContext, pack_call_stack
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


class _DummyNotifyingServer(_DummyServer):
    def __init__(self, root_path: str) -> None:
        super().__init__(root_path)
        self.notifications: list[tuple[str, dict[str, object]]] = []

    def send_notification(self, method: str, params: dict[str, object]) -> None:
        self.notifications.append((method, params))


@dataclass
class _CommandResult:
    exit_code: int
    violations: int


_TIMEOUT_PAYLOAD = {
    "analysis_timeout_ticks": 50_000,
    "analysis_timeout_tick_ns": 1_000_000,
}


def _with_timeout(payload: dict) -> dict:
    merged = {**_TIMEOUT_PAYLOAD, **payload}
    if (
        "analysis_timeout_ticks" not in payload
        and (
            "analysis_timeout_ms" in payload
            or "analysis_timeout_seconds" in payload
        )
    ):
        merged.pop("analysis_timeout_ticks", None)
        merged.pop("analysis_timeout_tick_ns", None)
    return merged


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


def _timeout_exc(
    *,
    progress: dict[str, object] | object | None = None,
) -> server.TimeoutExceeded:
    payload: dict[str, object] = {}
    if progress is not None:
        payload["progress"] = progress
    context = TimeoutContext(
        call_stack=pack_call_stack([{"path": "mod.py", "qual": "pkg.f"}]),
        progress=payload.get("progress") if isinstance(payload.get("progress"), dict) else None,
    )
    if progress is not None and not isinstance(progress, dict):
        class _ContextProxy:
            def as_payload(self) -> dict[str, object]:
                return {"progress": progress}

        return server.TimeoutExceeded(_ContextProxy())  # type: ignore[arg-type]
    return server.TimeoutExceeded(context)


def _empty_analysis_result() -> server.AnalysisResult:
    return server.AnalysisResult(
        groups_by_path={},
        param_spans_by_path={},
        bundle_sites_by_path={},
        type_suggestions=[],
        type_ambiguities=[],
        type_callsite_evidence=[],
        constant_smells=[],
        unused_arg_smells=[],
        forest=server.Forest(),
    )


def _execute_with_deps(
    ls: _DummyServer,
    payload: dict,
    **overrides: object,
) -> dict:
    deps = server._default_execute_command_deps().with_overrides(**overrides)
    return server.execute_command_with_deps(ls, payload, deps=deps)


def _progress_values(ls: _DummyNotifyingServer) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for method, params in ls.notifications:
        if method != server._LSP_PROGRESS_NOTIFICATION_METHOD:
            continue
        if not isinstance(params, dict):
            continue
        if params.get("token") != server._LSP_PROGRESS_TOKEN:
            continue
        value = params.get("value")
        if isinstance(value, dict):
            values.append(value)
    return values


def test_execute_command_emits_lsp_progress_success_terminal(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyNotifyingServer(str(tmp_path))

    result = _execute_with_deps(
        ls,
        _with_timeout({"root": str(tmp_path), "paths": [str(module_path)], "report": "-"}),
        analyze_paths_fn=lambda *_args, **_kwargs: _empty_analysis_result(),
    )

    assert result["analysis_state"] == "succeeded"
    progress_values = _progress_values(ls)
    assert progress_values
    assert all(value.get("schema") == "gabion/dataflow_progress_v1" for value in progress_values)
    assert all(value.get("format_version") == 1 for value in progress_values)
    assert any(
        "completed_files" in value
        and "in_progress_files" in value
        and "remaining_files" in value
        and "total_files" in value
        and "work_done" in value
        and "work_total" in value
        for value in progress_values
    )
    assert any(
        value.get("done") is True and value.get("analysis_state") == "succeeded"
        for value in progress_values
    )




def test_execute_command_emits_resume_progress_before_completion(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyNotifyingServer(str(tmp_path))

    result = _execute_with_deps(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": "-",
                "resume_checkpoint": str(tmp_path / "missing_resume.json"),
            }
        ),
        analyze_paths_fn=lambda *_args, **_kwargs: _empty_analysis_result(),
    )

    assert result["analysis_state"] == "succeeded"
    progress_values = _progress_values(ls)
    assert progress_values
    resume_progress = next(
        (
            value
            for value in progress_values
            if isinstance(value.get("resume_checkpoint"), dict)
        ),
        None,
    )
    assert isinstance(resume_progress, dict)
    resume_checkpoint = resume_progress.get("resume_checkpoint")
    assert isinstance(resume_checkpoint, dict)
    assert str(resume_checkpoint.get("status", "")).startswith("checkpoint_")
    assert resume_checkpoint.get("reused_files") == 0
    assert resume_checkpoint.get("total_files") == 1
    assert resume_progress.get("classification") == "resume_checkpoint_detected"
    assert resume_progress.get("done") is not True

def test_execute_command_emits_lsp_progress_timeout_terminal(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyNotifyingServer(str(tmp_path))

    result = _execute_with_deps(
        ls,
        _with_timeout({"root": str(tmp_path), "paths": [str(module_path)], "report": "-"}),
        analyze_paths_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            _timeout_exc(progress={"classification": "timed_out_no_progress"})
        ),
    )

    assert result["timeout"] is True
    progress_values = _progress_values(ls)
    terminal_value = progress_values[-1]
    assert terminal_value.get("done") is True
    analysis_state = str(terminal_value.get("analysis_state", ""))
    assert analysis_state.startswith("timed_out_")
    timeout_progress = (result.get("timeout_context") or {}).get("progress")
    assert isinstance(timeout_progress, dict)
    assert terminal_value.get("classification") == timeout_progress.get("classification")


def test_execute_command_emits_lsp_progress_failed_terminal(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyNotifyingServer(str(tmp_path))

    with pytest.raises(RuntimeError, match="boom"):
        _execute_with_deps(
            ls,
            _with_timeout({"root": str(tmp_path), "paths": [str(module_path)], "report": "-"}),
            analyze_paths_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    progress_values = _progress_values(ls)
    terminal_value = progress_values[-1]
    assert terminal_value.get("done") is True
    assert terminal_value.get("analysis_state") == "failed"
    assert terminal_value.get("classification") == "failed"


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


def test_analysis_timeout_budget_reserves_default_cleanup_grace() -> None:
    total_ns, analysis_ns, cleanup_ns = server._analysis_timeout_budget_ns(
        {
            "analysis_timeout_ticks": 100,
            "analysis_timeout_tick_ns": 1_000_000,
        }
    )
    assert total_ns == 100_000_000
    assert cleanup_ns == 20_000_000
    assert analysis_ns == 80_000_000


def test_analysis_timeout_budget_caps_configured_cleanup_grace() -> None:
    total_ns, analysis_ns, cleanup_ns = server._analysis_timeout_budget_ns(
        {
            "analysis_timeout_ticks": 100,
            "analysis_timeout_tick_ns": 1_000_000,
            "analysis_timeout_grace_ms": 90,
        }
    )
    assert total_ns == 100_000_000
    assert cleanup_ns == 20_000_000
    assert analysis_ns == 80_000_000


def test_collection_checkpoint_flush_due() -> None:
    now_ns = 20_000_000_000
    assert server._collection_checkpoint_flush_due(
        intro_changed=True,
        remaining_files=10,
        now_ns=now_ns,
        last_flush_ns=0,
    )
    assert server._collection_checkpoint_flush_due(
        intro_changed=False,
        remaining_files=0,
        now_ns=now_ns,
        last_flush_ns=0,
    )
    assert server._collection_checkpoint_flush_due(
        intro_changed=False,
        remaining_files=1,
        now_ns=now_ns,
        last_flush_ns=now_ns - server._COLLECTION_CHECKPOINT_FLUSH_INTERVAL_NS,
    )
    assert not server._collection_checkpoint_flush_due(
        intro_changed=False,
        remaining_files=1,
        now_ns=1,
        last_flush_ns=0,
    )


def test_collection_report_flush_due() -> None:
    now_ns = 20_000_000_000
    assert server._collection_report_flush_due(
        completed_files=1,
        remaining_files=99,
        now_ns=now_ns,
        last_flush_ns=0,
        last_flush_completed=-1,
    )
    assert server._collection_report_flush_due(
        completed_files=10,
        remaining_files=90,
        now_ns=now_ns,
        last_flush_ns=0,
        last_flush_completed=1,
    )
    assert server._collection_report_flush_due(
        completed_files=2,
        remaining_files=98,
        now_ns=now_ns,
        last_flush_ns=now_ns - server._COLLECTION_REPORT_FLUSH_INTERVAL_NS,
        last_flush_completed=1,
    )
    assert server._collection_report_flush_due(
        completed_files=2,
        remaining_files=0,
        now_ns=1,
        last_flush_ns=1,
        last_flush_completed=1,
    )
    assert not server._collection_report_flush_due(
        completed_files=2,
        remaining_files=10,
        now_ns=1,
        last_flush_ns=1,
        last_flush_completed=1,
    )


def test_projection_phase_flush_due() -> None:
    assert server._projection_phase_flush_due(
        phase="post",
        now_ns=1,
        last_flush_ns=1,
    )
    assert not server._projection_phase_flush_due(
        phase="forest",
        now_ns=1,
        last_flush_ns=1,
    )
    assert server._projection_phase_flush_due(
        phase="forest",
        now_ns=server._COLLECTION_REPORT_FLUSH_INTERVAL_NS + 1,
        last_flush_ns=1,
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
    timeout_budget = progress.get("timeout_budget")
    assert isinstance(timeout_budget, dict)
    assert int(timeout_budget["cleanup_grace_ns"]) >= 0


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
        "analysis_timeout_tick_ns": 200_000_000,
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
        include_wl_refinement=False,
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
    assert progress.get("cleanup_truncated") is True
    cleanup_steps = progress.get("cleanup_timeout_steps")
    assert cleanup_steps == ["render_timeout_report", "incremental_obligations"]


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
                    "analysis_timeout_ms": 250,
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
    if not obligations:
        assert progress.get("cleanup_truncated") is True
        cleanup_steps = progress.get("cleanup_timeout_steps")
        assert isinstance(cleanup_steps, list)
        assert "incremental_obligations" in cleanup_steps
        return
    progress_contract_entries = [
        entry
        for entry in obligations
        if isinstance(entry, dict)
        and entry.get("contract") == "progress_report_contract"
    ]
    assert progress_contract_entries
    assert any(
        isinstance(entry, dict) and entry.get("status") == "SATISFIED"
        for entry in progress_contract_entries
    )
    if isinstance(collection_phase, dict):
        assert (
            "Collection progress checkpoint (provisional)." in report_text
            or "Collection bootstrap checkpoint (provisional)." in report_text
        )
        assert "## Section `intro`\nPENDING" not in report_text
        if collection_phase.get("status") == "checkpointed":
            assert int(collection_phase.get("completed_files", 0)) >= 0
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
                "analysis_timeout_ms": 250,
            }
        ),
    )
    assert result.get("timeout") is True
    progress = (result.get("timeout_context") or {}).get("progress")
    assert isinstance(progress, dict)
    obligations = progress.get("incremental_obligations")
    assert isinstance(obligations, list)
    if not obligations:
        assert progress.get("cleanup_truncated") is True
        cleanup_steps = progress.get("cleanup_timeout_steps")
        assert isinstance(cleanup_steps, list)
        assert "incremental_obligations" in cleanup_steps
        return
    assert any(
        isinstance(entry, dict)
        and entry.get("contract") == "incremental_projection_contract"
        and entry.get("detail") == "stale_input"
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
                    "analysis_timeout_ticks": 50_000,
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
    assert post_phase.get("work_done") == 1
    assert post_phase.get("work_total") == 1
    edge_phase = phases["edge"]
    assert isinstance(edge_phase, dict)
    assert isinstance(edge_phase.get("work_done"), int)
    assert isinstance(edge_phase.get("work_total"), int)


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


def test_collection_semantic_progress_treats_completed_path_as_non_regression() -> None:
    progress = server._collection_semantic_progress(
        previous_collection_resume={
            "completed_paths": [],
            "in_progress_scan_by_path": {
                "a.py": {
                    "phase": "function_scan",
                    "processed_functions": ["f1", "f2"],
                }
            },
        },
        collection_resume={
            "completed_paths": ["a.py"],
            "in_progress_scan_by_path": {},
        },
        total_files=1,
    )
    assert progress["regressed_processed_functions_count"] == 0
    assert progress["completed_files_delta"] == 1
    assert progress["monotonic_progress"] is True
    assert progress["substantive_progress"] is True


def test_collection_semantic_progress_flags_state_loss_regression() -> None:
    progress = server._collection_semantic_progress(
        previous_collection_resume={
            "completed_paths": [],
            "in_progress_scan_by_path": {
                "a.py": {
                    "phase": "function_scan",
                    "processed_functions": ["f1", "f2"],
                }
            },
        },
        collection_resume={
            "completed_paths": [],
            "in_progress_scan_by_path": {},
        },
        total_files=1,
    )
    assert progress["regressed_processed_functions_count"] == 2
    assert progress["completed_files_delta"] == 0
    assert progress["monotonic_progress"] is False
    assert progress["substantive_progress"] is False


def test_collection_semantic_progress_tracks_analysis_index_hydration() -> None:
    progress = server._collection_semantic_progress(
        previous_collection_resume={
            "completed_paths": ["a.py"],
            "in_progress_scan_by_path": {},
            "analysis_index_resume": {
                "hydrated_paths": ["a.py"],
                "hydrated_paths_count": 1,
            },
        },
        collection_resume={
            "completed_paths": ["a.py"],
            "in_progress_scan_by_path": {},
            "analysis_index_resume": {
                "hydrated_paths": ["a.py", "b.py"],
                "hydrated_paths_count": 2,
            },
        },
        total_files=2,
    )
    assert progress["hydrated_paths_delta"] == 1
    assert progress["hydrated_paths_regressed"] == 0
    assert progress["monotonic_progress"] is True
    assert progress["substantive_progress"] is True


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


def test_externalize_resume_states_reject_path_order_regression(tmp_path: Path) -> None:
    with pytest.raises(NeverThrown):
        server._externalize_collection_resume_states(
            path=tmp_path / "resume.json",
            collection_resume={
                "in_progress_scan_by_path": {
                    "b.py": {"phase": "scan_pending"},
                    "a.py": {"phase": "scan_pending"},
                }
            },
        )


def test_inflate_resume_states_reject_path_order_regression(tmp_path: Path) -> None:
    with pytest.raises(NeverThrown):
        server._inflate_collection_resume_states(
            path=tmp_path / "resume.json",
            collection_resume={
                "in_progress_scan_by_path": {
                    "b.py": {"phase": "scan_pending"},
                    "a.py": {"phase": "scan_pending"},
                }
            },
        )


def test_externalize_and_inflate_analysis_index_resume_state_ref(
    tmp_path: Path,
) -> None:
    payload = {
        "in_progress_scan_by_path": {},
        "analysis_index_resume": {
            "format_version": 1,
            "phase": "analysis_index_hydration",
            "hydrated_paths": ["a.py"],
            "hydrated_paths_count": 1,
            "function_count": 1,
            "class_count": 0,
            "functions_by_qual": {
                "m.a": {
                    "name": "a",
                    "qual": "m.a",
                    "path": "a.py",
                    "params": [],
                    "annots": {},
                    "calls": [],
                    "unused_params": [],
                    "defaults": [],
                    "transparent": True,
                    "scope": [],
                    "lexical_scope": [],
                    "decision_params": [],
                    "value_decision_params": [],
                    "value_decision_reasons": [],
                    "positional_params": [],
                    "kwonly_params": [],
                    "param_spans": {},
                    "padding": "x" * 70000,
                }
            },
            "symbol_table": {
                "imports": [],
                "internal_roots": [],
                "external_filter": True,
                "star_imports": {},
                "module_exports": {},
                "module_export_map": {},
            },
            "class_index": {},
        },
    }
    externalized = server._externalize_collection_resume_states(
        path=tmp_path / "resume.json",
        collection_resume=payload,
    )
    raw_analysis_index_resume = externalized.get("analysis_index_resume")
    assert isinstance(raw_analysis_index_resume, dict)
    assert isinstance(raw_analysis_index_resume.get("state_ref"), str)
    inflated = server._inflate_collection_resume_states(
        path=tmp_path / "resume.json",
        collection_resume=externalized,
    )
    inflated_resume = inflated.get("analysis_index_resume")
    assert isinstance(inflated_resume, dict)
    assert inflated_resume.get("hydrated_paths_count") == 1


def test_analysis_index_resume_signature_prefers_resume_digest() -> None:
    signature = server._analysis_index_resume_signature(
        {
            "analysis_index_resume": {
                "phase": "analysis_index_hydration",
                "hydrated_paths": ["a.py"],
                "hydrated_paths_count": 1,
                "function_count": 2,
                "class_count": 1,
                "resume_digest": "abc123",
            }
        }
    )
    assert signature == (1, hashlib.sha1(b'[\"a.py\"]').hexdigest(), 2, 1, "analysis_index_hydration", "abc123")


def test_resolve_analysis_resume_checkpoint_path_variants(tmp_path: Path) -> None:
    assert server._resolve_analysis_resume_checkpoint_path(False, root=tmp_path) is None
    assert server._resolve_analysis_resume_checkpoint_path(None, root=tmp_path) == (
        tmp_path / server._DEFAULT_ANALYSIS_RESUME_CHECKPOINT
    )
    assert server._resolve_analysis_resume_checkpoint_path(True, root=tmp_path) == (
        tmp_path / server._DEFAULT_ANALYSIS_RESUME_CHECKPOINT
    )
    assert (
        server._resolve_analysis_resume_checkpoint_path("  ", root=tmp_path) is None
    )
    assert server._resolve_analysis_resume_checkpoint_path("resume.json", root=tmp_path) == (
        tmp_path / "resume.json"
    )
    absolute = tmp_path / "abs.json"
    assert server._resolve_analysis_resume_checkpoint_path(
        str(absolute), root=tmp_path
    ) == absolute
    with pytest.raises(NeverThrown):
        server._resolve_analysis_resume_checkpoint_path(123, root=tmp_path)


def test_analysis_timeout_grace_ns_validation_and_cap() -> None:
    assert server._analysis_timeout_grace_ns({}, total_ns=1) == 0
    assert server._analysis_timeout_grace_ns({}, total_ns=100) == 20
    assert (
        server._analysis_timeout_grace_ns(
            {"analysis_timeout_grace_ms": 1000},
            total_ns=100,
        )
        == 20
    )
    assert (
        server._analysis_timeout_grace_ns(
            {
                "analysis_timeout_grace_ticks": 5,
                "analysis_timeout_grace_tick_ns": 2,
            },
            total_ns=100,
        )
        == 10
    )
    assert (
        server._analysis_timeout_grace_ns(
            {"analysis_timeout_grace_seconds": "0.000000010"},
            total_ns=100,
        )
        == 10
    )
    with pytest.raises(NeverThrown):
        server._analysis_timeout_grace_ns(
            {"analysis_timeout_grace_ticks": 1},
            total_ns=100,
        )
    with pytest.raises(NeverThrown):
        server._analysis_timeout_grace_ns(
            {"analysis_timeout_grace_tick_ns": "bad", "analysis_timeout_grace_ticks": 1},
            total_ns=100,
        )
    with pytest.raises(NeverThrown):
        server._analysis_timeout_grace_ns(
            {"analysis_timeout_grace_ms": 0},
            total_ns=100,
        )
    with pytest.raises(NeverThrown):
        server._analysis_timeout_grace_ns(
            {"analysis_timeout_grace_seconds": "bad"},
            total_ns=100,
        )


def test_analysis_manifest_digest_from_witness_validation() -> None:
    witness = {
        "root": "/r",
        "recursive": True,
        "include_invariant_propositions": False,
        "include_wl_refinement": False,
        "config": {
            "exclude_dirs": ["a"],
            "ignore_params": ["b"],
            "strictness": "high",
            "external_filter": True,
            "transparent_decorators": [],
        },
        "files": [{"path": "a.py", "size": 1, "mtime_ns": 2}],
    }
    digest = server._analysis_manifest_digest_from_witness(witness)
    assert isinstance(digest, str)
    assert server._analysis_manifest_digest_from_witness({"files": "bad"}) is None
    assert server._analysis_manifest_digest_from_witness({"files": [{}], "config": {}}) is None
    assert (
        server._analysis_manifest_digest_from_witness(
            {"files": [{"path": "a.py"}], "config": {}, "root": 1}
        )
        is None
    )
    assert (
        server._analysis_manifest_digest_from_witness(
            {
                "files": [{"path": "a.py"}],
                "config": {"exclude_dirs": object()},
                "root": "/r",
                "recursive": True,
                "include_invariant_propositions": False,
                "include_wl_refinement": False,
            }
        )
        is None
    )


def test_load_analysis_resume_checkpoint_and_manifest_validation(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "resume.json"
    input_witness = {"witness_digest": "wd", "x": 1}
    assert (
        server._load_analysis_resume_checkpoint(path=checkpoint_path, input_witness=input_witness)
        is None
    )
    checkpoint_path.write_text("{")
    assert (
        server._load_analysis_resume_checkpoint(path=checkpoint_path, input_witness=input_witness)
        is None
    )
    checkpoint_path.write_text("[]")
    assert (
        server._load_analysis_resume_checkpoint(path=checkpoint_path, input_witness=input_witness)
        is None
    )
    checkpoint_path.write_text(
        json.dumps(
            {
                "format_version": 0,
                "input_witness": input_witness,
                "collection_resume": {},
            }
        )
    )
    assert (
        server._load_analysis_resume_checkpoint(path=checkpoint_path, input_witness=input_witness)
        is None
    )
    checkpoint_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "input_witness_digest": "other",
                "input_witness": input_witness,
                "collection_resume": {},
            }
        )
    )
    assert (
        server._load_analysis_resume_checkpoint(path=checkpoint_path, input_witness=input_witness)
        is None
    )
    checkpoint_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "input_witness_digest": "wd",
                "input_witness": {"witness_digest": "wd", "x": 2},
                "collection_resume": {},
            }
        )
    )
    assert (
        server._load_analysis_resume_checkpoint(path=checkpoint_path, input_witness=input_witness)
        is None
    )
    checkpoint_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "input_witness_digest": "wd",
                "input_witness": input_witness,
                "collection_resume": "bad",
            }
        )
    )
    assert (
        server._load_analysis_resume_checkpoint(path=checkpoint_path, input_witness=input_witness)
        is None
    )
    checkpoint_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "input_witness_digest": "wd",
                "input_witness": input_witness,
                "collection_resume": {"in_progress_scan_by_path": {}},
                "input_manifest_digest": "md",
            }
        )
    )
    loaded = server._load_analysis_resume_checkpoint(
        path=checkpoint_path,
        input_witness=input_witness,
    )
    assert isinstance(loaded, dict)
    manifest = server._load_analysis_resume_checkpoint_manifest(
        path=checkpoint_path,
        manifest_digest="md",
    )
    assert isinstance(manifest, tuple)
    witness, collection_resume = manifest
    assert witness == input_witness
    assert isinstance(collection_resume, dict)
    assert (
        server._load_analysis_resume_checkpoint_manifest(
            path=checkpoint_path,
            manifest_digest="other",
        )
        is None
    )


def test_analysis_input_witness_handles_missing_unreadable_and_syntax(
    tmp_path: Path,
) -> None:
    config = server.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
        transparent_decorators=set(),
    )
    missing = tmp_path / "missing.py"
    unreadable = tmp_path / "bad_utf.py"
    syntax = tmp_path / "syntax.py"
    unreadable.write_text("print('x')\n")
    syntax.write_text("def broken(:\n")

    original_read = server._read_text_profiled

    def _patched_read(path: Path, *, io_name: str, encoding: str | None = None) -> str:
        if path == unreadable:
            raise UnicodeError("boom")
        return original_read(path, io_name=io_name, encoding=encoding)

    witness = server._analysis_input_witness(
        root=tmp_path,
        file_paths=[missing, unreadable, syntax],
        recursive=True,
        include_invariant_propositions=False,
        include_wl_refinement=False,
        config=config,
        read_text_fn=_patched_read,
    )
    files = witness.get("files")
    assert isinstance(files, list)
    by_path = {entry["path"]: entry for entry in files if isinstance(entry, dict)}
    assert by_path[str(missing)].get("missing") is True
    unreadable_error = by_path[str(unreadable)].get("parse_error")
    assert isinstance(unreadable_error, dict)
    assert unreadable_error.get("kind") == "UnicodeError"
    syntax_error = by_path[str(syntax)].get("parse_error")
    assert isinstance(syntax_error, dict)
    assert syntax_error.get("kind") == "SyntaxError"


def test_analysis_input_witness_normalizes_non_scalar_ast_values(
    tmp_path: Path,
) -> None:
    class _CustomValue:
        def __repr__(self) -> str:
            return "CustomValue()"

    class _CustomNode(server.ast.AST):
        _fields = ("payload",)
        _attributes = ("lineno", "col_offset")

        def __init__(self) -> None:
            self.payload = {
                "tuple": (1, "a"),
                "set": {"z", "a"},
                "frozen": frozenset({"y", "b"}),
                "float": 1.5,
                "bytes": b"ab",
                "complex": complex(1, 2),
                "ellipsis": Ellipsis,
                "custom": _CustomValue(),
            }
            self.lineno = 1
            self.col_offset = 0

    module_path = tmp_path / "sample.py"
    module_path.write_text("x = 1\n")
    config = server.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
        transparent_decorators=set(),
    )
    witness = server._analysis_input_witness(
        root=tmp_path,
        file_paths=[module_path],
        recursive=True,
        include_invariant_propositions=False,
        include_wl_refinement=False,
        config=config,
        parse_source_fn=lambda *_args, **_kwargs: _CustomNode(),
    )
    table = witness.get("ast_intern_table")
    assert isinstance(table, dict)
    file_entry = next(entry for entry in witness["files"] if isinstance(entry, dict))
    ast_ref = file_entry.get("ast_ref")
    assert isinstance(ast_ref, str)
    normalized = table[ast_ref]
    assert isinstance(normalized, dict)
    fields = normalized.get("fields")
    assert isinstance(fields, dict)
    payload = fields.get("payload")
    assert isinstance(payload, dict)
    assert payload["bytes"] == {"_py": "bytes", "hex": "6162"}
    assert payload["ellipsis"] == {"_py": "ellipsis"}
    assert payload["tuple"] == {"_py": "tuple", "items": [1, "a"]}
    assert payload["custom"] == {"_py": "_CustomValue", "repr": "CustomValue()"}


def test_analysis_resume_progress_uses_observed_file_counts() -> None:
    progress = server._analysis_resume_progress(
        collection_resume={
            "completed_paths": ["a.py", "b.py"],
            "in_progress_scan_by_path": {"c.py": {"phase": "scan_pending"}},
        },
        total_files=0,
    )
    assert progress == {
        "completed_files": 2,
        "in_progress_files": 1,
        "remaining_files": 1,
        "total_files": 3,
    }


def test_in_progress_scan_states_filters_malformed_entries() -> None:
    states = server._in_progress_scan_states(
        {
            "in_progress_scan_by_path": {
                "a.py": {"phase": "scan_pending"},
                "b.py": "bad",
                1: {"phase": "ignored"},
            }
        }
    )
    assert states == {"a.py": {"phase": "scan_pending"}}
    with pytest.raises(NeverThrown):
        server._in_progress_scan_states(
            {
                "in_progress_scan_by_path": {
                    "b.py": {"phase": "scan_pending"},
                    "a.py": {"phase": "scan_pending"},
                }
            }
        )


def test_analysis_index_resume_helpers_fallbacks() -> None:
    resume = {
        "analysis_index_resume": {
            "hydrated_paths_count": -1,
            "hydrated_paths_digest": "digest",
            "function_count": "bad",
            "class_count": "bad",
            "phase": 1,
            "resume_digest": "",
        }
    }
    assert server._analysis_index_resume_hydrated_count(resume) == 0
    digest = server._analysis_index_resume_hydrated_digest(resume)
    assert isinstance(digest, str) and digest
    signature = server._analysis_index_resume_signature(resume)
    assert signature[2] == 0
    assert signature[3] == 0
    assert signature[4] == ""
    summary = server._analysis_index_resume_summary(resume)
    assert isinstance(summary, dict)
    assert summary["phase"] == "analysis_index_hydration"


def test_analysis_index_resume_hydrated_helpers_non_int_fallback() -> None:
    resume = {
        "analysis_index_resume": {
            "hydrated_paths_count": "bad",
            "hydrated_paths_digest": "",
        }
    }
    assert server._analysis_index_resume_hydrated_count(resume) == 0
    digest = server._analysis_index_resume_hydrated_digest(resume)
    expected = hashlib.sha1(
        server._canonical_json_text({"count": 0}).encode("utf-8")
    ).hexdigest()
    assert digest == expected


def test_load_report_section_journal_validation_paths(tmp_path: Path) -> None:
    journal_path = tmp_path / "sections.json"
    sections, reason = server._load_report_section_journal(
        path=journal_path,
        witness_digest="wd",
    )
    assert sections == {}
    assert reason is None
    journal_path.write_text("{")
    sections, reason = server._load_report_section_journal(
        path=journal_path,
        witness_digest="wd",
    )
    assert sections == {}
    assert reason == "policy"
    journal_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "witness_digest": "other",
                "sections": {},
            }
        )
    )
    sections, reason = server._load_report_section_journal(
        path=journal_path,
        witness_digest="wd",
    )
    assert sections == {}
    assert reason == "stale_input"
    journal_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "witness_digest": "wd",
                "sections": {
                    "intro": {"lines": ["ok"]},
                    "bad": {"lines": [1]},
                    1: {"lines": ["ignored"]},
                },
            }
        )
    )
    sections, reason = server._load_report_section_journal(
        path=journal_path,
        witness_digest="wd",
    )
    assert reason is None
    assert sections == {"intro": ["ok"], "1": ["ignored"]}


def test_load_report_phase_checkpoint_validation_paths(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "phases.json"
    assert (
        server._load_report_phase_checkpoint(
            path=checkpoint_path,
            witness_digest="wd",
        )
        == {}
    )
    checkpoint_path.write_text("{")
    assert (
        server._load_report_phase_checkpoint(
            path=checkpoint_path,
            witness_digest="wd",
        )
        == {}
    )
    checkpoint_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "witness_digest": "other",
                "phases": {},
            }
        )
    )
    assert (
        server._load_report_phase_checkpoint(
            path=checkpoint_path,
            witness_digest="wd",
        )
        == {}
    )
    checkpoint_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "witness_digest": "wd",
                "phases": {
                    "collection": {"status": "ok"},
                    "bad": "ignored",
                    1: {"status": "ignored"},
                },
            }
        )
    )
    phases = server._load_report_phase_checkpoint(
        path=checkpoint_path,
        witness_digest="wd",
    )
    assert phases == {"collection": {"status": "ok"}, "1": {"status": "ignored"}}


def test_render_incremental_report_marks_missing_dep_and_policy() -> None:
    report_text, pending = server._render_incremental_report(
        analysis_state="analysis_collection_in_progress",
        progress_payload={
            "phase": "forest",
            "work_done": 3,
            "work_total": 4,
            "classification": "timed_out_no_progress",
            "resume_supported": False,
        },
        projection_rows=[
            {"section_id": "intro", "phase": "collection", "deps": []},
            {"section_id": "components", "phase": "forest", "deps": ["intro", "missing"]},
            {"section_id": "violations", "phase": "post", "deps": []},
        ],
        sections={"intro": ["ready"]},
    )
    assert "Section `intro`" in report_text
    assert "- `phase`: `forest`" in report_text
    assert "- `work_done`: `3`" in report_text
    assert "- `work_total`: `4`" in report_text
    assert "- `work_percent`: `75.00`" in report_text
    assert pending["components"] == "missing_dep"
    assert pending["violations"] == "policy"


def test_write_bootstrap_incremental_artifacts_marks_existing_reason_policy(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.md"
    journal_path = tmp_path / "sections.json"
    phase_checkpoint_path = tmp_path / "phases.json"
    journal_path.write_text("{")
    phases: dict[str, object] = {}
    server._write_bootstrap_incremental_artifacts(
        report_output_path=report_path,
        report_section_journal_path=journal_path,
        report_phase_checkpoint_path=phase_checkpoint_path,
        witness_digest="wd",
        root=tmp_path,
        paths_requested=1,
        projection_rows=[
            {"section_id": "intro", "phase": "collection", "deps": []},
            {"section_id": "components", "phase": "forest", "deps": ["intro"]},
        ],
        phase_checkpoint_state=phases,
    )
    payload = json.loads(journal_path.read_text())
    components = payload["sections"]["components"]
    assert components["reason"] == "policy"
    assert report_path.exists()
    assert phase_checkpoint_path.exists()


def test_clear_analysis_resume_checkpoint_removes_checkpoint_and_chunks(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "resume.json"
    chunks_dir = server._analysis_resume_checkpoint_chunks_dir(checkpoint_path)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("{}")
    (chunks_dir / "one.json").write_text("{}")
    server._clear_analysis_resume_checkpoint(checkpoint_path)
    assert not checkpoint_path.exists()
    assert not chunks_dir.exists()
    # idempotent on missing paths
    server._clear_analysis_resume_checkpoint(checkpoint_path)


def test_write_analysis_resume_checkpoint_emits_analysis_index_hydration_summary(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "artifacts" / "audit_reports" / "resume.json"
    collection_resume = {
        "completed_paths": [],
        "in_progress_scan_by_path": {},
        "analysis_index_resume": {
            "format_version": 1,
            "phase": "analysis_index_hydration",
            "hydrated_paths": ["a.py"],
            "hydrated_paths_count": 1,
            "function_count": 2,
            "class_count": 1,
            "resume_digest": "abc123",
            "functions_by_qual": {},
            "symbol_table": {
                "imports": [],
                "internal_roots": [],
                "external_filter": True,
                "star_imports": {},
                "module_exports": {},
                "module_export_map": {},
            },
            "class_index": {},
        },
    }
    server._write_analysis_resume_checkpoint(
        path=checkpoint_path,
        input_witness={"witness_digest": "w1", "manifest_digest": "m1"},
        input_manifest_digest="m1",
        collection_resume=collection_resume,
    )
    payload = json.loads(checkpoint_path.read_text())
    summary = payload.get("analysis_index_hydration")
    assert isinstance(summary, dict)
    assert summary.get("hydrated_paths_count") == 1
    assert summary.get("function_count") == 2
    assert summary.get("class_count") == 1
    assert summary.get("resume_digest") == "abc123"


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
        include_wl_refinement=False,
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
        include_wl_refinement=False,
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
@pytest.mark.parametrize(
    ("timeout_field", "timeout_value"),
    [
        ("analysis_timeout_ms", 1000),
        ("analysis_timeout_seconds", "10"),
    ],
)
# dataflow-bundle: timeout_field, timeout_value
def test_execute_command_accepts_duration_timeout_fields(
    tmp_path: Path,
    timeout_field: str,
    timeout_value: object,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            timeout_field: timeout_value,
        },
    )
    assert result.get("exit_code") == 0


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_ignores_invalid_duration_timeout_fields(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    for timeout_field in ("analysis_timeout_ms", "analysis_timeout_seconds"):
        with pytest.raises(NeverThrown):
            server.execute_command(
                ls,
                {
                    "root": str(tmp_path),
                    "paths": [str(module_path)],
                    timeout_field: "nope",
                },
            )


def test_execute_structure_reuse_payload_non_dict() -> None:
    with pytest.raises(NeverThrown):
        server.execute_structure_reuse(None, [])  # type: ignore[arg-type]


def test_execute_decision_diff_payload_non_dict() -> None:
    with pytest.raises(NeverThrown):
        server.execute_decision_diff(None, [])  # type: ignore[arg-type]


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_structure_diff::payload
def test_execute_structure_diff_requires_timeout_payload() -> None:
    with pytest.raises(NeverThrown):
        server.execute_structure_diff(None, None)


def test_execute_structure_diff_rejects_non_dict_payload() -> None:
    with pytest.raises(NeverThrown):
        server.execute_structure_diff(None, [])  # type: ignore[arg-type]


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


def test_execute_command_rejects_non_dict_payload(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_command(ls, [])  # type: ignore[arg-type]


def test_execute_refactor_payload_non_dict(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_refactor(ls, [])  # type: ignore[arg-type]


def test_execute_synthesis_payload_non_dict(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(NeverThrown):
        server.execute_synthesis(ls, [])  # type: ignore[arg-type]


def test_write_text_profiled_writes_with_encoding(tmp_path: Path) -> None:
    output = tmp_path / "encoded.txt"
    server._write_text_profiled(
        output,
        "hello",
        io_name="test.write_profiled",
        encoding="utf-8",
    )
    assert output.read_text(encoding="utf-8") == "hello"


def test_analysis_input_manifest_marks_missing_files(tmp_path: Path) -> None:
    existing = tmp_path / "exists.py"
    existing.write_text("x = 1\n")
    missing = tmp_path / "missing.py"
    config = server.AuditConfig(project_root=tmp_path)
    manifest = server._analysis_input_manifest(
        root=tmp_path,
        file_paths=[existing, missing],
        recursive=True,
        include_invariant_propositions=False,
        include_wl_refinement=False,
        config=config,
    )
    files = manifest.get("files")
    assert isinstance(files, list)
    by_path = {entry["path"]: entry for entry in files if isinstance(entry, dict)}
    assert by_path[str(missing)]["missing"] is True
    assert isinstance(by_path[str(existing)]["size"], int)


def test_analysis_manifest_digest_from_witness_rejects_invalid_shapes() -> None:
    witness = {
        "root": "/repo",
        "recursive": True,
        "include_invariant_propositions": False,
        "include_wl_refinement": False,
        "files": [{"path": "a.py", "missing": True}],
        "config": {
            "exclude_dirs": [],
            "ignore_params": [],
            "strictness": "high",
            "external_filter": True,
            "transparent_decorators": [],
        },
    }
    digest = server._analysis_manifest_digest_from_witness(witness)
    assert isinstance(digest, str)
    assert server._analysis_manifest_digest_from_witness(
        {**witness, "files": [None]}  # type: ignore[list-item]
    ) is None
    assert server._analysis_manifest_digest_from_witness(
        {**witness, "config": None}  # type: ignore[arg-type]
    ) is None
    assert server._analysis_manifest_digest_from_witness(
        {**witness, "root": 1}  # type: ignore[arg-type]
    ) is None
    assert server._analysis_manifest_digest_from_witness(
        {**witness, "recursive": "yes"}  # type: ignore[arg-type]
    ) is None
    assert server._analysis_manifest_digest_from_witness(
        {**witness, "include_invariant_propositions": "no"}  # type: ignore[arg-type]
    ) is None
    assert server._analysis_manifest_digest_from_witness(
        {**witness, "include_wl_refinement": "no"}  # type: ignore[arg-type]
    ) is None


def test_analysis_timeout_grace_ns_rejects_invalid_numeric_shapes() -> None:
    with pytest.raises(NeverThrown):
        server._analysis_timeout_grace_ns(
            {"analysis_timeout_grace_ticks": "bad", "analysis_timeout_grace_tick_ns": 1},
            total_ns=10,
        )
    with pytest.raises(NeverThrown):
        server._analysis_timeout_grace_ns(
            {"analysis_timeout_grace_ticks": -1, "analysis_timeout_grace_tick_ns": 1},
            total_ns=10,
        )
    with pytest.raises(NeverThrown):
        server._analysis_timeout_grace_ns(
            {"analysis_timeout_grace_ms": "bad"},
            total_ns=10,
        )
    with pytest.raises(NeverThrown):
        server._analysis_timeout_grace_ns(
            {"analysis_timeout_grace_seconds": 0},
            total_ns=10,
        )


def test_externalize_collection_resume_states_handles_mixed_rows(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume.json"
    raw_state = {
        "phase": "function_scan",
        "processed_functions_digest": "abc",
        "fn_names": {"a": "x"},
        "padding": "x" * 70_000,
    }
    payload = {
        "in_progress_scan_by_path": {"a.py": raw_state, 1: {"phase": "x"}, "b.py": 1},
        "analysis_index_resume": {},
    }
    chunks_dir = checkpoint.with_name(f"{checkpoint.name}.chunks")
    stale_dir = chunks_dir / "stale.json"
    stale_dir.mkdir(parents=True)
    externalized = server._externalize_collection_resume_states(
        path=checkpoint,
        collection_resume=payload,
    )
    states = externalized.get("in_progress_scan_by_path")
    assert isinstance(states, dict)
    assert states["a.py"]["processed_functions_digest"] == "abc"
    assert states["a.py"]["function_count"] == 1
    assert states["b.py"] == 1
    assert stale_dir.exists()


def test_externalize_collection_resume_states_summarizes_processed_function_list(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "resume.json"
    payload = {
        "in_progress_scan_by_path": {
            "a.py": {
                "phase": "function_scan",
                "processed_functions": ["pkg.a", "pkg.a", "pkg.b", 1],
                "padding": "x" * 70_000,
            }
        }
    }
    externalized = server._externalize_collection_resume_states(
        path=checkpoint,
        collection_resume=payload,
    )
    states = externalized.get("in_progress_scan_by_path")
    assert isinstance(states, dict)
    summary = states["a.py"]
    assert summary["processed_functions_count"] == 2
    expected_digest = hashlib.sha1(
        server._canonical_json_text(["pkg.a", "pkg.b"]).encode("utf-8")
    ).hexdigest()
    assert summary["processed_functions_digest"] == expected_digest


def test_inflate_collection_resume_states_handles_chunk_failures(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume.json"
    chunks_dir = checkpoint.with_name(f"{checkpoint.name}.chunks")
    chunks_dir.mkdir(parents=True)
    bad_chunk = chunks_dir / "bad.json"
    bad_chunk.write_text("{")
    wrong_chunk = chunks_dir / "wrong.json"
    wrong_chunk.write_text(
        json.dumps(
            {
                "format_version": 1,
                "path": "other.py",
                "state": {"phase": "function_scan"},
            }
        )
    )
    payload = {
        "in_progress_scan_by_path": {
            "a.py": {"state_ref": "bad.json"},
            "b.py": {"state_ref": "wrong.json"},
            1: {"state_ref": "ignored.json"},
            "c.py": 1,
        },
        "analysis_index_resume": {"state_ref": "bad.json"},
    }
    inflated = server._inflate_collection_resume_states(
        path=checkpoint,
        collection_resume=payload,
    )
    states = inflated.get("in_progress_scan_by_path")
    assert isinstance(states, dict)
    assert states["a.py"]["state_ref"] == "bad.json"
    assert states["b.py"]["state_ref"] == "wrong.json"
    assert states["c.py"] == 1
    analysis_index_resume = inflated.get("analysis_index_resume")
    assert isinstance(analysis_index_resume, dict)
    assert analysis_index_resume["state_ref"] == "bad.json"


def test_timeout_cleanup_tracks_truncated_report_steps(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    report_path = tmp_path / "report.md"
    phase_checkpoint_path = tmp_path / "report_phase_checkpoint.json"

    def _raise_timeout(*_args: object, **_kwargs: object) -> None:
        raise server.TimeoutExceeded("timeout")

    result = server._execute_command_total(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(report_path),
                "resume_checkpoint": str(tmp_path / "resume.json"),
                "analysis_timeout_ms": 1,
                "analysis_timeout_grace_ms": 1,
            }
        ),
        deps=server._default_execute_command_deps().with_overrides(
            load_report_phase_checkpoint_fn=_raise_timeout
        ),
    )
    assert result["timeout"] is True
    progress = result["timeout_context"]["progress"]
    assert progress.get("cleanup_truncated") is True
    cleanup_steps = progress.get("cleanup_timeout_steps")
    assert isinstance(cleanup_steps, list)
    assert "render_timeout_report" in cleanup_steps
    assert report_path.exists()
    assert phase_checkpoint_path.exists()


def test_apply_journal_pending_reason_only_for_stale_or_policy() -> None:
    pending: dict[str, str] = {}
    rows: list[dict[str, object]] = [
        {"section_id": "intro"},
        {"section_id": "components"},
        {"section_id": ""},
    ]
    server._apply_journal_pending_reason(
        projection_rows=rows,
        sections={"intro": ["ok"]},
        pending_reasons=pending,
        journal_reason="stale_input",
    )
    assert pending == {"components": "stale_input"}
    server._apply_journal_pending_reason(
        projection_rows=rows,
        sections={"intro": ["ok"]},
        pending_reasons=pending,
        journal_reason="policy",
    )
    assert pending["components"] == "policy"
    server._apply_journal_pending_reason(
        projection_rows=rows,
        sections={},
        pending_reasons=pending,
        journal_reason="missing_dep",
    )
    assert pending["components"] == "policy"


def test_latest_report_phase_and_truthy_flag_edges() -> None:
    assert server._latest_report_phase(None) is None
    assert server._latest_report_phase({"post": {}, "forest": {}}) == "post"
    assert server._latest_report_phase({1: {}, "invalid": {}}) is None
    assert server._truthy_flag(0) is False
    assert server._truthy_flag(2) is True
    assert server._truthy_flag(0.0) is False
    assert server._truthy_flag(" on ") is True
    assert server._truthy_flag(" no ") is False


def test_report_section_journal_load_policy_and_stale_paths(tmp_path: Path) -> None:
    path = tmp_path / "sections.json"
    assert server._load_report_section_journal(path=None, witness_digest=None) == ({}, None)
    path.write_text("[]")
    assert server._load_report_section_journal(path=path, witness_digest=None) == ({}, "policy")
    path.write_text(
        json.dumps({"format_version": 0, "sections": {}}, sort_keys=True) + "\n"
    )
    assert server._load_report_section_journal(path=path, witness_digest=None) == ({}, "policy")
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "witness_digest": "old",
                "sections": {},
            },
            sort_keys=True,
        )
        + "\n"
    )
    assert server._load_report_section_journal(path=path, witness_digest="new") == (
        {},
        "stale_input",
    )
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "witness_digest": "same",
                "sections": {
                    "intro": {"lines": ["ok"]},
                    "drop": {"lines": [1]},
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    sections, reason = server._load_report_section_journal(path=path, witness_digest="same")
    assert reason is None
    assert sections == {"intro": ["ok"]}


def test_report_phase_checkpoint_load_and_write_filters_invalid_entries(tmp_path: Path) -> None:
    path = tmp_path / "phase.json"
    assert server._load_report_phase_checkpoint(path=None, witness_digest=None) == {}
    path.write_text("[]")
    assert server._load_report_phase_checkpoint(path=path, witness_digest=None) == {}
    path.write_text(
        json.dumps({"format_version": 0, "phases": {}}, sort_keys=True) + "\n"
    )
    assert server._load_report_phase_checkpoint(path=path, witness_digest=None) == {}
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "witness_digest": "old",
                "phases": {},
            },
            sort_keys=True,
        )
        + "\n"
    )
    assert server._load_report_phase_checkpoint(path=path, witness_digest="new") == {}
    path.write_text(
        json.dumps(
                {
                    "format_version": 1,
                    "witness_digest": "same",
                    "phases": {"collection": {"status": "ok"}},
                },
                sort_keys=True,
            )
        + "\n"
    )
    phases = server._load_report_phase_checkpoint(path=path, witness_digest="same")
    assert phases == {"collection": {"status": "ok"}}

    server._write_report_phase_checkpoint(
        path=path,
        witness_digest="w",
        phases={"collection": {"status": "ok"}, "bad": "skip", 1: {"status": "skip"}},  # type: ignore[dict-item]
    )
    payload = json.loads(path.read_text())
    assert payload["phases"] == {"collection": {"status": "ok"}}


def test_write_report_section_journal_handles_path_none_and_empty_section_id(tmp_path: Path) -> None:
    server._write_report_section_journal(
        path=None,
        witness_digest="w",
        projection_rows=[],
        sections={},
    )
    path = tmp_path / "sections.json"
    server._write_report_section_journal(
        path=path,
        witness_digest="w",
        projection_rows=[{"section_id": "", "phase": "collection", "deps": []}],
        sections={},
    )
    payload = json.loads(path.read_text())
    assert payload["sections"] == {}
    assert payload["projection_rows"] == []


def test_collection_component_and_group_projection_filters_invalid_shapes() -> None:
    assert server._collection_components_preview_lines(collection_resume={})[1] == "- `paths_with_groups`: `0`"
    resume = {
        "groups_by_path": {
            "a.py": {
                "f": [["x"], "skip"],
                1: [["y"]],
                "g": "skip",
            },
            1: {"f": [["x"]]},
            "b.py": "skip",
        }
    }
    preview = server._collection_components_preview_lines(collection_resume=resume)
    assert "- `paths_with_groups`: `1`" in preview
    assert "- `functions_with_groups`: `1`" in preview
    assert "- `bundle_alternatives`: `1`" in preview

    groups = server._groups_by_path_from_collection_resume(resume)
    assert list(groups) == [Path("a.py")]
    assert set(groups[Path("a.py")]) == {"f"}
    assert groups[Path("a.py")]["f"] == [{"x"}]


def test_collection_progress_intro_lines_counts_processed_and_hydrated() -> None:
    resume = {
        "completed_paths": ["a.py"],
        "in_progress_scan_by_path": {
            "b.py": {"phase": "scan_pending", "processed_functions": ["x"], "fn_names": {"f": []}},
            "c.py": {"phase": "scan_pending", "processed_functions_count": 1, "function_count": 2},
            "d.py": {"phase": "scan_pending", "processed_functions": ["z"], "fn_names": {"g": []}},
            "e.py": {"phase": "scan_pending", "processed_functions": ["q"], "fn_names": {"h": []}},
        },
        "analysis_index_resume": {"hydrated_paths": ["a.py", "b.py"], "function_count": 3, "class_count": 1},
    }
    lines = server._collection_progress_intro_lines(
        collection_resume=resume,
        total_files=5,
    )
    assert any("in_progress_detail" in line for line in lines)
    assert any("hydrated_paths_count" in line for line in lines)
    assert any("hydrated_function_count" in line for line in lines)
    assert any("hydrated_class_count" in line for line in lines)


def test_render_incremental_report_handles_missing_and_invalid_phases() -> None:
    report, pending = server._render_incremental_report(
        analysis_state="analysis_collection_in_progress",
        progress_payload={
            "classification": "timed_out_no_progress",
            "resume_supported": True,
            "retry_recommended": False,
        },
        projection_rows=[
            {"section_id": "", "phase": "collection", "deps": []},
            {"section_id": "intro", "phase": "collection", "deps": []},
            {"section_id": "weird", "phase": "unknown", "deps": ["intro"]},
            {"section_id": "blocked", "phase": "post", "deps": ["missing"]},
        ],
        sections={"intro": ["ready"]},
    )
    assert "## Section `intro`" in report
    assert "`retry_recommended`: `False`" in report
    assert pending["weird"] == "policy"
    assert pending["blocked"] == "missing_dep"


def test_externalize_collection_resume_states_passthrough_and_cleanup_oserror(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "resume.json"
    passthrough = server._externalize_collection_resume_states(
        path=checkpoint,
        collection_resume={"x": 1},
    )
    assert passthrough == {"x": 1}

    chunks_dir = checkpoint.with_name(f"{checkpoint.name}.chunks")
    (chunks_dir / "keep").mkdir(parents=True)
    payload = server._externalize_collection_resume_states(
        path=checkpoint,
        collection_resume={
            "in_progress_scan_by_path": [],
            "analysis_index_resume": {},
        },  # type: ignore[arg-type]
    )
    assert payload["in_progress_scan_by_path"] == {}
    assert chunks_dir.exists()


def test_inflate_collection_resume_states_passthrough_and_chunk_success(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "resume.json"
    passthrough = server._inflate_collection_resume_states(
        path=checkpoint,
        collection_resume={"x": 1},
    )
    assert passthrough == {"x": 1}

    chunks_dir = checkpoint.with_name(f"{checkpoint.name}.chunks")
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "state.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "path": "a.py",
                "state": {"phase": "function_scan", "processed_functions": ["f"]},
            },
            sort_keys=True,
        )
        + "\n"
    )
    (chunks_dir / "analysis.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "path": "analysis_index_resume",
                "state": 1,
            },
            sort_keys=True,
        )
        + "\n"
    )
    inflated = server._inflate_collection_resume_states(
        path=checkpoint,
        collection_resume={
            "in_progress_scan_by_path": {"a.py": {"state_ref": "state.json"}},
            "analysis_index_resume": {"state_ref": "analysis.json", "phase": "x"},
        },
    )
    states = inflated["in_progress_scan_by_path"]
    assert isinstance(states, dict)
    assert states["a.py"]["phase"] == "function_scan"
    analysis_resume = inflated["analysis_index_resume"]
    assert isinstance(analysis_resume, dict)
    assert analysis_resume["state_ref"] == "analysis.json"


def test_load_analysis_resume_checkpoint_manifest_invalid_shapes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume.json"
    assert (
        server._load_analysis_resume_checkpoint_manifest(
            path=checkpoint,
            manifest_digest="x",
        )
        is None
    )

    checkpoint.write_text("[]")
    assert (
        server._load_analysis_resume_checkpoint_manifest(
            path=checkpoint,
            manifest_digest="x",
        )
        is None
    )

    checkpoint.write_text(
        json.dumps({"format_version": 0, "input_manifest_digest": "x", "collection_resume": {}})
    )
    assert (
        server._load_analysis_resume_checkpoint_manifest(
            path=checkpoint,
            manifest_digest="x",
        )
        is None
    )

    checkpoint.write_text(
        json.dumps(
            {
                "format_version": 1,
                "input_manifest_digest": 1,
                "collection_resume": {},
            }
        )
    )
    assert (
        server._load_analysis_resume_checkpoint_manifest(
            path=checkpoint,
            manifest_digest="x",
        )
        is None
    )

    checkpoint.write_text(
        json.dumps(
            {
                "format_version": 1,
                "input_manifest_digest": "x",
                "collection_resume": "bad",
            }
        )
    )
    assert (
        server._load_analysis_resume_checkpoint_manifest(
            path=checkpoint,
            manifest_digest="x",
        )
        is None
    )

    checkpoint.write_text(
        json.dumps(
            {
                "format_version": 1,
                "input_manifest_digest": "x",
                "collection_resume": {},
            }
        )
    )
    loaded = server._load_analysis_resume_checkpoint_manifest(
        path=checkpoint,
        manifest_digest="x",
    )
    assert loaded == (None, {})


def test_resume_helpers_default_paths_and_digests() -> None:
    assert server._completed_path_set(None) == set()
    assert server._completed_path_set({"completed_paths": "bad"}) == set()
    assert server._in_progress_scan_states(None) == {}
    assert server._in_progress_scan_states({"in_progress_scan_by_path": []}) == {}
    assert server._state_processed_count({}) == 0
    assert server._state_processed_count({"processed_functions_count": 3}) == 3
    assert server._state_processed_digest({})
    assert server._state_processed_digest({"processed_functions_digest": "x"}) == "x"
    assert server._analysis_index_resume_hydrated_paths(None) == set()
    assert server._analysis_index_resume_hydrated_count(None) == 0
    assert server._analysis_index_resume_hydrated_digest(None) == hashlib.sha1(b"[]").hexdigest()
    summary = server._analysis_index_resume_summary(None)
    assert summary is None
    signature = server._analysis_index_resume_signature(None)
    assert signature[0] == 0


def test_misc_small_helpers_cover_validation_edges(tmp_path: Path) -> None:
    assert server._resolve_report_output_path(root=tmp_path, report_path="-") is None
    assert server._report_witness_digest(input_witness={"witness_digest": 1}, manifest_digest=1) is None
    assert server._coerce_section_lines("bad") == []
    assert server._groups_by_path_from_collection_resume({"groups_by_path": []}) == {}

    obligations = server._incremental_progress_obligations(
        analysis_state="timed_out_progress_resume",
        progress_payload={"classification": "timed_out_progress_resume", "resume_supported": True},
        resume_checkpoint_path=tmp_path / "missing.json",
        partial_report_written=False,
        report_requested=True,
        projection_rows=[{"section_id": "", "phase": "collection"}],
        sections={},
        pending_reasons={"intro": "stale_input"},
    )
    assert any(
        entry.get("kind") == "restart_required_on_witness_mismatch"
        and entry.get("detail") == "restart_required"
        for entry in obligations
    )
    obligations = server._incremental_progress_obligations(
        analysis_state="timed_out_progress_resume",
        progress_payload={"classification": "timed_out_progress_resume", "resume_supported": True},
        resume_checkpoint_path=tmp_path / "missing.json",
        partial_report_written=False,
        report_requested=True,
        projection_rows=[{"section_id": "intro", "phase": "collection"}],
        sections={},
        pending_reasons={"other": "stale_input", "intro": "policy"},
    )
    assert any(
        entry.get("contract") == "incremental_projection_contract"
        and entry.get("section_id") == "intro"
        and entry.get("detail") == "stale_input"
        for entry in obligations
    )

    resumability, incremental = server._split_incremental_obligations(
        [{"status": "SATISFIED"}, 1]  # type: ignore[list-item]
    )
    assert resumability == []
    assert incremental == []


def test_server_deadline_overhead_and_name_set_edges() -> None:
    assert server._server_deadline_overhead_ns(total_ns=0) == 0
    assert server._server_deadline_overhead_ns(total_ns=1, divisor=1) == 0
    with pytest.raises(NeverThrown):
        server._server_deadline_overhead_ns(total_ns=1, divisor=0)
    assert server._normalize_name_set(" a, b ") == {"a", "b"}
    assert server._normalize_name_set([" a ", "b,c"]) == {"a", "b", "c"}
    with pytest.raises(NeverThrown):
        server._normalize_name_set(["ok", 1])  # type: ignore[list-item]
    with pytest.raises(NeverThrown):
        server._normalize_name_set(1)  # type: ignore[arg-type]


def test_execute_structure_reuse_total_edges(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.json"
    snapshot.write_text("{}")
    result = server._execute_structure_reuse_total(
        _DummyServer(str(tmp_path)),
        _with_timeout({"snapshot": str(snapshot), "min_count": 0}),
    )
    assert result["exit_code"] == 2
    result = server._execute_structure_reuse_total(
        _DummyServer(str(tmp_path)),
        _with_timeout({"snapshot": str(snapshot), "min_count": 1, "lemma_stubs": "-"}),
    )
    assert "lemma_stubs" in result


def test_inflate_manifest_and_checkpoint_edge_paths(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume.json"
    inflated = server._inflate_collection_resume_states(
        path=checkpoint,
        collection_resume={"in_progress_scan_by_path": [], "analysis_index_resume": {}},  # type: ignore[arg-type]
    )
    assert inflated["in_progress_scan_by_path"] == {}

    checkpoint.write_text("{")
    assert (
        server._load_analysis_resume_checkpoint_manifest(
            path=checkpoint,
            manifest_digest="x",
        )
        is None
    )

    witness = {
        "root": "/repo",
        "recursive": True,
        "include_invariant_propositions": False,
        "include_wl_refinement": False,
        "files": [{"path": "a.py", "missing": True}],
        "config": {
            "exclude_dirs": [],
            "ignore_params": [],
            "strictness": "high",
            "external_filter": True,
            "transparent_decorators": [],
        },
    }
    manifest_digest = server._analysis_manifest_digest_from_witness(witness)
    assert isinstance(manifest_digest, str)
    checkpoint.write_text(
        json.dumps(
            {
                "format_version": 1,
                "input_witness": witness,
                "collection_resume": {},
            }
        )
    )
    loaded = server._load_analysis_resume_checkpoint_manifest(
        path=checkpoint,
        manifest_digest=manifest_digest,
    )
    assert loaded is not None
    with pytest.raises(NeverThrown):
        server._write_analysis_resume_checkpoint(
            path=checkpoint,
            input_witness={"x": 1},
            input_manifest_digest=None,
            collection_resume={},
        )


def test_clear_checkpoint_and_grace_tick_validation_edges(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume.json"
    chunks_dir = checkpoint.with_name(f"{checkpoint.name}.chunks")
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "bad.json").mkdir()
    (chunks_dir / "keep").mkdir()
    server._clear_analysis_resume_checkpoint(checkpoint)
    assert chunks_dir.exists()

    with pytest.raises(NeverThrown):
        server._analysis_timeout_grace_ns(
            {
                "analysis_timeout_grace_ticks": 1,
                "analysis_timeout_grace_tick_ns": 0,
            },
            total_ns=100,
        )


def test_collection_semantic_progress_and_journal_phase_edges(tmp_path: Path) -> None:
    progress = server._collection_semantic_progress(
        previous_collection_resume={},
        collection_resume={
            "in_progress_scan_by_path": {
                "a.py": {"processed_functions_count": 1},
            }
        },
        total_files=1,
        cumulative=None,
    )
    assert progress["new_processed_functions_count"] == 1

    journal_path = tmp_path / "sections.json"
    journal_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "sections": [],
            },
            sort_keys=True,
        )
        + "\n"
    )
    sections, reason = server._load_report_section_journal(path=journal_path, witness_digest=None)
    assert sections == {}
    assert reason == "policy"
    journal_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "sections": {"ok": {"lines": ["x"]}, "bad": []},
            },
            sort_keys=True,
        )
        + "\n"
    )
    sections, _ = server._load_report_section_journal(path=journal_path, witness_digest=None)
    assert sections == {"ok": ["x"]}

    phase_path = tmp_path / "phases.json"
    phase_path.write_text(
        json.dumps({"format_version": 1, "phases": []}, sort_keys=True) + "\n"
    )
    assert server._load_report_phase_checkpoint(path=phase_path, witness_digest=None) == {}
    server._write_report_phase_checkpoint(path=None, witness_digest=None, phases={})


def test_bootstrap_incremental_artifacts_existing_reason_policy(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    journal_path = tmp_path / "report_sections.json"
    phase_path = tmp_path / "report_phases.json"
    projection_rows = [{"section_id": "components", "phase": "forest", "deps": ["intro"]}]

    journal_path.write_text("[]")
    server._write_bootstrap_incremental_artifacts(
        report_output_path=report_path,
        report_section_journal_path=journal_path,
        report_phase_checkpoint_path=phase_path,
        witness_digest="w",
        root=tmp_path,
        paths_requested=1,
        projection_rows=projection_rows,
        phase_checkpoint_state={},
    )
    payload = json.loads(journal_path.read_text())
    assert payload["sections"]["components"]["reason"] == "policy"
    journal_path.write_text(json.dumps({"format_version": 0}, sort_keys=True) + "\n")
    server._write_bootstrap_incremental_artifacts(
        report_output_path=report_path,
        report_section_journal_path=journal_path,
        report_phase_checkpoint_path=phase_path,
        witness_digest="w",
        root=tmp_path,
        paths_requested=1,
        projection_rows=projection_rows,
        phase_checkpoint_state={},
    )
    payload = json.loads(journal_path.read_text())
    assert payload["sections"]["components"]["reason"] == "policy"


def test_bootstrap_incremental_artifacts_skips_non_string_deps(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    journal_path = tmp_path / "report_sections.json"
    phase_path = tmp_path / "report_phases.json"
    projection_rows = [
        {"section_id": "components", "phase": "forest", "deps": ["intro", 1, None]}
    ]
    server._write_bootstrap_incremental_artifacts(
        report_output_path=report_path,
        report_section_journal_path=journal_path,
        report_phase_checkpoint_path=phase_path,
        witness_digest="w",
        root=tmp_path,
        paths_requested=1,
        projection_rows=projection_rows,
        phase_checkpoint_state={},
    )
    payload = json.loads(journal_path.read_text())
    assert payload["sections"]["components"]["deps"] == ["intro"]


def test_execute_command_timeout_context_payload_fallback(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    class _BoomContext:
        def as_payload(self) -> dict[str, object]:
            raise server.TimeoutExceeded(self)  # type: ignore[arg-type]

    def _raise_timeout(*_args: object, **_kwargs: object) -> server.AnalysisResult:
        raise server.TimeoutExceeded(_BoomContext())  # type: ignore[arg-type]

    result = server._execute_command_total(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(tmp_path / "report.md"),
            }
        ),
        deps=server._default_execute_command_deps().with_overrides(
            analyze_paths_fn=_raise_timeout
        ),
    )
    assert result["timeout"] is True
    progress = result["timeout_context"]["progress"]
    assert progress["classification"] == "timed_out_no_progress"


def test_timeout_context_payload_helper_falls_back_without_payload_api() -> None:
    payload = server._timeout_context_payload(
        server.TimeoutExceeded("boom")  # type: ignore[arg-type]
    )
    assert payload["summary"] == "Analysis timed out."
    assert payload["progress"]["classification"] == "timed_out_no_progress"


def test_execute_command_timeout_context_payload_handles_missing_payload_api(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    def _raise_timeout(*_args: object, **_kwargs: object) -> server.AnalysisResult:
        raise server.TimeoutExceeded("boom")  # type: ignore[arg-type]

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(tmp_path / "report.md"),
            }
        ),
        analyze_paths_fn=_raise_timeout,
    )
    assert result["timeout"] is True
    timeout_context = result["timeout_context"]
    assert isinstance(timeout_context, dict)
    assert timeout_context["progress"]["classification"] == "timed_out_no_progress"


def test_execute_command_timeout_progress_payload_repaired_when_not_mapping(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    def _raise_timeout(*_args: object, **_kwargs: object) -> server.AnalysisResult:
        raise _timeout_exc(progress="bad")

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(tmp_path / "report.md"),
                "analysis_timeout_ms": 1,
                "analysis_timeout_grace_ms": 1,
            }
        ),
        analyze_paths_fn=_raise_timeout,
    )
    progress = result["timeout_context"]["progress"]
    assert isinstance(progress, dict)
    assert "timeout_budget" in progress


def test_execute_command_timeout_resume_payload_promotes_progress_with_witness(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    checkpoint_path = tmp_path / "resume.json"
    witness: dict[str, object] = {"witness_digest": "wd"}
    load_calls = {"count": 0}
    write_calls = {"count": 0}

    def _load_checkpoint(*_args: object, **_kwargs: object):
        load_calls["count"] += 1
        return {"completed_paths": [str(module_path)]}

    def _manifest_loader(*_args: object, **_kwargs: object):
        return witness, {"completed_paths": []}

    def _write_checkpoint(*_args: object, **_kwargs: object) -> None:
        write_calls["count"] += 1
        raise _timeout_exc()

    def _raise_timeout(*_args: object, **_kwargs: object) -> server.AnalysisResult:
        raise _timeout_exc(progress={"classification": "timed_out_no_progress"})

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(tmp_path / "report.md"),
                "resume_checkpoint": str(checkpoint_path),
                "analysis_timeout_grace_ms": 200,
            }
        ),
        load_analysis_resume_checkpoint_manifest_fn=_manifest_loader,
        write_analysis_resume_checkpoint_fn=_write_checkpoint,
        load_analysis_resume_checkpoint_fn=_load_checkpoint,
        analyze_paths_fn=_raise_timeout,
    )
    assert result["timeout"] is True
    progress = result["timeout_context"]["progress"]
    assert write_calls["count"] >= 1
    assert load_calls["count"] == 1
    assert progress["classification"] == "timed_out_progress_resume"
    resume_payload = progress["resume"]
    assert resume_payload["input_witness"] == witness
    assert resume_payload["resume_token"]["witness_digest"] == "wd"


def test_execute_command_resume_timeout_paths_cover_manifest_and_witness_fallbacks(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    checkpoint_path = tmp_path / "resume.json"
    report_path = tmp_path / "report.md"
    witness = {"witness_digest": "w"}
    write_calls = {"count": 0}

    def _manifest_loader(*_args: object, **_kwargs: object):
        return witness, {}

    def _write_checkpoint(*args: object, **kwargs: object) -> None:
        del args, kwargs
        write_calls["count"] += 1
        if write_calls["count"] >= 2:
            raise _timeout_exc()

    def _load_checkpoint(*_args: object, **_kwargs: object):
        return {"completed_paths": [str(module_path)]}

    def _raise_timeout(*_args: object, **_kwargs: object) -> server.AnalysisResult:
        raise _timeout_exc(progress={"classification": "timed_out_no_progress"})

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(report_path),
                "resume_checkpoint": str(checkpoint_path),
                "emit_timeout_progress_report": True,
                "analysis_timeout_ms": 10,
                "analysis_timeout_grace_ms": 200,
            }
        ),
        load_analysis_resume_checkpoint_manifest_fn=_manifest_loader,
        write_analysis_resume_checkpoint_fn=_write_checkpoint,
        load_analysis_resume_checkpoint_fn=_load_checkpoint,
        analyze_paths_fn=_raise_timeout,
    )
    assert result["timeout"] is True
    progress = result["timeout_context"]["progress"]
    assert progress.get("cleanup_truncated") is True
    cleanup_steps = progress.get("cleanup_timeout_steps")
    assert isinstance(cleanup_steps, list)
    assert cleanup_steps


def test_execute_command_timeout_manifest_fallback_branch_and_intro_fallback(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    checkpoint_path = tmp_path / "resume.json"
    report_path = tmp_path / "report.md"
    write_calls = {"count": 0}
    manifest_calls = {"count": 0}

    def _manifest_loader(*_args: object, **_kwargs: object):
        manifest_calls["count"] += 1
        if manifest_calls["count"] == 1:
            return None
        return None, {"completed_paths": [str(module_path)]}

    def _write_checkpoint(*args: object, **kwargs: object) -> None:
        del args, kwargs
        write_calls["count"] += 1
        if write_calls["count"] >= 2:
            raise _timeout_exc()

    def _raise_timeout(*_args: object, **_kwargs: object) -> server.AnalysisResult:
        raise _timeout_exc(progress={"classification": "timed_out_no_progress"})

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(report_path),
                "resume_checkpoint": str(checkpoint_path),
                "analysis_timeout_ms": 1,
                "analysis_timeout_grace_ms": 1,
            }
        ),
        load_analysis_resume_checkpoint_manifest_fn=_manifest_loader,
        write_analysis_resume_checkpoint_fn=_write_checkpoint,
        analyze_paths_fn=_raise_timeout,
    )
    assert result["timeout"] is True
    assert report_path.exists()
    assert "Collection bootstrap checkpoint" in report_path.read_text()


def test_execute_command_resolve_manifest_without_checkpoint_and_invalid_retry_payload(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    def _analyze(*_args: object, **kwargs: object) -> server.AnalysisResult:
        on_collection_progress = kwargs.get("on_collection_progress")
        if callable(on_collection_progress):
            on_collection_progress(
                {
                    "completed_paths": [],
                    "in_progress_scan_by_path": {
                        str(module_path): {
                            "processed_functions_count": 1,
                            "semantic_progress": {"current_witness_digest": 1},
                        }
                    },
                    "semantic_progress": {"current_witness_digest": 1},
                }
            )
            on_collection_progress(
                {
                    "completed_paths": [str(module_path)],
                    "in_progress_scan_by_path": {
                        str(module_path): {"processed_functions_count": 2}
                    },
                    "semantic_progress": {"current_witness_digest": 1},
                }
            )
        return _empty_analysis_result()

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(tmp_path / "report.md"),
                "resume_on_timeout": "bad",
            }
        ),
        analyze_paths_fn=_analyze,
    )
    assert result.get("analysis_state") == "succeeded"


def test_execute_command_bootstrap_seed_manifest_and_semantic_progress_edges(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    report_path = tmp_path / "report.md"
    manifest_digest_calls = {"count": 0}
    seed_calls = {"count": 0}

    def _manifest(**_kwargs: object) -> dict[str, object]:
        return {"format_version": 1, "root": str(tmp_path), "files": []}

    def _manifest_digest(_manifest_payload: object) -> str:
        manifest_digest_calls["count"] += 1
        return "digest"

    def _seed(*, in_progress_paths: list[Path]) -> dict[str, object]:
        seed_calls["count"] += 1
        return {
            "completed_paths": [],
            "in_progress_scan_by_path": {
                str(path): {"phase": "scan_pending"} for path in in_progress_paths
            },
            "semantic_progress": {"current_witness_digest": 1},
        }

    def _semantic_progress(**_kwargs: object) -> dict[str, object]:
        return {
            "current_witness_digest": 1,
            "new_processed_functions_count": 0,
            "total_processed_functions_count": 0,
            "substantive_progress": False,
        }

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(report_path),
            }
        ),
        analysis_input_manifest_fn=_manifest,
        analysis_input_manifest_digest_fn=_manifest_digest,
        build_analysis_collection_resume_seed_fn=_seed,
        collection_semantic_progress_fn=_semantic_progress,
        analyze_paths_fn=lambda *_args, **_kwargs: _empty_analysis_result(),
    )

    assert result.get("analysis_state") == "succeeded"
    assert manifest_digest_calls["count"] == 1
    assert seed_calls["count"] == 1


def test_execute_command_resume_checkpoint_seed_written_when_manifest_missing(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    manifest_digest_calls = {"count": 0}
    seed_calls = {"count": 0}

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "resume_checkpoint": False,
                "report": str(tmp_path / "report.md"),
            }
        ),
        analysis_input_manifest_fn=lambda **_kwargs: {
            "format_version": 1,
            "root": str(tmp_path),
            "files": [],
        },
        analysis_input_manifest_digest_fn=lambda _manifest: manifest_digest_calls.__setitem__(
            "count", manifest_digest_calls["count"] + 1
        )
        or "digest",
        build_analysis_collection_resume_seed_fn=lambda *, in_progress_paths: seed_calls.__setitem__(
            "count", seed_calls["count"] + 1
        )
        or {
            "completed_paths": [],
            "in_progress_scan_by_path": {
                str(path): {"phase": "scan_pending"} for path in in_progress_paths
            },
        },
        analyze_paths_fn=lambda *_args, **_kwargs: _empty_analysis_result(),
    )
    assert result.get("analysis_state") == "succeeded"
    assert manifest_digest_calls["count"] == 1
    assert seed_calls["count"] == 1


def test_execute_command_timeout_phase_preview_projection_edges(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    checkpoint_path = tmp_path / "resume.json"
    report_path = tmp_path / "report.md"

    def _manifest_loader(*_args: object, **_kwargs: object):
        return None, {
            "completed_paths": [str(module_path)],
            "groups_by_path": {str(module_path): {"caller": [["a"]]}},
            "param_spans_by_path": {str(module_path): {"caller": {"a": [1, 1, 1, 2]}}},
            "bundle_sites_by_path": {str(module_path): {"caller": []}},
            "in_progress_scan_by_path": {},
        }

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "resume_checkpoint": str(checkpoint_path),
                "report": str(report_path),
                "emit_timeout_progress_report": True,
                "analysis_timeout_ms": 2_000,
                "analysis_timeout_grace_ms": 2_000,
            }
        ),
        load_analysis_resume_checkpoint_manifest_fn=_manifest_loader,
        analyze_paths_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            _timeout_exc(progress={"classification": "timed_out_progress_resume"})
        ),
    )
    assert result["timeout"] is True
    assert report_path.exists()


def test_execute_command_timeout_phase_checkpoint_preview_and_classification_edges(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    report_path = tmp_path / "report.md"
    phase_checkpoint_loads = {"count": 0}
    preview_calls = {"count": 0}

    def _raise_timeout(*_args: object, **_kwargs: object) -> server.AnalysisResult:
        raise _timeout_exc(progress={"classification": "timed_out_progress_resume"})

    def _load_phase_checkpoint(**_kwargs: object) -> dict[str, object]:
        phase_checkpoint_loads["count"] += 1
        return {}

    original_project = server.project_report_sections

    def _project_sections(*args: object, **kwargs: object):
        if kwargs.get("preview_only"):
            preview_calls["count"] += 1
        return original_project(*args, **kwargs)

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(report_path),
                "emit_timeout_progress_report": True,
                "analysis_timeout_ms": 2_000,
                "analysis_timeout_grace_ms": 2_000,
            }
        ),
        analyze_paths_fn=_raise_timeout,
        load_report_phase_checkpoint_fn=_load_phase_checkpoint,
        project_report_sections_fn=_project_sections,
    )
    assert result["timeout"] is True
    assert result["analysis_state"] == "timed_out_no_progress"
    assert phase_checkpoint_loads["count"] >= 1
    assert preview_calls["count"] >= 1


def test_execute_command_total_timeout_context_payload_timeout_fallback(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    class _RaisingContext:
        def as_payload(self) -> dict[str, object]:
            raise server.TimeoutExceeded(
                TimeoutContext(
                    call_stack=pack_call_stack([{"path": str(module_path), "qual": "mod.f"}])
                )
            )

    result = server._execute_command_total(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(tmp_path / "report.md"),
                "analysis_timeout_ms": 2_000,
                "analysis_timeout_grace_ms": 2_000,
            }
        ),
        deps=server._default_execute_command_deps().with_overrides(
            analyze_paths_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                server.TimeoutExceeded(_RaisingContext())  # type: ignore[arg-type]
            )
        ),
    )
    assert result["timeout"] is True


def test_execute_command_total_timeout_uses_non_empty_classification(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    result = server._execute_command_total(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(tmp_path / "report.md"),
                "resume_checkpoint": False,
                "analysis_timeout_ms": 2_000,
                "analysis_timeout_grace_ms": 2_000,
            }
        ),
        deps=server._default_execute_command_deps().with_overrides(
            analyze_paths_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                _timeout_exc(progress={"classification": "timed_out_progress_resume"})
            )
        ),
    )
    assert result["analysis_state"] == "timed_out_progress_resume"


def test_execute_command_total_timeout_loads_phase_checkpoint_and_preview_projection(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    phase_checkpoint_loads = {"count": 0}
    preview_calls = {"count": 0}

    def _load_phase_checkpoint(**_kwargs: object) -> dict[str, object]:
        phase_checkpoint_loads["count"] += 1
        return {}

    original_project = server.project_report_sections

    def _project_sections(*args: object, **kwargs: object):
        if kwargs.get("preview_only"):
            preview_calls["count"] += 1
        return original_project(*args, **kwargs)

    result = server._execute_command_total(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(tmp_path / "report.md"),
                "emit_timeout_progress_report": True,
                "analysis_timeout_ms": 2_000,
                "analysis_timeout_grace_ms": 2_000,
            }
        ),
        deps=server._default_execute_command_deps().with_overrides(
            collection_checkpoint_flush_due_fn=lambda **_kwargs: False,
            load_report_phase_checkpoint_fn=_load_phase_checkpoint,
            project_report_sections_fn=_project_sections,
            analyze_paths_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                _timeout_exc(progress={"classification": "timed_out_no_progress"})
            ),
        ),
    )
    assert result["timeout"] is True
    assert phase_checkpoint_loads["count"] >= 1
    assert preview_calls["count"] >= 1


def test_execute_command_total_timeout_intro_from_resume_collection(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    report_path = tmp_path / "report.md"
    checkpoint_path = tmp_path / "resume.json"
    phase_checkpoint_loads = {"count": 0}

    def _load_phase_checkpoint(**_kwargs: object) -> dict[str, object]:
        phase_checkpoint_loads["count"] += 1
        if phase_checkpoint_loads["count"] == 1:
            return {}
        return {"loaded": "timeout"}

    resume_payload = server.build_analysis_collection_resume_seed(
        in_progress_paths=[module_path]
    )

    result = server._execute_command_total(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(report_path),
                "resume_checkpoint": str(checkpoint_path),
                "analysis_timeout_ms": 2_000,
                "analysis_timeout_grace_ms": 2_000,
            }
        ),
        deps=server._default_execute_command_deps().with_overrides(
            write_bootstrap_incremental_artifacts_fn=lambda **_kwargs: None,
            load_report_section_journal_fn=lambda **_kwargs: ({}, None),
            load_report_phase_checkpoint_fn=_load_phase_checkpoint,
            load_analysis_resume_checkpoint_manifest_fn=lambda **_kwargs: (
                None,
                resume_payload,
            ),
            analyze_paths_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                _timeout_exc(progress={"classification": "timed_out_no_progress"})
            ),
        ),
    )
    assert result["timeout"] is True
    assert phase_checkpoint_loads["count"] >= 1
    assert "Collection progress checkpoint (provisional)." in report_path.read_text(
        encoding="utf-8"
    )


def test_execute_command_total_timeout_intro_fallback_bootstrap(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    report_path = tmp_path / "report.md"

    result = server._execute_command_total(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(report_path),
                "resume_checkpoint": False,
                "analysis_timeout_ms": 2_000,
                "analysis_timeout_grace_ms": 2_000,
            }
        ),
        deps=server._default_execute_command_deps().with_overrides(
            write_bootstrap_incremental_artifacts_fn=lambda **_kwargs: None,
            load_report_section_journal_fn=lambda **_kwargs: ({}, None),
            load_report_phase_checkpoint_fn=lambda **_kwargs: {},
            analyze_paths_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                _timeout_exc(progress={"classification": "timed_out_no_progress"})
            ),
        ),
    )
    assert result["timeout"] is True
    report_text = report_path.read_text(encoding="utf-8")
    assert (
        "Collection bootstrap checkpoint (provisional)." in report_text
        or "Collection progress checkpoint (provisional)." in report_text
    )


def test_execute_command_timeout_cleanup_manifest_resume_and_projection_preview(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    checkpoint_path = tmp_path / "resume.json"
    report_path = tmp_path / "report.md"
    manifest_calls = {"count": 0}

    def _manifest_loader(*_args: object, **_kwargs: object):
        manifest_calls["count"] += 1
        if manifest_calls["count"] == 1:
            return None, {"completed_paths": []}
        return None, {"completed_paths": [str(module_path)]}

    def _write_checkpoint(*_args: object, **_kwargs: object) -> None:
        raise _timeout_exc()

    def _raise_timeout(*_args: object, **_kwargs: object) -> server.AnalysisResult:
        raise _timeout_exc(progress={"classification": "timed_out_no_progress"})

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(report_path),
                "resume_checkpoint": str(checkpoint_path),
                "analysis_timeout_ms": 2_000,
                "analysis_timeout_grace_ms": 2_000,
                "emit_timeout_progress_report": True,
            }
        ),
        load_analysis_resume_checkpoint_manifest_fn=_manifest_loader,
        write_analysis_resume_checkpoint_fn=_write_checkpoint,
        analyze_paths_fn=_raise_timeout,
    )

    assert result["timeout"] is True
    assert report_path.exists()
    report_text = report_path.read_text()
    assert (
        "Collection progress checkpoint (provisional)." in report_text
        or "Collection bootstrap checkpoint (provisional)." in report_text
    )
    assert (
        "Component summary (provisional)." in report_text
        or "## Section `components`" in report_text
    )


def test_execute_command_timeout_cleanup_load_resume_progress_timeout(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    checkpoint_path = tmp_path / "resume.json"

    manifest_calls = {"count": 0}

    def _manifest_loader(*_args: object, **_kwargs: object):
        manifest_calls["count"] += 1
        if manifest_calls["count"] == 1:
            return None, {}
        raise _timeout_exc()

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "resume_checkpoint": str(checkpoint_path),
                "report": str(tmp_path / "report.md"),
                "analysis_timeout_ms": 2_000,
                "analysis_timeout_grace_ms": 2_000,
            }
        ),
        load_analysis_resume_checkpoint_manifest_fn=_manifest_loader,
        write_analysis_resume_checkpoint_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            _timeout_exc()
        ),
        analyze_paths_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            _timeout_exc(progress={"classification": "timed_out_no_progress"})
        ),
    )

    assert result["timeout"] is True
    progress = result["timeout_context"]["progress"]
    steps = progress.get("cleanup_timeout_steps")
    assert isinstance(steps, list)
    assert "load_resume_progress" in steps


def test_execute_command_projection_phase_callback_no_rows(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    def _analyze(*_args: object, **kwargs: object) -> server.AnalysisResult:
        on_phase_progress = kwargs.get("on_phase_progress")
        if callable(on_phase_progress):
            on_phase_progress(
                "forest",
                {},
                server.ReportCarrier(forest=server.Forest()),
                0,
                1,
            )
        return _empty_analysis_result()

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": str(tmp_path / "report.md"),
                "emit_timeout_progress_report": True,
            }
        ),
        report_projection_spec_rows_fn=lambda: [],
        analyze_paths_fn=_analyze,
    )
    assert result.get("analysis_state") == "succeeded"


def test_externalize_collection_resume_state_summary_fallback_branches(
    tmp_path: Path,
) -> None:
    payload: dict[str, object] = {
        "in_progress_scan_by_path": {
            "a.py": {
                "phase": "function_scan",
                "processed_functions_digest": "",
                "padding": "x" * 70_000,
            }
        },
        "analysis_index_resume": {
            "phase": "analysis_index_hydration",
            "hydrated_paths": {"not": "a-sequence"},
            "function_count": "bad",
            "class_count": None,
            "padding": "x" * 70_000,
        },
    }
    externalized = server._externalize_collection_resume_states(
        path=tmp_path / "resume.json",
        collection_resume=payload,
    )
    in_progress = externalized["in_progress_scan_by_path"]["a.py"]
    assert isinstance(in_progress, dict)
    assert "processed_functions_digest" not in in_progress
    analysis_index = externalized["analysis_index_resume"]
    assert isinstance(analysis_index, dict)
    assert "hydrated_paths_count" not in analysis_index
    assert "function_count" not in analysis_index
    assert "class_count" not in analysis_index


def test_inflate_collection_resume_chunk_state_non_mapping_falls_back(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "resume.json"
    chunk_name = server._analysis_resume_state_chunk_name("a.py")
    chunk_path = server._analysis_resume_checkpoint_chunks_dir(checkpoint) / chunk_name
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_text(
        json.dumps(
            {
                "format_version": server._ANALYSIS_RESUME_STATE_REF_FORMAT_VERSION,
                "path": "a.py",
                "state": ["not", "a", "mapping"],
            }
        )
    )
    inflated = server._inflate_collection_resume_states(
        path=checkpoint,
        collection_resume={
            "in_progress_scan_by_path": {
                "a.py": {"phase": "scan_pending", "state_ref": chunk_name}
            }
        },
    )
    state = inflated["in_progress_scan_by_path"]["a.py"]
    assert isinstance(state, dict)
    assert state["state_ref"] == chunk_name


def test_inflate_collection_resume_analysis_index_invalid_chunk_falls_back(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "resume.json"
    chunk_name = server._analysis_resume_named_chunk_name("analysis_index_resume")
    chunk_path = server._analysis_resume_checkpoint_chunks_dir(checkpoint) / chunk_name
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_text(
        json.dumps(
            {
                "format_version": server._ANALYSIS_RESUME_STATE_REF_FORMAT_VERSION,
                "path": "wrong-path",
                "state": {"phase": "analysis_index_hydration"},
            }
        )
    )
    inflated = server._inflate_collection_resume_states(
        path=checkpoint,
        collection_resume={
            "in_progress_scan_by_path": {},
            "analysis_index_resume": {"state_ref": chunk_name, "phase": "fallback"},
        },
    )
    analysis_index = inflated["analysis_index_resume"]
    assert isinstance(analysis_index, dict)
    assert analysis_index["state_ref"] == chunk_name
    assert analysis_index["phase"] == "fallback"


def test_analysis_input_witness_reuses_ast_intern_keys_for_identical_trees(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    source = "def f(x):\n    return x\n"
    first.write_text(source)
    second.write_text(source)
    witness = server._analysis_input_witness(
        root=tmp_path,
        file_paths=[first, second],
        recursive=True,
        include_invariant_propositions=False,
        include_wl_refinement=False,
        config=server.AuditConfig(project_root=tmp_path),
    )
    files = witness["files"]
    assert isinstance(files, list)
    refs = [entry.get("ast_ref") for entry in files if isinstance(entry, dict)]
    assert len(set(refs)) == 1
    table = witness["ast_intern_table"]
    assert isinstance(table, dict)
    assert len(table) == 1


def test_load_analysis_resume_checkpoint_without_expected_digest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume.json"
    input_witness = {"root": str(tmp_path)}
    checkpoint.write_text(
        json.dumps(
            {
                "format_version": server._ANALYSIS_RESUME_CHECKPOINT_FORMAT_VERSION,
                "input_witness": input_witness,
                "collection_resume": {"in_progress_scan_by_path": {}},
            }
        )
    )
    loaded = server._load_analysis_resume_checkpoint(
        path=checkpoint,
        input_witness=input_witness,
    )
    assert isinstance(loaded, dict)


def test_analysis_resume_checkpoint_compatibility_manifest_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume.json"
    checkpoint.write_text(
        json.dumps(
            {
                "format_version": server._ANALYSIS_RESUME_CHECKPOINT_FORMAT_VERSION,
                "input_manifest_digest": "old-digest",
                "collection_resume": {"completed_paths": []},
            }
        )
    )
    status = server._analysis_resume_checkpoint_compatibility(
        path=checkpoint,
        manifest_digest="new-digest",
    )
    assert status == "checkpoint_manifest_mismatch"


def test_analysis_resume_checkpoint_compatibility_compatible(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume.json"
    checkpoint.write_text(
        json.dumps(
            {
                "format_version": server._ANALYSIS_RESUME_CHECKPOINT_FORMAT_VERSION,
                "input_manifest_digest": "same-digest",
                "collection_resume": {"completed_paths": []},
            }
        )
    )
    status = server._analysis_resume_checkpoint_compatibility(
        path=checkpoint,
        manifest_digest="same-digest",
    )
    assert status == "checkpoint_compatible"


def test_analysis_resume_progress_allows_negative_total_files() -> None:
    progress = server._analysis_resume_progress(
        collection_resume={
            "completed_paths": ["a.py"],
            "in_progress_scan_by_path": {"b.py": {"phase": "scan_pending"}},
        },
        total_files=-1,
    )
    assert progress == {
        "completed_files": 1,
        "in_progress_files": 1,
        "remaining_files": 0,
        "total_files": -1,
    }


def test_collection_semantic_progress_ignores_non_int_cumulative_values() -> None:
    progress = server._collection_semantic_progress(
        previous_collection_resume={"completed_paths": []},
        collection_resume={"completed_paths": ["a.py"]},
        total_files=1,
        cumulative={
            "cumulative_new_processed_functions": "bad",
            "cumulative_completed_files_delta": None,
            "cumulative_hydrated_paths_delta": 1.2,
            "cumulative_regressed_functions": {},
        },
    )
    assert progress["cumulative_new_processed_functions"] == 0
    assert progress["cumulative_completed_files_delta"] == 1
    assert progress["cumulative_hydrated_paths_delta"] == 0
    assert progress["cumulative_regressed_functions"] == 0


def test_write_report_section_journal_handles_non_list_deps(tmp_path: Path) -> None:
    journal_path = tmp_path / "sections.json"
    server._write_report_section_journal(
        path=journal_path,
        witness_digest="wd",
        projection_rows=[{"section_id": "intro", "phase": "collection", "deps": "bad"}],
        sections={},
    )
    payload = json.loads(journal_path.read_text())
    assert payload["sections"]["intro"]["deps"] == []
    assert payload["projection_rows"][0]["deps"] == []


def test_write_bootstrap_incremental_artifacts_without_journal_path(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.md"
    phase_path = tmp_path / "phases.json"
    phase_state: dict[str, object] = {}
    server._write_bootstrap_incremental_artifacts(
        report_output_path=report_path,
        report_section_journal_path=None,
        report_phase_checkpoint_path=phase_path,
        witness_digest="wd",
        root=tmp_path,
        paths_requested=1,
        projection_rows=[{"section_id": "components", "phase": "forest", "deps": "bad"}],
        phase_checkpoint_state=phase_state,
    )
    assert report_path.exists()
    assert phase_path.exists()
    assert phase_state.get("collection") is not None


def test_write_bootstrap_incremental_artifacts_existing_digest_variants(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.md"
    journal_path = tmp_path / "sections.json"
    phase_path = tmp_path / "phases.json"
    phase_state: dict[str, object] = {}

    journal_path.write_text(
        json.dumps(
            {
                "format_version": server._REPORT_SECTION_JOURNAL_FORMAT_VERSION,
                "witness_digest": "",
                "sections": {},
            }
        )
    )
    server._write_bootstrap_incremental_artifacts(
        report_output_path=report_path,
        report_section_journal_path=journal_path,
        report_phase_checkpoint_path=phase_path,
        witness_digest="wd",
        root=tmp_path,
        paths_requested=1,
        projection_rows=[{"section_id": "components", "phase": "forest", "deps": []}],
        phase_checkpoint_state=phase_state,
    )
    first_payload = json.loads(journal_path.read_text())
    assert first_payload["sections"]["components"]["reason"] == "policy"

    journal_path.write_text(
        json.dumps(
            {
                "format_version": server._REPORT_SECTION_JOURNAL_FORMAT_VERSION,
                "witness_digest": "wd",
                "sections": {},
            }
        )
    )
    server._write_bootstrap_incremental_artifacts(
        report_output_path=report_path,
        report_section_journal_path=journal_path,
        report_phase_checkpoint_path=phase_path,
        witness_digest="wd",
        root=tmp_path,
        paths_requested=1,
        projection_rows=[{"section_id": "components", "phase": "forest", "deps": []}],
        phase_checkpoint_state=phase_state,
    )
    second_payload = json.loads(journal_path.read_text())
    assert second_payload["sections"]["components"]["reason"] == "policy"


def test_render_incremental_report_handles_non_mapping_progress_and_non_list_deps() -> None:
    report_text, pending = server._render_incremental_report(
        analysis_state="timed_out_no_progress",
        progress_payload=None,
        projection_rows=[{"section_id": "components", "phase": "forest", "deps": "bad"}],
        sections={},
    )
    assert "classification" not in report_text
    assert pending["components"] == "policy"


def test_collection_progress_intro_lines_skip_non_numeric_optional_metrics() -> None:
    lines = server._collection_progress_intro_lines(
        collection_resume={
            "completed_paths": [],
            "in_progress_scan_by_path": {},
            "semantic_progress": {
                "new_processed_functions_count": "3",
                "substantive_progress": "true",
            },
            "analysis_index_resume": {
                "hydrated_paths_count": "bad",
                "function_count": "bad",
                "class_count": None,
            },
        },
        total_files=0,
    )
    assert all("new_processed_functions" not in line for line in lines)
    assert all("substantive_progress" not in line for line in lines)
    assert any("hydrated_paths_count" in line for line in lines)
    assert all("hydrated_function_count" not in line for line in lines)
    assert all("hydrated_class_count" not in line for line in lines)


def test_collection_progress_intro_lines_skips_non_string_scan_entries() -> None:
    lines = server._collection_progress_intro_lines(
        collection_resume={
            "completed_paths": [],
            "in_progress_scan_by_path": {
                "module.py": {
                    "phase": "scan_pending",
                    "processed_functions": ["fn", 1],
                    "fn_names": {"fn": [], 2: []},
                }
            },
        },
        total_files=1,
    )
    assert any("processed_functions=1" in line for line in lines)
    assert any("function_count=1" in line for line in lines)


def test_incremental_progress_obligations_ignore_non_boolean_semantic_flags() -> None:
    obligations = server._incremental_progress_obligations(
        analysis_state="timed_out_progress_resume",
        progress_payload={
            "classification": "timed_out_progress_resume",
            "resume_supported": True,
            "semantic_progress": {
                "monotonic_progress": "yes",
                "substantive_progress": "no",
            },
        },
        resume_checkpoint_path=None,
        partial_report_written=False,
        report_requested=False,
        projection_rows=[],
        sections={},
        pending_reasons={},
    )
    kinds = {
        entry.get("kind")
        for entry in obligations
        if isinstance(entry, dict) and isinstance(entry.get("kind"), str)
    }
    assert "progress_monotonicity" not in kinds
    assert "substantive_progress_required" not in kinds


def test_normalize_transparent_decorators_returns_none_for_invalid_payload() -> None:
    assert server._normalize_transparent_decorators(123) is None  # type: ignore[arg-type]


def test_timeout_context_payload_falls_back_for_non_mapping_payload() -> None:
    class _ContextProxy:
        def as_payload(self) -> list[str]:
            return ["bad"]

    payload = server._timeout_context_payload(server.TimeoutExceeded(_ContextProxy()))  # type: ignore[arg-type]
    assert payload["summary"] == "Analysis timed out."


def test_execute_command_writes_refactor_plan_json_file(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    refactor_plan_path = tmp_path / "refactor-plan.json"
    result = server.execute_command(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "refactor_plan_json": str(refactor_plan_path),
            }
        ),
    )
    assert result.get("exit_code") == 0
    assert refactor_plan_path.exists()


def test_execute_command_refactor_plan_without_json_path(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    result = server.execute_command(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "refactor_plan": True,
            }
        ),
    )
    assert result.get("exit_code") == 0
    assert "refactor_plan" in result


def test_execute_command_with_empty_fingerprint_index(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    config_path = tmp_path / "gabion.toml"
    config_path.write_text("[fingerprints]\nuser_context = 1\n")
    result = server.execute_command(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "config": str(config_path),
            }
        ),
    )
    assert result.get("exit_code") == 0


def test_execute_command_skips_markdown_write_when_report_output_is_dash(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    result = server.execute_command(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": "-",
            }
        ),
    )
    assert result.get("exit_code") == 0
    assert not (tmp_path / "report.md").exists()


def test_execute_command_skips_report_append_when_report_is_empty_string(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    result = server.execute_command(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "report": "",
            }
        ),
    )
    assert result.get("exit_code") == 0


def test_execute_command_timeout_cleanup_manifest_resume_none_branch(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    checkpoint_path = tmp_path / "resume.json"

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "resume_checkpoint": str(checkpoint_path),
                "report": "",
                "lint": True,
                "analysis_timeout_ms": 5_000,
            }
        ),
        load_analysis_resume_checkpoint_manifest_fn=lambda *_args, **_kwargs: None,
        write_analysis_resume_checkpoint_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            _timeout_exc()
        ),
        analyze_paths_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            _timeout_exc(progress={"classification": "timed_out_no_progress"})
        ),
    )
    assert result["timeout"] is True


def test_execute_command_timeout_cleanup_non_boolean_semantic_progress(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    checkpoint_path = tmp_path / "resume.json"

    manifest_resume = (
        {"witness_digest": "wd"},
        {
            "completed_paths": ["sample.py"],
            "in_progress_scan_by_path": {},
            "semantic_progress": {"substantive_progress": "unknown"},
        },
    )

    class _EmptyClassificationContext:
        def as_payload(self) -> dict[str, object]:
            return {"progress": {"classification": ""}}

    result = _execute_with_deps(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "resume_checkpoint": str(checkpoint_path),
                "report": "",
                "lint": True,
                "analysis_timeout_ms": 5_000,
            }
        ),
        load_analysis_resume_checkpoint_manifest_fn=lambda *_args, **_kwargs: manifest_resume,
        write_analysis_resume_checkpoint_fn=lambda *_args, **_kwargs: None,
        load_analysis_resume_checkpoint_fn=lambda *_args, **_kwargs: {
            "completed_paths": ["sample.py"],
            "in_progress_scan_by_path": {},
            "semantic_progress": {"substantive_progress": "unknown"},
        },
        analyze_paths_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            server.TimeoutExceeded(_EmptyClassificationContext())  # type: ignore[arg-type]
        ),
    )
    assert result["timeout"] is True


def test_execute_refactor_valid_payload_without_workspace_root(tmp_path: Path) -> None:
    module_path = tmp_path / "target.py"
    module_path.write_text("def f(a, b):\n    return a + b\n")
    result = server.execute_refactor(
        _DummyServer(""),
        _with_timeout(
            {
                "protocol_name": "ExampleProto",
                "bundle": ["a", "b"],
                "target_path": str(module_path),
                "target_functions": [],
            }
        ),
    )
    assert result.get("errors") == []


def test_execute_structure_reuse_total_success_without_lemma_stubs(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        "{\"format_version\": 1, \"root\": null, \"files\": []}"
    )
    result = server._execute_structure_reuse_total(
        _DummyServer(str(tmp_path)),
        _with_timeout({"snapshot": str(snapshot), "min_count": 1}),
    )
    assert result["exit_code"] == 0
    assert result["lemma_stubs"] is None


def test_execute_impact_query_groups_tests_and_docs(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (src / "app.py").write_text(
        "def target(value):\n"
        "    return value\n\n"
        "def helper(value):\n"
        "    return target(value)\n"
    )
    (tests_dir / "test_app.py").write_text(
        "from src.app import helper\n\n"
        "def test_helper():\n"
        "    assert helper(1) == 1\n"
    )
    (docs_dir / "impact.md").write_text(
        "# Impact\n"
        "The target function powers helper and tests.\n"
    )
    ls = _DummyServer(str(tmp_path))
    result = server.execute_impact(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "changes": ["src/app.py:1-2"],
                "confidence_threshold": 0.2,
            }
        ),
    )
    assert result.get("exit_code") == 0
    must = result.get("must_run_tests") or []
    assert any("tests/test_app.py::test_helper" in str(entry.get("id")) for entry in must)
    docs = result.get("impacted_docs") or []
    assert any(str(entry.get("path")) == "docs/impact.md" for entry in docs)


def test_execute_impact_query_accepts_git_diff(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("def f():\n    return 1\n")
    ls = _DummyServer(str(tmp_path))
    diff_text = """diff --git a/module.py b/module.py
+++ b/module.py
@@ -1,1 +1,2 @@
 def f():
+    return 2
"""
    result = server.execute_impact(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "git_diff": diff_text,
            }
        ),
    )
    assert result.get("exit_code") == 0
    assert result.get("changes")


def test_server_lint_normalization_helpers_cover_invalid_rows() -> None:
    assert server._parse_lint_line("not a lint row") is None
    assert server._parse_lint_line("pkg/mod.py:1:2:   ") is None
    entries = server._lint_entries_from_lines(
        [
            "pkg/mod.py:1:2: DF001 bad",
            "invalid row",
        ]
    )
    assert entries == [
        {
            "path": "pkg/mod.py",
            "line": 1,
            "col": 2,
            "code": "DF001",
            "message": "bad",
            "severity": "warning",
        }
    ]
    normalized = server._normalize_dataflow_response(
        {
            "exit_code": 0,
            "lint_lines": ["pkg/mod.py:1:2: DF001 bad"],
            "errors": "not-a-list",
        }
    )
    assert normalized["lint_entries"][0]["code"] == "DF001"


def test_execute_command_rejects_invalid_strictness(tmp_path: Path) -> None:
    module = tmp_path / "sample.py"
    _write_bundle_module(module)
    with pytest.raises(NeverThrown):
        server.execute_command(
            _DummyServer(str(tmp_path)),
            _with_timeout(
                {
                    "root": str(tmp_path),
                    "paths": [str(module)],
                    "strictness": "invalid",
                }
            ),
        )


def test_execute_refactor_accepts_structured_compatibility_shim(tmp_path: Path) -> None:
    module_path = tmp_path / "target.py"
    module_path.write_text("def f(a, b):\n    return a + b\n")
    result = server.execute_refactor(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "protocol_name": "ExampleProto",
                "bundle": ["a", "b"],
                "fields": [{"name": "a", "type_hint": "int"}, {"name": "b"}],
                "target_path": str(module_path),
                "target_functions": [],
                "compatibility_shim": {
                    "enabled": True,
                    "emit_deprecation_warning": False,
                    "emit_overload_stubs": True,
                },
            }
        ),
    )
    assert result.get("errors") == []


def test_impact_change_normalization_and_diff_range_edges() -> None:
    assert server._normalize_impact_change_entry({"path": ""}) is None
    assert server._normalize_impact_change_entry(
        {"path": "src/app.py", "start_line": "bad"}
    ) is None
    assert server._normalize_impact_change_entry(
        {"path": "src/app.py", "start_line": -1}
    ) is None

    swapped = server._normalize_impact_change_entry(
        {"path": "src/app.py", "start_line": 5, "end_line": 3}
    )
    assert swapped == server.ImpactSpan(path="src/app.py", start_line=3, end_line=5)
    assert server._normalize_impact_change_entry(
        {"path": "src/app.py", "start_line": 3, "end_line": 5}
    ) == server.ImpactSpan(path="src/app.py", start_line=3, end_line=5)

    assert server._normalize_impact_change_entry("") is None
    assert server._normalize_impact_change_entry("src/app.py\n1-2") is None
    assert server._normalize_impact_change_entry("src/app.py") == server.ImpactSpan(
        path="src/app.py",
        start_line=1,
        end_line=10**9,
    )
    assert server._normalize_impact_change_entry("src/app.py:9-3") == server.ImpactSpan(
        path="src/app.py",
        start_line=3,
        end_line=9,
    )

    diff_spans = server._parse_impact_diff_ranges(
        "+++ /dev/null\n"
        "@@ -1,0 +1,0 @@\n"
        "+++ b/src/app.py\n"
        "@@ -4,0 +4,0 @@\n"
    )
    assert diff_spans == [server.ImpactSpan(path="src/app.py", start_line=4, end_line=4)]
    assert server._impact_path_is_test("pkg/tests/unit_module.py")


def test_impact_function_and_edge_helpers_cover_guard_paths(tmp_path: Path) -> None:
    parsed_tree = server.ast.parse(
        "class Box:\n"
        "    def run(self):\n"
        "        return 1\n"
    )
    functions = server._impact_functions_from_tree("src/app.py", parsed_tree)
    assert [item.qual for item in functions] == ["Box.run"]

    empty_function = server.ast.FunctionDef(
        name="no_span",
        args=server.ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[server.ast.Pass()],
        decorator_list=[],
    )
    synthetic_tree = server.ast.Module(body=[empty_function], type_ignores=[])
    assert server._impact_functions_from_tree("src/synth.py", synthetic_tree) == []

    tree_a = server.ast.parse("def shared():\n    missing()\n")
    tree_b = server.ast.parse("def shared():\n    (lambda: None)()\n    unknown()\n")
    functions_by_qual = {
        "shared": server.ImpactFunction(
            path="b.py",
            qual="shared",
            name="shared",
            start_line=1,
            end_line=3,
            is_test=False,
        )
    }
    edges = server._impact_collect_edges(
        functions_by_qual=functions_by_qual,
        trees_by_path={"a.py": tree_a, "b.py": tree_b},
    )
    assert edges == []

    synthetic_call = server.ast.Call(
        func=server.ast.Name(id="shared", ctx=server.ast.Load()),
        args=[],
        keywords=[],
    )
    synthetic_expr = server.ast.Expr(value=synthetic_call)
    synthetic_fn = server.ast.FunctionDef(
        name="shared",
        args=server.ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[synthetic_expr],
        decorator_list=[],
    )
    synthetic_fn.lineno = 1
    synthetic_fn.end_lineno = 2
    synthetic_tree_with_call = server.ast.Module(body=[synthetic_fn], type_ignores=[])
    edges_without_line_info = server._impact_collect_edges(
        functions_by_qual={
            "shared": server.ImpactFunction(
                path="c.py",
                qual="shared",
                name="shared",
                start_line=1,
                end_line=2,
                is_test=False,
            )
        },
        trees_by_path={"c.py": synthetic_tree_with_call},
    )
    assert edges_without_line_info == []

    doc_path = tmp_path / "impact.md"
    doc_path.write_text("intro\n# One\nalpha\n# Two\nbeta\n", encoding="utf-8")
    sections = server._impact_parse_doc_sections(doc_path)
    assert sections == [("(preamble)", "intro"), ("One", "alpha"), ("Two", "beta")]

    heading_only = tmp_path / "heading_only.md"
    heading_only.write_text("# Heading\n", encoding="utf-8")
    assert server._impact_parse_doc_sections(heading_only) == []


def test_execute_impact_validation_and_depth_edges(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "broken.py").write_text("def bad(:\n    pass\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Heading\nNo symbol match here.\n", encoding="utf-8")
    ls = _DummyServer(str(tmp_path))

    bad_depth = server.execute_impact(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "changes": ["module.py:1-1"],
                "max_call_depth": "bad",
            }
        ),
    )
    assert bad_depth["exit_code"] == 2
    assert "max_call_depth must be an integer" in bad_depth["errors"][0]

    negative_depth = server.execute_impact(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "changes": ["module.py:1-1"],
                "max_call_depth": -1,
            }
        ),
    )
    assert negative_depth["exit_code"] == 2
    assert "max_call_depth must be non-negative" in negative_depth["errors"][0]

    bad_threshold = server.execute_impact(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "changes": ["module.py:1-1"],
                "confidence_threshold": "bad",
            }
        ),
    )
    assert bad_threshold["exit_code"] == 2
    assert "confidence_threshold must be numeric" in bad_threshold["errors"][0]

    missing_changes = server.execute_impact(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "changes": [{"path": "", "start_line": 1}],
            }
        ),
    )
    assert missing_changes["exit_code"] == 2
    assert "Provide at least one change span or git diff" in missing_changes["errors"][0]

    depth_limited = server.execute_impact(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "changes": ["module.py:1-1"],
                "max_call_depth": 0,
            }
        ),
    )
    assert depth_limited["exit_code"] == 0
    assert depth_limited["seed_functions"] == ["f"]
    assert depth_limited["must_run_tests"] == []
    assert depth_limited["likely_run_tests"] == []


def test_execute_impact_duplicate_test_edges_cover_seen_state_and_confidence_guard(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (src / "app.py").write_text(
        "def target(value):\n"
        "    return value\n\n"
        "def helper(value):\n"
        "    return target(value)\n",
        encoding="utf-8",
    )
    (tests_dir / "test_app.py").write_text(
        "from src.app import helper\n\n"
        "def test_helper_twice():\n"
        "    assert helper(1) == 1\n"
        "    assert helper(2) == 2\n",
        encoding="utf-8",
    )
    result = server.execute_impact(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "changes": ["src/app.py:1-2"],
            }
        ),
    )
    assert result.get("exit_code") == 0
    must = result.get("must_run_tests") or []
    assert len(must) == 1


def test_execute_impact_handles_change_without_seed_functions(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    result = server.execute_impact(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "changes": ["module.py:1-1"],
            }
        ),
    )
    assert result.get("exit_code") == 0
    assert result.get("seed_functions") == []
    assert result.get("must_run_tests") == []
    assert result.get("likely_run_tests") == []


def test_execute_impact_bfs_step_limit_handles_dense_reverse_edges(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text(
        "def seed(v):\n"
        "    return fan1(v) + fan2(v) + fan3(v) + fan4(v)\n"
        "def fan1(v):\n"
        "    return seed(v) + fan2(v) + fan3(v) + fan4(v)\n"
        "def fan2(v):\n"
        "    return seed(v) + fan1(v) + fan3(v) + fan4(v)\n"
        "def fan3(v):\n"
        "    return seed(v) + fan1(v) + fan2(v) + fan4(v)\n"
        "def fan4(v):\n"
        "    return seed(v) + fan1(v) + fan2(v) + fan3(v)\n",
        encoding="utf-8",
    )
    result = server.execute_impact(
        _DummyServer(str(tmp_path)),
        _with_timeout(
            {
                "root": str(tmp_path),
                "changes": ["module.py:1-2"],
            }
        ),
    )
    assert result.get("exit_code") == 0
    assert "seed" in (result.get("seed_functions") or [])


def test_normalize_progress_work_clamps_negative_and_overflow() -> None:
    assert server._normalize_progress_work(work_done=-3, work_total=-1) == (0, 0)
    assert server._normalize_progress_work(work_done=9, work_total=4) == (4, 4)
    assert server._normalize_progress_work(work_done=2, work_total=None) == (2, None)
    assert server._normalize_progress_work(work_done=None, work_total=3) == (None, 3)
