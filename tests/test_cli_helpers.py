from __future__ import annotations

from pathlib import Path
import io
import json
import os

import pytest
import typer

from gabion import cli


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._split_csv_entries::entries E:decision_surface/direct::cli.py::gabion.cli._split_csv::value
def test_split_csv_helpers() -> None:
    assert cli._split_csv_entries(["a, b", " ", "c"]) == ["a", "b", "c"]
    assert cli._split_csv_entries([" ", ""]) == []

    assert cli._split_csv("a, , b") == ["a", "b"]
    assert cli._split_csv(" ,") == []


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._write_lint_jsonl::target E:decision_surface/direct::cli.py::gabion.cli._write_lint_sarif::target
def test_lint_parsing_and_writers(tmp_path: Path, capsys) -> None:
    good_line = "mod.py:10:2: GABION_CODE something happened"
    parsed = cli._parse_lint_line(good_line)
    assert parsed is not None
    assert parsed["code"] == "GABION_CODE"
    assert cli._parse_lint_line("bad line") is None
    assert cli._parse_lint_line("mod.py:1:2:") is None

    entries = cli._collect_lint_entries([good_line, "bad"])
    assert len(entries) == 1

    cli._write_lint_jsonl("-", entries)
    out = capsys.readouterr().out
    assert "GABION_CODE" in out

    jsonl_path = tmp_path / "lint.jsonl"
    cli._write_lint_jsonl(str(jsonl_path), entries)
    assert jsonl_path.read_text().strip()

    sarif_path = tmp_path / "lint.sarif"
    cli._write_lint_sarif(str(sarif_path), entries)
    sarif_text = sarif_path.read_text()
    assert "sarif-2.1.0.json" in sarif_text
    cli._write_lint_sarif("-", entries)
    assert "sarif-2.1.0.json" in capsys.readouterr().out


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli._write_lint_jsonl::target E:decision_surface/direct::cli.py::gabion.cli._write_lint_sarif::target
def test_emit_lint_outputs_writes_artifacts(tmp_path: Path, capsys) -> None:
    lines = ["mod.py:1:1: GABION_CODE message"]
    jsonl_path = tmp_path / "lint.jsonl"
    sarif_path = tmp_path / "lint.sarif"
    cli._emit_lint_outputs(
        lines,
        lint=True,
        lint_jsonl=jsonl_path,
        lint_sarif=sarif_path,
    )
    out = capsys.readouterr().out
    assert "GABION_CODE" in out
    assert jsonl_path.exists()
    assert sarif_path.exists()


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_refactor_payload::bundle,input_payload,protocol_name,target_path
def test_build_refactor_payload_input_payload_passthrough() -> None:
    payload = {"protocol_name": "Bundle", "bundle": ["a"]}
    assert cli.build_refactor_payload(
        input_payload=payload,
        protocol_name=None,
        bundle=None,
        field=None,
        target_path=None,
        target_functions=None,
        compatibility_shim=False,
        rationale=None,
    ) == payload


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_refactor_payload::bundle,input_payload,protocol_name,target_path
def test_build_refactor_payload_requires_fields(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter):
        cli.build_refactor_payload(
            protocol_name=None,
            bundle=None,
            field=None,
            target_path=None,
            target_functions=None,
            compatibility_shim=False,
            rationale=None,
        )

    payload = cli.build_refactor_payload(
        protocol_name="Bundle",
        bundle=None,
        field=[" : ", "a:int", "b"],
        target_path=tmp_path / "target.py",
        target_functions=None,
        compatibility_shim=False,
        rationale=None,
    )
    assert payload["bundle"] == ["a", "b"]
    assert payload["fields"] == [
        {"name": "a", "type_hint": "int"},
        {"name": "b", "type_hint": None},
    ]


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._run_docflow_audit::fail_on_violations
def test_run_docflow_audit_missing_script(tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"
    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=False,
        script=missing,
    )
    assert exit_code == 2


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._run_docflow_audit::fail_on_violations
def test_run_docflow_audit_passes_flags(tmp_path: Path) -> None:
    script = tmp_path / "docflow.py"
    script.write_text("import sys\nsys.exit(0)\n")
    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=True,
        script=script,
    )
    assert exit_code == 0


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/value_encoded::cli.py::gabion.cli._dataflow_audit::request
def test_dataflow_audit_skips_type_audit_output() -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {"exit_code": 0}

    request = cli.DataflowAuditRequest(ctx=DummyCtx(), args=["sample.py"], runner=runner)
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/value_encoded::cli.py::gabion.cli._dataflow_audit::request
def test_dataflow_audit_type_audit_empty_findings() -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {"exit_code": 0, "type_suggestions": [], "type_ambiguities": []}

    request = cli.DataflowAuditRequest(
        ctx=DummyCtx(),
        args=["sample.py", "--type-audit", "--type-audit-max", "1"],
        runner=runner,
    )
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/value_encoded::cli.py::gabion.cli._dataflow_audit::request
def test_dataflow_audit_emits_lint_outputs(tmp_path: Path, capsys) -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {
            "exit_code": 0,
            "lint_lines": ["mod.py:1:1: GABION_CODE message"],
        }

    jsonl_path = tmp_path / "lint.jsonl"
    sarif_path = tmp_path / "lint.sarif"
    request = cli.DataflowAuditRequest(
        ctx=DummyCtx(),
        args=[
            "sample.py",
            "--lint",
            "--lint-jsonl",
            str(jsonl_path),
            "--lint-sarif",
            str(sarif_path),
        ],
        runner=runner,
    )
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0
    out = capsys.readouterr().out
    assert "GABION_CODE" in out
    assert jsonl_path.exists()
    assert sarif_path.exists()


def test_dataflow_audit_timeout_writes_deadline_profile(tmp_path: Path) -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {
            "exit_code": 2,
            "timeout": True,
            "timeout_context": {
                "deadline_profile": {
                    "checks_total": 3,
                    "total_elapsed_ns": 1000,
                    "unattributed_elapsed_ns": 10,
                    "sites": [],
                    "edges": [],
                }
            },
        }

    request = cli.DataflowAuditRequest(
        ctx=DummyCtx(),
        args=["sample.py", "--root", str(tmp_path)],
        runner=runner,
    )
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 2
    profile_json = tmp_path / "artifacts" / "out" / "deadline_profile.json"
    profile_md = tmp_path / "artifacts" / "out" / "deadline_profile.md"
    assert profile_json.exists()
    assert profile_md.exists()
    payload = json.loads(profile_json.read_text())
    assert payload["checks_total"] == 3


def test_dataflow_audit_timeout_progress_report_and_resume_retry(tmp_path: Path) -> None:
    class DummyCtx:
        args: list[str] = []

    calls = {"count": 0}

    def runner(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "exit_code": 2,
                "timeout": True,
                "analysis_state": "timed_out_progress_resume",
                "timeout_context": {
                    "deadline_profile": {
                        "checks_total": 5,
                        "total_elapsed_ns": 1000,
                        "unattributed_elapsed_ns": 10,
                        "sites": [],
                        "edges": [],
                    },
                    "progress": {
                        "classification": "timed_out_progress_resume",
                        "retry_recommended": True,
                        "resume_supported": True,
                        "resume": {
                            "resume_token": {
                                "phase": "analysis_collection",
                                "checkpoint_path": "resume.json",
                                "completed_files": 1,
                                "remaining_files": 0,
                                "total_files": 1,
                                "witness_digest": "abc",
                            }
                        },
                    },
                },
            }
        return {"exit_code": 0}

    request = cli.DataflowAuditRequest(
        ctx=DummyCtx(),
        args=[
            "sample.py",
            "--root",
            str(tmp_path),
            "--emit-timeout-progress-report",
            "--resume-on-timeout",
            "1",
        ],
        runner=runner,
    )
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0
    assert calls["count"] == 2
    progress_json = tmp_path / "artifacts" / "audit_reports" / "timeout_progress.json"
    progress_md = tmp_path / "artifacts" / "audit_reports" / "timeout_progress.md"
    assert progress_json.exists()
    assert progress_md.exists()
    payload = json.loads(progress_json.read_text())
    assert payload["analysis_state"] == "timed_out_progress_resume"


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/value_encoded::cli.py::gabion.cli._dataflow_audit::request
def test_dataflow_audit_emits_structure_tree(capsys) -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {
            "exit_code": 0,
            "structure_tree": {"format_version": 1, "root": ".", "files": []},
        }

    request = cli.DataflowAuditRequest(
        ctx=DummyCtx(),
        args=["sample.py", "--emit-structure-tree", "-"],
        runner=runner,
    )
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0
    captured = capsys.readouterr()
    assert "\"format_version\": 1" in captured.out


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/value_encoded::cli.py::gabion.cli._dataflow_audit::request
def test_dataflow_audit_emits_structure_metrics(capsys) -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {
            "exit_code": 0,
            "structure_metrics": {
                "files": 0,
                "functions": 0,
                "bundles": 0,
                "mean_bundle_size": 0.0,
                "max_bundle_size": 0,
                "bundle_size_histogram": {},
            },
        }

    request = cli.DataflowAuditRequest(
        ctx=DummyCtx(),
        args=["sample.py", "--emit-structure-metrics", "-"],
        runner=runner,
    )
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0
    captured = capsys.readouterr()
    assert "\"bundle_size_histogram\"" in captured.out


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/value_encoded::cli.py::gabion.cli._dataflow_audit::request
def test_dataflow_audit_emits_decision_snapshot(capsys) -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {
            "exit_code": 0,
            "decision_snapshot": {
                "format_version": 1,
                "root": ".",
                "decision_surfaces": [],
                "value_decision_surfaces": [],
                "summary": {},
            },
        }

    request = cli.DataflowAuditRequest(
        ctx=DummyCtx(),
        args=["sample.py", "--emit-decision-snapshot", "-"],
        runner=runner,
    )
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0
    captured = capsys.readouterr()
    assert "\"decision_surfaces\"" in captured.out


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/value_encoded::cli.py::gabion.cli._dataflow_audit::request
def test_dataflow_audit_emits_fingerprint_outputs(capsys) -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args: object, **_kwargs: object) -> dict[str, object]:
        # dataflow-bundle: _args, _kwargs
        return {
            "exit_code": 0,
            "fingerprint_synth_registry": {"version": "synth@1", "entries": []},
            "fingerprint_provenance": [{"path": "x.py", "bundle": ["a"]}],
            "fingerprint_deadness": [{"path": "x.py", "bundle": ["a"], "result": "UNREACHABLE"}],
            "fingerprint_coherence": [
                {
                    "site": {"path": "x.py", "function": "f", "bundle": ["a"]},
                    "result": "UNKNOWN",
                }
            ],
            "fingerprint_rewrite_plans": [
                {
                    "plan_id": "rewrite:x.py:f:a",
                    "site": {"path": "x.py", "function": "f", "bundle": ["a"]},
                    "status": "UNVERIFIED",
                }
            ],
            "fingerprint_exception_obligations": [
                {
                    "exception_path_id": "x.py:f:E0:1:0:raise",
                    "site": {"path": "x.py", "function": "f", "bundle": ["a"]},
                    "status": "UNKNOWN",
                }
            ],
            "fingerprint_handledness": [
                {
                    "handledness_id": "handled:x.py:f:E0:1:0:raise",
                    "exception_path_id": "x.py:f:E0:1:0:raise",
                }
            ],
        }

    request = cli.DataflowAuditRequest(
        ctx=DummyCtx(),
        args=[
            "sample.py",
            "--fingerprint-synth-json",
            "-",
            "--fingerprint-provenance-json",
            "-",
            "--fingerprint-deadness-json",
            "-",
            "--fingerprint-coherence-json",
            "-",
            "--fingerprint-rewrite-plans-json",
            "-",
            "--fingerprint-exception-obligations-json",
            "-",
            "--fingerprint-handledness-json",
            "-",
        ],
        runner=runner,
    )
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0
    captured = capsys.readouterr()
    assert "\"bundle\"" in captured.out
    assert "\"version\"" in captured.out
    assert "\"UNREACHABLE\"" in captured.out
    assert "\"fingerprint_coherence\"" not in captured.out
    assert "\"UNKNOWN\"" in captured.out
    assert "\"plan_id\"" in captured.out
    assert "\"exception_path_id\"" in captured.out
    assert "\"handledness_id\"" in captured.out


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._run_synth::config,exclude,ignore_params_csv,no_timestamp,paths,refactor_plan,strictness,synthesis_protocols_kind,transparent_decorators_csv
def test_run_synth_parses_optional_inputs(tmp_path: Path) -> None:
    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {"exit_code": 0}

    result, paths_out, timestamp = cli._run_synth(
        paths=None,
        root=tmp_path,
        out_dir=tmp_path / "out",
        no_timestamp=True,
        config=None,
        exclude=["a, b"],
        ignore_params_csv="x, y",
        transparent_decorators_csv="deco",
        allow_external=None,
        strictness=None,
        no_recursive=False,
        max_components=3,
        type_audit_report=True,
        type_audit_max=5,
        synthesis_max_tier=2,
        synthesis_min_bundle_size=1,
        synthesis_allow_singletons=False,
        synthesis_protocols_kind="dataclass",
        refactor_plan=False,
        fail_on_violations=False,
        runner=runner,
    )
    assert result["exit_code"] == 0
    assert timestamp is None
    assert paths_out["output_root"].exists()


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_synth_outputs::paths_out,refactor_plan,timestamp
def test_emit_synth_outputs_lists_optional_paths(tmp_path: Path, capsys) -> None:
    root = tmp_path / "out"
    root.mkdir()
    paths_out = {
        "output_root": root,
        "report": root / "dataflow_report.md",
        "dot": root / "graph.dot",
        "plan": root / "plan.json",
        "protocol": root / "protocols.py",
        "refactor": root / "refactor.json",
        "fingerprint_synth": root / "fingerprint_synth.json",
        "fingerprint_provenance": root / "fingerprint_provenance.json",
        "fingerprint_coherence": root / "fingerprint_coherence.json",
        "fingerprint_rewrite_plans": root / "fingerprint_rewrite_plans.json",
        "fingerprint_exception_obligations": root / "fingerprint_exception_obligations.json",
        "fingerprint_handledness": root / "fingerprint_handledness.json",
    }
    paths_out["fingerprint_synth"].write_text("{}")
    paths_out["fingerprint_provenance"].write_text("{}")
    paths_out["fingerprint_coherence"].write_text("{}")
    paths_out["fingerprint_rewrite_plans"].write_text("{}")
    paths_out["fingerprint_exception_obligations"].write_text("{}")
    paths_out["fingerprint_handledness"].write_text("{}")
    cli._emit_synth_outputs(
        paths_out=paths_out,
        timestamp=None,
        refactor_plan=False,
    )
    output = capsys.readouterr().out
    assert "fingerprint_synth.json" in output
    assert "fingerprint_provenance.json" in output
    assert "fingerprint_coherence.json" in output
    assert "fingerprint_rewrite_plans.json" in output
    assert "fingerprint_exception_obligations.json" in output
    assert "fingerprint_handledness.json" in output


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._run_synthesis_plan::input_path,output_path
def test_run_synthesis_plan_without_input(tmp_path: Path) -> None:
    captured = {}

    def runner(request, *, root=None):
        captured["request"] = request
        captured["root"] = root
        return {"protocols": []}

    output_path = tmp_path / "plan.json"
    cli._run_synthesis_plan(
        input_path=None,
        output_path=output_path,
        runner=runner,
    )
    assert captured["root"] is None
    assert output_path.read_text().strip()


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._run_synthesis_plan::input_path,output_path
def test_run_synthesis_plan_rejects_non_object_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("[]\n")
    with pytest.raises(typer.BadParameter) as exc:
        cli._run_synthesis_plan(
            input_path=payload_path,
            output_path=None,
            runner=lambda *_args, **_kwargs: {"protocols": []},
        )
    assert "json object" in str(exc.value).lower()


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._run_refactor_protocol::input_path,output_path
def test_refactor_protocol_rejects_non_object_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("[]\n")
    with pytest.raises(typer.BadParameter) as exc:
        cli.refactor_protocol(input_path=payload_path)
    assert "json object" in str(exc.value).lower()


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_refactor_payload::bundle,input_payload,protocol_name,target_path E:decision_surface/direct::cli.py::gabion.cli._run_refactor_protocol::input_path,output_path
def test_run_refactor_protocol_accepts_object_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("{\"protocol_name\": \"Bundle\", \"bundle\": [\"a\"]}\n")

    captured: dict[str, object] = {}

    def runner(request, *, root=None):
        captured["command"] = request.command
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"ok": True}

    output_path = tmp_path / "out.json"
    cli._run_refactor_protocol(
        input_path=payload_path,
        output_path=output_path,
        protocol_name=None,
        bundle=None,
        field=None,
        target_path=None,
        target_functions=None,
        compatibility_shim=False,
        rationale=None,
        runner=runner,
    )
    assert captured["command"] == cli.REFACTOR_COMMAND
    assert captured["root"] is None
    assert output_path.read_text().strip()


# gabion:evidence E:function_site::cli.py::gabion.cli.run_structure_diff
def test_run_structure_diff_uses_runner(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(request, *, root=None):
        captured["command"] = request.command
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"added_bundles": []}

    baseline = tmp_path / "base.json"
    current = tmp_path / "current.json"
    result = cli.run_structure_diff(
        baseline=baseline,
        current=current,
        root=tmp_path,
        runner=runner,
    )
    assert captured["command"] == cli.STRUCTURE_DIFF_COMMAND
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["baseline"] == str(baseline)
    assert payload["current"] == str(current)
    assert isinstance(payload.get("analysis_timeout_ticks"), int)
    assert payload["analysis_timeout_ticks"] > 0
    assert isinstance(payload.get("analysis_timeout_tick_ns"), int)
    assert payload["analysis_timeout_tick_ns"] > 0
    assert captured["root"] == tmp_path
    assert result == {"added_bundles": []}


# gabion:evidence E:function_site::cli.py::gabion.cli.run_decision_diff
def test_run_decision_diff_uses_runner(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(request, *, root=None):
        captured["command"] = request.command
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"exit_code": 0}

    baseline = tmp_path / "base.json"
    current = tmp_path / "current.json"
    result = cli.run_decision_diff(
        baseline=baseline,
        current=current,
        root=tmp_path,
        runner=runner,
    )
    assert captured["command"] == cli.DECISION_DIFF_COMMAND
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["baseline"] == str(baseline)
    assert payload["current"] == str(current)
    assert isinstance(payload.get("analysis_timeout_ticks"), int)
    assert payload["analysis_timeout_ticks"] > 0
    assert isinstance(payload.get("analysis_timeout_tick_ns"), int)
    assert payload["analysis_timeout_tick_ns"] > 0
    assert result == {"exit_code": 0}


def _rpc_message(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    return header + body


class _FakeProc:
    def __init__(self, stdout_bytes: bytes) -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout_bytes)
        self.stderr = io.BytesIO()
        self.returncode = 0

    def communicate(self, timeout: float | None = None):
        return (b"", b"")


def _extract_rpc_messages(buffer: bytes) -> list[dict]:
    messages: list[dict] = []
    offset = 0
    while True:
        header_end = buffer.find(b"\r\n\r\n", offset)
        if header_end < 0:
            break
        header = buffer[offset:header_end].decode("utf-8")
        length = None
        for line in header.split("\r\n"):
            if line.lower().startswith("content-length:"):
                length = int(line.split(":", 1)[1].strip())
                break
        if length is None:
            break
        body_start = header_end + 4
        body_end = body_start + length
        if body_end > len(buffer):
            break
        payload = json.loads(buffer[body_start:body_end].decode("utf-8"))
        messages.append(payload)
        offset = body_end
    return messages


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.dispatch_command
def test_dispatch_command_passes_timeout_ticks(tmp_path: Path) -> None:
    proc_holder: dict[str, _FakeProc] = {}

    def factory(*_args, **_kwargs):
        init = _rpc_message({"jsonrpc": "2.0", "id": 1, "result": {}})
        cmd = _rpc_message({"jsonrpc": "2.0", "id": 2, "result": {"ok": True}})
        shutdown = _rpc_message({"jsonrpc": "2.0", "id": 3, "result": {}})
        proc = _FakeProc(init + cmd + shutdown)
        proc_holder["proc"] = proc
        return proc

    previous = os.environ.get("GABION_DIRECT_RUN")
    os.environ.pop("GABION_DIRECT_RUN", None)
    try:
        result = cli.dispatch_command(
            command=cli.STRUCTURE_DIFF_COMMAND,
            payload={"baseline": str(tmp_path / "base.json"), "current": str(tmp_path / "current.json")},
            root=tmp_path,
            runner=cli.run_command,
            process_factory=factory,
        )
    finally:
        if previous is None:
            os.environ.pop("GABION_DIRECT_RUN", None)
        else:
            os.environ["GABION_DIRECT_RUN"] = previous
    assert result == {"ok": True}
    proc = proc_holder["proc"]
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert payload["analysis_timeout_ticks"] > 0
    assert payload["analysis_timeout_tick_ns"] > 0


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.run_structure_reuse::lemma_stubs
def test_run_structure_reuse_uses_runner(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(request, *, root=None):
        captured["command"] = request.command
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"exit_code": 0}

    snapshot = tmp_path / "snapshot.json"
    lemma = tmp_path / "lemmas.py"
    result = cli.run_structure_reuse(
        snapshot=snapshot,
        min_count=3,
        lemma_stubs=lemma,
        root=tmp_path,
        runner=runner,
    )
    assert captured["command"] == cli.STRUCTURE_REUSE_COMMAND
    assert captured["payload"]["snapshot"] == str(snapshot)
    assert captured["payload"]["min_count"] == 3
    assert captured["payload"]["lemma_stubs"] == str(lemma)
    assert result == {"exit_code": 0}


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.run_structure_reuse::lemma_stubs
def test_cli_diff_and_reuse_commands_use_default_runner(capsys) -> None:
    calls: list[str] = []

    def runner(request, root=None):
        calls.append(request.command)
        if request.command == cli.STRUCTURE_REUSE_COMMAND:
            return {"exit_code": 0, "reuse": {"format_version": 1}}
        return {"exit_code": 0, "diff": {"format_version": 1}}

    saved = cli.DEFAULT_RUNNER
    cli.DEFAULT_RUNNER = runner
    try:
        cli.structure_diff(
            baseline=Path("baseline.json"),
            current=Path("current.json"),
            root=None,
        )
        cli.decision_diff(
            baseline=Path("baseline.json"),
            current=Path("current.json"),
            root=None,
        )
        cli.structure_reuse(
            snapshot=Path("snapshot.json"),
            min_count=2,
            lemma_stubs=None,
            root=None,
        )
    finally:
        cli.DEFAULT_RUNNER = saved
    captured = capsys.readouterr().out
    assert "format_version" in captured
    assert cli.STRUCTURE_DIFF_COMMAND in calls
    assert cli.DECISION_DIFF_COMMAND in calls
    assert cli.STRUCTURE_REUSE_COMMAND in calls


# gabion:evidence E:function_site::cli.py::gabion.cli._emit_structure_diff
def test_emit_structure_diff_success(capsys) -> None:
    result = {"exit_code": 0, "diff": {"summary": {"added": 0}}}
    cli._emit_structure_diff(result)
    captured = capsys.readouterr()
    assert "\"exit_code\": 0" in captured.out
    assert captured.err == ""


# gabion:evidence E:function_site::cli.py::gabion.cli._emit_structure_diff
def test_emit_structure_diff_errors_exit(capsys) -> None:
    result = {"exit_code": 2, "errors": ["bad snapshot"], "diff": {}}
    with pytest.raises(typer.Exit) as exc:
        cli._emit_structure_diff(result)
    assert exc.value.exit_code == 2
    captured = capsys.readouterr()
    assert "\"exit_code\": 2" in captured.out
    assert "bad snapshot" in captured.err


# gabion:evidence E:function_site::cli.py::gabion.cli._emit_decision_diff
def test_emit_decision_diff_success(capsys) -> None:
    result = {"exit_code": 0, "diff": {"summary": {"added": 0}}}
    cli._emit_decision_diff(result)
    captured = capsys.readouterr()
    assert "\"exit_code\": 0" in captured.out


# gabion:evidence E:function_site::cli.py::gabion.cli._emit_decision_diff
def test_emit_decision_diff_errors_exit(capsys) -> None:
    result = {"exit_code": 2, "errors": ["bad decision"], "diff": {}}
    with pytest.raises(typer.Exit) as exc:
        cli._emit_decision_diff(result)
    assert exc.value.exit_code == 2
    captured = capsys.readouterr()
    assert "bad decision" in captured.err


# gabion:evidence E:function_site::cli.py::gabion.cli._emit_structure_reuse
def test_emit_structure_reuse_success(capsys) -> None:
    result = {"exit_code": 0, "reuse": {"summary": {}}}
    cli._emit_structure_reuse(result)
    captured = capsys.readouterr()
    assert "\"exit_code\": 0" in captured.out


# gabion:evidence E:function_site::cli.py::gabion.cli._emit_structure_reuse
def test_emit_structure_reuse_errors_exit(capsys) -> None:
    result = {"exit_code": 2, "errors": ["bad reuse"], "reuse": {}}
    with pytest.raises(typer.Exit) as exc:
        cli._emit_structure_reuse(result)
    assert exc.value.exit_code == 2
    captured = capsys.readouterr()
    assert "bad reuse" in captured.err
