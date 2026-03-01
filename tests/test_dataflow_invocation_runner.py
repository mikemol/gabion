from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from gabion.tooling.dataflow_invocation_runner import DataflowInvocationRunner
from gabion.tooling.execution_envelope import ExecutionEnvelope


def test_dataflow_invocation_runner_delta_bundle_uses_injected_run_check() -> None:
    captured: dict[str, object] = {}

    def _run_check(**kwargs):
        captured.update(kwargs)
        return {"exit_code": 0, "analysis_state": "succeeded"}

    runner = DataflowInvocationRunner(run_check_fn=_run_check)
    envelope = ExecutionEnvelope.for_delta_bundle(
        root=Path("."),
        report_path=Path("artifacts/audit_reports/dataflow_report.md"),
        strictness="low",
        allow_external=False,
        aspf_state_json=Path("out/state.json"),
        aspf_delta_jsonl=Path("out/delta.jsonl"),
        aspf_import_state=(Path("out/import.json"),),
    )
    result = runner.run_delta_bundle(envelope)
    assert result.exit_code == 0
    assert result.analysis_state == "succeeded"
    assert captured["strictness"] == "low"
    assert captured["aspf_import_state"] == [Path("out/import.json").resolve()]


def test_dataflow_invocation_runner_delta_bundle_rejects_raw_envelope() -> None:
    runner = DataflowInvocationRunner(run_check_fn=lambda **_kwargs: {})
    envelope = ExecutionEnvelope.for_raw(
        root=Path("."),
        aspf_state_json=None,
        aspf_delta_jsonl=None,
    )
    with pytest.raises(ValueError):
        runner.run_delta_bundle(envelope)


def test_dataflow_invocation_runner_raw_uses_cli_loader_and_dispatch() -> None:
    dispatched: dict[str, object] = {}

    def _dispatch(**kwargs):
        dispatched.update(kwargs)
        return {"exit_code": 3, "analysis_state": "failed"}

    fake_cli = SimpleNamespace(
        parse_dataflow_args_or_exit=lambda _args: SimpleNamespace(root="."),
        build_dataflow_payload=lambda _opts: {"seed": "payload"},
        run_command=lambda **_kwargs: {},
    )
    runner = DataflowInvocationRunner(
        dispatch_command_fn=_dispatch,
        cli_module_loader=lambda: fake_cli,
    )
    envelope = ExecutionEnvelope.for_raw(
        root=Path("."),
        aspf_state_json=Path("out/state.json"),
        aspf_delta_jsonl=Path("out/delta.jsonl"),
        aspf_import_state=(Path("out/import.json"),),
    )
    result = runner.run_raw(envelope, ["--", ".", "--root", "."])
    assert result.exit_code == 3
    assert result.analysis_state == "failed"
    payload = dispatched["payload"]
    assert isinstance(payload, dict)
    assert payload["seed"] == "payload"
    assert payload["aspf_state_json"].endswith("out/state.json")
    assert payload["aspf_import_state"] == [str(Path("out/import.json").resolve())]


def test_dataflow_invocation_runner_resolve_helpers_cover_fallback_paths() -> None:
    fake_cli = SimpleNamespace(
        run_check=lambda **_kwargs: {"exit_code": 0, "analysis_state": "ok"},
        dispatch_command=lambda **_kwargs: {"exit_code": 0, "analysis_state": "ok"},
    )
    runner = DataflowInvocationRunner(cli_module_loader=lambda: fake_cli)
    assert runner._resolve_run_check() is fake_cli.run_check
    assert runner._resolve_dispatch_command() is fake_cli.dispatch_command

    default_runner = DataflowInvocationRunner()
    cli_module = default_runner._resolve_cli_module()
    assert getattr(cli_module, "__name__", "") == "gabion.cli"


def test_dataflow_invocation_runner_ensures_repo_root_importable(tmp_path: Path) -> None:
    repo_root = (tmp_path / "repo").resolve()
    repo_root.mkdir(parents=True, exist_ok=True)
    root_text = str(repo_root)
    original_path = list(sys.path)
    try:
        sys.path = [entry for entry in original_path if entry != root_text]
        runner = DataflowInvocationRunner()
        runner._ensure_repo_root_importable(repo_root)
        assert sys.path[0] == root_text
        runner._ensure_repo_root_importable(repo_root)
        assert sys.path.count(root_text) == 1
    finally:
        sys.path = original_path


def test_dataflow_invocation_runner_raw_without_aspf_payload_passthrough() -> None:
    dispatched: dict[str, object] = {}
    fake_cli = SimpleNamespace(
        parse_dataflow_args_or_exit=lambda _args: SimpleNamespace(root="."),
        build_dataflow_payload=lambda _opts: {"seed": "payload"},
        run_command=lambda **_kwargs: {},
    )
    runner = DataflowInvocationRunner(
        dispatch_command_fn=lambda **kwargs: (
            dispatched.update(kwargs)
            or {"exit_code": 0, "analysis_state": "succeeded"}
        ),
        cli_module_loader=lambda: fake_cli,
    )
    envelope = ExecutionEnvelope.for_raw(
        root=Path("."),
        aspf_state_json=None,
        aspf_delta_jsonl=None,
    )
    result = runner.run_raw(envelope, ["--", ".", "--root", "."])
    assert result.exit_code == 0
    payload = dispatched["payload"]
    assert isinstance(payload, dict)
    assert "aspf_state_json" not in payload


def test_dataflow_invocation_runner_raw_rejects_delta_envelope() -> None:
    fake_cli = SimpleNamespace(
        parse_dataflow_args_or_exit=lambda _args: SimpleNamespace(root="."),
        build_dataflow_payload=lambda _opts: {},
        run_command=lambda **_kwargs: {},
    )
    runner = DataflowInvocationRunner(
        dispatch_command_fn=lambda **_kwargs: {},
        cli_module_loader=lambda: fake_cli,
    )
    envelope = ExecutionEnvelope.for_delta_bundle(
        root=Path("."),
        report_path=Path("artifacts/audit_reports/dataflow_report.md"),
        strictness=None,
        allow_external=None,
        aspf_state_json=None,
        aspf_delta_jsonl=None,
    )
    with pytest.raises(ValueError):
        runner.run_raw(envelope, ["--", ".", "--root", "."])
