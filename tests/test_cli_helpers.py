from __future__ import annotations

from pathlib import Path
import io
import json
import re
import subprocess
import sys
import urllib.error
import zipfile

import pytest
import typer
from typer.testing import CliRunner

from gabion import cli
from gabion.analysis.timeout_context import check_deadline
from gabion.commands import progress_contract as progress_timeline
from gabion.exceptions import NeverThrown
from tests.env_helpers import env_scope as _env_scope


_ANSI_ESCAPE = re.compile(r"\x1B\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._split_csv_entries::entries E:decision_surface/direct::cli.py::gabion.cli._split_csv::value E:decision_surface/direct::cli.py::gabion.cli._split_csv::stale_22e7d997b440
def test_split_csv_helpers() -> None:
    assert cli._split_csv_entries(["a, b", " ", "c"]) == ["a", "b", "c"]
    assert cli._split_csv_entries([" ", ""]) == []

    assert cli._split_csv("a, , b") == ["a", "b"]
    assert cli._split_csv(" ,") == []


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_phase_timeline_wrappers_delegate_to_shared_progress_module::cli.py::gabion.cli._phase_timeline_header_columns::cli.py::gabion.cli._phase_timeline_header_block::cli.py::gabion.cli._phase_progress_dimensions_summary
def test_phase_timeline_wrappers_delegate_to_shared_progress_module() -> None:
    assert cli._phase_timeline_header_columns() == progress_timeline.phase_timeline_header_columns()
    assert cli._phase_timeline_header_block() == progress_timeline.phase_timeline_header_block()
    assert (
        cli._phase_progress_dimensions_summary(
            {"dimensions": {"paths": {"done": 1, "total": 2}}}
        )
        == "paths=1/2"
    )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_parse_dataflow_args_or_exit_routes_help_to_parser::cli.py::gabion.cli.parse_dataflow_args_or_exit
def test_parse_dataflow_args_or_exit_routes_help_to_parser(capsys) -> None:
    with pytest.raises(typer.Exit) as exc:
        cli.parse_dataflow_args_or_exit(["--help"])
    assert exc.value.exit_code == 0
    assert "usage:" in capsys.readouterr().out.lower()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_parse_dataflow_args_or_exit_converts_parse_errors_to_typer_exit::cli.py::gabion.cli.parse_dataflow_args_or_exit
def test_parse_dataflow_args_or_exit_converts_parse_errors_to_typer_exit() -> None:
    with pytest.raises(typer.Exit) as exc:
        cli.parse_dataflow_args_or_exit([])
    assert exc.value.exit_code == 2


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_rejects_unknown_args_in_strict_profile::cli.py::gabion.cli.app
def test_check_rejects_unknown_args_in_strict_profile() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["check", "run", "sample.py", "--dot", "-"])
    assert result.exit_code != 0
    normalized_output = _strip_ansi(result.output)
    assert "No such option: --dot" in normalized_output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_rejects_unknown_profile::cli.py::gabion.cli.app
def test_check_rejects_unknown_profile() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["check", "--profile", "mystery", "run", "sample.py"],
    )
    assert result.exit_code != 0
    normalized_output = _strip_ansi(result.output)
    assert "Removed --profile flag." in normalized_output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_raw_profile_delegates_with_profile_defaults::cli.py::gabion.cli.app
def test_check_raw_profile_delegates_with_profile_defaults(
) -> None:
    captured: dict[str, object] = {}

    def _fake_run(argv: list[str]) -> None:
        captured["argv"] = list(argv)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["check", "raw", "--", "sample.py"],
        obj={"run_dataflow_raw_argv": _fake_run},
    )
    assert result.exit_code == 0
    assert captured["argv"] == ["sample.py"]


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_raw_profile_maps_common_flags_and_passthrough_args::cli.py::gabion.cli.app
def test_check_raw_profile_maps_common_flags_and_passthrough_args(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run(argv: list[str]) -> None:
        captured["argv"] = list(argv)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "check",
            "raw",
            "--",
            "sample.py",
            "--root",
            str(tmp_path),
            "--config",
            "cfg.toml",
            "--report",
            "report.md",
            "--decision-snapshot",
            "decision.json",
            "--baseline",
            "baseline.txt",
            "--baseline-write",
            "--exclude",
            "a,b",
            "--ignore-params",
            "x,y",
            "--transparent-decorators",
            "deco",
            "--allow-external",
            "--strictness",
            "low",
            "--resume-checkpoint",
            "resume.json",
            "--emit-timeout-progress-report",
            "--resume-on-timeout",
            "2",
            "--fail-on-violations",
            "--fail-on-type-ambiguities",
            "--lint",
            "--lint-jsonl",
            "lint.jsonl",
            "--lint-sarif",
            "lint.sarif",
            "--dot",
            "-",
            "--type-audit",
        ],
        obj={"run_dataflow_raw_argv": _fake_run},
    )
    assert result.exit_code == 0
    assert captured["argv"] == [
        "sample.py",
        "--root",
        str(tmp_path),
        "--config",
        "cfg.toml",
        "--report",
        "report.md",
        "--decision-snapshot",
        "decision.json",
        "--baseline",
        "baseline.txt",
        "--baseline-write",
        "--exclude",
        "a,b",
        "--ignore-params",
        "x,y",
        "--transparent-decorators",
        "deco",
        "--allow-external",
        "--strictness",
        "low",
        "--resume-checkpoint",
        "resume.json",
        "--emit-timeout-progress-report",
        "--resume-on-timeout",
        "2",
        "--fail-on-violations",
        "--fail-on-type-ambiguities",
        "--lint",
        "--lint-jsonl",
        "lint.jsonl",
        "--lint-sarif",
        "lint.sarif",
        "--dot",
        "-",
        "--type-audit",
    ]


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_raw_profile_maps_no_allow_external::cli.py::gabion.cli.app
def test_check_raw_profile_maps_no_allow_external(
) -> None:
    captured: dict[str, object] = {}

    def _fake_run(argv: list[str]) -> None:
        captured["argv"] = list(argv)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "check",
            "raw",
            "--",
            "sample.py",
            "--no-allow-external",
            "--no-fail-on-violations",
            "--no-fail-on-type-ambiguities",
        ],
        obj={"run_dataflow_raw_argv": _fake_run},
    )
    assert result.exit_code == 0
    assert captured["argv"] == [
        "sample.py",
        "--no-allow-external",
        "--no-fail-on-violations",
        "--no-fail-on-type-ambiguities",
    ]


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_raw_profile_rejects_check_only_flags::test_cli_helpers.py::tests.test_cli_helpers._strip_ansi
def test_check_raw_profile_rejects_check_only_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["check", "--emit-test-obsolescence", "raw", "--", "sample.py"],
    )
    normalized_output = _strip_ansi(result.output)
    assert result.exit_code != 0
    assert "Removed legacy check modality flags." in normalized_output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dataflow_audit_nonzero_exit_reports_explicit_causes::cli.py::gabion.cli._run_dataflow_raw_argv
def test_dataflow_audit_nonzero_exit_reports_explicit_causes(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {"exit_code": 1, "analysis_state": "succeeded", "violations": 2}

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(["sample.py"], runner=runner)
    assert exc.value.exit_code == 1
    assert "Non-zero exit (1) cause(s):" in capsys.readouterr().err


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dataflow_audit_nonzero_exit_fallback_is_explicit::cli.py::gabion.cli._run_dataflow_raw_argv
def test_dataflow_audit_nonzero_exit_fallback_is_explicit(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {"exit_code": 1, "analysis_state": "succeeded"}

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(["sample.py"], runner=runner)
    assert exc.value.exit_code == 1
    err = capsys.readouterr().err
    assert "no explicit violations/type ambiguities/errors were returned" in err
    assert "analysis_state=succeeded" in err


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_context_cli_deps_use_defaults_for_non_mapping_context::cli.py::gabion.cli._context_cli_deps
def test_context_cli_deps_use_defaults_for_non_mapping_context() -> None:
    class _CtxNoMapping:
        obj = None

    deps = cli._context_cli_deps(_CtxNoMapping())
    assert deps.run_dataflow_raw_argv_fn is cli._run_dataflow_raw_argv
    assert deps.run_check_fn is cli.run_check
    assert deps.run_sppf_sync_fn is cli._run_sppf_sync


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_context_cli_deps_accept_callable_overrides::cli.py::gabion.cli._context_cli_deps
def test_context_cli_deps_accept_callable_overrides() -> None:
    def _run_dataflow(argv: list[str]) -> None:
        _ = argv

    def _run_check(**_kwargs: object) -> dict[str, object]:
        return {}

    def _restore(**_kwargs: object) -> int:
        return 0

    def _run_sppf(**_kwargs: object) -> int:
        return 0

    class _Ctx:
        obj = {
            "run_dataflow_raw_argv": _run_dataflow,
            "run_check": _run_check,
            "run_sppf_sync": _run_sppf,
        }

    deps = cli._context_cli_deps(_Ctx())
    assert deps.run_dataflow_raw_argv_fn is _run_dataflow
    assert deps.run_check_fn is _run_check
    assert deps.run_sppf_sync_fn is _run_sppf


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_context_dependency_helpers_reject_noncallables::cli.py::gabion.cli._context_callable_dep
def test_context_dependency_helpers_reject_noncallables() -> None:
    class DummyCtx:
        obj = {
            "run_dataflow_raw_argv": "not-callable",
        }

    with pytest.raises(NeverThrown):
        cli._context_run_dataflow_raw_argv(DummyCtx())


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_context_dependency_helpers_reject_noncallables_across_check_helpers::cli.py::gabion.cli._context_run_check::cli.py::gabion.cli._context_run_sppf_sync
def test_context_dependency_helpers_reject_noncallables_across_check_helpers() -> None:
    class _Ctx:
        obj = {
            "run_check": "not-callable",
            "run_sppf_sync": "not-callable",
        }

    with pytest.raises(NeverThrown):
        cli._context_run_check(_Ctx())
    with pytest.raises(NeverThrown):
        cli._context_run_sppf_sync(_Ctx())


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._write_lint_jsonl::target E:decision_surface/direct::cli.py::gabion.cli._write_lint_sarif::target E:decision_surface/direct::cli.py::gabion.cli._write_lint_jsonl::stale_a0c064f7325b
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


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._write_lint_jsonl::target E:decision_surface/direct::cli.py::gabion.cli._write_lint_jsonl::stale_1061723ef45d
def test_lint_writers_accept_dev_stdout(capsys) -> None:
    entries = [
        {
            "path": "mod.py",
            "line": 3,
            "col": 1,
            "code": "GABION_CODE",
            "message": "note",
            "severity": "warning",
        }
    ]
    cli._write_lint_jsonl("/dev/stdout", entries)
    assert "GABION_CODE" in capsys.readouterr().out


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_target_stream_router_reopens_for_encoding_change::cli.py::gabion.cli._TargetStreamRouter._stream_for_target
def test_target_stream_router_reopens_for_encoding_change(tmp_path: Path) -> None:
    target_path = tmp_path / "enc.txt"
    router = cli._TargetStreamRouter(max_open_streams=2)
    try:
        router.write(target=str(target_path), payload="alpha", encoding="utf-8")
        router.write(target=str(target_path), payload="beta", encoding="utf-16")
        assert target_path.read_text(encoding="utf-16") == "beta"
    finally:
        router.close()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_target_stream_router_evicts_oldest_stream::cli.py::gabion.cli._TargetStreamRouter._stream_for_target
def test_target_stream_router_evicts_oldest_stream(tmp_path: Path) -> None:
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    router = cli._TargetStreamRouter(max_open_streams=1)
    try:
        router.write(target=str(first_path), payload="one")
        router.write(target=str(second_path), payload="two")
        assert list(router._streams.keys()) == [str(second_path)]
    finally:
        router.close()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_target_stream_router_close_closes_all_streams::cli.py::gabion.cli._TargetStreamRouter.close
def test_target_stream_router_close_closes_all_streams(tmp_path: Path) -> None:
    router = cli._TargetStreamRouter(max_open_streams=4)
    router.write(target=str(tmp_path / "a.txt"), payload="a")
    router.write(target=str(tmp_path / "b.txt"), payload="b")
    assert len(router._streams) == 2
    router.close()
    assert len(router._streams) == 0


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_normalize_optional_output_target_handles_empty::cli.py::gabion.cli._normalize_optional_output_target
def test_normalize_optional_output_target_handles_empty() -> None:
    assert cli._normalize_optional_output_target(None) is None
    assert cli._normalize_optional_output_target("   ") is None
    assert (
        cli._normalize_optional_output_target("-")
        == "/dev/stdout"
    )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_write_text_to_target_reuses_stream_and_preserves_overwrite_semantics::cli.py::gabion.cli._write_text_to_target
def test_write_text_to_target_reuses_stream_and_preserves_overwrite_semantics(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "out.txt"
    cli._write_text_to_target(target_path, "first")
    cli._write_text_to_target(target_path, "second")
    assert target_path.read_text(encoding="utf-8") == "second"


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_write_lint_sarif_rejects_duplicate_rule_codes::cli.py::gabion.cli._write_lint_sarif
def test_write_lint_sarif_rejects_duplicate_rule_codes(tmp_path: Path) -> None:
    entries = [
        {"path": "mod.py", "line": 1, "col": 1, "code": "GABION_CODE", "message": "m1"},
        {"path": "mod.py", "line": 2, "col": 1, "code": "GABION_CODE", "message": "m2"},
    ]
    sarif_path = tmp_path / "lint.sarif"
    with pytest.raises(ValueError) as exc:
        cli._write_lint_sarif(str(sarif_path), entries)
    assert "duplicate SARIF rule code(s): GABION_CODE" in str(exc.value)


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


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_lint_outputs_jsonl_only::cli.py::gabion.cli._emit_lint_outputs
def test_emit_lint_outputs_jsonl_only(tmp_path: Path) -> None:
    lines = ["mod.py:1:1: GABION_CODE message"]
    jsonl_path = tmp_path / "lint.jsonl"
    cli._emit_lint_outputs(
        lines,
        lint=False,
        lint_jsonl=jsonl_path,
        lint_sarif=None,
    )
    assert jsonl_path.exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_lint_outputs_sarif_only::cli.py::gabion.cli._emit_lint_outputs
def test_emit_lint_outputs_sarif_only(tmp_path: Path) -> None:
    lines = ["mod.py:1:1: GABION_CODE message"]
    sarif_path = tmp_path / "lint.sarif"
    cli._emit_lint_outputs(
        lines,
        lint=False,
        lint_jsonl=None,
        lint_sarif=sarif_path,
    )
    assert sarif_path.exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_cli_deadline_scope_yields::cli.py::gabion.cli._cli_deadline_scope
def test_cli_deadline_scope_yields() -> None:
    with cli._cli_deadline_scope():
        assert True


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_refactor_payload::bundle,input_payload,protocol_name,target_path E:decision_surface/direct::cli.py::gabion.cli.build_refactor_payload::stale_7d8b74e626fe
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
        compatibility_shim_warnings=True,
        compatibility_shim_overloads=True,
        ambient_rewrite=False,
        rationale=None,
    ) == payload


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.build_refactor_payload::bundle,input_payload,protocol_name,target_path E:decision_surface/direct::cli.py::gabion.cli.build_refactor_payload::stale_e040ac567f92
def test_build_refactor_payload_requires_fields(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter):
        cli.build_refactor_payload(
            protocol_name=None,
            bundle=None,
            field=None,
            target_path=None,
            target_functions=None,
            compatibility_shim=False,
            compatibility_shim_warnings=True,
            compatibility_shim_overloads=True,
            ambient_rewrite=False,
            rationale=None,
        )

    payload = cli.build_refactor_payload(
        protocol_name="Bundle",
        bundle=None,
        field=[" : ", "a:int", "b"],
        target_path=tmp_path / "target.py",
        target_functions=None,
        compatibility_shim=False,
        compatibility_shim_warnings=True,
        compatibility_shim_overloads=True,
        ambient_rewrite=False,
        rationale=None,
    )
    assert payload["bundle"] == ["a", "b"]
    assert payload["fields"] == [
        {"name": "a", "type_hint": "int"},
        {"name": "b", "type_hint": None},
    ]


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_governance_runner_success_and_exception::cli.py::gabion.cli._run_governance_runner
def test_run_governance_runner_success_and_exception(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        cli._run_governance_runner(
            runner_name="ok",
            runner=lambda _argv: 3,
            args=["--flag"],
        )
        == 3
    )

    assert (
        cli._run_governance_runner(
            runner_name="boom",
            runner=lambda _argv: (_ for _ in ()).throw(RuntimeError("boom")),
            args=[],
        )
        == 1
    )
    assert "governance command failed (boom): boom" in capsys.readouterr().err


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_docflow_audit_passes_flags_to_governance_module::cli.py::gabion.cli._run_docflow_audit
def test_run_docflow_audit_passes_flags_to_governance_module(tmp_path: Path) -> None:
    calls: list[tuple[str, list[str]]] = []
    orig_docflow = cli.tooling_governance_audit.run_docflow_cli
    orig_sppf = cli.tooling_governance_audit.run_sppf_graph_cli

    def _docflow(argv: list[str] | None = None) -> int:
        calls.append(("docflow", list(argv or [])))
        return 0

    def _sppf(argv: list[str] | None = None) -> int:
        calls.append(("sppf", list(argv or [])))
        return 0

    try:
        cli.tooling_governance_audit.run_docflow_cli = _docflow
        cli.tooling_governance_audit.run_sppf_graph_cli = _sppf
        exit_code = cli._run_docflow_audit(
            root=tmp_path,
            fail_on_violations=True,
            sppf_gh_ref_mode="required",
            extra_path=["in"],
        )
    finally:
        cli.tooling_governance_audit.run_docflow_cli = orig_docflow
        cli.tooling_governance_audit.run_sppf_graph_cli = orig_sppf

    assert exit_code == 0
    assert calls[0] == (
        "docflow",
        ["--root", str(tmp_path), "--extra-path", "in", "--fail-on-violations", "--sppf-gh-ref-mode", "required"],
    )
    assert calls[1] == ("sppf", [])


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_docflow_audit_nonzero_short_circuits_sppf::cli.py::gabion.cli._run_docflow_audit
def test_run_docflow_audit_nonzero_short_circuits_sppf(tmp_path: Path) -> None:
    calls: list[str] = []
    orig_docflow = cli.tooling_governance_audit.run_docflow_cli
    orig_sppf = cli.tooling_governance_audit.run_sppf_graph_cli

    def _docflow(_argv: list[str] | None = None) -> int:
        calls.append("docflow")
        return 7

    def _sppf(_argv: list[str] | None = None) -> int:
        calls.append("sppf")
        return 0

    try:
        cli.tooling_governance_audit.run_docflow_cli = _docflow
        cli.tooling_governance_audit.run_sppf_graph_cli = _sppf
        exit_code = cli._run_docflow_audit(
            root=tmp_path,
            fail_on_violations=False,
            sppf_gh_ref_mode="required",
        )
    finally:
        cli.tooling_governance_audit.run_docflow_cli = orig_docflow
        cli.tooling_governance_audit.run_sppf_graph_cli = orig_sppf

    assert exit_code == 7
    assert calls == ["docflow"]


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::stale_5b469ca25d0a
def test_dataflow_audit_skips_type_audit_output() -> None:
    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {"exit_code": 0}

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(["sample.py"], runner=runner)
    assert exc.value.exit_code == 0


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::stale_f562194a2932
def test_dataflow_audit_type_audit_empty_findings() -> None:
    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {"exit_code": 0, "type_suggestions": [], "type_ambiguities": []}

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            ["sample.py", "--type-audit", "--type-audit-max", "1"],
            runner=runner,
        )
    assert exc.value.exit_code == 0


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::stale_b1d435f0c525
def test_dataflow_audit_emits_lint_outputs(tmp_path: Path, capsys) -> None:
    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {
            "exit_code": 0,
            "lint_lines": ["mod.py:1:1: GABION_CODE message"],
        }

    jsonl_path = tmp_path / "lint.jsonl"
    sarif_path = tmp_path / "lint.sarif"
    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
                "sample.py",
                "--lint",
                "--lint-jsonl",
                str(jsonl_path),
                "--lint-sarif",
                str(sarif_path),
            ],
            runner=runner,
        )
    assert exc.value.exit_code == 0
    out = capsys.readouterr().out
    assert "GABION_CODE" in out
    assert jsonl_path.exists()
    assert sarif_path.exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dataflow_audit_timeout_writes_deadline_profile::cli.py::gabion.cli._run_dataflow_raw_argv
def test_dataflow_audit_timeout_writes_deadline_profile(tmp_path: Path) -> None:
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

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            ["sample.py", "--root", str(tmp_path)],
            runner=runner,
        )
    assert exc.value.exit_code == 2
    profile_json = tmp_path / "artifacts" / "out" / "deadline_profile.json"
    profile_md = tmp_path / "artifacts" / "out" / "deadline_profile.md"
    assert profile_json.exists()
    assert profile_md.exists()
    payload = json.loads(profile_json.read_text())
    assert payload["checks_total"] == 3


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_timeout_profile_artifacts_no_profile_is_noop::cli.py::gabion.cli._emit_timeout_profile_artifacts
def test_emit_timeout_profile_artifacts_no_profile_is_noop(tmp_path: Path) -> None:
    cli._emit_timeout_profile_artifacts(
        {"timeout_context": {"deadline_profile": "bad"}},
        root=tmp_path,
    )
    assert not (tmp_path / "artifacts" / "out" / "deadline_profile.json").exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dataflow_audit_timeout_exits_without_builtin_retry::cli.py::gabion.cli._run_dataflow_raw_argv
def test_dataflow_audit_timeout_exits_without_builtin_retry(tmp_path: Path) -> None:
    calls = {"count": 0}

    def runner(*_args, **_kwargs):
        calls["count"] += 1
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
            },
        }

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
                "sample.py",
                "--root",
                str(tmp_path),
            ],
            runner=runner,
        )
    assert exc.value.exit_code == 2
    assert calls["count"] == 1
    profile_json = tmp_path / "artifacts" / "out" / "deadline_profile.json"
    assert profile_json.exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_timeout_exits_and_emits_profile_artifacts::cli.py::gabion.cli._run_with_timeout_retries
def test_check_timeout_exits_and_emits_profile_artifacts(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    emitted_profile: list[dict[str, object]] = []

    def _fake_run_once() -> dict[str, object]:
        calls.append({"attempt": len(calls) + 1})
        return {
            "timeout": True,
            "analysis_state": "timed_out_progress_resume",
            "exit_code": 2,
        }

    with pytest.raises(typer.Exit) as exc:
        cli._run_with_timeout_retries(
            run_once=_fake_run_once,
            root=tmp_path,
            emit_timeout_profile_artifacts_fn=lambda result, *, root: emitted_profile.append(
            {"result": dict(result), "root": Path(root)}
            ),
        )
    assert exc.value.exit_code == 2
    assert len(calls) == 1
    assert len(emitted_profile) == 1
    assert emitted_profile[0]["root"] == tmp_path


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_timeout_no_retry_exits_with_timeout_code::cli.py::gabion.cli._run_with_timeout_retries
def test_check_timeout_no_retry_exits_with_timeout_code(
    tmp_path: Path,
) -> None:
    profile_calls = 0

    def _fake_run_once() -> dict[str, object]:
        return {
            "timeout": True,
            "analysis_state": "timed_out_no_progress",
            "exit_code": 7,
        }

    def _record_profile(*_args: object, **_kwargs: object) -> None:
        nonlocal profile_calls
        profile_calls += 1

    with pytest.raises(typer.Exit) as exc:
        cli._run_with_timeout_retries(
            run_once=_fake_run_once,
            root=tmp_path,
            emit_timeout_profile_artifacts_fn=_record_profile,
        )
    assert exc.value.exit_code == 7
    assert profile_calls == 1


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_render_timeout_progress_markdown_skips_empty_resume_token_fields::cli.py::gabion.cli._render_timeout_progress_markdown
def test_render_timeout_progress_markdown_skips_empty_resume_token_fields() -> None:
    rendered = cli._render_timeout_progress_markdown(
        analysis_state="timed_out_progress_resume",
        progress={
            "classification": "timed_out_progress_resume",
            "resume": {"resume_token": {"phase": "analysis_collection", "witness_digest": None}},
        },
    )
    assert "`phase`: `analysis_collection`" in rendered
    assert "witness_digest" not in rendered


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_render_timeout_progress_markdown_skips_non_mapping_obligation_entries::cli.py::gabion.cli._render_timeout_progress_markdown
def test_render_timeout_progress_markdown_skips_non_mapping_obligation_entries() -> None:
    rendered = cli._render_timeout_progress_markdown(
        analysis_state="timed_out_progress_resume",
        progress={
            "incremental_obligations": ["bad", {"status": "SATISFIED", "contract": "c", "kind": "k", "detail": "d"}]
        },
    )
    assert "`SATISFIED` `c` `k`" in rendered


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_render_timeout_progress_markdown_includes_tick_metrics::cli.py::gabion.cli._render_timeout_progress_markdown
def test_render_timeout_progress_markdown_includes_tick_metrics() -> None:
    rendered = cli._render_timeout_progress_markdown(
        analysis_state="timed_out_progress_resume",
        progress={
            "classification": "timed_out_progress_resume",
            "ticks_consumed": 11,
            "tick_limit": 20,
            "ticks_remaining": 9,
            "ticks_per_ns": 0.25,
        },
        deadline_profile={"ticks_per_ns": 0.125},
    )
    assert "`ticks_consumed`: `11`" in rendered
    assert "`tick_limit`: `20`" in rendered
    assert "`ticks_remaining`: `9`" in rendered
    assert "`ticks_per_ns`: `0.250000000`" in rendered
    assert "`ticks_per_ns`: `0.125000000`" not in rendered


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_render_timeout_progress_markdown_falls_back_to_profile_tick_metric::cli.py::gabion.cli._render_timeout_progress_markdown
def test_render_timeout_progress_markdown_falls_back_to_profile_tick_metric() -> None:
    rendered = cli._render_timeout_progress_markdown(
        analysis_state="timed_out_progress_resume",
        progress={
            "classification": "timed_out_progress_resume",
            "ticks_consumed": 11,
            "tick_limit": 20,
            "ticks_remaining": 9,
        },
        deadline_profile={"ticks_per_ns": 0.125},
    )
    assert "`ticks_per_ns`: `0.125000000`" in rendered


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dataflow_audit_timeout_without_retry_raises_exit::cli.py::gabion.cli._run_dataflow_raw_argv
def test_dataflow_audit_timeout_without_retry_raises_exit(tmp_path: Path) -> None:
    def runner(*_args, **_kwargs):
        return {
            "exit_code": 2,
            "timeout": True,
            "analysis_state": "timed_out_no_progress",
            "timeout_context": {"deadline_profile": {"checks_total": 1, "sites": [], "edges": []}},
        }

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
                "sample.py",
                "--root",
                str(tmp_path),
                "--resume-on-timeout",
                "1",
            ],
            runner=runner,
        )
    assert exc.value.exit_code == 2


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dataflow_audit_timeout_progress_resume_is_single_attempt::cli.py::gabion.cli._run_dataflow_raw_argv
def test_dataflow_audit_timeout_progress_resume_is_single_attempt(
    tmp_path: Path,
) -> None:
    calls = {"count": 0}

    def runner(*_args, **_kwargs):
        calls["count"] += 1
        return {
            "exit_code": 2,
            "timeout": True,
            "analysis_state": "timed_out_progress_resume",
            "timeout_context": {
                "deadline_profile": {"checks_total": 1, "sites": [], "edges": []},
                "progress": {"classification": "timed_out_progress_resume"},
            },
        }

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
                "sample.py",
                "--root",
                str(tmp_path),
            ],
            runner=runner,
        )
    assert exc.value.exit_code == 2
    assert calls["count"] == 1


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dataflow_audit_timeout_uses_single_attempt_budget::cli.py::gabion.cli._run_dataflow_raw_argv::timeout_context.py::gabion.analysis.timeout_context.check_deadline
def test_dataflow_audit_timeout_uses_single_attempt_budget(
    tmp_path: Path,
    env_scope,
    restore_env,
) -> None:
    calls = {"count": 0}

    def runner(*_args, **_kwargs):
        calls["count"] += 1
        check_deadline()
        return {
            "exit_code": 2,
            "timeout": True,
            "analysis_state": "timed_out_progress_resume",
        }

    previous = env_scope(
        {
            "GABION_LSP_TIMEOUT_TICKS": "3",
            "GABION_LSP_TIMEOUT_TICK_NS": "1000000000",
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": None,
        }
    )
    try:
        with pytest.raises(typer.Exit) as exc:
            cli._run_dataflow_raw_argv(
                [
                    "sample.py",
                    "--root",
                    str(tmp_path),
                ],
                runner=runner,
            )
    finally:
        restore_env(previous)
    assert exc.value.exit_code == 2
    assert calls["count"] == 1


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_check_delta_gates_default_runner_with_deadline_budget::cli.py::gabion.cli._run_check_delta_gates
def test_run_check_delta_gates_default_runner_with_deadline_budget(
    env_scope,
    restore_env,
) -> None:
    previous = env_scope(
        {
            "GABION_LSP_TIMEOUT_TICKS": "10000",
            "GABION_LSP_TIMEOUT_TICK_NS": "1000000",
            "GABION_LSP_TIMEOUT_MS": None,
            "GABION_LSP_TIMEOUT_SECONDS": None,
        }
    )
    try:
        assert cli._run_check_delta_gates() == 0
    finally:
        restore_env(previous)


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_phase_progress_from_progress_notification::cli.py::gabion.cli._phase_progress_from_progress_notification
def test_phase_progress_from_progress_notification() -> None:
    payload = cli._phase_progress_from_progress_notification(
        {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "phase": "forest",
                    "work_done": 3,
                    "work_total": 8,
                    "completed_files": 282,
                    "remaining_files": 0,
                    "total_files": 282,
                    "analysis_state": "analysis_forest_in_progress",
                    "classification": "forest_projection",
                    "done": False,
                },
            },
        }
    )
    assert payload == {
        "phase": "forest",
        "work_done": 3,
        "work_total": 8,
        "completed_files": 282,
        "remaining_files": 0,
        "total_files": 282,
        "analysis_state": "analysis_forest_in_progress",
        "classification": "forest_projection",
        "event_kind": "",
        "event_seq": None,
        "ts_utc": "",
        "stale_for_s": None,
        "phase_progress_v2": None,
        "progress_marker": "",
        "phase_timeline_header": "",
        "phase_timeline_row": "",
        "done": False,
    }


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_phase_timeline_from_progress_notification_wrapper::cli.py::gabion.cli._phase_timeline_from_progress_notification::progress_contract.py::gabion.commands.progress_contract.phase_timeline_from_progress_notification
def test_phase_timeline_from_progress_notification_wrapper() -> None:
    timeline = cli._phase_timeline_from_progress_notification(
        {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "phase": "collection",
                    "work_done": 1,
                    "work_total": 2,
                },
            },
        }
    )
    assert isinstance(timeline, dict)
    assert "header" in timeline
    assert "row" in timeline


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_phase_timeline_from_progress_notification_wrapper_invalid_notification::cli.py::gabion.cli._phase_timeline_from_progress_notification::progress_contract.py::gabion.commands.progress_contract.phase_timeline_from_progress_notification
def test_phase_timeline_from_progress_notification_wrapper_invalid_notification() -> None:
    assert (
        cli._phase_timeline_from_progress_notification(
            {"method": "textDocument/publishDiagnostics"}
        )
        is None
    )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_phase_progress_from_progress_notification_rejects_invalid_shapes::cli.py::gabion.cli._phase_progress_from_progress_notification
def test_phase_progress_from_progress_notification_rejects_invalid_shapes() -> None:
    assert (
        cli._phase_progress_from_progress_notification(
            {"method": "textDocument/publishDiagnostics"}
        )
        is None
    )
    assert (
        cli._phase_progress_from_progress_notification(
            {"method": "$/progress", "params": "bad"}
        )
        is None
    )
    assert (
        cli._phase_progress_from_progress_notification(
            {"method": "$/progress", "params": {"token": "wrong", "value": {}}}
        )
        is None
    )
    assert (
        cli._phase_progress_from_progress_notification(
            {
                "method": "$/progress",
                "params": {"token": "gabion.dataflowAudit/progress-v1", "value": "bad"},
            }
        )
        is None
    )
    assert (
        cli._phase_progress_from_progress_notification(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {"phase": ""},
                },
            }
        )
        is None
    )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_phase_timeline_row_from_phase_progress_formats_dimensions_and_staleness::cli.py::gabion.cli._phase_timeline_row_from_phase_progress
def test_phase_timeline_row_from_phase_progress_formats_dimensions_and_staleness() -> None:
    row = cli._phase_timeline_row_from_phase_progress(
        {
            "ts_utc": "2026-02-20T00:00:00Z",
            "event_seq": 7,
            "event_kind": "heartbeat",
            "phase": "forest",
            "analysis_state": "analysis_forest_in_progress",
            "classification": "forest_projection",
            "progress_marker": "bundle_groups",
            "phase_progress_v2": {
                "primary_unit": "forest_mutable_steps",
                "primary_done": 9,
                "primary_total": 4,
                "dimensions": {
                    "forest_mutable_steps": {"done": 9, "total": 4},
                    "callsite_inventory": {"done": 55, "total": 55},
                    "invalid_payload": "skip",
                    "invalid_done": {"done": "x", "total": 5},
                },
            },
            "completed_files": 5,
            "remaining_files": 1,
            "total_files": 6,
            "stale_for_s": 6.75,
        }
    )
    assert "4/4 forest_mutable_steps" in row
    assert "5/6 rem=1" in row
    assert "forest_mutable_steps=4/4" in row
    assert "callsite_inventory=55/55" in row
    assert "6.8" in row

    row_unit_only = cli._phase_timeline_row_from_phase_progress(
        {
            "phase": "collection",
            "phase_progress_v2": {
                "primary_unit": "collection_files",
                "dimensions": {},
            },
        }
    )
    assert "collection_files" in row_unit_only

    row_known_resume = cli._phase_timeline_row_from_phase_progress(
        {
            "phase": "collection",
            "phase_progress_v2": {
                "primary_unit": "collection_files",
                "dimensions": "bad",
            },
        }
    )
    assert "collection_files" in row_known_resume


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_phase_progress_line_ignores_missing_phase::cli.py::gabion.cli._emit_phase_progress_line
def test_emit_phase_progress_line_ignores_missing_phase(capsys) -> None:
    cli._emit_phase_progress_line({})
    assert capsys.readouterr().out == ""


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_phase_progress_line_formats_payload_fields::cli.py::gabion.cli._emit_phase_progress_line
def test_emit_phase_progress_line_formats_payload_fields(capsys) -> None:
    cli._emit_phase_progress_line(
        {
            "phase": "edge",
            "analysis_state": "analysis_edge_in_progress",
            "classification": "edge_projection",
            "work_done": 2,
            "work_total": 3,
            "completed_files": 8,
            "remaining_files": 1,
            "total_files": 9,
            "done": False,
        }
    )
    cli._emit_phase_progress_line(
        {
            "phase": "post",
            "analysis_state": "succeeded",
            "classification": "succeeded",
            "work_done": 3,
            "work_total": 3,
            "completed_files": 9,
            "remaining_files": 0,
            "total_files": 9,
            "done": True,
        }
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("progress ")
    assert "phase=edge" in lines[0]
    assert "work=2/3" in lines[0]
    assert "files=8/9 remaining=1" in lines[0]
    assert lines[1].startswith("progress done ")
    assert "phase=post" in lines[1]
    assert "work=3/3" in lines[1]


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_phase_progress_line_omits_optional_fragments_when_values_missing::cli.py::gabion.cli._emit_phase_progress_line
def test_emit_phase_progress_line_omits_optional_fragments_when_values_missing(capsys) -> None:
    cli._emit_phase_progress_line({"phase": "collection"})
    line = capsys.readouterr().out.strip()
    assert line == "progress phase=collection"


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_resume_state_startup_line::cli.py::gabion.cli._emit_resume_state_startup_line
def test_emit_resume_state_startup_line(capsys) -> None:
    cli._emit_resume_state_startup_line(
        checkpoint_path="artifacts/audit_reports/resume.json",
        status="checkpoint_loaded",
        reused_files=3,
        total_files=7,
    )
    output = capsys.readouterr().out
    assert "resume state detected..." in output
    assert "status=checkpoint_loaded" in output
    assert "reused_files=3/7" in output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_resume_state_startup_line_unknown_pending::cli.py::gabion.cli._emit_resume_state_startup_line
def test_emit_resume_state_startup_line_unknown_pending(capsys) -> None:
    cli._emit_resume_state_startup_line(
        checkpoint_path="artifacts/audit_reports/resume.json",
        status="pending",
        reused_files=None,
        total_files=None,
    )
    output = capsys.readouterr().out
    assert "resume state detected..." in output
    assert "status=pending" in output
    assert "reused_files=unknown" in output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_dataflow_raw_argv_rejects_removed_resume_checkpoint_flag::cli.py::gabion.cli._run_dataflow_raw_argv
def test_run_dataflow_raw_argv_rejects_removed_resume_checkpoint_flag(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "resume.json"
    module_path = tmp_path / "sample.py"
    module_path.write_text("def sample():\n    return 1\n", encoding="utf-8")

    def _fake_execute_command(ls, _payload=None):
        ls.send_notification(
            "$/progress",
            {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "resume_checkpoint": {
                        "state_path": str(checkpoint_path),
                        "status": "checkpoint_loaded",
                        "reused_files": 2,
                        "total_files": 5,
                    }
                },
            },
        )
        return {"exit_code": 0}

    def _runner(request, *, root=None, notification_callback=None):
        return cli.run_command_direct(
            request,
            root=root,
            notification_callback=notification_callback,
            execute_dataflow_fn=_fake_execute_command,
        )

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
                str(module_path),
                "--root",
                str(tmp_path),
                "--resume-checkpoint",
                str(checkpoint_path),
            ],
            runner=_runner,
        )
    assert exc.value.exit_code == 2


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_dataflow_raw_argv_emits_phase_timeline_rows::cli.py::gabion.cli._run_dataflow_raw_argv
def test_run_dataflow_raw_argv_emits_phase_timeline_rows(
    tmp_path: Path,
    capsys,
) -> None:
    module_path = tmp_path / "sample.py"
    module_path.write_text("def sample():\n    return 1\n", encoding="utf-8")

    def _fake_runner(_request, *, root=None, notification_callback=None):
        _ = root
        assert callable(notification_callback)
        notification_callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "collection",
                        "phase_timeline_header": "| ts | phase |",
                        "phase_timeline_row": "| t0 | collection |",
                    },
                },
            }
        )
        notification_callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "forest",
                        "phase_timeline_row": "| t1 | forest |",
                    },
                },
            }
        )
        return {"exit_code": 0}

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
                str(module_path),
                "--root",
                str(tmp_path),
            ],
            runner=_fake_runner,
        )
    assert exc.value.exit_code == 0
    output = capsys.readouterr().out.splitlines()
    assert "| ts | phase |" in output
    assert "| t0 | collection |" in output
    assert "| t1 | forest |" in output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_dataflow_raw_argv_dedupes_duplicate_event_seq::cli.py::gabion.cli._run_dataflow_raw_argv
def test_run_dataflow_raw_argv_dedupes_duplicate_event_seq(
    tmp_path: Path,
    capsys,
) -> None:
    module_path = tmp_path / "sample.py"
    module_path.write_text("def sample():\n    return 1\n", encoding="utf-8")

    def _fake_runner(_request, *, root=None, notification_callback=None):
        _ = root
        assert callable(notification_callback)
        notification_callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "collection",
                        "event_seq": 11,
                        "phase_timeline_header": "| ts | phase |",
                        "phase_timeline_row": "| t0 | collection |",
                    },
                },
            }
        )
        notification_callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "forest",
                        "event_seq": 11,
                        "phase_timeline_header": "| ts | phase |",
                        "phase_timeline_row": "| t1 | forest |",
                    },
                },
            }
        )
        return {"exit_code": 0}

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
                str(module_path),
                "--root",
                str(tmp_path),
            ],
            runner=_fake_runner,
        )
    assert exc.value.exit_code == 0
    lines = capsys.readouterr().out.splitlines()
    assert "| ts | phase |" in lines
    assert "| t0 | collection |" in lines
    assert "| t1 | forest |" not in lines


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_dataflow_raw_argv_ignores_empty_checkpoint_intro_timeline_row::cli.py::gabion.cli._run_dataflow_raw_argv
def test_run_dataflow_raw_argv_ignores_empty_checkpoint_intro_timeline_row(
    tmp_path: Path,
    capsys,
) -> None:
    module_path = tmp_path / "sample.py"
    module_path.write_text("def sample():\n    return 1\n", encoding="utf-8")

    def _fake_runner(_request, *, root=None, notification_callback=None):
        _ = root
        assert callable(notification_callback)
        notification_callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "checkpoint_intro_timeline_header": "| ts | done |",
                        "checkpoint_intro_timeline_row": "",
                    },
                },
            }
        )
        return {"exit_code": 0}

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
                str(module_path),
                "--root",
                str(tmp_path),
            ],
            runner=_fake_runner,
        )
    assert exc.value.exit_code == 0
    output = capsys.readouterr().out
    assert "| ts | done |" not in output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_dataflow_raw_argv_emits_phase_timeline_rows_for_collection_and_non_collection_updates::cli.py::gabion.cli._run_dataflow_raw_argv
def test_run_dataflow_raw_argv_emits_phase_timeline_rows_for_collection_and_non_collection_updates(
    tmp_path: Path,
    capsys,
) -> None:
    module_path = tmp_path / "sample.py"
    module_path.write_text("def sample():\n    return 1\n", encoding="utf-8")

    def _fake_runner(_request, *, root=None, notification_callback=None):
        _ = root
        assert callable(notification_callback)
        collection_notification = {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "phase": "collection",
                    "work_done": 5,
                    "work_total": 10,
                    "completed_files": 5,
                    "remaining_files": 5,
                    "total_files": 10,
                },
            },
        }
        forest_notification = {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "phase": "forest",
                    "work_done": 1,
                    "work_total": 3,
                    "completed_files": 10,
                    "remaining_files": 0,
                    "total_files": 10,
                    "analysis_state": "analysis_forest_in_progress",
                },
            },
        }
        post_done_notification = {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "phase": "post",
                    "work_done": 3,
                    "work_total": 3,
                    "completed_files": 10,
                    "remaining_files": 0,
                    "total_files": 10,
                    "analysis_state": "succeeded",
                    "classification": "succeeded",
                    "done": True,
                },
            },
        }
        notification_callback(collection_notification)
        notification_callback(forest_notification)
        notification_callback(forest_notification)
        notification_callback(post_done_notification)
        return {"exit_code": 0}

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
                str(module_path),
                "--root",
                str(tmp_path),
            ],
            runner=_fake_runner,
    )
    assert exc.value.exit_code == 0
    lines = capsys.readouterr().out.splitlines()
    table_rows = [line for line in lines if line.startswith("| ") and not line.startswith("| ---")]
    assert any("| collection |" in line for line in table_rows)
    assert sum(1 for line in table_rows if "| forest |" in line) == 1
    assert any("| post |" in line for line in table_rows)


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_dataflow_raw_argv_emits_distinct_post_markers_even_when_work_is_unchanged::cli.py::gabion.cli._run_dataflow_raw_argv
def test_run_dataflow_raw_argv_emits_distinct_post_markers_even_when_work_is_unchanged(
    tmp_path: Path,
    capsys,
) -> None:
    module_path = tmp_path / "sample.py"
    module_path.write_text("def sample():\n    return 1\n", encoding="utf-8")

    def _fake_runner(_request, *, root=None, notification_callback=None):
        _ = root
        assert callable(notification_callback)
        post_progress_a = {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "phase": "post",
                    "work_done": 0,
                    "work_total": 1,
                    "completed_files": 10,
                    "remaining_files": 0,
                    "total_files": 10,
                    "analysis_state": "analysis_post_in_progress",
                    "progress_marker": "deadline_obligations:start",
                },
            },
        }
        post_progress_b = {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "phase": "post",
                    "work_done": 0,
                    "work_total": 1,
                    "completed_files": 10,
                    "remaining_files": 0,
                    "total_files": 10,
                    "analysis_state": "analysis_post_in_progress",
                    "progress_marker": "deadline_obligations:index_ready:1",
                },
            },
        }
        notification_callback(post_progress_a)
        notification_callback(post_progress_b)
        return {"exit_code": 0}

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
                str(module_path),
                "--root",
                str(tmp_path),
            ],
            runner=_fake_runner,
        )
    assert exc.value.exit_code == 0
    lines = capsys.readouterr().out.splitlines()
    post_lines = [
        line
        for line in lines
        if line.startswith("| ") and not line.startswith("| ---") and "| post |" in line
    ]
    assert len(post_lines) == 2
    assert "deadline_obligations:start" in post_lines[0]
    assert "deadline_obligations:index_ready:1" in post_lines[1]


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_analysis_resume_summary::cli.py::gabion.cli._emit_analysis_resume_summary
@pytest.mark.parametrize("cache_verdict", ["hit", "miss", "invalidated", "seeded"])
def test_emit_analysis_resume_summary(cache_verdict: str, capsys) -> None:
    cli._emit_analysis_resume_summary(
        {
            "analysis_resume": {
                "state_path": "artifacts/audit_reports/resume.json",
                "status": "checkpoint_loaded",
                "reused_files": 3,
                "total_files": 5,
                "remaining_files": 2,
                "cache_verdict": cache_verdict,
            }
        }
    )
    output = capsys.readouterr().out
    assert "Resume state:" in output
    assert "status=checkpoint_loaded" in output
    assert "reused_files=3/5" in output
    assert f"cache_verdict={cache_verdict}" in output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_analysis_resume_summary_skips_missing_payload::cli.py::gabion.cli._emit_analysis_resume_summary
def test_emit_analysis_resume_summary_skips_missing_payload(capsys) -> None:
    cli._emit_analysis_resume_summary({"exit_code": 0})
    output = capsys.readouterr().out
    assert output == ""


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_render_timeout_progress_markdown_includes_incremental_obligations::cli.py::gabion.cli._render_timeout_progress_markdown
def test_render_timeout_progress_markdown_includes_incremental_obligations() -> None:
    progress = {
        "classification": "timed_out_progress_resume",
        "retry_recommended": True,
        "resume_supported": True,
        "incremental_obligations": [
            {
                "status": "SATISFIED",
                "contract": "resume_contract",
                "kind": "classification_matches_resume_support",
                "detail": "ok",
            },
            {
                "status": "OBLIGATION",
                "contract": "incremental_projection_contract",
                "kind": "section_projection_state",
                "section_id": "components",
                "detail": "missing_dep_or_not_yet_computed",
            },
        ],
    }
    rendered = cli._render_timeout_progress_markdown(
        analysis_state="timed_out_progress_resume",
        progress=progress,
    )
    assert "Incremental Obligations" in rendered
    assert "resume_contract" in rendered
    assert "components" in rendered


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::stale_09ceb3645a45
def test_dataflow_audit_emits_structure_tree(capsys) -> None:
    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {
            "exit_code": 0,
            "structure_tree": {"format_version": 1, "root": ".", "files": []},
        }

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            ["sample.py", "--emit-structure-tree", "-"],
            runner=runner,
        )
    assert exc.value.exit_code == 0
    captured = capsys.readouterr()
    assert "\"format_version\": 1" in captured.out


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::stale_3ee3d4401f7c
def test_dataflow_audit_emits_structure_tree_dev_stdout(capsys) -> None:
    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {
            "exit_code": 0,
            "structure_tree": {"format_version": 1, "root": ".", "files": []},
        }

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            ["sample.py", "--emit-structure-tree", "/dev/stdout"],
            runner=runner,
        )
    assert exc.value.exit_code == 0
    captured = capsys.readouterr()
    assert "\"format_version\": 1" in captured.out


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::stale_c3f2f0d13aec
def test_dataflow_audit_emits_structure_metrics(capsys) -> None:
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

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            ["sample.py", "--emit-structure-metrics", "-"],
            runner=runner,
        )
    assert exc.value.exit_code == 0
    captured = capsys.readouterr()
    assert "\"bundle_size_histogram\"" in captured.out


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::stale_ca75522a5338_2081cc39
def test_dataflow_audit_emits_decision_snapshot(capsys) -> None:
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

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            ["sample.py", "--emit-decision-snapshot", "-"],
            runner=runner,
        )
    assert exc.value.exit_code == 0
    captured = capsys.readouterr()
    assert "\"decision_surfaces\"" in captured.out


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::stale_f9a3416893cb
def test_dataflow_audit_emits_fingerprint_outputs(capsys) -> None:
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

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
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


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._run_synth::config,exclude,filter_bundle,no_timestamp,paths,refactor_plan,strictness,synthesis_protocols_kind
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
        filter_bundle=cli.DataflowFilterBundle("x, y", "deco"),
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


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._run_synthesis_plan::input_path,output_path E:decision_surface/direct::cli.py::gabion.cli._run_synthesis_plan::stale_71198c0357eb
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
    assert captured["root"] == Path(".")
    assert output_path.read_text().strip()


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._run_synthesis_plan::input_path,output_path E:decision_surface/direct::cli.py::gabion.cli._run_synthesis_plan::stale_a51a81557205_b963adf2
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


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._run_refactor_protocol::input_path,output_path E:decision_surface/direct::cli.py::gabion.cli._run_refactor_protocol::stale_b51675818f31
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
        compatibility_shim_warnings=True,
        compatibility_shim_overloads=True,
        ambient_rewrite=False,
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
    request = cli.SnapshotDiffRequest(baseline=baseline, current=current)
    result = cli.run_structure_diff(
        request=request,
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
    request = cli.SnapshotDiffRequest(baseline=baseline, current=current)
    result = cli.run_decision_diff(
        request=request,
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


# gabion:evidence E:function_site::cli.py::gabion.cli.dispatch_command
def test_dispatch_command_passes_timeout_ticks(tmp_path: Path) -> None:
    proc_holder: dict[str, _FakeProc] = {}

    def factory(*_args, **_kwargs):
        init = _rpc_message({"jsonrpc": "2.0", "id": 1, "result": {}})
        cmd = _rpc_message({"jsonrpc": "2.0", "id": 2, "result": {"ok": True}})
        shutdown = _rpc_message({"jsonrpc": "2.0", "id": 3, "result": {}})
        proc = _FakeProc(init + cmd + shutdown)
        proc_holder["proc"] = proc
        return proc

    with _env_scope({"GABION_DIRECT_RUN": None}):
        result = cli.dispatch_command(
            command=cli.STRUCTURE_DIFF_COMMAND,
            payload={"baseline": str(tmp_path / "base.json"), "current": str(tmp_path / "current.json")},
            root=tmp_path,
            runner=cli.run_command,
            process_factory=factory,
        )
    assert result == {"ok": True}
    proc = proc_holder["proc"]
    messages = _extract_rpc_messages(proc.stdin.getvalue())
    execute = next(msg for msg in messages if msg.get("method") == "workspace/executeCommand")
    payload = execute["params"]["arguments"][0]
    assert payload["analysis_timeout_ticks"] > 0
    assert payload["analysis_timeout_tick_ns"] > 0


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dispatch_command_preserves_existing_timeout_ms::cli.py::gabion.cli.dispatch_command


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dispatch_command_blocks_direct_transport_for_beta_without_override::cli.py::gabion.cli._resolve_command_transport
def test_dispatch_command_blocks_direct_transport_for_beta_without_override(tmp_path: Path) -> None:
    with _env_scope({"GABION_DIRECT_RUN": "1", "GABION_DIRECT_RUN_OVERRIDE_EVIDENCE": None}):
        with pytest.raises(NeverThrown):
            cli.dispatch_command(
                command=cli.CHECK_COMMAND,
                payload={"paths": [str(tmp_path / "x.py")]},
                root=tmp_path,
                runner=cli.run_command,
            )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dispatch_command_allows_direct_transport_for_debug_maturity::cli.py::gabion.cli._resolve_command_transport
def test_dispatch_command_allows_direct_transport_for_debug_maturity(tmp_path: Path) -> None:
    with _env_scope({"GABION_DIRECT_RUN": "1", "GABION_DIRECT_RUN_OVERRIDE_EVIDENCE": None}):
        result = cli.dispatch_command(
            command=cli.LSP_PARITY_GATE_COMMAND,
            payload={},
            root=tmp_path,
            runner=cli.run_command,
        )
    assert isinstance(result, dict)


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dispatch_command_allows_beta_direct_with_override_evidence::cli.py::gabion.cli._resolve_command_transport
def test_dispatch_command_blocks_beta_direct_with_override_evidence_missing_record(tmp_path: Path) -> None:
    with _env_scope({
        "GABION_DIRECT_RUN": "1",
        "GABION_DIRECT_RUN_OVERRIDE_EVIDENCE": "audit://ci/transport-override/123",
        "GABION_OVERRIDE_RECORD_JSON": None,
    }):
        with pytest.raises(NeverThrown):
            cli.dispatch_command(
                command=cli.CHECK_COMMAND,
                payload={"paths": [str(tmp_path / "x.py")]},
                root=tmp_path,
                runner=cli.run_command,
            )


# gabion:evidence E:function_site::test_cli_helpers.py::tests.test_cli_helpers.test_dispatch_command_blocks_beta_direct_with_expired_override_record
def test_dispatch_command_blocks_beta_direct_with_expired_override_record(tmp_path: Path) -> None:
    with _env_scope({
        "GABION_DIRECT_RUN": "1",
        "GABION_DIRECT_RUN_OVERRIDE_EVIDENCE": "audit://ci/transport-override/123",
        "GABION_OVERRIDE_RECORD_JSON": json.dumps(
            {
                "actor": "ci",
                "rationale": "temporary",
                "scope": "direct_transport",
                "start": "2024-01-01T00:00:00Z",
                "expiry": "2024-01-02T00:00:00Z",
                "rollback_condition": "fix merged",
                "evidence_links": ["artifact://x"],
            }
        ),
    }):
        with pytest.raises(NeverThrown):
            cli.dispatch_command(
                command=cli.CHECK_COMMAND,
                payload={"paths": [str(tmp_path / "x.py")]},
                root=tmp_path,
                runner=cli.run_command,
            )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dispatch_command_allows_beta_direct_with_override_evidence_and_valid_record::cli.py::gabion.cli._resolve_command_transport
def test_dispatch_command_allows_beta_direct_with_override_evidence_and_valid_record(tmp_path: Path) -> None:
    with _env_scope({
        "GABION_DIRECT_RUN": "1",
        "GABION_DIRECT_RUN_OVERRIDE_EVIDENCE": "audit://ci/transport-override/123",
        "GABION_OVERRIDE_RECORD_JSON": json.dumps(
            {
                "actor": "ci",
                "rationale": "temporary",
                "scope": "direct_transport",
                "start": "2024-01-01T00:00:00Z",
                "expiry": "2999-01-02T00:00:00Z",
                "rollback_condition": "fix merged",
                "evidence_links": ["artifact://x"],
            }
        ),
    }):
        result = cli.dispatch_command(
            command=cli.CHECK_COMMAND,
            payload={"paths": [str(tmp_path / "x.py")]},
            root=tmp_path,
            runner=cli.run_command,
        )
    assert isinstance(result, dict)
# gabion:evidence E:function_site::test_cli_helpers.py::tests.test_cli_helpers.test_dispatch_command_preserves_existing_timeout_ms
def test_dispatch_command_preserves_existing_timeout_ms(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(request, *, root=None):
        captured["command"] = request.command
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"ok": True}

    result = cli.dispatch_command(
        command=cli.STRUCTURE_DIFF_COMMAND,
        payload={
            "baseline": str(tmp_path / "base.json"),
            "current": str(tmp_path / "current.json"),
            "analysis_timeout_ms": 1,
        },
        root=tmp_path,
        runner=runner,
    )
    assert result == {"ok": True}
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["analysis_timeout_ms"] == 1
    assert "analysis_timeout_ticks" not in payload


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dispatch_command_handles_signature_introspection_failure::cli.py::gabion.cli.dispatch_command
def test_dispatch_command_handles_signature_introspection_failure(tmp_path: Path) -> None:
    class _Runner:
        __signature__ = "bad-signature"

        def __call__(self, request, *, root=None):
            _ = request
            _ = root
            return {"ok": True}

    runner = _Runner()
    result = cli.dispatch_command(
        command=cli.STRUCTURE_DIFF_COMMAND,
        payload={
            "baseline": str(tmp_path / "base.json"),
            "current": str(tmp_path / "current.json"),
        },
        root=tmp_path,
        runner=runner,
        notification_callback=lambda _payload: None,
    )
    assert result == {"ok": True}


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dispatch_command_rejects_non_mapping_custom_runner_result::cli.py::gabion.cli.dispatch_command
def test_dispatch_command_rejects_non_mapping_custom_runner_result(tmp_path: Path) -> None:
    def runner(_request, *, root=None):
        _ = root
        return []

    with pytest.raises(NeverThrown):
        cli.dispatch_command(
            command=cli.STRUCTURE_DIFF_COMMAND,
            payload={
                "baseline": str(tmp_path / "base.json"),
                "current": str(tmp_path / "current.json"),
            },
            root=tmp_path,
            runner=runner,
        )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dispatch_command_rejects_non_mapping_custom_runner_with_callback::cli.py::gabion.cli.dispatch_command
def test_dispatch_command_rejects_non_mapping_custom_runner_with_callback(
    tmp_path: Path,
) -> None:
    def runner(_request, *, root=None, notification_callback=None):
        _ = root
        assert callable(notification_callback)
        notification_callback({"kind": "progress"})
        return []

    with pytest.raises(NeverThrown):
        cli.dispatch_command(
            command=cli.STRUCTURE_DIFF_COMMAND,
            payload={
                "baseline": str(tmp_path / "base.json"),
                "current": str(tmp_path / "current.json"),
            },
            root=tmp_path,
            runner=runner,
            notification_callback=lambda _payload: None,
        )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dispatch_command_execution_plan_payload_skips_non_mapping_inputs_and_policy_metadata::cli.py::gabion.cli.dispatch_command
def test_dispatch_command_execution_plan_payload_skips_non_mapping_inputs_and_policy_metadata(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class _Plan:
        def to_payload(self) -> dict[str, object]:
            return {
                "inputs": "invalid",
                "policy_metadata": "invalid",
            }

    def runner(request, *, root=None):
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"ok": True}

    result = cli.dispatch_command(
        command=cli.STRUCTURE_DIFF_COMMAND,
        payload={
            "baseline": str(tmp_path / "base.json"),
            "current": str(tmp_path / "current.json"),
        },
        root=tmp_path,
        runner=runner,
        execution_plan_request=_Plan(),  # type: ignore[arg-type]
    )
    assert result == {"ok": True}
    sent_payload = captured["payload"]
    assert isinstance(sent_payload, dict)
    execution_plan_request = sent_payload.get("execution_plan_request")
    assert isinstance(execution_plan_request, dict)
    assert execution_plan_request["inputs"] == "invalid"
    assert execution_plan_request["policy_metadata"] == "invalid"


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.run_structure_reuse::lemma_stubs E:decision_surface/direct::cli.py::gabion.cli.run_structure_reuse::stale_6424f9623b7c
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


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli.run_structure_reuse::lemma_stubs E:decision_surface/direct::cli.py::gabion.cli.run_structure_reuse::stale_9512fb3adc80_4011a505
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


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_impact_query_uses_runner_and_optional_fields::cli.py::gabion.cli.run_impact_query
def test_run_impact_query_uses_runner_and_optional_fields(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(request, *, root=None):
        captured["command"] = request.command
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"exit_code": 0}

    result = cli.run_impact_query(
        changes=["src/app.py:1-3"],
        git_diff="diff --git a/src/app.py b/src/app.py",
        max_call_depth=2,
        confidence_threshold=0.75,
        root=tmp_path,
        runner=runner,
    )
    assert result["exit_code"] == 0
    assert captured["command"] == cli.IMPACT_COMMAND
    assert captured["root"] == tmp_path
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["changes"] == ["src/app.py:1-3"]
    assert payload["git_diff"] == "diff --git a/src/app.py b/src/app.py"
    assert payload["max_call_depth"] == 2
    assert payload["confidence_threshold"] == 0.75
    assert payload["root"] == str(tmp_path)


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_impact_human_output_and_exit::cli.py::gabion.cli._emit_impact
def test_emit_impact_human_output_and_exit(capsys) -> None:
    cli._emit_impact(
        {
            "exit_code": 0,
            "must_run_tests": [{"id": "tests/test_a.py::test_one", "depth": 1, "confidence": 1.0}],
            "likely_run_tests": [{"id": "tests/test_b.py::test_two", "depth": 2, "confidence": 0.4}],
            "impacted_docs": [{"path": "docs/a.md", "section": "Intro", "symbols": ["target"]}],
        },
        json_output=False,
    )
    output = capsys.readouterr().out
    assert "must-run tests:" in output
    assert "tests/test_a.py::test_one" in output
    assert "likely-run tests:" in output
    assert "tests/test_b.py::test_two" in output
    assert "impacted docs sections:" in output
    assert "docs/a.md#Intro [target]" in output

    with pytest.raises(typer.Exit) as excinfo:
        cli._emit_impact(
            {
                "exit_code": 2,
                "errors": ["boom"],
                "must_run_tests": [],
                "likely_run_tests": [],
                "impacted_docs": [],
            },
            json_output=False,
        )
    assert excinfo.value.exit_code == 2
    captured = capsys.readouterr()
    assert "- (none)" in captured.out
    assert "boom" in captured.err


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_synth_outputs_skips_absent_optional_paths::cli.py::gabion.cli._emit_synth_outputs
def test_emit_synth_outputs_skips_absent_optional_paths(tmp_path: Path, capsys) -> None:
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
    cli._emit_synth_outputs(
        paths_out=paths_out,
        timestamp=root,
        refactor_plan=True,
    )
    output = capsys.readouterr().out
    assert "fingerprint_coherence.json" not in output
    assert "fingerprint_rewrite_plans.json" not in output
    assert "fingerprint_exception_obligations.json" not in output
    assert "fingerprint_handledness.json" not in output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_render_timeout_progress_markdown_handles_non_mapping_resume_token::cli.py::gabion.cli._render_timeout_progress_markdown
def test_render_timeout_progress_markdown_handles_non_mapping_resume_token() -> None:
    rendered = cli._render_timeout_progress_markdown(
        analysis_state=None,
        progress={
            "resume": {"resume_token": "bad"},
            "incremental_obligations": [],
        },
    )
    assert "analysis_state" not in rendered
    assert "Resume Token" not in rendered


# gabion:evidence E:function_site::cli.py::gabion.cli._emit_structure_diff E:decision_surface/direct::cli.py::gabion.cli._emit_structure_diff::stale_0d2be4c7ed9c
def test_emit_structure_diff_success(capsys) -> None:
    result = {"exit_code": 0, "diff": {"summary": {"added": 0}}}
    cli._emit_structure_diff(result)
    captured = capsys.readouterr()
    assert "\"exit_code\": 0" in captured.out
    assert captured.err == ""


# gabion:evidence E:function_site::cli.py::gabion.cli._emit_structure_diff E:decision_surface/direct::cli.py::gabion.cli._emit_structure_diff::stale_98a6bf69ab7f_0440a891
def test_emit_structure_diff_errors_exit(capsys) -> None:
    result = {"exit_code": 2, "errors": ["bad snapshot"], "diff": {}}
    with pytest.raises(typer.Exit) as exc:
        cli._emit_structure_diff(result)
    assert exc.value.exit_code == 2
    captured = capsys.readouterr()
    assert "\"exit_code\": 2" in captured.out
    assert "bad snapshot" in captured.err


# gabion:evidence E:function_site::cli.py::gabion.cli._emit_decision_diff E:decision_surface/direct::cli.py::gabion.cli._emit_decision_diff::stale_278718bd685f
def test_emit_decision_diff_success(capsys) -> None:
    result = {"exit_code": 0, "diff": {"summary": {"added": 0}}}
    cli._emit_decision_diff(result)
    captured = capsys.readouterr()
    assert "\"exit_code\": 0" in captured.out


# gabion:evidence E:function_site::cli.py::gabion.cli._emit_decision_diff E:decision_surface/direct::cli.py::gabion.cli._emit_decision_diff::stale_eab18e59dce2_5a66a6e5
def test_emit_decision_diff_errors_exit(capsys) -> None:
    result = {"exit_code": 2, "errors": ["bad decision"], "diff": {}}
    with pytest.raises(typer.Exit) as exc:
        cli._emit_decision_diff(result)
    assert exc.value.exit_code == 2
    captured = capsys.readouterr()
    assert "bad decision" in captured.err


# gabion:evidence E:function_site::cli.py::gabion.cli._emit_structure_reuse E:decision_surface/direct::cli.py::gabion.cli._emit_structure_reuse::stale_907053caf6e8
def test_emit_structure_reuse_success(capsys) -> None:
    result = {"exit_code": 0, "reuse": {"summary": {}}}
    cli._emit_structure_reuse(result)
    captured = capsys.readouterr()
    assert "\"exit_code\": 0" in captured.out


# gabion:evidence E:function_site::cli.py::gabion.cli._emit_structure_reuse E:decision_surface/direct::cli.py::gabion.cli._emit_structure_reuse::stale_6e9d6dd3c001_96b57ce7
def test_emit_structure_reuse_errors_exit(capsys) -> None:
    result = {"exit_code": 2, "errors": ["bad reuse"], "reuse": {}}
    with pytest.raises(typer.Exit) as exc:
        cli._emit_structure_reuse(result)
    assert exc.value.exit_code == 2
    captured = capsys.readouterr()
    assert "bad reuse" in captured.err


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_aspf_state_from_github_artifacts_restores_files::cli.py::gabion.cli._restore_aspf_state_from_github_artifacts
def test_restore_aspf_state_from_github_artifacts_restores_files(
    tmp_path: Path,
) -> None:
    checkpoint_name = "aspf_state_ci.json"
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr(
            f"dataflow-report/{checkpoint_name}",
            '{"completed_paths": ["src/gabion/cli.py"]}',
        )
        zf.writestr(
            f"dataflow-report/{checkpoint_name}.chunks/part-000.json",
            "{}",
        )

    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    payload = {
        "artifacts": [
            {
                "expired": False,
                "archive_download_url": "https://example.invalid/archive.zip",
                "workflow_run": {
                    "id": 101,
                    "head_branch": "stage",
                    "event": "push",
                },
            }
        ]
    }

    def _fake_urlopen(req, timeout=0):
        url = str(getattr(req, "full_url", req))
        if "actions/artifacts" in url:
            return _Resp(json.dumps(payload).encode("utf-8"))
        return _Resp(zip_buf.getvalue())

    exit_code = cli._restore_aspf_state_from_github_artifacts(
        token="token",
        repo="owner/repo",
        output_dir=tmp_path,
        ref_name="stage",
        current_run_id="999",
        urlopen_fn=_fake_urlopen,
    )

    assert exit_code == 0
    assert (tmp_path / checkpoint_name).exists()
    assert (tmp_path / f"{checkpoint_name}.chunks/part-000.json").exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_dataflow_resume_checkpoint_accepts_workflow_dispatch_artifacts::cli.py::gabion.cli._restore_aspf_state_from_github_artifacts
def test_restore_dataflow_resume_checkpoint_accepts_workflow_dispatch_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint_name = "aspf_state_ci.json"
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr(
            f"dataflow-report/{checkpoint_name}",
            '{"completed_paths": ["src/gabion/server.py"]}',
        )

    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    payload = {
        "artifacts": [
            {
                "expired": False,
                "archive_download_url": "https://example.invalid/archive.zip",
                "workflow_run": {
                    "id": 101,
                    "head_branch": "stage",
                    "event": "workflow_dispatch",
                },
            }
        ]
    }

    def _fake_urlopen(req, timeout=0):
        url = str(getattr(req, "full_url", req))
        if "actions/artifacts" in url:
            return _Resp(json.dumps(payload).encode("utf-8"))
        return _Resp(zip_buf.getvalue())

    exit_code = cli._restore_aspf_state_from_github_artifacts(
        token="token",
        repo="owner/repo",
        output_dir=tmp_path,
        ref_name="stage",
        current_run_id="999",
        urlopen_fn=_fake_urlopen,
    )

    assert exit_code == 0
    assert (tmp_path / checkpoint_name).exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_dataflow_resume_checkpoint_accepts_artifacts_with_missing_event::cli.py::gabion.cli._restore_aspf_state_from_github_artifacts
def test_restore_dataflow_resume_checkpoint_accepts_artifacts_with_missing_event(
    tmp_path: Path,
) -> None:
    checkpoint_name = "aspf_state_ci.json"
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr(
            f"dataflow-report/{checkpoint_name}",
            '{"completed_paths": ["src/gabion/server.py"]}',
        )

    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    payload = {
        "artifacts": [
            {
                "expired": False,
                "archive_download_url": "https://example.invalid/archive.zip",
                "workflow_run": {
                    "id": 101,
                    "head_branch": "stage",
                },
            }
        ]
    }

    def _fake_urlopen(req, timeout=0):
        url = str(getattr(req, "full_url", req))
        if "actions/artifacts" in url:
            return _Resp(json.dumps(payload).encode("utf-8"))
        return _Resp(zip_buf.getvalue())

    exit_code = cli._restore_aspf_state_from_github_artifacts(
        token="token",
        repo="owner/repo",
        output_dir=tmp_path,
        ref_name="stage",
        current_run_id="999",
        urlopen_fn=_fake_urlopen,
    )

    assert exit_code == 0
    assert (tmp_path / checkpoint_name).exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_dataflow_resume_checkpoint_falls_back_from_incomplete_chunks::cli.py::gabion.cli._restore_aspf_state_from_github_artifacts
def test_restore_dataflow_resume_checkpoint_falls_back_from_incomplete_chunks(
    tmp_path: Path,
) -> None:
    checkpoint_name = "aspf_state_ci.json"
    state_ref = "chunk-state.json"
    incomplete_zip = io.BytesIO()
    with zipfile.ZipFile(incomplete_zip, "w") as zf:
        zf.writestr(
            f"dataflow-report/{checkpoint_name}",
            json.dumps(
                {
                    "collection_resume": {
                        "analysis_index_resume": {"state_ref": state_ref}
                    }
                }
            ),
        )
    complete_zip = io.BytesIO()
    with zipfile.ZipFile(complete_zip, "w") as zf:
        zf.writestr(
            f"dataflow-report/{checkpoint_name}",
            json.dumps(
                {
                    "collection_resume": {
                        "analysis_index_resume": {"state_ref": state_ref}
                    }
                }
            ),
        )
        zf.writestr(
            f"dataflow-report/{checkpoint_name}.chunks/{state_ref}",
            '{"ok": true}',
        )

    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    payload = {
        "artifacts": [
            {
                "expired": False,
                "archive_download_url": "https://example.invalid/incomplete.zip",
                "workflow_run": {
                    "id": 101,
                    "head_branch": "stage",
                    "event": "push",
                },
            },
            {
                "expired": False,
                "archive_download_url": "https://example.invalid/complete.zip",
                "workflow_run": {
                    "id": 100,
                    "head_branch": "stage",
                    "event": "push",
                },
            },
        ]
    }

    def _fake_urlopen(req, timeout=0):
        _ = timeout
        url = str(getattr(req, "full_url", req))
        if "actions/artifacts" in url:
            return _Resp(json.dumps(payload).encode("utf-8"))
        if "incomplete.zip" in url:
            return _Resp(incomplete_zip.getvalue())
        if "complete.zip" in url:
            return _Resp(complete_zip.getvalue())
        raise AssertionError(f"unexpected url: {url}")

    exit_code = cli._restore_aspf_state_from_github_artifacts(
        token="token",
        repo="owner/repo",
        output_dir=tmp_path,
        ref_name="stage",
        current_run_id="999",
        urlopen_fn=_fake_urlopen,
    )

    assert exit_code == 0
    assert (tmp_path / checkpoint_name).exists()
    assert (tmp_path / f"{checkpoint_name}.chunks/{state_ref}").exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_dataflow_resume_checkpoint_overwrites_existing_output_files::cli.py::gabion.cli._restore_aspf_state_from_github_artifacts
def test_restore_dataflow_resume_checkpoint_overwrites_existing_output_files(
    tmp_path: Path,
) -> None:
    checkpoint_name = "aspf_state_ci.json"
    state_ref = "chunk-state.json"
    archive = io.BytesIO()
    checkpoint_payload = json.dumps(
        {"collection_resume": {"analysis_index_resume": {"state_ref": state_ref}}}
    )
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"dataflow-report/{checkpoint_name}", checkpoint_payload)
        zf.writestr(f"dataflow-report/{checkpoint_name}.chunks/{state_ref}", '{"ok": true}')

    checkpoint_path = tmp_path / checkpoint_name
    checkpoint_path.write_text('{"old": true}', encoding="utf-8")
    chunk_dir = tmp_path / f"{checkpoint_name}.chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    stale_chunk = chunk_dir / "stale.json"
    stale_chunk.write_text("stale", encoding="utf-8")

    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    payload = {
        "artifacts": [
            {
                "expired": False,
                "archive_download_url": "https://example.invalid/archive.zip",
                "workflow_run": {
                    "id": 101,
                    "head_branch": "stage",
                    "event": "push",
                },
            }
        ]
    }

    def _fake_urlopen(req, timeout=0):
        _ = timeout
        url = str(getattr(req, "full_url", req))
        if "actions/artifacts" in url:
            return _Resp(json.dumps(payload).encode("utf-8"))
        return _Resp(archive.getvalue())

    exit_code = cli._restore_aspf_state_from_github_artifacts(
        token="token",
        repo="owner/repo",
        output_dir=tmp_path,
        ref_name="stage",
        current_run_id="999",
        urlopen_fn=_fake_urlopen,
    )

    assert exit_code == 0
    assert checkpoint_path.read_text(encoding="utf-8") == checkpoint_payload
    assert not stale_chunk.exists()
    assert (chunk_dir / state_ref).exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_dataflow_resume_checkpoint_ignores_non_chunk_members_and_preserves_non_file_chunk_entries::cli.py::gabion.cli._restore_aspf_state_from_github_artifacts
def test_restore_dataflow_resume_checkpoint_ignores_non_chunk_members_and_preserves_non_file_chunk_entries(
    tmp_path: Path,
) -> None:
    checkpoint_name = "aspf_state_ci.json"
    archive = io.BytesIO()
    checkpoint_payload = json.dumps(
        {"collection_resume": {"analysis_index_resume": {"state_ref": "chunk.json"}}}
    )
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"dataflow-report/{checkpoint_name}", checkpoint_payload)
        zf.writestr(f"dataflow-report/{checkpoint_name}.chunks/chunk.json", '{"ok": true}')
        zf.writestr("dataflow-report/notes.txt", "ignore")

    chunk_dir = tmp_path / f"{checkpoint_name}.chunks"
    nested_dir = chunk_dir / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "stale.json").write_text("stale", encoding="utf-8")

    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    payload = {
        "artifacts": [
            {
                "expired": False,
                "archive_download_url": "https://example.invalid/archive.zip",
                "workflow_run": {
                    "id": 101,
                    "head_branch": "stage",
                    "event": "push",
                },
            }
        ]
    }

    def _fake_urlopen(req, timeout=0):
        _ = timeout
        url = str(getattr(req, "full_url", req))
        if "actions/artifacts" in url:
            return _Resp(json.dumps(payload).encode("utf-8"))
        return _Resp(archive.getvalue())

    exit_code = cli._restore_aspf_state_from_github_artifacts(
        token="token",
        repo="owner/repo",
        output_dir=tmp_path,
        ref_name="stage",
        current_run_id="999",
        urlopen_fn=_fake_urlopen,
    )

    assert exit_code == 0
    assert (tmp_path / checkpoint_name).exists()
    assert (chunk_dir / "chunk.json").exists()
    assert nested_dir.exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_download_artifact_archive_bytes_follows_blob_redirect_without_auth::cli.py::gabion.cli._download_artifact_archive_bytes
def test_download_artifact_archive_bytes_follows_blob_redirect_without_auth() -> None:
    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def _no_redirect_open(_req, timeout=0):
        _ = timeout
        raise urllib.error.HTTPError(
            url="https://api.github.com/repos/owner/repo/actions/artifacts/1/zip",
            code=302,
            msg="Found",
            hdrs={
                "Location": (
                    "https://productionresultssa15.blob.core.windows.net/"
                    "actions-results/example.zip?sig=abc"
                )
            },
            fp=None,
        )

    seen_headers: list[dict[str, str]] = []

    def _follow_redirect(req, timeout=0):
        _ = timeout
        seen_headers.append({str(key): str(value) for key, value in req.headers.items()})
        return _Resp(b"zip-bytes")

    archive_bytes = cli._download_artifact_archive_bytes(
        download_url="https://api.github.com/repos/owner/repo/actions/artifacts/1/zip",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer token",
        },
        no_redirect_open_fn=_no_redirect_open,
        follow_redirect_open_fn=_follow_redirect,
    )

    assert archive_bytes == b"zip-bytes"
    assert seen_headers
    lowered = {key.lower(): value for key, value in seen_headers[0].items()}
    assert "authorization" not in lowered


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_download_artifact_archive_bytes_keeps_auth_for_github_redirect::cli.py::gabion.cli._download_artifact_archive_bytes
def test_download_artifact_archive_bytes_keeps_auth_for_github_redirect() -> None:
    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def _no_redirect_open(_req, timeout=0):
        _ = timeout
        raise urllib.error.HTTPError(
            url="https://api.github.com/repos/owner/repo/actions/artifacts/1/zip",
            code=302,
            msg="Found",
            hdrs={"Location": "https://api.github.com/repos/owner/repo/actions/artifacts/1/zip?retry=1"},
            fp=None,
        )

    seen_headers: list[dict[str, str]] = []

    def _follow_redirect(req, timeout=0):
        _ = timeout
        seen_headers.append({str(key): str(value) for key, value in req.headers.items()})
        return _Resp(b"zip-bytes")

    archive_bytes = cli._download_artifact_archive_bytes(
        download_url="https://api.github.com/repos/owner/repo/actions/artifacts/1/zip",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer token",
        },
        no_redirect_open_fn=_no_redirect_open,
        follow_redirect_open_fn=_follow_redirect,
    )

    assert archive_bytes == b"zip-bytes"
    assert seen_headers
    lowered = {key.lower(): value for key, value in seen_headers[0].items()}
    assert lowered.get("authorization") == "Bearer token"


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_download_artifact_archive_bytes_default_no_redirect_path_uses_data_url::cli.py::gabion.cli._download_artifact_archive_bytes
def test_download_artifact_archive_bytes_default_no_redirect_path_uses_data_url() -> None:
    archive_bytes = cli._download_artifact_archive_bytes(
        download_url="data:text/plain;base64,YXJjaGl2ZS1ieXRlcw==",
        headers={"Accept": "application/vnd.github+json"},
    )
    assert archive_bytes == b"archive-bytes"


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_no_redirect_handler_redirect_request_returns_none::cli.py::gabion.cli._NoRedirectHandler
def test_no_redirect_handler_redirect_request_returns_none() -> None:
    handler = cli._NoRedirectHandler()
    request = urllib.request.Request("https://example.invalid/archive.zip")
    assert (
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {"Location": "https://example.invalid/redirect"},
            "https://example.invalid/redirect",
        )
        is None
    )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_download_artifact_archive_bytes_raises_when_redirect_location_missing::cli.py::gabion.cli._download_artifact_archive_bytes
def test_download_artifact_archive_bytes_raises_when_redirect_location_missing() -> None:
    def _no_redirect_open(_req, timeout=0):
        _ = timeout
        raise urllib.error.HTTPError(
            url="https://api.github.com/repos/owner/repo/actions/artifacts/1/zip",
            code=302,
            msg="Found",
            hdrs={},
            fp=None,
        )

    with pytest.raises(urllib.error.HTTPError):
        cli._download_artifact_archive_bytes(
            download_url="https://api.github.com/repos/owner/repo/actions/artifacts/1/zip",
            headers={"Accept": "application/vnd.github+json"},
            no_redirect_open_fn=_no_redirect_open,
        )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_state_requires_chunk_artifacts_invalid_payload_shapes::cli.py::gabion.cli._state_requires_chunk_artifacts
def test_state_requires_chunk_artifacts_invalid_payload_shapes() -> None:
    assert cli._state_requires_chunk_artifacts(checkpoint_bytes=b"{not-json") is False
    assert cli._state_requires_chunk_artifacts(checkpoint_bytes=b"[]") is False
    assert (
        cli._state_requires_chunk_artifacts(
            checkpoint_bytes=json.dumps({"collection_resume": {}}).encode("utf-8")
        )
        is False
    )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_resume_checkpoint_command_removed::cli.py::gabion.cli.app
def test_restore_resume_checkpoint_command_removed() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["restore-resume-checkpoint"])
    assert result.exit_code != 0
    assert "No such command 'restore-resume-checkpoint'" in result.output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_derived_artifacts_includes_all_optional_outputs::cli.py::gabion.cli._check_derived_artifacts
def test_check_derived_artifacts_includes_all_optional_outputs() -> None:
    derived = cli._check_derived_artifacts(
        report=Path("artifacts/audit_reports/dataflow_report.md"),
        decision_snapshot=Path("artifacts/out/decision_snapshot.json"),
        artifact_flags=cli.CheckArtifactFlags(
            emit_test_obsolescence=True,
            emit_test_evidence_suggestions=True,
            emit_call_clusters=True,
            emit_call_cluster_consolidation=True,
            emit_test_annotation_drift=True,
            emit_semantic_coverage_map=True,
        ),
        emit_test_obsolescence_state=True,
        emit_test_obsolescence_delta=True,
        emit_test_annotation_drift_delta=True,
        emit_ambiguity_delta=True,
        emit_ambiguity_state=True,
        aspf_trace_json=None,
        aspf_opportunities_json=None,
        aspf_state_json=None,
        aspf_delta_jsonl=None,
        aspf_equivalence_enabled=False,
    )
    assert "artifacts/out/decision_snapshot.json" in derived
    assert "artifacts/out/test_obsolescence_report.json" in derived
    assert "artifacts/out/test_obsolescence_state.json" in derived
    assert "artifacts/out/test_obsolescence_delta.json" in derived
    assert "artifacts/out/test_evidence_suggestions.json" in derived
    assert "artifacts/out/call_clusters.json" in derived
    assert "artifacts/out/call_cluster_consolidation.json" in derived
    assert "artifacts/out/test_annotation_drift.json" in derived
    assert "artifacts/out/semantic_coverage_map.json" in derived
    assert "artifacts/out/test_annotation_drift_delta.json" in derived
    assert "artifacts/out/ambiguity_delta.json" in derived
    assert "artifacts/out/ambiguity_state.json" in derived


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_nonzero_exit_causes_formats_timeout_ambiguity_and_errors::cli.py::gabion.cli._nonzero_exit_causes
def test_nonzero_exit_causes_formats_timeout_ambiguity_and_errors() -> None:
    causes = cli._nonzero_exit_causes(
        {
            "timeout": True,
            "analysis_state": "timed_out",
            "type_ambiguities": ["a", "b"],
            "errors": ["first", "second"],
        }
    )
    assert "timeout (analysis_state=timed_out)" in causes
    assert "type ambiguities=2" in causes
    assert "errors=2 (first: first)" in causes

    causes_single = cli._nonzero_exit_causes({"errors": ["only"]})
    assert "error: only" in causes_single


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_dataflow_raw_argv_rejects_removed_resume_checkpoint_flag_once::cli.py::gabion.cli._run_dataflow_raw_argv
def test_run_dataflow_raw_argv_rejects_removed_resume_checkpoint_flag_once(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "resume.json"
    module_path = tmp_path / "sample.py"
    module_path.write_text("def sample():\n    return 1\n", encoding="utf-8")

    def _fake_runner(_request, *, root=None, notification_callback=None):
        _ = root
        assert callable(notification_callback)
        payload = {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "resume_checkpoint": {
                        "state_path": str(checkpoint_path),
                        "status": "checkpoint_loaded",
                        "reused_files": 2,
                        "total_files": 5,
                    }
                }
            },
        }
        notification_callback(payload)
        notification_callback(payload)
        return {"exit_code": 0}

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
                str(module_path),
                "--root",
                str(tmp_path),
                "--resume-checkpoint",
                str(checkpoint_path),
            ],
            runner=_fake_runner,
        )
    assert exc.value.exit_code == 2


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_rejects_removed_resume_checkpoint_flag::cli.py::gabion.cli.app
def test_check_rejects_removed_resume_checkpoint_flag(
    tmp_path: Path,
) -> None:
    def _fake_run_check(**kwargs):
        callback = kwargs.get("notification_callback")
        assert callable(callback)
        payload = {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "resume_checkpoint": {
                        "state_path": "resume.json",
                        "status": "checkpoint_loaded",
                        "reused_files": 1,
                        "total_files": 2,
                    }
                },
            },
        }
        callback(payload)
        callback(payload)
        return {"exit_code": 0}

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "check",
            "run",
            "sample.py",
            "--root",
            str(tmp_path),
            "--resume-checkpoint",
                str(tmp_path / "resume.json"),
                "--gate",
                "none",
            ],
        obj={
            "run_check": _fake_run_check,
            "run_with_timeout_retries": lambda run_once, **_kwargs: run_once(),
        },
    )
    assert result.exit_code != 0
    normalized_output = _strip_ansi(result.output)
    assert "No such option" in normalized_output
    assert "--resume-checkpoint" in normalized_output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_emits_checkpoint_intro_timeline_header_once::cli.py::gabion.cli.app
def test_check_emits_checkpoint_intro_timeline_header_once(
    tmp_path: Path,
) -> None:
    def _fake_run_check(**kwargs):
        callback = kwargs.get("notification_callback")
        assert callable(callback)
        callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "collection",
                        "phase_timeline_header": "| ts | phase |",
                        "phase_timeline_row": "| t0 | collection |",
                    },
                },
            }
        )
        callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "forest",
                        "phase_timeline_header": "| ts | phase |",
                        "phase_timeline_row": "| t1 | forest |",
                    },
                },
            }
        )
        return {"exit_code": 0}

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "check",
            "run",
            "sample.py",
            "--root",
            str(tmp_path),
            "--gate",
            "none",
        ],
        obj={
            "run_check": _fake_run_check,
            "run_with_timeout_retries": lambda run_once, **_kwargs: run_once(),
        },
    )
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines.count("| ts | phase |") == 1
    assert "| t0 | collection |" in lines
    assert "| t1 | forest |" in lines


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_dedupes_duplicate_event_seq::cli.py::gabion.cli.app
def test_check_dedupes_duplicate_event_seq(
    tmp_path: Path,
) -> None:
    def _fake_run_check(**kwargs):
        callback = kwargs.get("notification_callback")
        assert callable(callback)
        callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "collection",
                        "event_seq": 21,
                        "phase_timeline_header": "| ts | phase |",
                        "phase_timeline_row": "| t0 | collection |",
                    },
                },
            }
        )
        callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "forest",
                        "event_seq": 21,
                        "phase_timeline_header": "| ts | phase |",
                        "phase_timeline_row": "| t1 | forest |",
                    },
                },
            }
        )
        return {"exit_code": 0}

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "check",
            "run",
            "sample.py",
            "--root",
            str(tmp_path),
            "--gate",
            "none",
        ],
        obj={
            "run_check": _fake_run_check,
            "run_with_timeout_retries": lambda run_once, **_kwargs: run_once(),
        },
    )
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert "| ts | phase |" in lines
    assert "| t0 | collection |" in lines
    assert "| t1 | forest |" not in lines


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_ignores_empty_checkpoint_intro_timeline_row::cli.py::gabion.cli.app
def test_check_ignores_empty_checkpoint_intro_timeline_row(
    tmp_path: Path,
) -> None:
    def _fake_run_check(**kwargs):
        callback = kwargs.get("notification_callback")
        assert callable(callback)
        callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "checkpoint_intro_timeline_header": "| ts | done |",
                        "checkpoint_intro_timeline_row": "",
                    },
                },
            }
        )
        return {"exit_code": 0}

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "check",
            "run",
            "sample.py",
            "--root",
            str(tmp_path),
            "--gate",
            "none",
        ],
        obj={
            "run_check": _fake_run_check,
            "run_with_timeout_retries": lambda run_once, **_kwargs: run_once(),
        },
    )
    assert result.exit_code == 0
    assert "| ts | done |" not in result.stdout


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_emits_non_collection_phase_progress_lines::cli.py::gabion.cli.app
def test_check_emits_non_collection_phase_progress_lines(
    tmp_path: Path,
) -> None:
    def _fake_run_check(**kwargs):
        callback = kwargs.get("notification_callback")
        assert callable(callback)
        callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "edge",
                        "work_done": 2,
                        "work_total": 4,
                        "completed_files": 10,
                        "remaining_files": 0,
                        "total_files": 10,
                        "analysis_state": "analysis_edge_in_progress",
                    },
                },
            }
        )
        callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "edge",
                        "work_done": 2,
                        "work_total": 4,
                        "completed_files": 10,
                        "remaining_files": 0,
                        "total_files": 10,
                        "analysis_state": "analysis_edge_in_progress",
                    },
                },
            }
        )
        callback(
            {
                "method": "$/progress",
                "params": {
                    "token": "gabion.dataflowAudit/progress-v1",
                    "value": {
                        "phase": "post",
                        "work_done": 4,
                        "work_total": 4,
                        "completed_files": 10,
                        "remaining_files": 0,
                        "total_files": 10,
                        "analysis_state": "succeeded",
                        "classification": "succeeded",
                        "done": True,
                    },
                },
            }
        )
        return {"exit_code": 0}

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "check",
            "run",
            "sample.py",
            "--root",
            str(tmp_path),
            "--gate",
            "none",
        ],
        obj={
            "run_check": _fake_run_check,
            "run_with_timeout_retries": lambda run_once, **_kwargs: run_once(),
        },
    )
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    table_rows = [line for line in lines if line.startswith("| ") and not line.startswith("| ---")]
    assert sum(1 for line in table_rows if "| edge |" in line) == 1
    assert any("| post |" in line for line in table_rows)


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_docflow_audit_returns_one_when_sppf_graph_fails::cli.py::gabion.cli._run_docflow_audit
def test_run_docflow_audit_returns_one_when_sppf_graph_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    orig_docflow = cli.tooling_governance_audit.run_docflow_cli
    orig_sppf = cli.tooling_governance_audit.run_sppf_graph_cli

    try:
        cli.tooling_governance_audit.run_docflow_cli = lambda _argv=None: 0
        cli.tooling_governance_audit.run_sppf_graph_cli = (
            lambda _argv=None: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        exit_code = cli._run_docflow_audit(
            root=tmp_path,
            fail_on_violations=False,
            sppf_gh_ref_mode="required",
        )
    finally:
        cli.tooling_governance_audit.run_docflow_cli = orig_docflow
        cli.tooling_governance_audit.run_sppf_graph_cli = orig_sppf

    assert exit_code == 1
    assert "docflow: sppf-graph failed: boom" in capsys.readouterr().err


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_resume_checkpoint_handles_guard_and_error_branches::cli.py::gabion.cli._restore_aspf_state_from_github_artifacts
def test_restore_resume_checkpoint_handles_guard_and_error_branches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        cli._restore_aspf_state_from_github_artifacts(
            token="",
            repo="",
            output_dir=tmp_path,
        )
        == 0
    )
    assert "token/repository unavailable" in capsys.readouterr().out

    def _raise_urlopen(_req, timeout=0):
        raise RuntimeError("query boom")

    assert (
        cli._restore_aspf_state_from_github_artifacts(
            token="token",
            repo="owner/repo",
            output_dir=tmp_path,
            urlopen_fn=_raise_urlopen,
        )
        == 0
    )

    payload = {
        "artifacts": [
            1,
            {"expired": True, "archive_download_url": "u", "workflow_run": {"event": "push"}},
            {"expired": False, "archive_download_url": "u", "workflow_run": "bad"},
            {
                "expired": False,
                "archive_download_url": "u",
                "workflow_run": {"id": "123", "head_branch": "main", "event": "push"},
            },
            {
                "expired": False,
                "archive_download_url": "u",
                "workflow_run": {"id": "200", "head_branch": "other", "event": "push"},
            },
            {
                "expired": False,
                "archive_download_url": "u",
                "workflow_run": {"id": "201", "head_branch": "main", "event": "pull_request"},
            },
        ]
    }

    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def _query_only(_req, timeout=0):
        return _Resp(json.dumps(payload).encode("utf-8"))

    assert (
        cli._restore_aspf_state_from_github_artifacts(
            token="token",
            repo="owner/repo",
            output_dir=tmp_path,
            ref_name="main",
            current_run_id="123",
            urlopen_fn=_query_only,
        )
        == 0
    )

    payload_download = {
        "artifacts": [
            {
                "expired": False,
                "archive_download_url": "https://example.invalid/archive.zip",
                "workflow_run": {"id": "100", "head_branch": "main", "event": "push"},
            }
        ]
    }

    def _download_fails(req, timeout=0):
        url = str(getattr(req, "full_url", req))
        if "actions/artifacts" in url:
            return _Resp(json.dumps(payload_download).encode("utf-8"))
        raise RuntimeError("download boom")

    assert (
        cli._restore_aspf_state_from_github_artifacts(
            token="token",
            repo="owner/repo",
            output_dir=tmp_path,
            ref_name="main",
            urlopen_fn=_download_fails,
        )
        == 0
    )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("dataflow-report/aspf_state_ci.json.chunks/", "")

    def _zip_with_only_directory(req, timeout=0):
        url = str(getattr(req, "full_url", req))
        if "actions/artifacts" in url:
            return _Resp(json.dumps(payload_download).encode("utf-8"))
        return _Resp(zip_buf.getvalue())

    assert (
        cli._restore_aspf_state_from_github_artifacts(
            token="token",
            repo="owner/repo",
            output_dir=tmp_path,
            ref_name="main",
            urlopen_fn=_zip_with_only_directory,
        )
        == 0
    )

    def _invalid_zip(req, timeout=0):
        url = str(getattr(req, "full_url", req))
        if "actions/artifacts" in url:
            return _Resp(json.dumps(payload_download).encode("utf-8"))
        return _Resp(b"not-a-zip")

    assert (
        cli._restore_aspf_state_from_github_artifacts(
            token="token",
            repo="owner/repo",
            output_dir=tmp_path,
            ref_name="main",
            urlopen_fn=_invalid_zip,
        )
        == 0
    )

# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_sppf_sync_label_only_branch::cli.py::gabion.cli._run_sppf_sync
def test_run_sppf_sync_label_only_branch() -> None:
    calls: list[list[str]] = []
    commits = [
        cli.SppfSyncCommitInfo(
            sha="abc",
            subject="Link #12",
            body="Refs #12",
        )
    ]
    exit_code = cli._run_sppf_sync(
        rev_range="HEAD~1..HEAD",
        comment=False,
        close=False,
        label="sppf",
        dry_run=False,
        default_rev_range_fn=lambda: "HEAD~1..HEAD",
        collect_sppf_commits_fn=lambda _range: commits,
        run_sppf_gh_fn=lambda args: calls.append(args),
    )
    assert exit_code == 0
    assert calls == [["issue", "edit", "12", "--add-label", "sppf"]]


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_sppf_sync_command_handles_runner_errors::cli.py::gabion.cli.app
def test_sppf_sync_command_handles_runner_errors() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["sppf-sync", "--range", "__not_a_rev_range__"])
    assert result.exit_code == 2


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_governance_commands_include_optional_cli_args::cli.py::gabion.cli.app
def test_governance_commands_include_optional_cli_args(tmp_path: Path) -> None:
    calls: list[tuple[str, list[str]]] = []
    orig_sppf = cli.tooling_governance_audit.run_sppf_graph_cli
    orig_status = cli.tooling_governance_audit.run_status_consistency_cli
    orig_tiers = cli.tooling_governance_audit.run_decision_tiers_cli
    orig_consolidation = cli.tooling_governance_audit.run_consolidation_cli
    orig_lint = cli.tooling_governance_audit.run_lint_summary_cli

    def _capture(name: str):
        def _runner(argv: list[str] | None = None) -> int:
            calls.append((name, list(argv or [])))
            return 0
        return _runner

    try:
        cli.tooling_governance_audit.run_sppf_graph_cli = _capture("sppf")
        cli.tooling_governance_audit.run_status_consistency_cli = _capture("status")
        cli.tooling_governance_audit.run_decision_tiers_cli = _capture("tiers")
        cli.tooling_governance_audit.run_consolidation_cli = _capture("consolidation")
        cli.tooling_governance_audit.run_lint_summary_cli = _capture("lint")

        runner = CliRunner()
        assert runner.invoke(
            cli.app,
            [
                "sppf-graph",
                "--root",
                str(tmp_path),
                "--json-output",
                str(tmp_path / "graph.json"),
                "--dot-output",
                str(tmp_path / "graph.dot"),
                "--issues-json",
                str(tmp_path / "issues.json"),
            ],
        ).exit_code == 0
        assert runner.invoke(
            cli.app,
            [
                "status-consistency",
                "--root",
                str(tmp_path),
                "--extra-path",
                "in",
                "--fail-on-violations",
            ],
        ).exit_code == 0
        assert runner.invoke(
            cli.app,
            ["decision-tiers", "--root", str(tmp_path), "--lint", str(tmp_path / "lint.txt"), "--format", "lines"],
        ).exit_code == 0
        assert runner.invoke(
            cli.app,
            [
                "consolidation",
                "--root",
                str(tmp_path),
                "--decision",
                str(tmp_path / "decision.json"),
                "--lint",
                str(tmp_path / "lint.txt"),
                "--output",
                str(tmp_path / "report.md"),
                "--json-output",
                str(tmp_path / "report.json"),
            ],
        ).exit_code == 0
        assert runner.invoke(
            cli.app,
            ["lint-summary", "--root", str(tmp_path), "--lint", str(tmp_path / "lint.txt"), "--json", "--top", "7"],
        ).exit_code == 0
        assert runner.invoke(
            cli.app,
            ["decision-tiers", "--root", str(tmp_path)],
        ).exit_code == 0
        assert runner.invoke(
            cli.app,
            ["consolidation", "--root", str(tmp_path)],
        ).exit_code == 0
        assert runner.invoke(
            cli.app,
            ["lint-summary", "--root", str(tmp_path)],
        ).exit_code == 0
    finally:
        cli.tooling_governance_audit.run_sppf_graph_cli = orig_sppf
        cli.tooling_governance_audit.run_status_consistency_cli = orig_status
        cli.tooling_governance_audit.run_decision_tiers_cli = orig_tiers
        cli.tooling_governance_audit.run_consolidation_cli = orig_consolidation
        cli.tooling_governance_audit.run_lint_summary_cli = orig_lint

    assert any(name == "sppf" and "--dot-output" in argv and "--issues-json" in argv for name, argv in calls)
    assert any(name == "status" and "--extra-path" in argv and "--fail-on-violations" in argv for name, argv in calls)
    assert any(name == "tiers" and "--format" in argv and "lines" in argv for name, argv in calls)
    assert any(name == "tiers" and "--lint" not in argv for name, argv in calls)
    assert any(name == "consolidation" and "--json-output" in argv for name, argv in calls)
    assert any(
        name == "consolidation"
        and "--decision" not in argv
        and "--lint" not in argv
        and "--output" not in argv
        and "--json-output" not in argv
        for name, argv in calls
    )
    assert any(name == "lint" and "--json" in argv and "--top" in argv for name, argv in calls)
    assert any(name == "lint" and "--lint" not in argv and "--json" not in argv for name, argv in calls)
