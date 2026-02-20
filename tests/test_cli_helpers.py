from __future__ import annotations

from pathlib import Path
import io
import json
import re
import subprocess
import sys
import types
import urllib.error
import zipfile

import pytest
import typer
from typer.testing import CliRunner

from gabion import cli
from gabion.analysis.timeout_context import check_deadline
from tests.env_helpers import env_scope as _env_scope


_ANSI_ESCAPE = re.compile(r"\x1B\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._split_csv_entries::entries E:decision_surface/direct::cli.py::gabion.cli._split_csv::value
def test_split_csv_helpers() -> None:
    assert cli._split_csv_entries(["a, b", " ", "c"]) == ["a", "b", "c"]
    assert cli._split_csv_entries([" ", ""]) == []

    assert cli._split_csv("a, , b") == ["a", "b"]
    assert cli._split_csv(" ,") == []


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
    result = runner.invoke(cli.app, ["check", "sample.py", "--dot", "-"])
    assert result.exit_code != 0
    assert "Unknown arguments for strict profile" in result.output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_rejects_unknown_profile::cli.py::gabion.cli.app
def test_check_rejects_unknown_profile() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["check", "sample.py", "--profile", "mystery"])
    assert result.exit_code != 0
    assert "profile must be 'strict' or 'raw'" in result.output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_raw_profile_delegates_with_profile_defaults::cli.py::gabion.cli.app
def test_check_raw_profile_delegates_with_profile_defaults(
) -> None:
    captured: dict[str, object] = {}

    def _fake_run(argv: list[str]) -> None:
        captured["argv"] = list(argv)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["check", "--profile", "raw", "sample.py"],
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
            "--profile",
            "raw",
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
        "--dot",
        "-",
        "--type-audit",
        "--root",
        str(tmp_path),
        "--config",
        "cfg.toml",
        "--report",
        "report.md",
        "--emit-decision-snapshot",
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
            "--profile",
            "raw",
            "sample.py",
            "--no-allow-external",
            "--no-fail-on-violations",
            "--no-fail-on-type-ambiguities",
        ],
        obj={"run_dataflow_raw_argv": _fake_run},
    )
    assert result.exit_code == 0
    assert captured["argv"] == ["sample.py", "--no-allow-external"]


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_raw_profile_rejects_check_only_flags::test_cli_helpers.py::tests.test_cli_helpers._strip_ansi
def test_check_raw_profile_rejects_check_only_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["check", "--profile", "raw", "sample.py", "--emit-test-obsolescence"],
    )
    normalized_output = _strip_ansi(result.output)
    assert result.exit_code != 0
    assert "check-only options" in normalized_output
    assert "--emit-test-obsolescence" in normalized_output


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


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_context_dependency_helpers_ignore_noncallables::cli.py::gabion.cli._context_run_dataflow_raw_argv
def test_context_dependency_helpers_ignore_noncallables() -> None:
    class DummyCtx:
        obj = {
            "run_dataflow_raw_argv": "not-callable",
        }

    ctx = DummyCtx()
    assert cli._context_run_dataflow_raw_argv(ctx) is cli._run_dataflow_raw_argv


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


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_write_lint_sarif_reuses_existing_rule::cli.py::gabion.cli._write_lint_sarif
def test_write_lint_sarif_reuses_existing_rule(tmp_path: Path) -> None:
    entries = [
        {"path": "mod.py", "line": 1, "col": 1, "code": "GABION_CODE", "message": "m1"},
        {"path": "mod.py", "line": 2, "col": 1, "code": "GABION_CODE", "message": "m2"},
    ]
    sarif_path = tmp_path / "lint.sarif"
    cli._write_lint_sarif(str(sarif_path), entries)
    payload = json.loads(sarif_path.read_text(encoding="utf-8"))
    rules = payload["runs"][0]["tool"]["driver"]["rules"]
    assert len(rules) == 1


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
        compatibility_shim_warnings=True,
        compatibility_shim_overloads=True,
        ambient_rewrite=False,
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


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._run_docflow_audit::fail_on_violations
def test_run_docflow_audit_missing_script(tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"
    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=False,
        sppf_gh_ref_mode="required",
        audit_tools_path=missing,
    )
    assert exit_code == 2


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._run_docflow_audit::fail_on_violations
def test_run_docflow_audit_passes_flags(tmp_path: Path) -> None:
    module_path = tmp_path / "audit_tools.py"
    calls_path = tmp_path / "docflow_calls.txt"
    module_path.write_text(
        f"calls_path = {str(calls_path)!r}\n"
        "def _record(label, argv):\n"
        "    with open(calls_path, 'a', encoding='utf-8') as handle:\n"
        "        handle.write(label + ':' + '|'.join(argv or []) + '\\n')\n"
        "def run_docflow_cli(argv=None):\n"
        "    _record('docflow', argv)\n"
        "    return 0\n"
        "def run_sppf_graph_cli(argv=None):\n"
        "    _record('sppf', argv)\n"
        "    return 0\n"
    )
    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=True,
        sppf_gh_ref_mode="required",
        audit_tools_path=module_path,
    )
    assert exit_code == 0
    lines = calls_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"docflow:--root|{tmp_path}|--fail-on-violations|--sppf-gh-ref-mode|required"
    assert lines[1] == "sppf:"


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_docflow_audit_cleans_import_state::cli.py::gabion.cli._run_docflow_audit
def test_run_docflow_audit_cleans_import_state(tmp_path: Path) -> None:
    module_name = "gabion_repo_audit_tools"
    module_path = tmp_path / "audit_tools.py"
    module_path.write_text(
        "def run_docflow_cli(argv=None):\n"
        "    return 0\n"
        "def run_sppf_graph_cli(argv=None):\n"
        "    return 0\n"
    )
    scripts_root = str(tmp_path)
    sys_path_list: list[str] = []
    sys_modules_map: dict[str, object] = {}
    assert scripts_root not in sys_path_list
    assert module_name not in sys_modules_map

    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=False,
        sppf_gh_ref_mode="required",
        audit_tools_path=module_path,
        sys_path_list=sys_path_list,
        sys_modules_map=sys_modules_map,
    )

    assert exit_code == 0
    assert scripts_root not in sys_path_list
    assert module_name not in sys_modules_map


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_docflow_audit_restores_previous_module::cli.py::gabion.cli._run_docflow_audit
def test_run_docflow_audit_restores_previous_module(tmp_path: Path) -> None:
    module_name = "gabion_repo_audit_tools"
    module_path = tmp_path / "audit_tools.py"
    module_path.write_text(
        "def run_docflow_cli(argv=None):\n"
        "    return 0\n"
        "def run_sppf_graph_cli(argv=None):\n"
        "    return 0\n"
    )
    previous_module = types.ModuleType(module_name)
    sys_modules_map: dict[str, object] = {module_name: previous_module}
    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=False,
        sppf_gh_ref_mode="required",
        audit_tools_path=module_path,
        sys_modules_map=sys_modules_map,
    )
    assert exit_code == 0
    assert sys_modules_map.get(module_name) is previous_module


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_docflow_audit_cleanup_ignores_missing_sys_path::cli.py::gabion.cli._run_docflow_audit
def test_run_docflow_audit_cleanup_ignores_missing_sys_path(tmp_path: Path) -> None:
    module_path = tmp_path / "audit_tools.py"
    module_path.write_text(
        "def run_docflow_cli(argv=None):\n"
        "    return 0\n"
        "def run_sppf_graph_cli(argv=None):\n"
        "    return 0\n"
    )
    scripts_root = str(tmp_path)
    class _PathList(list[str]):
        def remove(self, value: str) -> None:  # pragma: no cover - exercised in test
            super().remove(value)
            raise ValueError("simulated missing sys.path entry")

    sys_path_list: list[str] = _PathList()
    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=False,
        sppf_gh_ref_mode="required",
        audit_tools_path=module_path,
        sys_path_list=sys_path_list,
    )
    assert exit_code == 0
    assert scripts_root not in sys_path_list


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_docflow_audit_keeps_existing_sys_path_entry::cli.py::gabion.cli._run_docflow_audit
def test_run_docflow_audit_keeps_existing_sys_path_entry(tmp_path: Path) -> None:
    module_path = tmp_path / "audit_tools.py"
    module_path.write_text(
        "def run_docflow_cli(argv=None):\n"
        "    return 0\n"
        "def run_sppf_graph_cli(argv=None):\n"
        "    return 0\n"
    )
    scripts_root = str(tmp_path)
    sys_path_list = [scripts_root]
    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=False,
        sppf_gh_ref_mode="required",
        audit_tools_path=module_path,
        sys_path_list=sys_path_list,
    )
    assert exit_code == 0
    assert sys_path_list == [scripts_root]


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_docflow_audit_returns_one_when_sppf_graph_fails::cli.py::gabion.cli._run_docflow_audit
def test_run_docflow_audit_returns_one_when_sppf_graph_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module_path = tmp_path / "audit_tools.py"
    module_path.write_text(
        "def run_docflow_cli(argv=None):\n"
        "    return 0\n"
        "def run_sppf_graph_cli(argv=None):\n"
        "    raise RuntimeError('boom')\n"
    )
    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=False,
        sppf_gh_ref_mode="required",
        audit_tools_path=module_path,
    )
    assert exit_code == 1
    assert "docflow: sppf-graph failed: boom" in capsys.readouterr().err


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_docflow_audit_returns_nonzero_docflow_status_without_sppf::cli.py::gabion.cli._run_docflow_audit
def test_run_docflow_audit_returns_nonzero_docflow_status_without_sppf(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "audit_tools.py"
    module_path.write_text(
        "def run_docflow_cli(argv=None):\n"
        "    return 7\n"
        "def run_sppf_graph_cli(argv=None):\n"
        "    raise RuntimeError('should not run')\n"
    )
    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=False,
        sppf_gh_ref_mode="required",
        audit_tools_path=module_path,
    )
    assert exit_code == 7


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_docflow_audit_returns_two_when_loader_creation_fails::cli.py::gabion.cli._run_docflow_audit
def test_run_docflow_audit_returns_two_when_loader_creation_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module_path = tmp_path / "audit_tools.py"
    module_path.write_text("def run_docflow_cli(argv=None):\n    return 0\n")
    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=False,
        sppf_gh_ref_mode="required",
        audit_tools_path=module_path,
        spec_from_file_location_fn=lambda *_a, **_k: None,
    )
    assert exit_code == 2
    assert "failed to load audit_tools module" in capsys.readouterr().err


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts
def test_dataflow_audit_skips_type_audit_output() -> None:
    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {"exit_code": 0}

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(["sample.py"], runner=runner)
    assert exc.value.exit_code == 0


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts
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


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts
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


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_timeout_progress_artifacts_no_progress_is_noop::cli.py::gabion.cli._emit_timeout_progress_artifacts
def test_emit_timeout_progress_artifacts_no_progress_is_noop(tmp_path: Path) -> None:
    cli._emit_timeout_progress_artifacts({"timeout_context": {"progress": "bad"}}, root=tmp_path)
    assert not (tmp_path / "artifacts" / "audit_reports" / "timeout_progress.json").exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_timeout_progress_artifacts_missing_timeout_context_is_noop::cli.py::gabion.cli._emit_timeout_progress_artifacts
def test_emit_timeout_progress_artifacts_missing_timeout_context_is_noop(tmp_path: Path) -> None:
    cli._emit_timeout_progress_artifacts({"timeout_context": "bad"}, root=tmp_path)
    assert not (tmp_path / "artifacts" / "audit_reports" / "timeout_progress.json").exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dataflow_audit_timeout_progress_report_and_resume_retry::cli.py::gabion.cli._run_dataflow_raw_argv
def test_dataflow_audit_timeout_progress_report_and_resume_retry(tmp_path: Path) -> None:
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

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
                "sample.py",
                "--root",
                str(tmp_path),
                "--emit-timeout-progress-report",
                "--resume-on-timeout",
                "1",
            ],
            runner=runner,
        )
    assert exc.value.exit_code == 0
    assert calls["count"] == 2
    progress_json = tmp_path / "artifacts" / "audit_reports" / "timeout_progress.json"
    progress_md = tmp_path / "artifacts" / "audit_reports" / "timeout_progress.md"
    assert progress_json.exists()
    assert progress_md.exists()
    payload = json.loads(progress_json.read_text())
    assert payload["analysis_state"] == "timed_out_progress_resume"


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_timeout_retry_branch_and_progress_artifacts::cli.py::gabion.cli._run_with_timeout_retries
def test_check_timeout_retry_branch_and_progress_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, object]] = []
    emitted_profile: list[dict[str, object]] = []
    emitted_progress: list[dict[str, object]] = []

    def _fake_run_once() -> dict[str, object]:
        calls.append({"attempt": len(calls) + 1})
        if len(calls) == 1:
            return {
                "timeout": True,
                "analysis_state": "timed_out_progress_resume",
                "exit_code": 2,
            }
        return {"timeout": False, "exit_code": 0}

    result = cli._run_with_timeout_retries(
        run_once=_fake_run_once,
        root=tmp_path,
        emit_timeout_progress_report=True,
        resume_on_timeout=1,
        emit_timeout_profile_artifacts_fn=lambda result, *, root: emitted_profile.append(
            {"result": dict(result), "root": Path(root)}
        ),
        emit_timeout_progress_artifacts_fn=lambda result, *, root: emitted_progress.append(
            {"result": dict(result), "root": Path(root)}
        ),
    )
    assert result["exit_code"] == 0
    assert len(calls) == 2
    assert len(emitted_profile) == 1
    assert len(emitted_progress) == 1
    assert emitted_profile[0]["root"] == tmp_path
    assert emitted_progress[0]["root"] == tmp_path
    assert "Retrying after timeout with progress (1/1)..." in capsys.readouterr().out


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
            emit_timeout_progress_report=False,
            resume_on_timeout=3,
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


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dataflow_audit_timeout_progress_resume_retries_when_attempts_remain::cli.py::gabion.cli._run_dataflow_raw_argv
def test_dataflow_audit_timeout_progress_resume_retries_when_attempts_remain(
    tmp_path: Path,
) -> None:
    calls = {"count": 0}

    def runner(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "exit_code": 2,
                "timeout": True,
                "analysis_state": "timed_out_progress_resume",
                "timeout_context": {
                    "deadline_profile": {"checks_total": 1, "sites": [], "edges": []},
                    "progress": {"classification": "timed_out_progress_resume"},
                },
            }
        return {"exit_code": 0}

    with pytest.raises(typer.Exit) as exc:
        cli._run_dataflow_raw_argv(
            [
                "sample.py",
                "--root",
                str(tmp_path),
                "--emit-timeout-progress-report",
                "--resume-on-timeout",
                "2",
            ],
            runner=runner,
        )
    assert exc.value.exit_code == 0
    assert calls["count"] == 2


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_dataflow_audit_retry_uses_fresh_cli_budget::cli.py::gabion.cli._run_dataflow_raw_argv::timeout_context.py::gabion.analysis.timeout_context.check_deadline
def test_dataflow_audit_retry_uses_fresh_cli_budget(
    tmp_path: Path,
    env_scope,
    restore_env,
) -> None:
    calls = {"count": 0}

    def runner(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            check_deadline()
            return {
                "exit_code": 2,
                "timeout": True,
                "analysis_state": "timed_out_progress_resume",
            }
        return {"exit_code": 0}

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
                    "--resume-on-timeout",
                    "1",
                ],
                runner=runner,
            )
    finally:
        restore_env(previous)
    assert exc.value.exit_code == 0
    assert calls["count"] == 2


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_resume_checkpoint_from_progress_notification::cli.py::gabion.cli._resume_checkpoint_from_progress_notification
def test_resume_checkpoint_from_progress_notification() -> None:
    payload = cli._resume_checkpoint_from_progress_notification(
        {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "resume_checkpoint": {
                        "checkpoint_path": "artifacts/audit_reports/resume.json",
                        "status": "checkpoint_loaded",
                        "reused_files": 3,
                        "total_files": 7,
                    }
                },
            },
        }
    )
    assert payload == {
        "checkpoint_path": "artifacts/audit_reports/resume.json",
        "status": "checkpoint_loaded",
        "reused_files": 3,
        "total_files": 7,
    }


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_checkpoint_intro_timeline_from_progress_notification::cli.py::gabion.cli._checkpoint_intro_timeline_from_progress_notification
def test_checkpoint_intro_timeline_from_progress_notification() -> None:
    payload = cli._checkpoint_intro_timeline_from_progress_notification(
        {
            "method": "$/progress",
            "params": {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {
                    "checkpoint_intro_timeline_header": "| a | b |",
                    "checkpoint_intro_timeline_row": "| 1 | 2 |",
                },
            },
        }
    )
    assert payload == {
        "header": "| a | b |",
        "row": "| 1 | 2 |",
    }


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_checkpoint_intro_timeline_from_progress_notification_rejects_invalid_shapes::cli.py::gabion.cli._checkpoint_intro_timeline_from_progress_notification
def test_checkpoint_intro_timeline_from_progress_notification_rejects_invalid_shapes() -> None:
    assert (
        cli._checkpoint_intro_timeline_from_progress_notification(
            {"method": "textDocument/publishDiagnostics"}
        )
        is None
    )
    assert (
        cli._checkpoint_intro_timeline_from_progress_notification(
            {"method": "$/progress", "params": "bad"}
        )
        is None
    )
    assert (
        cli._checkpoint_intro_timeline_from_progress_notification(
            {"method": "$/progress", "params": {"token": "wrong", "value": {}}}
        )
        is None
    )
    assert (
        cli._checkpoint_intro_timeline_from_progress_notification(
            {
                "method": "$/progress",
                "params": {"token": "gabion.dataflowAudit/progress-v1", "value": "bad"},
            }
        )
        is None
    )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_resume_checkpoint_startup_line::cli.py::gabion.cli._emit_resume_checkpoint_startup_line
def test_emit_resume_checkpoint_startup_line(capsys) -> None:
    cli._emit_resume_checkpoint_startup_line(
        checkpoint_path="artifacts/audit_reports/resume.json",
        status="checkpoint_loaded",
        reused_files=3,
        total_files=7,
    )
    output = capsys.readouterr().out
    assert "resume checkpoint detected..." in output
    assert "status=checkpoint_loaded" in output
    assert "reused_files=3/7" in output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_resume_checkpoint_startup_line_unknown_pending::cli.py::gabion.cli._emit_resume_checkpoint_startup_line
def test_emit_resume_checkpoint_startup_line_unknown_pending(capsys) -> None:
    cli._emit_resume_checkpoint_startup_line(
        checkpoint_path="artifacts/audit_reports/resume.json",
        status="pending",
        reused_files=None,
        total_files=None,
    )
    output = capsys.readouterr().out
    assert "resume checkpoint detected..." in output
    assert "status=pending" in output
    assert "reused_files=unknown" in output


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_dataflow_raw_argv_emits_pending_unknown_then_first_resume_update::cli.py::gabion.cli._run_dataflow_raw_argv
def test_run_dataflow_raw_argv_emits_pending_unknown_then_first_resume_update(
    tmp_path: Path,
    capsys,
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
                        "checkpoint_path": str(checkpoint_path),
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
    assert exc.value.exit_code == 0
    startup_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "resume checkpoint detected..." in line
    ]
    assert len(startup_lines) == 2
    assert "status=pending" in startup_lines[0]
    assert "reused_files=unknown" in startup_lines[0]
    assert "status=checkpoint_loaded" in startup_lines[1]
    assert "reused_files=2/5" in startup_lines[1]


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_dataflow_raw_argv_emits_checkpoint_intro_timeline_rows::cli.py::gabion.cli._run_dataflow_raw_argv
def test_run_dataflow_raw_argv_emits_checkpoint_intro_timeline_rows(
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
                        "checkpoint_intro_timeline_row": "| t0 | 0 |",
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
                        "checkpoint_intro_timeline_row": "| t1 | 1 |",
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
    assert "| ts | done |" in output
    assert "| t0 | 0 |" in output
    assert "| t1 | 1 |" in output


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


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_emit_analysis_resume_summary::cli.py::gabion.cli._emit_analysis_resume_summary
@pytest.mark.parametrize("cache_verdict", ["hit", "miss", "invalidated", "seeded"])
def test_emit_analysis_resume_summary(cache_verdict: str, capsys) -> None:
    cli._emit_analysis_resume_summary(
        {
            "analysis_resume": {
                "checkpoint_path": "artifacts/audit_reports/resume.json",
                "status": "checkpoint_loaded",
                "reused_files": 3,
                "total_files": 5,
                "remaining_files": 2,
                "cache_verdict": cache_verdict,
            }
        }
    )
    output = capsys.readouterr().out
    assert "Resume checkpoint:" in output
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


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts
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


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts
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


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts
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


# gabion:evidence E:decision_surface/direct::cli.py::gabion.cli._emit_lint_outputs::lint,lint_jsonl,lint_sarif E:decision_surface/direct::cli.py::gabion.cli.build_dataflow_payload::opts
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
    assert captured["root"] == Path(".")
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


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_dataflow_resume_checkpoint_from_github_artifacts_restores_files::cli.py::gabion.cli._restore_dataflow_resume_checkpoint_from_github_artifacts
def test_restore_dataflow_resume_checkpoint_from_github_artifacts_restores_files(
    tmp_path: Path,
) -> None:
    checkpoint_name = "dataflow_resume_checkpoint_ci.json"
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

    exit_code = cli._restore_dataflow_resume_checkpoint_from_github_artifacts(
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


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_dataflow_resume_checkpoint_accepts_workflow_dispatch_artifacts::cli.py::gabion.cli._restore_dataflow_resume_checkpoint_from_github_artifacts
def test_restore_dataflow_resume_checkpoint_accepts_workflow_dispatch_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint_name = "dataflow_resume_checkpoint_ci.json"
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

    exit_code = cli._restore_dataflow_resume_checkpoint_from_github_artifacts(
        token="token",
        repo="owner/repo",
        output_dir=tmp_path,
        ref_name="stage",
        current_run_id="999",
        urlopen_fn=_fake_urlopen,
    )

    assert exit_code == 0
    assert (tmp_path / checkpoint_name).exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_dataflow_resume_checkpoint_accepts_artifacts_with_missing_event::cli.py::gabion.cli._restore_dataflow_resume_checkpoint_from_github_artifacts
def test_restore_dataflow_resume_checkpoint_accepts_artifacts_with_missing_event(
    tmp_path: Path,
) -> None:
    checkpoint_name = "dataflow_resume_checkpoint_ci.json"
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

    exit_code = cli._restore_dataflow_resume_checkpoint_from_github_artifacts(
        token="token",
        repo="owner/repo",
        output_dir=tmp_path,
        ref_name="stage",
        current_run_id="999",
        urlopen_fn=_fake_urlopen,
    )

    assert exit_code == 0
    assert (tmp_path / checkpoint_name).exists()


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_dataflow_resume_checkpoint_falls_back_from_incomplete_chunks::cli.py::gabion.cli._restore_dataflow_resume_checkpoint_from_github_artifacts
def test_restore_dataflow_resume_checkpoint_falls_back_from_incomplete_chunks(
    tmp_path: Path,
) -> None:
    checkpoint_name = "dataflow_resume_checkpoint_ci.json"
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

    exit_code = cli._restore_dataflow_resume_checkpoint_from_github_artifacts(
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


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_dataflow_resume_checkpoint_overwrites_existing_output_files::cli.py::gabion.cli._restore_dataflow_resume_checkpoint_from_github_artifacts
def test_restore_dataflow_resume_checkpoint_overwrites_existing_output_files(
    tmp_path: Path,
) -> None:
    checkpoint_name = "dataflow_resume_checkpoint_ci.json"
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

    exit_code = cli._restore_dataflow_resume_checkpoint_from_github_artifacts(
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


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_checkpoint_requires_chunk_artifacts_invalid_payload_shapes::cli.py::gabion.cli._checkpoint_requires_chunk_artifacts
def test_checkpoint_requires_chunk_artifacts_invalid_payload_shapes() -> None:
    assert cli._checkpoint_requires_chunk_artifacts(checkpoint_bytes=b"{not-json") is False
    assert cli._checkpoint_requires_chunk_artifacts(checkpoint_bytes=b"[]") is False
    assert (
        cli._checkpoint_requires_chunk_artifacts(
            checkpoint_bytes=json.dumps({"collection_resume": {}}).encode("utf-8")
        )
        is False
    )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_resume_checkpoint_cli_maps_options::cli.py::gabion.cli.app
def test_restore_resume_checkpoint_cli_maps_options(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_restore(**kwargs):
        captured.update(kwargs)
        return 0

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "restore-resume-checkpoint",
            "--token",
            "abc",
            "--repo",
            "owner/repo",
            "--output-dir",
            str(tmp_path),
            "--ref-name",
            "stage",
            "--run-id",
            "123",
            "--artifact-name",
            "dataflow-report",
            "--checkpoint-name",
            "dataflow_resume_checkpoint_ci.json",
        ],
        obj={"restore_resume_checkpoint": _fake_restore},
    )

    assert result.exit_code == 0
    assert captured == {
        "token": "abc",
        "repo": "owner/repo",
        "output_dir": tmp_path,
        "ref_name": "stage",
        "current_run_id": "123",
        "artifact_name": "dataflow-report",
        "checkpoint_name": "dataflow_resume_checkpoint_ci.json",
    }


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


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_resume_checkpoint_from_progress_notification_rejects_invalid_shapes::cli.py::gabion.cli._resume_checkpoint_from_progress_notification
def test_resume_checkpoint_from_progress_notification_rejects_invalid_shapes() -> None:
    assert (
        cli._resume_checkpoint_from_progress_notification({"method": "textDocument/publishDiagnostics"})
        is None
    )
    assert cli._resume_checkpoint_from_progress_notification({"method": "$/progress", "params": "bad"}) is None
    assert (
        cli._resume_checkpoint_from_progress_notification(
            {"method": "$/progress", "params": {"token": "wrong", "value": {}}}
        )
        is None
    )
    assert (
        cli._resume_checkpoint_from_progress_notification(
            {
                "method": "$/progress",
                "params": {"token": "gabion.dataflowAudit/progress-v1", "value": "bad"},
            }
        )
        is None
    )


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_dataflow_raw_argv_emits_resume_update_once::cli.py::gabion.cli._run_dataflow_raw_argv
def test_run_dataflow_raw_argv_emits_resume_update_once(
    tmp_path: Path,
    capsys,
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
                        "checkpoint_path": str(checkpoint_path),
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
    assert exc.value.exit_code == 0
    startup_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "resume checkpoint detected..." in line
    ]
    assert len(startup_lines) == 2


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_check_emits_resume_startup_and_first_progress_once::cli.py::gabion.cli.app
def test_check_emits_resume_startup_and_first_progress_once(
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
                        "checkpoint_path": "resume.json",
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
            "sample.py",
            "--root",
            str(tmp_path),
            "--resume-checkpoint",
            str(tmp_path / "resume.json"),
            "--no-fail-on-violations",
            "--no-fail-on-type-ambiguities",
        ],
        obj={
            "run_check": _fake_run_check,
            "run_with_timeout_retries": lambda run_once, **_kwargs: run_once(),
        },
    )
    assert result.exit_code == 0
    startup_lines = [
        line
        for line in result.stdout.splitlines()
        if "resume checkpoint detected..." in line
    ]
    assert len(startup_lines) == 2


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
                        "checkpoint_intro_timeline_header": "| ts | done |",
                        "checkpoint_intro_timeline_row": "| t0 | 0 |",
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
                        "checkpoint_intro_timeline_header": "| ts | done |",
                        "checkpoint_intro_timeline_row": "| t1 | 1 |",
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
            "sample.py",
            "--root",
            str(tmp_path),
            "--no-fail-on-violations",
            "--no-fail-on-type-ambiguities",
        ],
        obj={
            "run_check": _fake_run_check,
            "run_with_timeout_retries": lambda run_once, **_kwargs: run_once(),
        },
    )
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines.count("| ts | done |") == 1
    assert "| t0 | 0 |" in lines
    assert "| t1 | 1 |" in lines


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
            "sample.py",
            "--root",
            str(tmp_path),
            "--no-fail-on-violations",
            "--no-fail-on-type-ambiguities",
        ],
        obj={
            "run_check": _fake_run_check,
            "run_with_timeout_retries": lambda run_once, **_kwargs: run_once(),
        },
    )
    assert result.exit_code == 0
    assert "| ts | done |" not in result.stdout


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_governance_cli_error_paths::cli.py::gabion.cli._run_governance_cli
def test_run_governance_cli_error_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        cli._run_governance_cli(
            runner_name="run_docflow_cli",
            args=[],
            audit_tools_path=tmp_path / "missing.py",
        )
        == 2
    )

    class _BadPath:
        pass

    assert (
        cli._run_governance_cli(
            runner_name="run_docflow_cli",
            args=[],
            audit_tools_path=_BadPath(),  # type: ignore[arg-type]
        )
        == 2
    )

    missing_runner = tmp_path / "missing_runner.py"
    missing_runner.write_text("value = 1\n", encoding="utf-8")
    assert (
        cli._run_governance_cli(
            runner_name="run_docflow_cli",
            args=[],
            audit_tools_path=missing_runner,
        )
        == 2
    )

    failing_runner = tmp_path / "failing_runner.py"
    failing_runner.write_text(
        "def run_docflow_cli(argv=None):\n"
        "    raise RuntimeError('runner boom')\n",
        encoding="utf-8",
    )
    assert (
        cli._run_governance_cli(
            runner_name="run_docflow_cli",
            args=[],
            audit_tools_path=failing_runner,
        )
        == 1
    )
    assert "governance command failed" in capsys.readouterr().err


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_restore_resume_checkpoint_handles_guard_and_error_branches::cli.py::gabion.cli._restore_dataflow_resume_checkpoint_from_github_artifacts
def test_restore_resume_checkpoint_handles_guard_and_error_branches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        cli._restore_dataflow_resume_checkpoint_from_github_artifacts(
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
        cli._restore_dataflow_resume_checkpoint_from_github_artifacts(
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
        cli._restore_dataflow_resume_checkpoint_from_github_artifacts(
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
        cli._restore_dataflow_resume_checkpoint_from_github_artifacts(
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
        zf.writestr("dataflow-report/dataflow_resume_checkpoint_ci.json.chunks/", "")

    def _zip_with_only_directory(req, timeout=0):
        url = str(getattr(req, "full_url", req))
        if "actions/artifacts" in url:
            return _Resp(json.dumps(payload_download).encode("utf-8"))
        return _Resp(zip_buf.getvalue())

    assert (
        cli._restore_dataflow_resume_checkpoint_from_github_artifacts(
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
        cli._restore_dataflow_resume_checkpoint_from_github_artifacts(
            token="token",
            repo="owner/repo",
            output_dir=tmp_path,
            ref_name="main",
            urlopen_fn=_invalid_zip,
        )
        == 0
    )

# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_context_restore_resume_checkpoint_falls_back_to_default::cli.py::gabion.cli._context_restore_resume_checkpoint
def test_context_restore_resume_checkpoint_falls_back_to_default() -> None:
    class _Ctx:
        obj = {"restore_resume_checkpoint": "not-callable"}

    fn = cli._context_restore_resume_checkpoint(_Ctx())
    assert fn is cli._restore_dataflow_resume_checkpoint_from_github_artifacts


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_docflow_audit_handles_loader_exception_and_extra_path::cli.py::gabion.cli._run_docflow_audit
def test_run_docflow_audit_handles_loader_exception_and_extra_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module_path = tmp_path / "audit_tools.py"
    calls_path = tmp_path / "calls.txt"
    module_path.write_text(
        f"calls_path = {str(calls_path)!r}\n"
        "def run_docflow_cli(argv=None):\n"
        "    with open(calls_path, 'w', encoding='utf-8') as handle:\n"
        "        handle.write('|'.join(argv or []))\n"
        "    return 0\n"
        "def run_sppf_graph_cli(argv=None):\n"
        "    return 0\n",
        encoding="utf-8",
    )

    assert (
        cli._run_docflow_audit(
            root=tmp_path,
            fail_on_violations=False,
            sppf_gh_ref_mode="required",
            extra_path=["extra/docs"],
            audit_tools_path=module_path,
        )
        == 0
    )
    assert "--extra-path|extra/docs" in calls_path.read_text(encoding="utf-8")

    assert (
        cli._run_docflow_audit(
            root=tmp_path,
            fail_on_violations=False,
            sppf_gh_ref_mode="required",
            audit_tools_path=module_path,
            spec_from_file_location_fn=lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("loader boom")
            ),
        )
        == 2
    )
    assert "failed to load audit_tools module" in capsys.readouterr().err


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_run_docflow_audit_handles_loader_creation_exception::cli.py::gabion.cli._run_docflow_audit
def test_run_docflow_audit_handles_loader_creation_exception(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _BadPath:
        pass

    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=False,
        sppf_gh_ref_mode="required",
        audit_tools_path=_BadPath(),  # type: ignore[arg-type]
    )
    assert exit_code == 2
    assert "failed to load audit_tools module" in capsys.readouterr().err


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_sppf_sync_command_handles_runner_errors::cli.py::gabion.cli.app
def test_sppf_sync_command_handles_runner_errors() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["sppf-sync", "--range", "__not_a_rev_range__"])
    assert result.exit_code == 2


# gabion:evidence E:call_footprint::tests/test_cli_helpers.py::test_sppf_graph_and_status_consistency_include_optional_cli_args::cli.py::gabion.cli.app
def test_sppf_graph_and_status_consistency_include_optional_cli_args(tmp_path: Path) -> None:
    calls: list[tuple[str, list[str]]] = []

    def _fake_run_governance_cli(*, runner_name: str, args: list[str], **_kwargs) -> int:
        calls.append((runner_name, list(args)))
        return 0

    runner = CliRunner()
    result = runner.invoke(
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
        obj={"run_governance_cli": _fake_run_governance_cli},
    )
    assert result.exit_code == 0

    result = runner.invoke(
        cli.app,
        [
            "status-consistency",
            "--root",
            str(tmp_path),
            "--extra-path",
            "in",
            "--fail-on-violations",
        ],
        obj={"run_governance_cli": _fake_run_governance_cli},
    )
    assert result.exit_code == 0
    assert any("--dot-output" in args for _, args in calls)
    assert any("--issues-json" in args for _, args in calls)
    assert any("--extra-path" in args and "--fail-on-violations" in args for _, args in calls)
