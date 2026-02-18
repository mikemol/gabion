from __future__ import annotations

from pathlib import Path
import io
import json
import re
import sys
import types

import pytest
import typer
from typer.testing import CliRunner

from gabion import cli
from gabion.analysis.timeout_context import check_deadline
from tests.env_helpers import env_scope as _env_scope


_ANSI_ESCAPE = re.compile(r"\x1B\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def test_parse_dataflow_args_or_exit_routes_help_to_parser(capsys) -> None:
    with pytest.raises(typer.Exit) as exc:
        cli.parse_dataflow_args_or_exit(["--help"])
    assert exc.value.exit_code == 0
    assert "usage:" in capsys.readouterr().out.lower()


def test_parse_dataflow_args_or_exit_converts_parse_errors_to_typer_exit() -> None:
    with pytest.raises(typer.Exit) as exc:
        cli.parse_dataflow_args_or_exit([])
    assert exc.value.exit_code == 2


def test_check_rejects_unknown_args_in_strict_profile() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["check", "sample.py", "--dot", "-"])
    assert result.exit_code != 0
    assert "Unknown arguments for strict profile" in result.output


def test_check_rejects_unknown_profile() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["check", "sample.py", "--profile", "mystery"])
    assert result.exit_code != 0
    assert "profile must be 'strict' or 'raw'" in result.output


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


def test_dataflow_audit_command_delegates_to_raw_runner(
) -> None:
    captured: dict[str, object] = {}

    def _fake_run(argv: list[str]) -> None:
        captured["argv"] = list(argv)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["dataflow-audit", "sample.py", "--dot", "-", "--root", "."],
        obj={
            "warn_dataflow_audit_alias": (lambda: None),
            "run_dataflow_raw_argv": _fake_run,
        },
    )
    assert result.exit_code == 0
    assert captured["argv"] == ["sample.py", "--dot", "-", "--root", "."]


def test_dataflow_audit_alias_matches_check_raw_args(
) -> None:
    captured: list[list[str]] = []

    def _fake_run(argv: list[str]) -> None:
        captured.append(list(argv))

    runner = CliRunner()
    check_result = runner.invoke(
        cli.app,
        ["check", "--profile", "raw", "sample.py", "--dot", "-"],
        obj={
            "warn_dataflow_audit_alias": (lambda: None),
            "run_dataflow_raw_argv": _fake_run,
        },
    )
    dataflow_result = runner.invoke(
        cli.app,
        ["dataflow-audit", "sample.py", "--dot", "-"],
        obj={
            "warn_dataflow_audit_alias": (lambda: None),
            "run_dataflow_raw_argv": _fake_run,
        },
    )
    assert check_result.exit_code == 0
    assert dataflow_result.exit_code == 0
    assert len(captured) == 2
    assert captured[0] == captured[1]


def test_dataflow_audit_emits_alias_warning() -> None:
    warned = {"count": 0}

    def _fake_warn() -> None:
        warned["count"] += 1

    def _fake_run(_argv: list[str]) -> None:
        return None

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["dataflow-audit", "sample.py"],
        obj={
            "warn_dataflow_audit_alias": _fake_warn,
            "run_dataflow_raw_argv": _fake_run,
        },
    )
    assert result.exit_code == 0
    assert warned["count"] == 1


def test_dataflow_audit_help_does_not_emit_alias_warning(
) -> None:
    warned = {"count": 0}

    def _fake_warn() -> None:
        warned["count"] += 1

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["dataflow-audit", "--help"],
        obj={"warn_dataflow_audit_alias": _fake_warn},
    )
    assert result.exit_code == 0
    assert warned["count"] == 0


def test_dataflow_alias_migration_epilog_contains_map() -> None:
    epilog = cli._dataflow_alias_migration_epilog()
    assert "gabion check --profile raw" in epilog
    assert "--emit-decision-snapshot -> --decision-snapshot" in epilog


def test_context_dependency_helpers_ignore_noncallables() -> None:
    class DummyCtx:
        obj = {
            "run_dataflow_raw_argv": "not-callable",
            "warn_dataflow_audit_alias": "not-callable",
        }

    ctx = DummyCtx()
    assert cli._context_run_dataflow_raw_argv(ctx) is cli._run_dataflow_raw_argv
    assert cli._context_warn_dataflow_audit_alias(ctx) is cli._warn_dataflow_audit_alias


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


def test_cli_deadline_scope_yields() -> None:
    with cli._cli_deadline_scope():
        assert True


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
        audit_tools_path=module_path,
        sys_path_list=sys_path_list,
        sys_modules_map=sys_modules_map,
    )

    assert exit_code == 0
    assert scripts_root not in sys_path_list
    assert module_name not in sys_modules_map


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
        audit_tools_path=module_path,
        sys_modules_map=sys_modules_map,
    )
    assert exit_code == 0
    assert sys_modules_map.get(module_name) is previous_module


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
        audit_tools_path=module_path,
        sys_path_list=sys_path_list,
    )
    assert exit_code == 0
    assert scripts_root not in sys_path_list


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
        audit_tools_path=module_path,
        sys_path_list=sys_path_list,
    )
    assert exit_code == 0
    assert sys_path_list == [scripts_root]


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
        audit_tools_path=module_path,
    )
    assert exit_code == 1
    assert "docflow: sppf-graph failed: boom" in capsys.readouterr().err


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
        audit_tools_path=module_path,
    )
    assert exit_code == 7


def test_run_docflow_audit_returns_two_when_loader_creation_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module_path = tmp_path / "audit_tools.py"
    module_path.write_text("def run_docflow_cli(argv=None):\n    return 0\n")
    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=False,
        audit_tools_path=module_path,
        spec_from_file_location_fn=lambda *_a, **_k: None,
    )
    assert exit_code == 2
    assert "failed to load audit_tools module" in capsys.readouterr().err


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


def test_emit_timeout_profile_artifacts_no_profile_is_noop(tmp_path: Path) -> None:
    cli._emit_timeout_profile_artifacts(
        {"timeout_context": {"deadline_profile": "bad"}},
        root=tmp_path,
    )
    assert not (tmp_path / "artifacts" / "out" / "deadline_profile.json").exists()


def test_emit_timeout_progress_artifacts_no_progress_is_noop(tmp_path: Path) -> None:
    cli._emit_timeout_progress_artifacts({"timeout_context": {"progress": "bad"}}, root=tmp_path)
    assert not (tmp_path / "artifacts" / "audit_reports" / "timeout_progress.json").exists()


def test_emit_timeout_progress_artifacts_missing_timeout_context_is_noop(tmp_path: Path) -> None:
    cli._emit_timeout_progress_artifacts({"timeout_context": "bad"}, root=tmp_path)
    assert not (tmp_path / "artifacts" / "audit_reports" / "timeout_progress.json").exists()


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


def test_render_timeout_progress_markdown_skips_non_mapping_obligation_entries() -> None:
    rendered = cli._render_timeout_progress_markdown(
        analysis_state="timed_out_progress_resume",
        progress={
            "incremental_obligations": ["bad", {"status": "SATISFIED", "contract": "c", "kind": "k", "detail": "d"}]
        },
    )
    assert "`SATISFIED` `c` `k`" in rendered


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


