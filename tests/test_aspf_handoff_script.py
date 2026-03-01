from __future__ import annotations

import json
from pathlib import Path

from gabion.runtime import env_policy
from gabion.tooling import dataflow_invocation_runner
from scripts import aspf_handoff as aspf_handoff_script


# gabion:evidence E:call_footprint::tests/test_aspf_handoff_script.py::test_script_prepare_and_record_roundtrip::aspf_handoff.py::scripts.aspf_handoff.main
def test_script_prepare_and_record_roundtrip(
    tmp_path: Path,
    capsys,
) -> None:
    manifest = tmp_path / "artifacts/out/aspf_handoff_manifest.json"
    state_root = tmp_path / "artifacts/out/aspf_state"
    root = tmp_path

    prepare_exit = aspf_handoff_script.main(
        [
            "prepare",
            "--root",
            str(root),
            "--session-id",
            "session-script",
            "--step-id",
            "script.check.run",
            "--command-profile",
            "check.run",
            "--manifest",
            str(manifest),
            "--state-root",
            str(state_root),
        ]
    )
    assert prepare_exit == 0
    prepare_payload = json.loads(capsys.readouterr().out)
    assert prepare_payload["session_id"] == "session-script"
    assert prepare_payload["sequence"] == 1
    assert prepare_payload["aspf_cli_args"][0] == "--aspf-state-json"

    record_exit = aspf_handoff_script.main(
        [
            "record",
            "--manifest",
            str(manifest),
            "--session-id",
            "session-script",
            "--sequence",
            str(prepare_payload["sequence"]),
            "--status",
            "success",
            "--exit-code",
            "0",
            "--analysis-state",
            "succeeded",
        ]
    )
    assert record_exit == 0
    record_payload = json.loads(capsys.readouterr().out)
    assert record_payload == {"ok": True}


# gabion:evidence E:call_footprint::tests/test_aspf_handoff_script.py::test_script_record_returns_nonzero_when_entry_missing::aspf_handoff.py::scripts.aspf_handoff.main
def test_script_record_returns_nonzero_when_entry_missing(
    tmp_path: Path,
    capsys,
) -> None:
    manifest = tmp_path / "artifacts/out/aspf_handoff_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "session_id": "session-script",
                "root": str(tmp_path),
                "entries": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = aspf_handoff_script.main(
        [
            "record",
            "--manifest",
            str(manifest),
            "--session-id",
            "session-script",
            "--sequence",
            "9",
            "--status",
            "failed",
            "--exit-code",
            "1",
            "--analysis-state",
            "failed",
        ]
    )
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"ok": False}


# gabion:evidence E:function_site::tests/test_aspf_handoff_script.py::test_analysis_state_from_state_path_reads_top_level_without_canonicalization
def test_analysis_state_from_state_path_reads_top_level_without_canonicalization(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "z_key": {"nested": [3, 1, 2]},
                "analysis_state": "timed_out_progress_resume",
                "a_key": [{"k": "v"}, {"k": "w"}],
            }
        ),
        encoding="utf-8",
    )
    assert (
        aspf_handoff_script._analysis_state_from_state_path(state_path)
        == "timed_out_progress_resume"
    )


# gabion:evidence E:function_site::tests/test_aspf_handoff_script.py::test_analysis_state_from_state_path_uses_resume_projection
def test_analysis_state_from_state_path_uses_resume_projection(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "resume_projection": {
                    "analysis_state": "timed_out_progress_resume",
                }
            }
        ),
        encoding="utf-8",
    )
    assert (
        aspf_handoff_script._analysis_state_from_state_path(state_path)
        == "timed_out_progress_resume"
    )


# gabion:evidence E:function_site::tests/test_aspf_handoff_script.py::test_analysis_state_from_state_path_returns_none_on_invalid_json
def test_analysis_state_from_state_path_returns_none_on_invalid_json(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text("{invalid", encoding="utf-8")
    assert aspf_handoff_script._analysis_state_from_state_path(state_path) == "none"


# gabion:evidence E:function_site::tests/test_aspf_handoff_script.py::test_command_timeout_text_supports_split_and_equals_forms
def test_command_timeout_text_supports_split_and_equals_forms() -> None:
    assert (
        aspf_handoff_script._command_timeout_text(
            ["python", "-m", "gabion", "--timeout", "130000000000000ns", "check"]
        )
        == "130000000000000ns"
    )
    assert (
        aspf_handoff_script._command_timeout_text(
            ["python", "-m", "gabion", "--timeout=2m", "check"]
        )
        == "2m"
    )
    assert (
        aspf_handoff_script._command_timeout_text(["python", "-m", "gabion", "check"])
        is None
    )


# gabion:evidence E:function_site::tests/test_aspf_handoff_script.py::test_run_preserves_timeout_override_and_runner_analysis_state_hint
def test_run_preserves_timeout_override_and_runner_analysis_state_hint(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    captured_timeout = {"ticks": 0, "tick_ns": 0}

    class _FakeRunner:
        def run_delta_bundle(self, _envelope):
            raise AssertionError("delta-bundle path should not be selected")

        def run_raw(self, _envelope, _raw_args):
            override = env_policy.lsp_timeout_override()
            assert override is not None
            captured_timeout["ticks"] = int(override.ticks)
            captured_timeout["tick_ns"] = int(override.tick_ns)
            return dataflow_invocation_runner.DataflowInvocationResult(
                exit_code=2,
                analysis_state="timed_out_progress_resume",
                payload={"exit_code": 2, "analysis_state": "timed_out_progress_resume"},
            )

    monkeypatch.setattr(
        aspf_handoff_script.dataflow_invocation_runner,
        "DataflowInvocationRunner",
        _FakeRunner,
    )

    manifest = tmp_path / "artifacts/out/aspf_handoff_manifest.json"
    state_root = tmp_path / "artifacts/out/aspf_state"
    report = tmp_path / "artifacts/dataflow_grammar/report.md"
    dot = tmp_path / "artifacts/dataflow_grammar/graph.dot"

    exit_code = aspf_handoff_script.main(
        [
            "run",
            "--root",
            str(tmp_path),
            "--session-id",
            "session-timeout-hint",
            "--step-id",
            "pr-dataflow.render-check.raw",
            "--command-profile",
            "pr-dataflow.check.raw",
            "--manifest",
            str(manifest),
            "--state-root",
            str(state_root),
            "--",
            ".venv/bin/python",
            "-m",
            "gabion",
            "--timeout",
            "130000000000000ns",
            "check",
            "raw",
            "--",
            ".",
            "--root",
            ".",
            "--report",
            str(report),
            "--dot",
            str(dot),
            "--type-audit-report",
            "--baseline",
            "baselines/dataflow_baseline.txt",
        ]
    )
    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["analysis_state"] == "timed_out_progress_resume"
    assert captured_timeout == {"ticks": 130000000, "tick_ns": 1000000}
