from __future__ import annotations

from pathlib import Path

from gabion.tooling.runtime import checks_runtime


def _options(tmp_path: Path) -> checks_runtime.ChecksCommandOptions:
    return checks_runtime.ChecksCommandOptions(
        root=tmp_path,
        python_bin=Path("/tmp/python"),
        run_docflow=True,
        run_dataflow=True,
        run_tests=True,
        run_status_watch=False,
        docflow_mode="required",
        aspf_handoff_enabled=True,
        aspf_handoff_manifest=tmp_path / "artifacts" / "out" / "aspf_handoff_manifest.json",
        aspf_handoff_session="session-1",
        aspf_state_root=tmp_path / "artifacts" / "out" / "aspf_state",
        status_watch_branch="stage",
        status_watch_workflow="ci",
        list_only=False,
    )


# gabion:behavior primary=desired
def test_build_local_checks_steps_cover_canonical_lane_surface(tmp_path: Path) -> None:
    (tmp_path / "baselines").mkdir(parents=True, exist_ok=True)
    (tmp_path / "baselines" / "dataflow_baseline.txt").write_text("", encoding="utf-8")

    steps = checks_runtime.build_local_checks_steps(_options(tmp_path))
    commands = [tuple(step.command) for step in steps]

    assert any("lsp-parity-gate" in command for command in commands)
    assert any("check" in command and "run" in command for command in commands)
    assert any("scripts/policy/docflow_packetize.py" in command for command in commands)
    assert any("scripts/policy/docflow_packet_enforce.py" in command for command in commands)
    assert any("scripts.sppf.sppf_status_audit" in command for command in commands)
    assert any("pytest" in command for command in commands)


# gabion:behavior primary=desired
def test_checks_main_list_respects_lane_selection(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(checks_runtime, "_repo_root", lambda: tmp_path)

    exit_code = checks_runtime.main(
        ["--docflow-only", "--list", "--no-aspf-handoff"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Checks to run:" in captured.err
    assert "docflow" in captured.err
    assert "tests (pytest)" not in captured.err
    assert "lsp parity gate" not in captured.err


# gabion:behavior primary=desired
def test_checks_wrapper_is_thin_delegate() -> None:
    script = (Path(__file__).resolve().parents[3] / "scripts" / "checks.sh").read_text(
        encoding="utf-8"
    )

    assert "-m gabion checks" in script
    assert "lsp-parity-gate" not in script
    assert "docflow_packetize.py" not in script
    assert "gabion check run" not in script
