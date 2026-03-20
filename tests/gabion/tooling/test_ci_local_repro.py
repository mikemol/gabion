from __future__ import annotations

from pathlib import Path

from gabion.tooling.runtime import ci_local_repro


def _options(tmp_path: Path) -> ci_local_repro.LocalCiReproOptions:
    root = tmp_path
    return ci_local_repro.LocalCiReproOptions(
        root=root,
        python_bin=Path("/tmp/python"),
        log_dir=root / "artifacts" / "test_runs" / "local_ci",
        run_checks=True,
        run_dataflow=False,
        run_pr_dataflow=False,
        run_extended_checks=False,
        run_sppf_sync_mode="auto",
        sppf_range="origin/stage..HEAD",
        skip_gabion_check_step=False,
        pr_base_sha="base",
        pr_head_sha="head",
        pr_body_file=None,
        verify_pr_stage_ci=True,
        pr_stage_ci_timeout_minutes=70,
        step_timing_enabled=False,
        observability_enabled=False,
        step_timing_artifact=root / "artifacts" / "audit_reports" / "ci_step_timings.json",
        step_timing_run_id="run-1",
        ci_event_name="push",
        aspf_handoff_enabled=True,
        aspf_handoff_manifest=root / "artifacts" / "out" / "aspf_handoff_manifest.json",
        aspf_handoff_session="session-1",
        aspf_state_root=root / "artifacts" / "out" / "aspf_state",
        before_sha="before",
        after_sha="after",
        timeout_ns="130000000000000ns",
        impact_gate_must_run=False,
    )


# gabion:behavior primary=desired
def test_checks_steps_cover_ci_parity_surface(tmp_path: Path) -> None:
    steps = ci_local_repro._checks_steps(_options(tmp_path))
    commands = [tuple(step.command) for step in steps]

    assert any("--tier2-residue-contract" in command for command in commands)
    assert any("docflow-packetize" in command for command in commands)
    assert any("docflow-packet-enforce" in command for command in commands)
    assert any("scripts/ci/ci_override_record_emit.py" in command for command in commands)
    assert any("scripts/ci/ci_controller_drift_gate.py" in command for command in commands)
    assert any("lsp-parity-gate" in command for command in commands)
    assert any("scripts/policy/structural_hash_policy_check.py" in command for command in commands)
    assert any(
        "scripts/policy/deprecated_nonerasability_policy_check.py" in command
        for command in commands
    )


# gabion:behavior primary=desired
def test_pr_dataflow_steps_cover_stage_ci_and_render_report(tmp_path: Path) -> None:
    steps = ci_local_repro._pr_dataflow_steps(_options(tmp_path))
    commands = [tuple(step.command) for step in steps]

    assert any("impact-select-tests" in command for command in commands)
    assert any(("aspf", "handoff", "run") == command[3:6] for command in commands)
    assert any(
        "artifacts/dataflow_grammar/report.md" in command and "raw" in command
        for command in commands
    )
