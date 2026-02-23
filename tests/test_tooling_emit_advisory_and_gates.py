from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from gabion.execution_plan import DocflowFacet, ExecutionPlan
from gabion.exceptions import NeverThrown
from gabion.tooling import ambiguity_delta_gate
from gabion.tooling import annotation_drift_orphaned_gate
from gabion.tooling import deadline_runtime
from gabion.tooling import delta_advisory
from gabion.tooling import delta_state_emit
from gabion.tooling import docflow_delta_emit
from gabion.tooling import docflow_delta_gate
from gabion.tooling import obsolescence_delta_gate
from gabion.tooling import obsolescence_delta_unmapped_gate
from tests.env_helpers import env_scope


@contextmanager
def _cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


# gabion:evidence E:call_footprint::tests/test_tooling_emit_advisory_and_gates.py::test_emit_build_payload_handles_state_resume_and_timeout_defaults::delta_state_emit.py::gabion.tooling.delta_state_emit._build_payload_for_emitter
@pytest.mark.parametrize(
    ("emitter_id", "state_path", "state_key"),
    [
        (
            "obsolescence_delta_emit",
            Path("artifacts/out/test_obsolescence_state.json"),
            "test_obsolescence_state",
        ),
        (
            "annotation_drift_delta_emit",
            Path("artifacts/out/test_annotation_drift.json"),
            "test_annotation_drift_state",
        ),
        (
            "ambiguity_delta_emit",
            Path("artifacts/out/ambiguity_state.json"),
            "ambiguity_state",
        ),
    ],
)
def test_emit_build_payload_handles_state_resume_and_timeout_defaults(
    tmp_path: Path,
    emitter_id: str,
    state_path: Path,
    state_key: str,
) -> None:
    with _cwd(tmp_path):
        with env_scope(
            {
                "GABION_LSP_TIMEOUT_TICKS": "bad",
                "GABION_LSP_TIMEOUT_TICK_NS": "bad",
            }
        ):
            payload = delta_state_emit._build_payload_for_emitter(emitter_id)
        assert payload["analysis_timeout_ticks"] == int(delta_state_emit._DEFAULT_TIMEOUT_TICKS)
        assert payload["analysis_timeout_tick_ns"] == int(delta_state_emit._DEFAULT_TIMEOUT_TICK_NS)
        assert payload.get(state_key) is None
        assert payload["resume_checkpoint"] is False

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{}\n", encoding="utf-8")
        delta_state_emit._DEFAULT_RESUME_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        delta_state_emit._DEFAULT_RESUME_CHECKPOINT_PATH.write_text("{}\n", encoding="utf-8")
        payload = delta_state_emit._build_payload_for_emitter(emitter_id)
        assert payload[state_key] == str(state_path)
        assert payload["resume_checkpoint"] == str(delta_state_emit._DEFAULT_RESUME_CHECKPOINT_PATH)

        override_resume = tmp_path / f"{emitter_id}.resume.json"
        payload = delta_state_emit._build_payload_for_emitter(
            emitter_id,
            resume_checkpoint=override_resume,
        )
        assert payload["resume_checkpoint"] == str(override_resume)
        payload = delta_state_emit._build_payload_for_emitter(
            emitter_id,
            resume_checkpoint=False,
        )
        assert payload["resume_checkpoint"] is False


# gabion:evidence E:call_footprint::tests/test_tooling_emit_advisory_and_gates.py::test_emit_main_covers_exit_and_missing_output_branches::delta_state_emit.py::gabion.tooling.delta_state_emit.obsolescence_main::delta_state_emit.py::gabion.tooling.delta_state_emit.annotation_drift_main::delta_state_emit.py::gabion.tooling.delta_state_emit.ambiguity_main
@pytest.mark.parametrize(
    "run_main",
    [
        delta_state_emit.obsolescence_main,
        delta_state_emit.annotation_drift_main,
        delta_state_emit.ambiguity_main,
    ],
)
def test_emit_main_covers_exit_and_missing_output_branches(
    tmp_path: Path,
    run_main,
    capsys: pytest.CaptureFixture[str],
) -> None:
    delta_path = tmp_path / "delta.json"

    def _ok(_request, *, root: Path) -> dict[str, object]:
        assert root == tmp_path
        return {"exit_code": 0}

    def _fail(_request, *, root: Path) -> dict[str, object]:
        assert root == tmp_path
        return {"exit_code": 3}

    assert run_main(
        run_command_direct_fn=_ok,
        root_path=tmp_path,
        delta_path=delta_path,
    ) == 1
    assert "missing output" in capsys.readouterr().out

    delta_path.write_text("{}\n", encoding="utf-8")
    assert run_main(
        run_command_direct_fn=_ok,
        root_path=tmp_path,
        delta_path=delta_path,
    ) == 0

    assert run_main(
        run_command_direct_fn=_fail,
        root_path=tmp_path,
        delta_path=delta_path,
    ) == 3
    assert "failed (exit 3)" in capsys.readouterr().out


# gabion:evidence E:call_footprint::tests/test_tooling_emit_advisory_and_gates.py::test_advisory_main_covers_missing_summary_skip_and_error::delta_advisory.py::gabion.tooling.delta_advisory.main_for_advisory
@pytest.mark.parametrize(
    ("run_main", "default_delta_path", "payload", "env_flag"),
    [
        (
            delta_advisory.obsolescence_main,
            Path("artifacts/out/test_obsolescence_delta.json"),
            {
                "summary": {
                    "counts": {
                        "baseline": {"unmapped": 1},
                        "current": {"unmapped": 2},
                        "delta": {"unmapped": 1},
                    },
                    "opaque_evidence": {"baseline": 0, "current": 1, "delta": 1},
                }
            },
            delta_advisory.OBSOLESCENCE_ENV_FLAG,
        ),
        (
            delta_advisory.annotation_drift_main,
            Path("artifacts/out/test_annotation_drift_delta.json"),
            {
                "summary": {
                    "baseline": {"orphaned": 1},
                    "current": {"orphaned": 2},
                    "delta": {"orphaned": 1},
                }
            },
            delta_advisory.ANNOTATION_DRIFT_ENV_FLAG,
        ),
        (
            delta_advisory.ambiguity_main,
            Path("artifacts/out/ambiguity_delta.json"),
            {
                "summary": {
                    "total": {"baseline": 1, "current": 2, "delta": 1},
                    "by_kind": {
                        "baseline": {"call": 1},
                        "current": {"call": 2},
                        "delta": {"call": 1},
                    },
                }
            },
            delta_advisory.AMBIGUITY_ENV_FLAG,
        ),
        (
            delta_advisory.docflow_main,
            Path("artifacts/out/docflow_compliance_delta.json"),
            {
                "summary": {
                    "baseline": {"compliant": 1},
                    "current": {"compliant": 2},
                    "delta": {"compliant": 1},
                }
            },
            "",
        ),
    ],
)
def test_advisory_main_covers_missing_summary_skip_and_error(
    tmp_path: Path,
    run_main: Any,
    default_delta_path: Path,
    payload: dict[str, object],
    env_flag: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _cwd(tmp_path):
        # Missing file branch.
        assert run_main() == 0
        missing_output = capsys.readouterr().out
        assert "missing" in missing_output.lower()

        # Skip-by-env branch for env-gated advisories.
        if env_flag:
            with env_scope({env_flag: "1"}):
                assert run_main() == 0
            skip_output = capsys.readouterr().out
            assert "skipped" in skip_output

        # Normal summary branch.
        default_delta_path.parent.mkdir(parents=True, exist_ok=True)
        default_delta_path.write_text(json.dumps(payload), encoding="utf-8")
        with env_scope({env_flag: "0"} if env_flag else {}):
            assert run_main() == 0
        summary_output = capsys.readouterr().out
        assert "summary" in summary_output.lower()

        # Exception branch.
        default_delta_path.write_text("{bad", encoding="utf-8")
        with env_scope({env_flag: "0"} if env_flag else {}):
            assert run_main() == 0
        error_output = capsys.readouterr().out
        assert "error" in error_output.lower()


# gabion:evidence E:call_footprint::tests/test_tooling_emit_advisory_and_gates.py::test_docflow_delta_emit_main_covers_all_paths::docflow_delta_emit.py::gabion.tooling.docflow_delta_emit.main
def test_docflow_delta_emit_main_covers_all_paths(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    delta_path = tmp_path / "delta.json"
    writes: list[tuple[Path, str]] = []

    plan = ExecutionPlan().with_docflow(
        DocflowFacet(changed_paths=("docs/sppf_checklist.md",))
    )

    def _write(path: Path, content: str) -> None:
        writes.append((path, content))

    def _load(path: Path) -> tuple[dict[str, int], bool]:
        if path == baseline_path:
            return {"compliant": 1, "contradicts": 1, "excess": 0, "proposed": 0}, False
        return {"compliant": 2, "contradicts": 1, "excess": 0, "proposed": 1}, False

    # Docflow command failure branch.
    def _raise_called_process_error() -> None:
        raise subprocess.CalledProcessError(1, ["gabion", "docflow"])

    assert (
        docflow_delta_emit.main(
            build_execution_plan_fn=lambda: plan,
            run_docflow_audit_fn=_raise_called_process_error,
            baseline_path=baseline_path,
            current_path=current_path,
            delta_path=delta_path,
            write_text_fn=_write,
        )
        == 0
    )
    assert not writes

    # Missing-current branch.
    assert (
        docflow_delta_emit.main(
            build_execution_plan_fn=lambda: plan,
            run_docflow_audit_fn=lambda: None,
            baseline_path=baseline_path,
            current_path=current_path,
            delta_path=delta_path,
            write_text_fn=_write,
        )
        == 0
    )
    assert not writes

    # Success branch.
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text("{}\n", encoding="utf-8")
    assert (
        docflow_delta_emit.main(
            build_execution_plan_fn=lambda: plan,
            run_docflow_audit_fn=lambda: None,
            load_summary_fn=_load,
            baseline_path=baseline_path,
            current_path=current_path,
            delta_path=delta_path,
            write_text_fn=_write,
        )
        == 0
    )
    assert writes
    written_path, written_json = writes[-1]
    assert written_path == delta_path
    payload = json.loads(written_json)
    assert payload["summary"]["delta"]["compliant"] == 1
    assert payload["facets"]["docflow"]["changed_paths"] == ["docs/sppf_checklist.md"]

    # Helper branches.
    with _cwd(tmp_path):
        assert docflow_delta_emit._changed_paths_from_git() == ()
    assert (
        docflow_delta_emit._delta_counts(
            {"compliant": 1, "contradicts": 0, "excess": 0, "proposed": 0},
            {"compliant": 3, "contradicts": 0, "excess": 0, "proposed": 0},
        )["compliant"]
        == 2
    )
    counts, missing = docflow_delta_emit._load_summary(tmp_path / "missing.json")
    assert missing is True
    assert counts["compliant"] == 0


# gabion:evidence E:call_footprint::tests/test_tooling_emit_advisory_and_gates.py::test_docflow_delta_emit_load_summary_handles_non_mapping_summary::docflow_delta_emit.py::gabion.tooling.docflow_delta_emit._load_summary
def test_docflow_delta_emit_load_summary_handles_non_mapping_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"summary": []}), encoding="utf-8")
    counts, missing = docflow_delta_emit._load_summary(summary_path)
    assert missing is False
    assert counts == {
        "compliant": 0,
        "contradicts": 0,
        "excess": 0,
        "proposed": 0,
    }


# gabion:evidence E:call_footprint::tests/test_tooling_emit_advisory_and_gates.py::test_advisory_main_env_gate_without_skip_message::delta_advisory.py::gabion.tooling.delta_advisory.main_for_advisory
def test_advisory_main_env_gate_without_skip_message() -> None:
    original = delta_advisory._ADVISORY_CONFIGS["docflow"]
    delta_advisory._ADVISORY_CONFIGS["docflow"] = delta_advisory.AdvisoryConfig(
        id="docflow",
        delta_path=original.delta_path,
        missing_message=original.missing_message,
        error_prefix=original.error_prefix,
        summary_renderer=original.summary_renderer,
        env_flag="GABION_TEST_ADVISORY_FLAG",
        skip_message=None,
    )
    lines: list[str] = []
    try:
        with env_scope({"GABION_TEST_ADVISORY_FLAG": "1"}):
            assert delta_advisory.main_for_advisory("docflow", print_fn=lines.append) == 0
        assert lines == []
    finally:
        delta_advisory._ADVISORY_CONFIGS["docflow"] = original


# gabion:evidence E:call_footprint::tests/test_tooling_emit_advisory_and_gates.py::test_docflow_delta_gate_branches::docflow_delta_gate.py::gabion.tooling.docflow_delta_gate.check_gate
def test_docflow_delta_gate_branches(tmp_path: Path) -> None:
    path = tmp_path / "docflow_delta.json"

    assert docflow_delta_gate.check_gate(path, enabled=False) == 0
    assert docflow_delta_gate.check_gate(path, enabled=True) == 0

    path.write_text("{bad", encoding="utf-8")
    assert docflow_delta_gate.check_gate(path, enabled=True) == 0

    path.write_text("[]", encoding="utf-8")
    assert docflow_delta_gate.check_gate(path, enabled=True) == 0

    path.write_text(json.dumps({"baseline_missing": True}), encoding="utf-8")
    assert docflow_delta_gate.check_gate(path, enabled=True) == 0

    path.write_text(
        json.dumps(
            {
                "summary": {
                    "baseline": {"contradicts": 1},
                    "current": {"contradicts": 2},
                    "delta": {"contradicts": 1, "excess": 0, "proposed": 0},
                }
            }
        ),
        encoding="utf-8",
    )
    assert docflow_delta_gate.check_gate(path, enabled=True) == 1

    path.write_text(
        json.dumps(
            {
                "summary": {
                    "baseline": {"contradicts": 2},
                    "current": {"contradicts": 2},
                    "delta": {"contradicts": 0, "excess": 0, "proposed": 0},
                }
            }
        ),
        encoding="utf-8",
    )
    assert docflow_delta_gate.check_gate(path, enabled=True) == 0
    assert docflow_delta_gate._enabled("1") is True
    assert docflow_delta_gate._enabled("0") is False


# gabion:evidence E:call_footprint::tests/test_tooling_emit_advisory_and_gates.py::test_docflow_delta_gate_helper_and_main_branches::docflow_delta_gate.py::gabion.tooling.docflow_delta_gate._delta_value::docflow_delta_gate.py::gabion.tooling.docflow_delta_gate.main
def test_docflow_delta_gate_helper_and_main_branches(tmp_path: Path) -> None:
    with env_scope({docflow_delta_gate.ENV_FLAG: "true"}):
        assert docflow_delta_gate._enabled() is True
    with env_scope({docflow_delta_gate.ENV_FLAG: "false"}):
        assert docflow_delta_gate._enabled() is False

    assert docflow_delta_gate._delta_value({}, "contradicts") == 0
    assert docflow_delta_gate._delta_value({"summary": []}, "contradicts") == 0
    assert docflow_delta_gate._delta_value({"summary": {"delta": []}}, "contradicts") == 0
    assert (
        docflow_delta_gate._delta_value(
            {"summary": {"delta": {"contradicts": "bad"}}},
            "contradicts",
        )
        == 0
    )

    with _cwd(tmp_path):
        target = Path("artifacts/out/docflow_compliance_delta.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"summary":{"delta":{"contradicts":0}}}', encoding="utf-8")
        assert docflow_delta_gate.main() == 0


# gabion:evidence E:call_footprint::tests/test_tooling_emit_advisory_and_gates.py::test_other_gate_non_mapping_and_main_paths::obsolescence_delta_gate.py::gabion.tooling.obsolescence_delta_gate.check_gate::ambiguity_delta_gate.py::gabion.tooling.ambiguity_delta_gate.main
@pytest.mark.parametrize(
    ("module", "default_delta_path"),
    [
        (
            obsolescence_delta_gate,
            Path("artifacts/out/test_obsolescence_delta.json"),
        ),
        (
            obsolescence_delta_unmapped_gate,
            Path("artifacts/out/test_obsolescence_delta.json"),
        ),
        (ambiguity_delta_gate, Path("artifacts/out/ambiguity_delta.json")),
        (
            annotation_drift_orphaned_gate,
            Path("artifacts/out/test_annotation_drift_delta.json"),
        ),
    ],
)
def test_other_gate_non_mapping_and_main_paths(
    tmp_path: Path,
    module: Any,
    default_delta_path: Path,
) -> None:
    path = tmp_path / "delta.json"
    path.write_text("[]", encoding="utf-8")
    assert module.check_gate(path, enabled=True) == 2

    with _cwd(tmp_path):
        # Ensure main entrypoint path exists and is non-failing for a zero delta payload.
        default_delta_path.parent.mkdir(parents=True, exist_ok=True)
        default_delta_path.write_text("{}", encoding="utf-8")
        assert module.main() in {0, 2}


# gabion:evidence E:call_footprint::tests/test_tooling_emit_advisory_and_gates.py::test_deadline_runtime_invalid_budget_and_gas_limit::deadline_runtime.py::gabion.tooling.deadline_runtime.deadline_scope_from_ticks
def test_deadline_runtime_invalid_budget_and_gas_limit() -> None:
    with pytest.raises(NeverThrown):
        deadline_runtime.DeadlineBudget(ticks=0, tick_ns=1)
    with pytest.raises(NeverThrown):
        deadline_runtime.DeadlineBudget(ticks=1, tick_ns=0)
    with pytest.raises(NeverThrown):
        with deadline_runtime.deadline_scope_from_ticks(
            deadline_runtime.DeadlineBudget(ticks=1, tick_ns=1),
            gas_limit=0,
        ):
            pass


# gabion:evidence E:call_footprint::tests/test_tooling_emit_advisory_and_gates.py::test_docflow_delta_emit_helper_and_default_write_paths::docflow_delta_emit.py::gabion.tooling.docflow_delta_emit._run_docflow_audit::docflow_delta_emit.py::gabion.tooling.docflow_delta_emit._build_execution_plan
def test_docflow_delta_emit_helper_and_default_write_paths(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def _run(args, *, check: bool, env: dict[str, str]) -> None:
        observed["args"] = list(args)
        observed["check"] = check
        observed["env"] = dict(env)
        return None

    docflow_delta_emit._run_docflow_audit(run_fn=_run)
    assert observed["args"][:4] == [docflow_delta_emit.sys.executable, "-m", "gabion", "docflow"]
    assert observed["check"] is True
    assert isinstance(observed["env"], dict)
    assert observed["env"]["GABION_DIRECT_RUN"] == "1"

    changed_paths = docflow_delta_emit._build_execution_plan(
        changed_paths_fn=lambda: ("docs/a.md", "docs/b.md")
    ).docflow.changed_paths
    assert changed_paths == ("docs/a.md", "docs/b.md")

    git_dir = tmp_path / "repo"
    git_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=git_dir, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=git_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=git_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_file = git_dir / "tracked.txt"
    tracked_file.write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=git_dir, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=git_dir, check=True, capture_output=True, text=True)
    tracked_file.write_text("a\nb\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=git_dir, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "update"], cwd=git_dir, check=True, capture_output=True, text=True)

    with _cwd(git_dir):
        changed = docflow_delta_emit._changed_paths_from_git()
    assert changed == ("tracked.txt",)

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({"summary": {"compliant": "bad", "contradicts": 2}}),
        encoding="utf-8",
    )
    counts, missing = docflow_delta_emit._load_summary(summary_path)
    assert missing is False
    assert counts == {"compliant": 0, "contradicts": 2, "excess": 0, "proposed": 0}

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    delta_path = tmp_path / "delta.json"
    baseline_path.write_text(json.dumps({"summary": {"compliant": 1}}), encoding="utf-8")
    current_path.write_text(json.dumps({"summary": {"compliant": 2}}), encoding="utf-8")
    assert (
        docflow_delta_emit.main(
            build_execution_plan_fn=lambda: ExecutionPlan().with_docflow(
                DocflowFacet(changed_paths=("docs/a.md",))
            ),
            run_docflow_audit_fn=lambda: None,
            baseline_path=baseline_path,
            current_path=current_path,
            delta_path=delta_path,
        )
        == 0
    )
    assert delta_path.exists()
