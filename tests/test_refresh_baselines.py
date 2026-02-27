from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any

from tests.env_helpers import env_scope as _env_scope


def _load_refresh_baselines():
    from scripts import refresh_baselines

    return refresh_baselines


@contextmanager
def _cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


# gabion:evidence E:call_footprint::tests/test_refresh_baselines.py::test_refresh_subprocess_env_injects_timeout_budget_without_mutating_process_env::env_helpers.py::tests.env_helpers.env_scope::test_refresh_baselines.py::tests.test_refresh_baselines._load_refresh_baselines
def test_refresh_subprocess_env_injects_timeout_budget_without_mutating_process_env(
) -> None:
    module = _load_refresh_baselines()
    timeout_env = module._refresh_lsp_timeout_env(None, None)
    calls: list[dict[str, Any]] = []

    # dataflow-bundle: check, cmd, env, timeout
    def _fake_run(cmd, *, check, timeout, env):
        calls.append(
            {
                "cmd": cmd,
                "check": check,
                "timeout": timeout,
                "env": env,
            }
        )

    with _env_scope(
        {
            "GABION_LSP_TIMEOUT_TICKS": "7",
            "GABION_LSP_TIMEOUT_TICK_NS": "9",
            "GABION_DIRECT_RUN": None,
        }
    ):
        original_ticks = os.environ.get("GABION_LSP_TIMEOUT_TICKS")
        original_tick_ns = os.environ.get("GABION_LSP_TIMEOUT_TICK_NS")
        module._run_check(
            [
                "ambiguity",
                "delta",
                "--baseline",
                str(module.AMBIGUITY_BASELINE_PATH),
            ],
            timeout=5,
            timeout_env=timeout_env,
            run_fn=_fake_run,
        )
        assert os.environ.get("GABION_LSP_TIMEOUT_TICKS") == original_ticks
        assert os.environ.get("GABION_LSP_TIMEOUT_TICK_NS") == original_tick_ns
        assert "GABION_DIRECT_RUN" not in os.environ

    assert len(calls) == 1
    call = calls[0]
    assert call["timeout"] == 5
    assert call["cmd"][3:5] == ["--timeout", "120000000000ns"]


# gabion:evidence E:call_footprint::tests/test_refresh_baselines.py::test_refresh_lsp_timeout_env_overrides_defaults::test_refresh_baselines.py::tests.test_refresh_baselines._load_refresh_baselines
def test_refresh_lsp_timeout_env_overrides_defaults() -> None:
    module = _load_refresh_baselines()

    timeout_env = module._refresh_lsp_timeout_env(123, 456)

    assert timeout_env.ticks == 123
    assert timeout_env.tick_ns == 456


# gabion:evidence E:call_footprint::tests/test_refresh_baselines.py::test_main_uses_cli_timeout_overrides_for_refresh_operations::test_refresh_baselines.py::tests.test_refresh_baselines._load_refresh_baselines
def test_main_uses_cli_timeout_overrides_for_refresh_operations() -> None:
    module = _load_refresh_baselines()
    captured: list[module._RefreshLspTimeoutEnv] = []

    def _capture_run_check(
        subcommand: list[str],
        timeout: int | None,
        timeout_env: module._RefreshLspTimeoutEnv,
        *,
        report_path=module.DEFAULT_CHECK_REPORT_PATH,
        extra: list[str] | None = None,
        run_fn=module.subprocess.run,
    ) -> None:
        _ = (
            subcommand,
            timeout,
            report_path,
            extra,
            run_fn,
        )
        captured.append(timeout_env)

    exit_code = module.main(
        [
            "--obsolescence",
            "--lsp-timeout-ticks",
            "222",
            "--lsp-timeout-tick-ns",
            "333",
        ],
        deadline_scope_factory=lambda: module.deadline_scope_from_lsp_env(
            default_budget=module.DeadlineBudget(ticks=10, tick_ns=1_000_000)
        ),
        run_check_fn=_capture_run_check,
        guard_obsolescence_delta_fn=lambda *args, **kwargs: None,
    )

    assert exit_code == 0
    assert captured
    assert captured[0].ticks == 222
    assert captured[0].tick_ns == 333


# gabion:evidence E:function_site::tests/test_refresh_baselines.py::test_requires_block_is_monotonic_without_override
def test_requires_block_is_monotonic_without_override() -> None:
    module = _load_refresh_baselines()
    with _env_scope(
        {
            "GABION_POLICY_OVERRIDE_TOKEN": None,
            "GABION_POLICY_OVERRIDE_RATIONALE": None,
        }
    ):
        assert module._requires_block("obsolescence_opaque", 0) is False
        assert module._requires_block("obsolescence_opaque", 1) is True
        assert module._requires_block("obsolescence_opaque", 2) is True


# gabion:evidence E:function_site::tests/test_refresh_baselines.py::test_requires_block_allows_override_token_with_rationale
def test_requires_block_allows_override_token_with_rationale() -> None:
    module = _load_refresh_baselines()
    with _env_scope(
        {
            "GABION_POLICY_OVERRIDE_TOKEN": "policy-override-123",
            "GABION_POLICY_OVERRIDE_RATIONALE": "temporary strictness reduction for controlled migration",
        }
    ):
        assert module._requires_block("obsolescence_opaque", 1) is False


# gabion:evidence E:call_footprint::tests/test_refresh_baselines.py::test_main_enables_default_aspf_handoff_and_writes_manifest::refresh_baselines.py::scripts.refresh_baselines.main
def test_main_enables_default_aspf_handoff_and_writes_manifest(tmp_path: Path) -> None:
    module = _load_refresh_baselines()
    captured: list[list[str]] = []

    def _capture_run_check(
        subcommand: list[str],
        timeout: int | None,
        timeout_env: module._RefreshLspTimeoutEnv,
        *,
        report_path=module.DEFAULT_CHECK_REPORT_PATH,
        extra: list[str] | None = None,
        run_fn=module.subprocess.run,
    ) -> None:
        _ = (
            subcommand,
            timeout,
            timeout_env,
            report_path,
            run_fn,
        )
        captured.append(list(extra or []))

    with _cwd(tmp_path):
        exit_code = module.main(
            [
                "--obsolescence",
                "--aspf-handoff-session",
                "session-refresh",
                "--aspf-handoff-manifest",
                "artifacts/out/aspf_handoff_manifest.json",
                "--aspf-state-root",
                "artifacts/out/aspf_state",
            ],
            deadline_scope_factory=lambda: module.deadline_scope_from_lsp_env(
                default_budget=module.DeadlineBudget(ticks=10, tick_ns=1_000_000)
            ),
            run_check_fn=_capture_run_check,
            guard_obsolescence_delta_fn=lambda *args, **kwargs: None,
            guard_annotation_drift_delta_fn=lambda *args, **kwargs: None,
            guard_ambiguity_delta_fn=lambda *args, **kwargs: None,
        )

    assert exit_code == 0
    assert captured
    first_extra = captured[0]
    assert "--aspf-state-json" in first_extra
    assert "--aspf-import-state" not in first_extra

    manifest_path = tmp_path / "artifacts/out/aspf_handoff_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "session-refresh"
    assert payload["entries"][0]["status"] == "success"


# gabion:evidence E:call_footprint::tests/test_refresh_baselines.py::test_main_aspf_handoff_imports_prior_successful_state::refresh_baselines.py::scripts.refresh_baselines.main
def test_main_aspf_handoff_imports_prior_successful_state(tmp_path: Path) -> None:
    module = _load_refresh_baselines()
    captures: list[list[str]] = []

    def _capture_run_check(
        subcommand: list[str],
        timeout: int | None,
        timeout_env: module._RefreshLspTimeoutEnv,
        *,
        report_path=module.DEFAULT_CHECK_REPORT_PATH,
        extra: list[str] | None = None,
        run_fn=module.subprocess.run,
    ) -> None:
        _ = (
            subcommand,
            timeout,
            timeout_env,
            report_path,
            run_fn,
        )
        captured_extra = list(extra or [])
        if "--aspf-state-json" in captured_extra:
            state_index = captured_extra.index("--aspf-state-json")
            state_path = Path(captured_extra[state_index + 1])
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text("{}", encoding="utf-8")
        captures.append(captured_extra)

    args = [
        "--obsolescence",
        "--aspf-handoff-session",
        "session-refresh",
        "--aspf-handoff-manifest",
        "artifacts/out/aspf_handoff_manifest.json",
        "--aspf-state-root",
        "artifacts/out/aspf_state",
    ]
    with _cwd(tmp_path):
        first_exit = module.main(
            args,
            deadline_scope_factory=lambda: module.deadline_scope_from_lsp_env(
                default_budget=module.DeadlineBudget(ticks=10, tick_ns=1_000_000)
            ),
            run_check_fn=_capture_run_check,
            guard_obsolescence_delta_fn=lambda *args, **kwargs: None,
            guard_annotation_drift_delta_fn=lambda *args, **kwargs: None,
            guard_ambiguity_delta_fn=lambda *args, **kwargs: None,
        )
        first_extra = captures.pop()
        first_state_path = first_extra[first_extra.index("--aspf-state-json") + 1]
        second_exit = module.main(
            args,
            deadline_scope_factory=lambda: module.deadline_scope_from_lsp_env(
                default_budget=module.DeadlineBudget(ticks=10, tick_ns=1_000_000)
            ),
            run_check_fn=_capture_run_check,
            guard_obsolescence_delta_fn=lambda *args, **kwargs: None,
            guard_annotation_drift_delta_fn=lambda *args, **kwargs: None,
            guard_ambiguity_delta_fn=lambda *args, **kwargs: None,
        )
        second_extra = captures.pop()

    assert first_exit == 0
    assert second_exit == 0
    assert "--aspf-import-state" in second_extra
    import_index = second_extra.index("--aspf-import-state")
    assert second_extra[import_index + 1] == first_state_path
