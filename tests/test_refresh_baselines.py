from __future__ import annotations

import os
from typing import Any

from tests.env_helpers import env_scope as _env_scope


def _load_refresh_baselines():
    from scripts import refresh_baselines

    return refresh_baselines


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
            "--emit-ambiguity-delta",
            timeout=5,
            timeout_env=timeout_env,
            resume_on_timeout=1,
            resume_checkpoint=None,
            run_fn=_fake_run,
        )
        assert os.environ.get("GABION_LSP_TIMEOUT_TICKS") == original_ticks
        assert os.environ.get("GABION_LSP_TIMEOUT_TICK_NS") == original_tick_ns
        assert "GABION_DIRECT_RUN" not in os.environ

    assert len(calls) == 1
    call = calls[0]
    assert call["timeout"] == 5
    assert call["env"]["GABION_DIRECT_RUN"] == "1"
    assert call["env"]["GABION_LSP_TIMEOUT_TICKS"] == str(module._DEFAULT_TIMEOUT_TICKS)
    assert call["env"]["GABION_LSP_TIMEOUT_TICK_NS"] == str(module._DEFAULT_TIMEOUT_TICK_NS)


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
        flag: str,
        timeout: int | None,
        timeout_env: module._RefreshLspTimeoutEnv,
        resume_on_timeout: int,
        *,
        resume_checkpoint,
        report_path=module.DEFAULT_CHECK_REPORT_PATH,
        extra: list[str] | None = None,
        run_fn=module.subprocess.run,
    ) -> None:
        _ = (flag, timeout, resume_on_timeout, resume_checkpoint, report_path, extra, run_fn)
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
