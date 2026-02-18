from __future__ import annotations

import os
from typing import Any

from tests.env_helpers import env_scope as _env_scope


def _load_refresh_baselines():
    from scripts import refresh_baselines

    return refresh_baselines


def test_refresh_subprocess_env_injects_timeout_budget_without_mutating_process_env(
    monkeypatch,
) -> None:
    module = _load_refresh_baselines()
    timeout_env = module._refresh_lsp_timeout_env(None, None)
    calls: list[dict[str, Any]] = []

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
        monkeypatch.setattr(module.subprocess, "run", _fake_run)
        module._run_check("--emit-ambiguity-delta", timeout=5, timeout_env=timeout_env)
        assert os.environ.get("GABION_LSP_TIMEOUT_TICKS") == original_ticks
        assert os.environ.get("GABION_LSP_TIMEOUT_TICK_NS") == original_tick_ns
        assert "GABION_DIRECT_RUN" not in os.environ

    assert len(calls) == 1
    call = calls[0]
    assert call["timeout"] == 5
    assert call["env"]["GABION_DIRECT_RUN"] == "1"
    assert call["env"]["GABION_LSP_TIMEOUT_TICKS"] == str(module._DEFAULT_TIMEOUT_TICKS)
    assert call["env"]["GABION_LSP_TIMEOUT_TICK_NS"] == str(module._DEFAULT_TIMEOUT_TICK_NS)


def test_refresh_lsp_timeout_env_overrides_defaults() -> None:
    module = _load_refresh_baselines()

    timeout_env = module._refresh_lsp_timeout_env(123, 456)

    assert timeout_env.ticks == 123
    assert timeout_env.tick_ns == 456


def test_main_uses_cli_timeout_overrides_for_refresh_operations(monkeypatch) -> None:
    module = _load_refresh_baselines()
    captured: list[module._RefreshLspTimeoutEnv] = []

    monkeypatch.setattr(module, "_guard_obsolescence_delta", lambda *_: None)

    def _capture_run_check(
        flag: str,
        timeout: int | None,
        timeout_env: module._RefreshLspTimeoutEnv,
        extra: list[str] | None = None,
    ) -> None:
        captured.append(timeout_env)

    monkeypatch.setattr(module, "_run_check", _capture_run_check)
    monkeypatch.setattr(
        module,
        "_deadline_scope",
        lambda: module.deadline_scope_from_lsp_env(
            default_budget=module.DeadlineBudget(ticks=10, tick_ns=1_000_000)
        ),
    )

    with monkeypatch.context() as context:
        context.setattr(
            module.sys,
            "argv",
            [
                "refresh_baselines.py",
                "--obsolescence",
                "--lsp-timeout-ticks",
                "222",
                "--lsp-timeout-tick-ns",
                "333",
            ],
        )
        exit_code = module.main()

    assert exit_code == 0
    assert captured
    assert captured[0].ticks == 222
    assert captured[0].tick_ns == 333
