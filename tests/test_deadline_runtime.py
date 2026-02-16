from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from gabion.analysis.timeout_context import TimeoutExceeded, check_deadline
from gabion.exceptions import NeverThrown


def _load_deadline_runtime():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import deadline_runtime

    return deadline_runtime


def test_timeout_ticks_from_lsp_env_uses_defaults() -> None:
    dr = _load_deadline_runtime()
    previous_ticks = os.environ.get("GABION_LSP_TIMEOUT_TICKS")
    previous_tick_ns = os.environ.get("GABION_LSP_TIMEOUT_TICK_NS")
    previous_ms = os.environ.get("GABION_LSP_TIMEOUT_MS")
    previous_seconds = os.environ.get("GABION_LSP_TIMEOUT_SECONDS")
    try:
        os.environ.pop("GABION_LSP_TIMEOUT_TICKS", None)
        os.environ.pop("GABION_LSP_TIMEOUT_TICK_NS", None)
        os.environ.pop("GABION_LSP_TIMEOUT_MS", None)
        os.environ.pop("GABION_LSP_TIMEOUT_SECONDS", None)
        budget = dr.timeout_budget_from_lsp_env(
            default_budget=dr.DeadlineBudget(ticks=7, tick_ns=9),
        )
    finally:
        if previous_ticks is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICKS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICKS"] = previous_ticks
        if previous_tick_ns is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICK_NS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICK_NS"] = previous_tick_ns
        if previous_ms is None:
            os.environ.pop("GABION_LSP_TIMEOUT_MS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_MS"] = previous_ms
        if previous_seconds is None:
            os.environ.pop("GABION_LSP_TIMEOUT_SECONDS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_SECONDS"] = previous_seconds
    assert budget.ticks == 7
    assert budget.tick_ns == 9


def test_timeout_ticks_from_lsp_env_uses_env_override_and_rejects_invalid() -> None:
    dr = _load_deadline_runtime()
    previous_ticks = os.environ.get("GABION_LSP_TIMEOUT_TICKS")
    previous_tick_ns = os.environ.get("GABION_LSP_TIMEOUT_TICK_NS")
    try:
        os.environ["GABION_LSP_TIMEOUT_TICKS"] = "11"
        os.environ["GABION_LSP_TIMEOUT_TICK_NS"] = "13"
        budget = dr.timeout_budget_from_lsp_env(
            default_budget=dr.DeadlineBudget(ticks=1, tick_ns=1),
        )
        assert budget.ticks == 11
        assert budget.tick_ns == 13

        os.environ["GABION_LSP_TIMEOUT_TICKS"] = ""
        os.environ["GABION_LSP_TIMEOUT_TICK_NS"] = "13"
        with pytest.raises(NeverThrown):
            dr.timeout_budget_from_lsp_env(
                default_budget=dr.DeadlineBudget(ticks=1, tick_ns=1),
            )
    finally:
        if previous_ticks is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICKS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICKS"] = previous_ticks
        if previous_tick_ns is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICK_NS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICK_NS"] = previous_tick_ns


def test_deadline_budget_rejects_non_positive_fields() -> None:
    dr = _load_deadline_runtime()
    with pytest.raises(NeverThrown):
        dr.DeadlineBudget(ticks=0, tick_ns=1)
    with pytest.raises(NeverThrown):
        dr.DeadlineBudget(ticks=1, tick_ns=0)


def test_deadline_scope_from_ticks_supports_deadline_checks() -> None:
    dr = _load_deadline_runtime()
    with dr.deadline_scope_from_ticks(
        dr.DeadlineBudget(ticks=5, tick_ns=1_000_000_000),
    ):
        check_deadline()


def test_deadline_scope_from_ticks_respects_gas_limit() -> None:
    dr = _load_deadline_runtime()
    with dr.deadline_scope_from_ticks(
        dr.DeadlineBudget(ticks=5, tick_ns=1_000_000_000),
        gas_limit=1,
    ):
        with pytest.raises(TimeoutExceeded):
            check_deadline()


def test_deadline_scope_from_ticks_rejects_invalid_gas_limit() -> None:
    dr = _load_deadline_runtime()
    with pytest.raises(NeverThrown):
        with dr.deadline_scope_from_ticks(
            dr.DeadlineBudget(ticks=5, tick_ns=1_000_000_000),
            gas_limit=0,
        ):
            check_deadline()


def test_deadline_scope_from_ticks_unwinds_on_inner_exception() -> None:
    dr = _load_deadline_runtime()
    with pytest.raises(RuntimeError):
        with dr.deadline_scope_from_ticks(
            dr.DeadlineBudget(ticks=5, tick_ns=1_000_000_000),
        ):
            raise RuntimeError("boom")


def test_deadline_scope_from_lsp_env_uses_default_and_explicit_gas_limit() -> None:
    dr = _load_deadline_runtime()
    previous_ticks = os.environ.get("GABION_LSP_TIMEOUT_TICKS")
    previous_tick_ns = os.environ.get("GABION_LSP_TIMEOUT_TICK_NS")
    previous_ms = os.environ.get("GABION_LSP_TIMEOUT_MS")
    previous_seconds = os.environ.get("GABION_LSP_TIMEOUT_SECONDS")
    try:
        os.environ.pop("GABION_LSP_TIMEOUT_TICKS", None)
        os.environ.pop("GABION_LSP_TIMEOUT_TICK_NS", None)
        os.environ.pop("GABION_LSP_TIMEOUT_MS", None)
        os.environ.pop("GABION_LSP_TIMEOUT_SECONDS", None)
        with dr.deadline_scope_from_lsp_env(
            default_budget=dr.DeadlineBudget(ticks=4, tick_ns=1_000_000_000),
            gas_limit=2,
        ):
            check_deadline()
            with pytest.raises(TimeoutExceeded):
                check_deadline()
    finally:
        if previous_ticks is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICKS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICKS"] = previous_ticks
        if previous_tick_ns is None:
            os.environ.pop("GABION_LSP_TIMEOUT_TICK_NS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_TICK_NS"] = previous_tick_ns
        if previous_ms is None:
            os.environ.pop("GABION_LSP_TIMEOUT_MS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_MS"] = previous_ms
        if previous_seconds is None:
            os.environ.pop("GABION_LSP_TIMEOUT_SECONDS", None)
        else:
            os.environ["GABION_LSP_TIMEOUT_SECONDS"] = previous_seconds
