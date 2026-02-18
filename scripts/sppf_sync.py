#!/usr/bin/env python3
"""Compatibility shim for `gabion sppf-sync`."""
from __future__ import annotations

import sys

try:  # pragma: no cover - import form depends on invocation mode
    from scripts.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env

from gabion.cli import run_sppf_sync_compat

_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_TIMEOUT_TICK_NS,
)


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_TIMEOUT_BUDGET,
    )


def main() -> int:
    with _deadline_scope():
        return run_sppf_sync_compat(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
