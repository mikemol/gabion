#!/usr/bin/env python3
from __future__ import annotations

try:  # pragma: no cover - import form depends on invocation mode
    from scripts.audit_tools import run_decision_tiers_cli
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from audit_tools import run_decision_tiers_cli


if __name__ == "__main__":
    raise SystemExit(run_decision_tiers_cli())
