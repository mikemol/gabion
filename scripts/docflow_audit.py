#!/usr/bin/env python3
"""Docflow audit for governance documents."""
from __future__ import annotations

import sys

try:  # pragma: no cover - import form depends on invocation mode
    from scripts.audit_tools import run_docflow_cli, run_sppf_graph_cli
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from audit_tools import run_docflow_cli, run_sppf_graph_cli


if __name__ == "__main__":
    status = run_docflow_cli()
    if status == 0:
        try:
            run_sppf_graph_cli([])
        except Exception as exc:
            print(f"docflow: sppf-graph failed: {exc}", file=sys.stderr)
            raise SystemExit(1)
    raise SystemExit(status)
