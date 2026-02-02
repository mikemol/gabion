#!/usr/bin/env python3
"""Smoke test for the LSP command path."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from gabion.lsp_client import run_command
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.lsp_client import run_command


DATAFLOW_COMMAND = "gabion.dataflowAudit"
SYNTHESIS_COMMAND = "gabion.synthesisPlan"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    payload = {
        "paths": [args.root],
        "fail_on_violations": False,
    }
    result = run_command(DATAFLOW_COMMAND, [payload], root=Path(args.root))
    if "exit_code" not in result:
        raise SystemExit("Missing exit_code in LSP result")
    synth_payload = {
        "bundles": [{"bundle": ["ctx"], "tier": 2}],
        "min_bundle_size": 1,
        "allow_singletons": True,
        "existing_names": ["CtxBundle"],
    }
    synth_result = run_command(SYNTHESIS_COMMAND, [synth_payload], root=Path(args.root))
    if "protocols" not in synth_result:
        raise SystemExit("Missing protocols in synthesis result")
    return int(result.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
