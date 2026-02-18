#!/usr/bin/env python3
"""Legacy wrapper for `gabion sppf-graph`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gabion.cli import _run_governance_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit SPPF dependency graph artifacts.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("artifacts/sppf_dependency_graph.json"),
        help="JSON output path",
    )
    parser.add_argument("--dot-output", type=Path, default=None, help="Optional DOT output path")
    parser.add_argument("--issues-json", type=Path, default=None, help="Optional GH issues JSON")
    args = parser.parse_args(argv)

    call_args = ["--root", str(args.root), "--json-output", str(args.json_output)]
    if args.dot_output is not None:
        call_args.extend(["--dot-output", str(args.dot_output)])
    if args.issues_json is not None:
        call_args.extend(["--issues-json", str(args.issues_json)])
    return _run_governance_cli(runner_name="run_sppf_graph_cli", args=call_args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
