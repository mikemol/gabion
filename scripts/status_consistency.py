#!/usr/bin/env python3
"""Legacy wrapper for `gabion status-consistency`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gabion.cli import _run_governance_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SPPF status consistency checks.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root")
    parser.add_argument(
        "--extra-path",
        action="append",
        default=None,
        help="Additional markdown path to include in docflow parsing.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("artifacts/out/status_consistency.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=Path("artifacts/audit_reports/status_consistency.md"),
        help="Markdown output path",
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero when violations are detected.",
    )
    args = parser.parse_args(argv)

    call_args = [
        "--root",
        str(args.root),
        "--json-output",
        str(args.json_output),
        "--md-output",
        str(args.md_output),
    ]
    for entry in args.extra_path or []:
        call_args.extend(["--extra-path", entry])
    if args.fail_on_violations:
        call_args.append("--fail-on-violations")
    return _run_governance_cli(runner_name="run_status_consistency_cli", args=call_args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
