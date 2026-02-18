#!/usr/bin/env python3
"""Legacy wrapper for `gabion docflow-audit`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gabion.cli import _run_docflow_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run governance docflow audit.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root")
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero when violations are detected.",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        default=None,
        help="Additional markdown path to include in docflow parsing.",
    )
    args = parser.parse_args(argv)
    return _run_docflow_audit(
        root=args.root,
        fail_on_violations=bool(args.fail_on_violations),
        extra_path=args.extra_path,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
