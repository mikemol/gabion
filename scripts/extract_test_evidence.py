#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gabion.analysis.timeout_context import Deadline, deadline_scope
from gabion.lsp_client import _env_timeout_ticks


def _add_repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    return root


def _deadline_scope_from_env():
    ticks, tick_ns = _env_timeout_ticks()
    return deadline_scope(Deadline.from_timeout_ticks(ticks, tick_ns))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract gabion evidence tags from tests.")
    parser.add_argument(
        "--tests",
        nargs="*",
        default=["tests"],
        help="Test files or directories to scan (default: tests/).",
    )
    parser.add_argument("--out", required=True, help="Write JSON output to this path.")
    parser.add_argument("--root", default=".", help="Repo root for relative paths.")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude path prefix.")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    _add_repo_root()
    from gabion.analysis import test_evidence

    paths = [Path(item) for item in args.tests]
    with _deadline_scope_from_env():
        payload = test_evidence.build_test_evidence_payload(
            paths,
            root=root,
            include=args.tests,
            exclude=args.exclude,
        )
        test_evidence.write_test_evidence(payload, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
