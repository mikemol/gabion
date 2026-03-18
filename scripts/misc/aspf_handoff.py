#!/usr/bin/env python3
"""Removed: scripts/misc/aspf_handoff.py is retired.

Use 'gabion aspf handoff [prepare|record|run] [args]' directly.
"""
from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    print(
        "Removed: scripts/misc/aspf_handoff.py is retired. "
        "Use 'gabion aspf handoff [prepare|record|run] [args]' directly. "
        "See docs/user_workflows.md#user_workflows and "
        "docs/normative_clause_index.md#clause-command-maturity-parity.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
