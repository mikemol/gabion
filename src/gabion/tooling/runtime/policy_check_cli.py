"""CLI entry point for `gabion policy check`.

Thin shim delegating to `scripts.policy.policy_check.main(argv)` through the
`policy.check` tooling runner key.
"""
from __future__ import annotations

from typing import Sequence

from scripts.policy.policy_check import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    return _main(list(argv) if argv is not None else None)
