"""CLI entry point for `gabion policy scanner`.

Thin shim delegating to `scripts.policy.policy_scanner_suite.main(argv)` through
the `policy.scanner` tooling runner key.
"""
from __future__ import annotations

from typing import Sequence

from scripts.policy.policy_scanner_suite import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    return _main(list(argv) if argv is not None else None)
