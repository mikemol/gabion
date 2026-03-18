"""CLI entry point for `gabion governance controller-audit`.

Thin shim delegating to `scripts.governance.governance_controller_audit.main()`
through the `governance.controller-audit` tooling runner key.

Note: `governance_controller_audit.main()` reads from `sys.argv`, so this shim
temporarily overrides `sys.argv[1:]` to pass the provided argv.
"""
from __future__ import annotations

import sys
from typing import Sequence

from scripts.governance.governance_controller_audit import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        return _main()
    old_argv = sys.argv[1:]
    sys.argv[1:] = list(argv)
    try:
        return _main()
    finally:
        sys.argv[1:] = old_argv
