"""CLI entry point for `gabion sppf status-audit`.

Thin shim delegating to `scripts.sppf.sppf_status_audit.main(argv)` through the
`sppf.status-audit` tooling runner key.
"""
from __future__ import annotations

from typing import Sequence

from scripts.sppf.sppf_status_audit import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    return _main(list(argv) if argv is not None else None)
