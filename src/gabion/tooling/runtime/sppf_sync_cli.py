"""CLI entry point for `gabion sppf sync`.

Thin shim delegating to `gabion.tooling.sppf.sync_core.main(argv)` through the
`sppf.sync` tooling runner key.
"""
from __future__ import annotations

from typing import Sequence

from gabion.tooling.sppf import sync_core


def main(argv: Sequence[str] | None = None) -> int:
    return sync_core.main(list(argv) if argv is not None else None)
