"""CLI entry point for `gabion policy docflow-packetize`.

Thin shim delegating to `scripts.policy.docflow_packetize.main(argv)` through
the `policy.docflow-packetize` tooling runner key.
"""
from __future__ import annotations

from typing import Sequence

from scripts.policy.docflow_packetize import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    return _main(list(argv) if argv is not None else None)
