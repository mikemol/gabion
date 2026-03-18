"""CLI entry point for `gabion governance telemetry-emit`.

Thin shim delegating to `scripts.governance.governance_telemetry_emit.main(argv)`
through the `governance.telemetry-emit` tooling runner key.
"""
from __future__ import annotations

from typing import Sequence

from scripts.governance.governance_telemetry_emit import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    return _main(list(argv) if argv is not None else None)
