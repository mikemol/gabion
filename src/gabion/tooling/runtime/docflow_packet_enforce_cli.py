"""CLI entry point for `gabion policy docflow-packet-enforce`.

Thin shim delegating to `scripts.policy.docflow_packet_enforce.main(argv)`
through the `policy.docflow-packet-enforce` tooling runner key.
"""
from __future__ import annotations

from typing import Sequence

from scripts.policy.docflow_packet_enforce import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    return _main(list(argv) if argv is not None else None)
