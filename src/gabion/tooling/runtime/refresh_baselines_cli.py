"""CLI entry point for `gabion repo refresh-baselines`."""
from __future__ import annotations

from typing import Sequence

from scripts.misc.refresh_baselines import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    return _main(list(argv) if argv is not None else None)
