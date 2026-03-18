"""CLI entry point for `gabion repo extract-test-behavior`."""
from __future__ import annotations

from typing import Sequence

from scripts.misc.extract_test_behavior import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    return _main(list(argv) if argv is not None else None)
