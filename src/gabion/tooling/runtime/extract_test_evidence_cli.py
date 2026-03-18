"""CLI entry point for `gabion repo extract-test-evidence`."""
from __future__ import annotations

from typing import Sequence

from scripts.misc.extract_test_evidence import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    return _main(list(argv) if argv is not None else None)
