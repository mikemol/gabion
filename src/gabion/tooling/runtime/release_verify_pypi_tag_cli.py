"""CLI entry point for `gabion release verify-pypi-tag`."""
from __future__ import annotations

import sys
from typing import Sequence

from scripts.release.release_verify_pypi_tag import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    old_argv = sys.argv[1:]
    sys.argv[1:] = list(argv or [])
    try:
        _main()
    finally:
        sys.argv[1:] = old_argv
    return 0
