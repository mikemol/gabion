#!/usr/bin/env python3
"""Thin wrapper for the packaged dataflow audit."""
from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap_src() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))


_bootstrap_src()

from gabion.analysis.dataflow_audit import main  # noqa: E402


if __name__ == "__main__":
    main()
