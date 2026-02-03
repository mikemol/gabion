from __future__ import annotations

from pathlib import Path
import runpy
import sys

import pytest


def test_dataflow_audit_main_executes(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("def f(a, b):\n    return a + b\n")
    original_argv = sys.argv
    sys.argv = [
        "dataflow_audit",
        str(sample),
        "--root",
        str(tmp_path),
        "--dot",
        "-",
    ]
    try:
        with pytest.raises(SystemExit):
            runpy.run_module("gabion.analysis.dataflow_audit", run_name="__main__")
    finally:
        sys.argv = original_argv
