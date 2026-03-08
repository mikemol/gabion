from __future__ import annotations

from pathlib import Path
import runpy
import sys

import pytest

from gabion.analysis.dataflow.engine import dataflow_raw_runtime as da
from gabion.exceptions import NeverThrown


# gabion:evidence E:call_footprint::tests/test_dataflow_main.py::test_legacy_dataflow_monolith_main_executes::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan
# gabion:behavior primary=allowed_unwanted facets=legacy
def test_legacy_dataflow_monolith_main_executes(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("def f(a, b):\n    return a + b\n")
    original_argv = sys.argv
    sys.argv = [
        "legacy_dataflow_monolith",
        str(sample),
        "--root",
        str(tmp_path),
        "--dot",
        "-",
    ]
    try:
        with pytest.raises(SystemExit):
            runpy.run_module(
                "gabion.analysis.dataflow.engine.dataflow_raw_runtime",
                run_name="__main__",
            )
    finally:
        sys.argv = original_argv


# gabion:evidence E:call_footprint::tests/test_dataflow_main.py::test_legacy_dataflow_monolith_run_dot_only_returns_success::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.run
# gabion:behavior primary=allowed_unwanted facets=legacy
def test_legacy_dataflow_monolith_run_dot_only_returns_success(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    dot_path = tmp_path / "sample.dot"
    sample.write_text("def f(a, b):\n    return a + b\n", encoding="utf-8")

    exit_code = da.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--dot",
            str(dot_path),
        ]
    )

    assert exit_code == 0
    assert dot_path.exists()
    assert dot_path.read_text(encoding="utf-8").startswith("digraph dataflow_grammar")


# gabion:evidence E:call_footprint::tests/test_dataflow_main.py::test_legacy_dataflow_monolith_parser_accepts_tick_options::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._build_parser
# gabion:behavior primary=allowed_unwanted facets=legacy
def test_legacy_dataflow_monolith_parser_accepts_tick_options() -> None:
    parser = da._build_parser()
    args = parser.parse_args(
        [
            "sample.py",
            "--analysis-timeout-ticks",
            "123",
            "--analysis-timeout-tick-ns",
            "456",
            "--analysis-tick-limit",
            "789",
        ]
    )
    assert args.analysis_timeout_ticks == 123
    assert args.analysis_timeout_tick_ns == 456
    assert args.analysis_tick_limit == 789


# gabion:evidence E:call_footprint::tests/test_dataflow_main.py::test_legacy_dataflow_monolith_run_uses_tick_limit_timeout::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.run
# gabion:behavior primary=allowed_unwanted facets=legacy,timeout
def test_legacy_dataflow_monolith_run_uses_tick_limit_timeout(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("def f(a, b):\n    return a + b\n", encoding="utf-8")
    with pytest.raises(da.TimeoutExceeded):
        da.run(
            [
                str(sample),
                "--root",
                str(tmp_path),
                "--analysis-timeout-ticks",
                "1000",
                "--analysis-timeout-tick-ns",
                "1000000",
                "--analysis-tick-limit",
                "1",
            ]
        )


# gabion:evidence E:call_footprint::tests/test_dataflow_main.py::test_legacy_dataflow_monolith_run_rejects_invalid_tick_config::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.run
# gabion:behavior primary=allowed_unwanted facets=legacy
def test_legacy_dataflow_monolith_run_rejects_invalid_tick_config(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("def f(a, b):\n    return a + b\n", encoding="utf-8")
    with pytest.raises(NeverThrown):
        da.run(
            [
                str(sample),
                "--root",
                str(tmp_path),
                "--analysis-timeout-ticks",
                "0",
            ]
        )


# gabion:evidence E:call_footprint::tests/test_dataflow_main.py::test_legacy_dataflow_monolith_run_rejects_invalid_tick_ns::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.run
# gabion:behavior primary=allowed_unwanted facets=legacy
def test_legacy_dataflow_monolith_run_rejects_invalid_tick_ns(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("def f(a, b):\n    return a + b\n", encoding="utf-8")
    with pytest.raises(NeverThrown):
        da.run(
            [
                str(sample),
                "--root",
                str(tmp_path),
                "--analysis-timeout-tick-ns",
                "0",
            ]
        )


# gabion:evidence E:call_footprint::tests/test_dataflow_main.py::test_legacy_dataflow_monolith_run_rejects_invalid_tick_limit::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.run
# gabion:behavior primary=allowed_unwanted facets=legacy
def test_legacy_dataflow_monolith_run_rejects_invalid_tick_limit(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("def f(a, b):\n    return a + b\n", encoding="utf-8")
    with pytest.raises(NeverThrown):
        da.run(
            [
                str(sample),
                "--root",
                str(tmp_path),
                "--analysis-timeout-ticks",
                "1000",
                "--analysis-timeout-tick-ns",
                "1000000",
                "--analysis-tick-limit",
                "0",
            ]
        )
