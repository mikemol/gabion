from __future__ import annotations

import importlib.util
from pathlib import Path
import json

import pytest
from typer.testing import CliRunner

from gabion import cli


def _has_pygls() -> bool:
    return importlib.util.find_spec("pygls") is not None


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands._has_pygls
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_check_and_dataflow_audit(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text(
        "def callee_int(x: int):\n"
        "    return x\n"
        "def callee_str(x: str):\n"
        "    return x\n"
        "def caller_single(a):\n"
        "    return callee_int(a)\n"
        "def caller_multi(b):\n"
        "    callee_int(b)\n"
        "    callee_str(b)\n"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "check",
            str(module),
            "--root",
            str(tmp_path),
            "--no-fail-on-violations",
            "--no-fail-on-type-ambiguities",
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        cli.app,
        [
            "dataflow-audit",
            str(module),
            "--root",
            str(tmp_path),
            "--type-audit",
            "--type-audit-max",
            "10",
            "--dot",
            "-",
            "--synthesis-plan",
            "-",
            "--synthesis-protocols",
            "-",
            "--synthesis-min-bundle-size",
            "1",
            "--synthesis-allow-singletons",
            "--refactor-plan-json",
            "-",
        ],
    )
    assert result.exit_code == 0
    assert "Type tightening candidates" in result.output
    assert "Type ambiguities" in result.output


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_docflow_audit::cli.py::gabion.cli.app
def test_cli_docflow_audit() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "docflow-audit",
            "--root",
            str(repo_root),
            "--no-fail-on-violations",
        ],
    )
    assert result.exit_code == 0


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_dataflow_audit_requires_paths::cli.py::gabion.cli.app
def test_cli_dataflow_audit_requires_paths() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["dataflow-audit"])
    assert result.exit_code != 0


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands._has_pygls
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_synth_and_synthesis_plan(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text(
        "def callee(x, y):\n"
        "    return x, y\n"
        "def caller(a, b):\n"
        "    return callee(a, b)\n"
    )
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "synth",
            str(module),
            "--root",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--synthesis-min-bundle-size",
            "1",
            "--synthesis-allow-singletons",
        ],
    )
    assert result.exit_code == 0
    assert "Snapshot:" in result.output


    result = runner.invoke(
        cli.app,
        [
            "synth",
            str(module),
            "--root",
            str(tmp_path),
            "--no-timestamp",
            "--no-refactor-plan",
            "--synthesis-min-bundle-size",
            "1",
            "--synthesis-allow-singletons",
        ],
    )
    assert result.exit_code == 0
    assert "Snapshot:" not in result.output

    payload_path = tmp_path / "synth.json"
    payload_path.write_text(
        '{"bundles":[{"bundle":["x"],"tier":2}],"allow_singletons":true,"min_bundle_size":1}'
    )
    output_path = tmp_path / "synth_out.json"
    result = runner.invoke(
        cli.app,
        [
            "synthesis-plan",
            "--input",
            str(payload_path),
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0
    assert output_path.exists()


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands._has_pygls
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_structure_diff(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps({"files": []}))
    current.write_text(json.dumps({"files": [{"functions": [{"bundles": [["a"]]}]}]}))
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "structure-diff",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert "\"diff\"" in result.output


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands._has_pygls
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_refactor_protocol(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def f(a, b):\n    return a + b\n")
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "refactor-protocol",
            "--protocol-name",
            "Bundle",
            "--bundle",
            "a",
            "--bundle",
            "b",
            "--target-path",
            str(module),
        ],
    )
    assert result.exit_code == 0


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_synthesis_plan_invalid_json::cli.py::gabion.cli.app
def test_cli_synthesis_plan_invalid_json(tmp_path: Path) -> None:
    payload_path = tmp_path / "bad.json"
    payload_path.write_text("{bad")
    runner = CliRunner()
    result = runner.invoke(cli.app, ["synthesis-plan", "--input", str(payload_path)])
    assert result.exit_code != 0
    assert "Invalid JSON payload" in result.output


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_refactor_protocol_invalid_json::cli.py::gabion.cli.app
def test_cli_refactor_protocol_invalid_json(tmp_path: Path) -> None:
    payload_path = tmp_path / "bad.json"
    payload_path.write_text("{bad")
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "refactor-protocol",
            "--input",
            str(payload_path),
        ],
    )
    assert result.exit_code != 0
    assert "Invalid JSON payload" in result.output


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands._has_pygls
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_synthesis_plan_stdout(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text('{"bundles":[{"bundle":["x"],"tier":2}]}')
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "synthesis-plan",
            "--input",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0
    assert result.output.strip().startswith("{")


# gabion:evidence E:function_site::test_cli_commands.py::tests.test_cli_commands._has_pygls
@pytest.mark.skipif(not _has_pygls(), reason="pygls not installed")
def test_cli_refactor_protocol_output_file(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def f(a, b):\n    return a + b\n")
    out_path = tmp_path / "out.json"
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "refactor-protocol",
            "--protocol-name",
            "Bundle",
            "--bundle",
            "a",
            "--bundle",
            "b",
            "--target-path",
            str(module),
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0
    assert out_path.exists()


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_synth_invalid_strictness::cli.py::gabion.cli.app
def test_cli_synth_invalid_strictness(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def f(a, b):\n    return a\n")
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["synth", str(module), "--root", str(tmp_path), "--strictness", "weird"],
    )
    assert result.exit_code != 0
    assert "strictness" in result.output


# gabion:evidence E:call_footprint::tests/test_cli_commands.py::test_cli_synth_invalid_protocols_kind::cli.py::gabion.cli.app
def test_cli_synth_invalid_protocols_kind(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("def f(a, b):\n    return a\n")
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "synth",
            str(module),
            "--root",
            str(tmp_path),
            "--synthesis-protocols-kind",
            "nope",
        ],
    )
    assert result.exit_code != 0
    assert "synthesis-protocols-kind" in result.output
