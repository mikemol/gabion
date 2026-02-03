from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gabion import server


class _DummyWorkspace:
    def __init__(self, root_path: str) -> None:
        self.root_path = root_path


class _DummyServer:
    def __init__(self, root_path: str) -> None:
        self.workspace = _DummyWorkspace(root_path)


@dataclass
class _CommandResult:
    exit_code: int
    violations: int


def _write_bundle_module(path: Path) -> None:
    path.write_text(
        "def callee(x):\n"
        "    return x\n"
        "\n"
        "def caller(a, b):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )


def _write_type_conflict_module(path: Path) -> None:
    path.write_text(
        "def callee_int(x: int):\n"
        "    return x\n"
        "\n"
        "def callee_str(x: str):\n"
        "    return x\n"
        "\n"
        "def caller_conflict(b):\n"
        "    callee_int(b)\n"
        "    callee_str(b)\n"
    )


def test_execute_command_dash_outputs(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "dot": "-",
            "synthesis_plan": "-",
            "synthesis_protocols": "-",
            "refactor_plan_json": "-",
        },
    )
    assert "dot" in result
    assert "synthesis_plan" in result
    assert "synthesis_protocols" in result
    assert "refactor_plan" in result


def test_execute_command_baseline_apply(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("preexisting\n")
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "baseline": str(baseline_path),
            "fail_on_violations": True,
        },
    )
    assert result.get("baseline_written") is False
    assert _CommandResult(
        exit_code=result.get("exit_code", -1),
        violations=result.get("violations", -1),
    ) == _CommandResult(exit_code=1, violations=result.get("violations", -1))


def test_execute_command_fail_on_type_ambiguities(tmp_path: Path) -> None:
    module_path = tmp_path / "types.py"
    _write_type_conflict_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "fail_on_type_ambiguities": True,
        },
    )
    assert result.get("exit_code") == 1
    assert result.get("type_ambiguities")


def test_execute_command_report_baseline_write(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    report_path = tmp_path / "report.md"
    baseline_path = tmp_path / "baseline.txt"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "report": str(report_path),
            "baseline": str(baseline_path),
            "baseline_write": True,
            "fail_on_violations": True,
        },
    )
    assert result.get("baseline_written") is True
    assert result.get("exit_code") == 0
    assert report_path.exists()
    assert baseline_path.exists()


def test_execute_command_report_appends_sections(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    report_path = tmp_path / "report.md"
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("legacy\n")
    refactor_json = tmp_path / "refactor.json"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "report": str(report_path),
            "baseline": str(baseline_path),
            "synthesis_report": True,
            "refactor_plan": True,
            "refactor_plan_json": str(refactor_json),
        },
    )
    assert result.get("baseline_written") is False
    assert report_path.exists()
    report_text = report_path.read_text()
    assert "Synthesis plan" in report_text
    assert "Refactoring plan" in report_text
    assert refactor_json.exists()


def test_execute_command_defaults_payload(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(ls, None)
    assert "violations" in result
    assert "exit_code" in result


def test_execute_refactor_valid_payload(tmp_path: Path) -> None:
    module_path = tmp_path / "target.py"
    module_path.write_text("def f(a, b):\n    return a + b\n")
    ls = _DummyServer(str(tmp_path))
    result = server.execute_refactor(
        ls,
        {
            "protocol_name": "ExampleProto",
            "bundle": ["a", "b"],
            "target_path": str(module_path),
            "target_functions": [],
        },
    )
    assert result.get("errors") == []
    edits = result.get("edits", [])
    assert edits


def test_execute_refactor_invalid_payload(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_refactor(ls, {"protocol_name": 123})
    assert result.get("errors")


def test_execute_refactor_payload_none(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_refactor(ls, None)
    assert result.get("errors")


def test_execute_synthesis_invalid_payload(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_synthesis(ls, {"bundles": "not-a-list"})
    assert result.get("errors")


def test_execute_synthesis_records_bundle_tiers(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_synthesis(
        ls,
        {
            "bundles": [
                {"bundle": ["a", "b"], "tier": 2},
            ],
            "existing_names": [],
        },
    )
    assert result.get("errors") == []
    assert "protocols" in result


def test_execute_synthesis_payload_none(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_synthesis(ls, None)
    assert result.get("errors")


def test_execute_synthesis_skips_empty_bundle(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_synthesis(
        ls,
        {
            "bundles": [{"bundle": [], "tier": 2}],
        },
    )
    assert result.get("protocols") == []
