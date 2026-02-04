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


def test_normalize_transparent_decorators() -> None:
    assert server._normalize_transparent_decorators(None) is None
    assert server._normalize_transparent_decorators("a, b") == {"a", "b"}
    assert server._normalize_transparent_decorators(["a", "b,c"]) == {"a", "b", "c"}
    assert server._normalize_transparent_decorators([]) is None


def test_uri_to_path_file_scheme() -> None:
    path = server._uri_to_path("file:///tmp/example.py")
    assert str(path).endswith("/tmp/example.py")


def test_uri_to_path_plain_path() -> None:
    path = server._uri_to_path("/tmp/example.py")
    assert str(path).endswith("/tmp/example.py")


def _write_minimal_module(path: Path) -> None:
    path.write_text(
        "def alpha():\n"
        "    return 1\n"
    )


def test_execute_command_no_violations(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "fail_on_violations": True,
            "type_audit": False,
        },
    )
    assert _CommandResult(
        exit_code=result.get("exit_code", -1),
        violations=result.get("violations", -1),
    ) == _CommandResult(exit_code=0, violations=0)


def test_execute_command_baseline_write(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    baseline_path = tmp_path / "baseline.txt"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "baseline": str(baseline_path),
            "baseline_write": True,
        },
    )
    assert result.get("baseline_written") is True
    assert baseline_path.exists()
    baseline_text = baseline_path.read_text()
    assert baseline_text.startswith("# gabion baseline")


def test_execute_command_report_and_dot(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    report_path = tmp_path / "report.md"
    dot_path = tmp_path / "graph.dot"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "report": str(report_path),
            "dot": str(dot_path),
        },
    )
    assert report_path.exists()
    assert dot_path.exists()
    assert "violations" in result


def test_execute_command_structure_tree(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    snapshot_path = tmp_path / "snapshot.json"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "structure_tree": str(snapshot_path),
        },
    )
    assert snapshot_path.exists()
    assert "violations" in result


def test_execute_command_structure_tree_stdout(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "structure_tree": "-",
        },
    )
    assert "structure_tree" in result
    assert result["structure_tree"]["files"]


def test_execute_structure_diff(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text("{\"files\": [{\"path\": \"a.py\", \"functions\": []}]}")
    current.write_text("{\"files\": [{\"path\": \"a.py\", \"functions\": [{\"name\": \"f\", \"bundles\": [[\"a\"]]}]}]}")
    result = server.execute_structure_diff(
        None,
        {"baseline": str(baseline), "current": str(current)},
    )
    assert result["exit_code"] == 0
    assert result["diff"]["added"][0]["bundle"] == ["a"]


def test_execute_structure_diff_missing_payload() -> None:
    result = server.execute_structure_diff(None, None)
    assert result["exit_code"] == 2


def test_execute_structure_diff_missing_paths() -> None:
    result = server.execute_structure_diff(None, {})
    assert result["exit_code"] == 2
    assert "required" in result["errors"][0]


def test_execute_structure_diff_invalid_snapshot(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text("{bad-json")
    current.write_text("{\"files\": []}")
    result = server.execute_structure_diff(
        None,
        {"baseline": str(baseline), "current": str(current)},
    )
    assert result["exit_code"] == 2
    assert "Invalid snapshot JSON" in result["errors"][0]


def test_execute_command_structure_metrics(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    metrics_path = tmp_path / "metrics.json"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "structure_metrics": str(metrics_path),
        },
    )
    assert metrics_path.exists()
    assert "violations" in result


def test_execute_command_structure_metrics_stdout(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "structure_metrics": "-",
        },
    )
    assert "structure_metrics" in result


def test_execute_command_synthesis_outputs(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    plan_path = tmp_path / "plan.json"
    protocol_path = tmp_path / "protocols.py"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "synthesis_plan": str(plan_path),
            "synthesis_protocols": str(protocol_path),
        },
    )
    assert plan_path.exists()
    assert protocol_path.exists()
    assert "violations" in result


def test_execute_synthesis_minimal_payload() -> None:
    result = server.execute_synthesis(None, {"bundles": []})
    assert result["protocols"] == []
    assert result["errors"] == []


def test_execute_refactor_invalid_payload() -> None:
    result = server.execute_refactor(None, {})
    assert result["errors"]
