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


def test_execute_command_invalid_synth_min_occurrences(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    config_path = tmp_path / "gabion.toml"
    config_path.write_text(
        "[fingerprints]\n"
        "user_context = [\"int\", \"str\"]\n"
        "synth_min_occurrences = \"bad\"\n"
    )
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "config": str(config_path),
        },
    )
    assert result.get("exit_code") == 0


def test_execute_command_fingerprint_outputs_and_decision_snapshot(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "def callee(x: int):\n"
        "    return x\n"
        "\n"
        "def caller_one(a: int, b: int):\n"
        "    callee(a)\n"
        "    callee(b)\n"
        "\n"
        "def caller_two(a: int, b: int):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )
    config_path = tmp_path / "gabion.toml"
    config_path.write_text(
        "[fingerprints]\n"
        "user_context = [\"int\"]\n"
        "synth_min_occurrences = 2\n"
        "\n"
        "[decision]\n"
        "tier2 = [\"a\"]\n"
    )
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "config": str(config_path),
            "fingerprint_synth_json": "-",
            "fingerprint_provenance_json": "-",
            "decision_snapshot": "-",
        },
    )
    assert "fingerprint_synth_registry" in result
    assert "fingerprint_provenance" in result
    assert "decision_snapshot" in result


def test_execute_command_writes_fingerprint_outputs(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "def callee(x: int):\n"
        "    return x\n"
        "\n"
        "def caller_one(a: int, b: int):\n"
        "    callee(a)\n"
        "    callee(b)\n"
        "\n"
        "def caller_two(a: int, b: int):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )
    synth_registry_path = tmp_path / "synth_registry.json"
    synth_registry_path.write_text(
        "{\"version\": \"synth@1\", \"min_occurrences\": 2, \"entries\": []}"
    )
    config_path = tmp_path / "gabion.toml"
    config_path.write_text(
        "[fingerprints]\n"
        "user_context = [\"int\"]\n"
        "synth_min_occurrences = 2\n"
        f"synth_registry_path = \"{synth_registry_path}\"\n"
    )
    synth_path = tmp_path / "fingerprints.json"
    provenance_path = tmp_path / "provenance.json"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "config": str(config_path),
            "fingerprint_synth_json": str(synth_path),
            "fingerprint_provenance_json": str(provenance_path),
        },
    )
    assert result.get("exit_code") == 0
    assert synth_path.exists()
    assert provenance_path.exists()


def test_execute_command_writes_decision_snapshot(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    decision_path = tmp_path / "decision.json"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "decision_snapshot": str(decision_path),
        },
    )
    assert result.get("exit_code") == 0
    assert decision_path.exists()


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


def test_execute_structure_reuse_missing_snapshot() -> None:
    result = server.execute_structure_reuse(None, {})
    assert result.get("exit_code") == 2


def test_execute_structure_reuse_payload_none() -> None:
    result = server.execute_structure_reuse(None, None)
    assert result.get("exit_code") == 2


def test_execute_structure_reuse_invalid_snapshot(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    result = server.execute_structure_reuse(None, {"snapshot": str(bad)})
    assert result.get("exit_code") == 2


def test_execute_structure_reuse_writes_lemma_stubs(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        "{\"format_version\": 1, \"root\": null, \"files\": []}"
    )
    result = server.execute_structure_reuse(
        None,
        {"snapshot": str(snapshot_path), "lemma_stubs": "-", "min_count": "bad"},
    )
    assert result.get("exit_code") == 0
    assert "lemma_stubs" in result


def test_execute_structure_reuse_writes_lemma_stubs_file(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        "{\"format_version\": 1, \"root\": null, \"files\": []}"
    )
    lemma_path = tmp_path / "lemmas.py"
    result = server.execute_structure_reuse(
        None,
        {"snapshot": str(snapshot_path), "lemma_stubs": str(lemma_path)},
    )
    assert result.get("exit_code") == 0
    assert lemma_path.exists()


def test_execute_decision_diff_missing_paths() -> None:
    result = server.execute_decision_diff(None, {})
    assert result.get("exit_code") == 2


def test_execute_decision_diff_payload_none() -> None:
    result = server.execute_decision_diff(None, None)
    assert result.get("exit_code") == 2


def test_execute_decision_diff_invalid_snapshot(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    result = server.execute_decision_diff(
        None,
        {"baseline": str(bad), "current": str(bad)},
    )
    assert result.get("exit_code") == 2


def test_execute_decision_diff_valid_snapshot(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(
        "{\"format_version\": 1, \"root\": null, \"decision_surfaces\": [\"a\"], \"value_decision_surfaces\": []}"
    )
    current.write_text(
        "{\"format_version\": 1, \"root\": null, \"decision_surfaces\": [\"a\", \"b\"], \"value_decision_surfaces\": [\"x\"]}"
    )
    result = server.execute_decision_diff(
        None,
        {"baseline": str(baseline), "current": str(current)},
    )
    assert result.get("exit_code") == 0
    diff = result.get("diff") or {}
    assert "decision_surfaces" in diff


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
