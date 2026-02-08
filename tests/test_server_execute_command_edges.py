from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from gabion import server
from gabion.analysis import (
    ambiguity_delta,
    ambiguity_state,
    evidence_keys,
    test_annotation_drift,
    test_annotation_drift_delta,
    test_obsolescence,
    test_obsolescence_delta,
    test_obsolescence_state,
)


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


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
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


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
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


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_ignores_invalid_timeout(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "analysis_timeout_seconds": "nope",
        },
    )
    assert result.get("exit_code") == 0


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_reports_timeout(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "report": str(tmp_path / "report.md"),
            "analysis_timeout_seconds": "1e-9",
        },
    )
    assert result.get("exit_code") == 2
    assert result.get("timeout") is True
    assert "timeout_context" in result


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
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
            "fingerprint_deadness_json": "-",
            "fingerprint_coherence_json": "-",
            "fingerprint_rewrite_plans_json": "-",
            "fingerprint_exception_obligations_json": "-",
            "fingerprint_handledness_json": "-",
            "decision_snapshot": "-",
        },
    )
    assert "fingerprint_synth_registry" in result
    assert "fingerprint_provenance" in result
    assert "fingerprint_deadness" in result
    assert "fingerprint_coherence" in result
    assert "fingerprint_rewrite_plans" in result
    assert "fingerprint_exception_obligations" in result
    assert "fingerprint_handledness" in result
    assert "decision_snapshot" in result


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
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
    deadness_path = tmp_path / "deadness.json"
    coherence_path = tmp_path / "coherence.json"
    rewrite_plans_path = tmp_path / "rewrite_plans.json"
    exception_obligations_path = tmp_path / "exception_obligations.json"
    handledness_path = tmp_path / "handledness.json"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "config": str(config_path),
            "fingerprint_synth_json": str(synth_path),
            "fingerprint_provenance_json": str(provenance_path),
            "fingerprint_deadness_json": str(deadness_path),
            "fingerprint_coherence_json": str(coherence_path),
            "fingerprint_rewrite_plans_json": str(rewrite_plans_path),
            "fingerprint_exception_obligations_json": str(exception_obligations_path),
            "fingerprint_handledness_json": str(handledness_path),
        },
    )
    assert result.get("exit_code") == 0
    assert synth_path.exists()
    assert provenance_path.exists()
    assert deadness_path.exists()
    assert coherence_path.exists()
    assert rewrite_plans_path.exists()
    assert exception_obligations_path.exists()
    assert handledness_path.exists()


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
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


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
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


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
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


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_includes_lint_lines(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "lint": True,
        },
    )
    assert "lint_lines" in result


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
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


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count E:decision_surface/direct::server.py::gabion.server.execute_structure_reuse::payload
def test_execute_structure_reuse_missing_snapshot() -> None:
    result = server.execute_structure_reuse(None, {})
    assert result.get("exit_code") == 2


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count E:decision_surface/direct::server.py::gabion.server.execute_structure_reuse::payload
def test_execute_structure_reuse_payload_none() -> None:
    result = server.execute_structure_reuse(None, None)
    assert result.get("exit_code") == 2


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count E:decision_surface/direct::server.py::gabion.server.execute_structure_reuse::payload
def test_execute_structure_reuse_invalid_snapshot(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    result = server.execute_structure_reuse(None, {"snapshot": str(bad)})
    assert result.get("exit_code") == 2


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count E:decision_surface/direct::server.py::gabion.server.execute_structure_reuse::payload
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


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count E:decision_surface/direct::server.py::gabion.server.execute_structure_reuse::payload
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


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_decision_diff::payload
def test_execute_decision_diff_missing_paths() -> None:
    result = server.execute_decision_diff(None, {})
    assert result.get("exit_code") == 2


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_decision_diff::payload
def test_execute_decision_diff_payload_none() -> None:
    result = server.execute_decision_diff(None, None)
    assert result.get("exit_code") == 2


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_decision_diff::payload
def test_execute_decision_diff_invalid_snapshot(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    result = server.execute_decision_diff(
        None,
        {"baseline": str(bad), "current": str(bad)},
    )
    assert result.get("exit_code") == 2


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_decision_diff::payload
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


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
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


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_emits_test_reports(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_test_evidence_suggestions": True,
            "emit_test_obsolescence": True,
        },
    )
    assert (out_dir / "test_evidence_suggestions.json").exists()
    assert (out_dir / "test_obsolescence_report.json").exists()
    assert "test_evidence_suggestions_summary" in result
    assert "test_obsolescence_summary" in result


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_emits_obsolescence_delta(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [
                    {"key": {"k": "paramset", "params": ["x"]}, "display": "E:paramset::x"}
                ],
                "status": "mapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    baseline_payload = test_obsolescence_delta.build_baseline_payload_from_paths(
        str(out_dir / "test_evidence.json"),
        str(out_dir / "evidence_risk_registry.json"),
    )
    baseline_path = test_obsolescence_delta.resolve_baseline_path(tmp_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    test_obsolescence_delta.write_baseline(str(baseline_path), baseline_payload)

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_test_obsolescence_delta": True,
        },
    )
    assert (out_dir / "test_obsolescence_delta.json").exists()
    assert (out_dir / "test_obsolescence_delta.md").exists()
    assert "test_obsolescence_delta_summary" in result


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_obsolescence_delta_from_state(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = evidence_keys.make_paramset_key(["x"])
    ref = test_obsolescence.EvidenceRef(
        key=key,
        identity=evidence_keys.key_identity(key),
        display=evidence_keys.render_display(key),
        opaque=False,
    )
    evidence_by_test = {"tests/test_sample.py::test_alpha": [ref]}
    status_by_test = {"tests/test_sample.py::test_alpha": "mapped"}
    candidates, summary = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    state_payload = test_obsolescence_state.build_state_payload(
        evidence_by_test, status_by_test, candidates, summary
    )
    state_path = out_dir / "test_obsolescence_state.json"
    state_path.write_text(json.dumps(state_payload, indent=2, sort_keys=True) + "\n")

    baseline_payload = state_payload["baseline"]
    baseline_path = test_obsolescence_delta.resolve_baseline_path(tmp_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    test_obsolescence_delta.write_baseline(str(baseline_path), baseline_payload)

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_test_obsolescence_delta": True,
            "test_obsolescence_state": str(state_path),
        },
    )
    assert (out_dir / "test_obsolescence_delta.json").exists()
    assert (out_dir / "test_obsolescence_delta.md").exists()
    assert "test_obsolescence_delta_summary" in result


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_rejects_missing_obsolescence_state(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_obsolescence_delta": True,
                "test_obsolescence_state": str(tmp_path / "out" / "missing.json"),
            },
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_obsolescence_state(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )
    ls = _DummyServer(str(tmp_path))
    server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_test_obsolescence_state": True,
        },
    )
    assert (out_dir / "test_obsolescence_state.json").exists()


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_writes_obsolescence_baseline(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "write_test_obsolescence_baseline": True,
        },
    )
    baseline_path = test_obsolescence_delta.resolve_baseline_path(tmp_path)
    assert baseline_path.exists()
    assert result.get("test_obsolescence_baseline_written") is True


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_annotation_drift_delta(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_sample.py"
    test_file.write_text(
        "# gabion:evidence E:function_site::sample.py::pkg.fn\n"
        "def test_alpha():\n"
        "    assert True\n"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = {"k": "function_site", "site": {"path": "sample.py", "qual": "pkg.fn"}}
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [{"key": key, "display": "E:function_site::sample.py::pkg.fn"}],
                "status": "mapped",
            }
        ],
        "evidence_index": [
            {
                "key": key,
                "display": "E:function_site::sample.py::pkg.fn",
                "tests": ["tests/test_sample.py::test_alpha"],
            }
        ],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    drift_payload = test_annotation_drift.build_annotation_drift_payload(
        [tests_dir],
        root=tmp_path,
        evidence_path=out_dir / "test_evidence.json",
    )
    baseline_payload = test_annotation_drift_delta.build_baseline_payload(
        drift_payload.get("summary", {})
    )
    baseline_path = test_annotation_drift_delta.resolve_baseline_path(tmp_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    test_annotation_drift_delta.write_baseline(str(baseline_path), baseline_payload)

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(tests_dir)],
            "emit_test_annotation_drift_delta": True,
        },
    )
    assert (out_dir / "test_annotation_drift_delta.json").exists()
    assert (out_dir / "test_annotation_drift_delta.md").exists()
    assert "test_annotation_drift_delta_summary" in result


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_rejects_missing_annotation_drift_state(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_annotation_drift_delta": True,
                "test_annotation_drift_state": str(tmp_path / "out" / "missing.json"),
            },
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_rejects_invalid_annotation_drift_state(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "test_annotation_drift.json"
    state_path.write_text(json.dumps(["bad"]), encoding="utf-8")
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(ValueError):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_annotation_drift_delta": True,
                "test_annotation_drift_state": str(state_path),
            },
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_annotation_drift_delta_from_state(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    drift_payload = {
        "version": 1,
        "summary": {
            "legacy_ambiguous": 0,
            "legacy_tag": 0,
            "ok": 1,
            "orphaned": 0,
        },
        "entries": [],
        "generated_by_spec_id": "spec",
        "generated_by_spec": {},
    }
    state_path = out_dir / "test_annotation_drift.json"
    state_path.write_text(json.dumps(drift_payload, indent=2, sort_keys=True) + "\n")

    baseline_payload = test_annotation_drift_delta.build_baseline_payload(
        drift_payload["summary"]
    )
    baseline_path = test_annotation_drift_delta.resolve_baseline_path(tmp_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    test_annotation_drift_delta.write_baseline(str(baseline_path), baseline_payload)

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_test_annotation_drift_delta": True,
            "test_annotation_drift_state": str(state_path),
        },
    )
    assert (out_dir / "test_annotation_drift_delta.json").exists()
    assert (out_dir / "test_annotation_drift_delta.md").exists()
    assert "test_annotation_drift_delta_summary" in result


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_emits_annotation_drift(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        "# gabion:evidence E:function_site::sample.py::pkg.fn\n"
        "def test_alpha():\n"
        "    assert True\n"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = {"k": "function_site", "site": {"path": "sample.py", "qual": "pkg.fn"}}
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [{"key": key, "display": "E:function_site::sample.py::pkg.fn"}],
                "status": "mapped",
            }
        ],
        "evidence_index": [
            {
                "key": key,
                "display": "E:function_site::sample.py::pkg.fn",
                "tests": ["tests/test_sample.py::test_alpha"],
            }
        ],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(tests_dir)],
            "emit_test_annotation_drift": True,
        },
    )
    assert (out_dir / "test_annotation_drift.json").exists()
    assert (out_dir / "test_annotation_drift.md").exists()
    assert "test_annotation_drift_summary" in result


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_emits_call_clusters(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        "from sample import caller\n"
        "\n"
        "def test_alpha():\n"
        "    caller(1, 2)\n"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(tests_dir), str(module_path)],
            "emit_call_clusters": True,
        },
    )
    assert (out_dir / "call_clusters.json").exists()
    assert (out_dir / "call_clusters.md").exists()
    assert "call_clusters_summary" in result


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_emits_call_cluster_consolidation(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        "from sample import caller\n"
        "\n"
        "def test_alpha():\n"
        "    caller(1, 2)\n"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    call_footprint = evidence_keys.make_call_footprint_key(
        path="tests/test_sample.py",
        qual="test_alpha",
        targets=[{"path": "sample.py", "qual": "caller"}],
    )
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [
                    {
                        "key": call_footprint,
                        "display": evidence_keys.render_display(call_footprint),
                    }
                ],
                "status": "mapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(tests_dir), str(module_path)],
            "emit_call_cluster_consolidation": True,
        },
    )
    assert (out_dir / "call_cluster_consolidation.json").exists()
    assert (out_dir / "call_cluster_consolidation.md").exists()
    assert "call_cluster_consolidation_summary" in result


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_requires_annotation_drift_baseline(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        "# gabion:evidence E:function_site::sample.py::pkg.fn\n"
        "def test_alpha():\n"
        "    assert True\n"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = {"k": "function_site", "site": {"path": "sample.py", "qual": "pkg.fn"}}
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [],
        "evidence_index": [
            {
                "key": key,
                "display": "E:function_site::sample.py::pkg.fn",
                "tests": [],
            }
        ],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(tests_dir)],
                "emit_test_annotation_drift_delta": True,
            },
        )


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_writes_annotation_drift_baseline(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text("def test_alpha():\n    assert True\n")

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "write_test_annotation_drift_baseline": True,
        },
    )
    baseline_path = test_annotation_drift_delta.resolve_baseline_path(tmp_path)
    assert baseline_path.exists()
    assert result.get("test_annotation_drift_baseline_written") is True


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_rejects_annotation_drift_flags(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(ValueError):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_annotation_drift_delta": True,
                "write_test_annotation_drift_baseline": True,
            },
        )


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_requires_ambiguity_baseline(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_ambiguity_delta": True,
            },
        )


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_ambiguity_delta(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    baseline_payload = ambiguity_delta.build_baseline_payload([])
    baseline_path = ambiguity_delta.resolve_baseline_path(tmp_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    ambiguity_delta.write_baseline(str(baseline_path), baseline_payload)

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_ambiguity_delta": True,
        },
    )
    out_dir = tmp_path / "out"
    assert (out_dir / "ambiguity_delta.json").exists()
    assert (out_dir / "ambiguity_delta.md").exists()
    assert "ambiguity_delta_summary" in result


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_ambiguity_delta_from_state(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    witnesses = [
        {
            "kind": "local_resolution_ambiguous",
            "site": {
                "path": "sample.py",
                "function": "caller",
                "span": [1, 0, 1, 5],
            },
            "candidate_count": 2,
        }
    ]
    state_payload = ambiguity_state.build_state_payload(witnesses)
    state_path = out_dir / "ambiguity_state.json"
    state_path.write_text(json.dumps(state_payload, indent=2, sort_keys=True) + "\n")

    baseline_payload = ambiguity_delta.build_baseline_payload(witnesses)
    baseline_path = ambiguity_delta.resolve_baseline_path(tmp_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    ambiguity_delta.write_baseline(str(baseline_path), baseline_payload)

    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_ambiguity_delta": True,
            "ambiguity_state": str(state_path),
        },
    )
    assert (out_dir / "ambiguity_delta.json").exists()
    assert (out_dir / "ambiguity_delta.md").exists()
    assert "ambiguity_delta_summary" in result


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_rejects_missing_ambiguity_state(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_ambiguity_delta": True,
                "ambiguity_state": str(tmp_path / "out" / "missing.json"),
            },
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_emits_ambiguity_state(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "emit_ambiguity_state": True,
        },
    )
    out_dir = tmp_path / "out"
    assert (out_dir / "ambiguity_state.json").exists()


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_writes_ambiguity_baseline(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "write_ambiguity_baseline": True,
        },
    )
    baseline_path = ambiguity_delta.resolve_baseline_path(tmp_path)
    assert baseline_path.exists()
    assert result.get("ambiguity_baseline_written") is True


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_command::payload
# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_rejects_ambiguity_flags(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(ValueError):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_ambiguity_delta": True,
                "write_ambiguity_baseline": True,
            },
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_rejects_ambiguity_state_conflict(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(ValueError):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_ambiguity_state": True,
                "ambiguity_state": "out/ambiguity_state.json",
            },
        )

# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_rejects_conflicting_obsolescence_flags(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(ValueError):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_obsolescence_delta": True,
                "write_test_obsolescence_baseline": True,
            },
        )


# gabion:evidence E:function_site::server.py::gabion.server.execute_command
def test_execute_command_rejects_obsolescence_state_conflict(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)
    ls = _DummyServer(str(tmp_path))
    with pytest.raises(ValueError):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_obsolescence_state": True,
                "test_obsolescence_state": "out/test_obsolescence_state.json",
            },
        )


# gabion:evidence E:call_cluster::server.py::gabion.server.execute_command::test_server_execute_command_edges.py::tests.test_server_execute_command_edges._write_bundle_module
def test_execute_command_requires_obsolescence_baseline(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_bundle_module(module_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_sample.py::test_alpha",
                "file": "tests/test_sample.py",
                "line": 1,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    ls = _DummyServer(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        server.execute_command(
            ls,
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "emit_test_obsolescence_delta": True,
            },
        )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_defaults_payload(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(ls, None)
    assert "violations" in result
    assert "exit_code" in result


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_refactor::ls,payload
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


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_refactor::ls,payload
def test_execute_refactor_invalid_payload(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_refactor(ls, {"protocol_name": 123})
    assert result.get("errors")


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_refactor::ls,payload
def test_execute_refactor_payload_none(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_refactor(ls, None)
    assert result.get("errors")


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_synthesis::payload
def test_execute_synthesis_invalid_payload(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_synthesis(ls, {"bundles": "not-a-list"})
    assert result.get("errors")


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_synthesis::payload
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


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_synthesis::payload
def test_execute_synthesis_payload_none(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_synthesis(ls, None)
    assert result.get("errors")


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_synthesis::payload
def test_execute_synthesis_skips_empty_bundle(tmp_path: Path) -> None:
    ls = _DummyServer(str(tmp_path))
    result = server.execute_synthesis(
        ls,
        {
            "bundles": [{"bundle": [], "tier": 2}],
        },
    )
    assert result.get("protocols") == []
