from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis import test_evidence_suggestions


def test_suggests_alias_invariance() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_alias_attribute.py::test_alias_attribute_forwarding",
        file="tests/test_alias_attribute.py",
        line=10,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.total == 1
    assert summary.suggested == 1
    assert summary.skipped_mapped == 0
    assert summary.skipped_no_match == 0
    assert suggestions[0].suggested == ("E:bundle/alias_invariance",)
    assert suggestions[0].matches == ("alias_invariance",)
    assert summary.unmapped_modules == (("tests/test_alias_attribute.py", 1),)
    assert summary.unmapped_prefixes == (("test_alias", 1),)


def test_suggests_aspf_bundle_pair() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_aspf.py::test_paramset_packed_reuse",
        file="tests/test_aspf.py",
        line=12,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.total == 1
    assert summary.suggested == 1
    assert suggestions[0].suggested == (
        "E:forest/canonical_paramset",
        "E:forest/packed_reuse",
    )


def test_skips_mapped_entries() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_baseline_ratchet.py::test_baseline_write_and_apply",
        file="tests/test_baseline_ratchet.py",
        line=2,
        evidence=("E:baseline/ratchet_monotonicity",),
        status="mapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert suggestions == []
    assert summary.total == 1
    assert summary.skipped_mapped == 1


def test_skips_no_match_entries() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_misc.py::test_smoke",
        file="tests/test_misc.py",
        line=1,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert suggestions == []
    assert summary.total == 1
    assert summary.skipped_no_match == 1


def test_suggests_decision_surface_direct() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_decision_surfaces.py::test_decision_surface_params_collects_names",
        file="tests/test_decision_surfaces.py",
        line=1,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.suggested == 1
    assert suggestions[0].suggested == ("E:decision_surface/direct",)


def test_suggests_value_encoded_decision() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_decision_surfaces.py::test_value_encoded_decision_params_collects_names",
        file="tests/test_decision_surfaces.py",
        line=1,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.suggested == 1
    assert suggestions[0].suggested == ("E:decision_surface/value_encoded",)


def test_suggests_docflow_contract() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_cli_commands.py::test_cli_docflow_audit",
        file="tests/test_cli_commands.py",
        line=1,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.suggested == 1
    assert "E:policy/docflow_contract" in suggestions[0].suggested


def test_suggests_fingerprint_registry_determinism() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_type_fingerprints.py::test_prime_registry_assigns_stable_primes",
        file="tests/test_type_fingerprints.py",
        line=1,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.suggested == 1
    assert suggestions[0].suggested == ("E:fingerprint/registry_determinism",)


def test_suggests_fingerprint_provenance() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_dataflow_report_helpers.py::test_emit_report_fingerprint_provenance_summary",
        file="tests/test_dataflow_report_helpers.py",
        line=1,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.suggested == 1
    assert suggestions[0].suggested == ("E:fingerprint/match_provenance",)


def test_suggests_fingerprint_rewrite_plan() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_dataflow_report_helpers.py::test_emit_report_rewrite_plan_summary",
        file="tests/test_dataflow_report_helpers.py",
        line=1,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.suggested == 1
    assert suggestions[0].suggested == ("E:fingerprint/rewrite_plan_verification",)


def test_suggests_schema_surfaces() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_schema_audit.py::test_find_anonymous_schema_surfaces_finds_common_sites",
        file="tests/test_schema_audit.py",
        line=1,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.suggested == 1
    assert suggestions[0].suggested == ("E:schema/anonymous_payload_surfaces",)


def test_suggests_cli_command_surface_integrity() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_cli_helpers.py::test_run_docflow_audit_missing_script",
        file="tests/test_cli_helpers.py",
        line=1,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.suggested == 1
    assert "E:cli/command_surface_integrity" in suggestions[0].suggested


def test_suggests_server_command_dispatch() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_server_execute_command_edges.py::test_execute_command_baseline_apply",
        file="tests/test_server_execute_command_edges.py",
        line=1,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.suggested == 1
    assert "E:transport/server_command_dispatch" in suggestions[0].suggested


def test_suggests_transport_payload_roundtrip() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_dataflow_run_edges.py::test_run_decision_snapshot_writes_file",
        file="tests/test_dataflow_run_edges.py",
        line=1,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence([entry])
    assert summary.suggested == 1
    assert suggestions[0].suggested == ("E:transport/payload_roundtrip",)


def test_load_test_evidence_payload(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_policy_check.py::test_policy_check_runs",
                "file": "tests/test_policy_check.py",
                "line": 5,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    path = tmp_path / "test_evidence.json"
    path.write_text(json.dumps(payload))
    entries = test_evidence_suggestions.load_test_evidence(str(path))
    assert entries[0].test_id.endswith("test_policy_check.py::test_policy_check_runs")
