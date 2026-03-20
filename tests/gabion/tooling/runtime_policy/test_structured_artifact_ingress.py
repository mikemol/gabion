from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling.policy_substrate.structured_artifact_ingress import (
    GitStateLineSpan,
    load_delivery_flow_summary_artifact,
    load_governance_telemetry_history_artifact,
    StructuredArtifactDecompositionKind,
    StructuredArtifactIdentitySpace,
    StructuredArtifactKind,
    build_ingress_merge_parity_artifact,
    load_cross_origin_witness_contract_artifact,
    load_git_state_artifact,
    load_controller_drift_artifact,
    load_docflow_compliance_artifact,
    load_docflow_packet_enforcement_artifact,
    load_identity_grammar_completion_artifact,
    load_ingress_merge_parity_artifact,
    load_junit_failure_artifact,
    load_kernel_vm_alignment_artifact,
    load_local_ci_repro_contract_artifact,
    load_local_repro_closure_ledger_artifact,
    load_observability_violations_artifact,
    load_test_evidence_artifact,
    write_ingress_merge_parity_artifact,
)


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# gabion:behavior primary=desired
def test_load_docflow_packet_enforcement_artifact_uses_typed_row_identities(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "docflow_packet_enforcement.json",
        {
            "summary": {
                "active_packets": 1,
                "active_rows": 1,
                "new_rows": 1,
                "drifted_rows": 0,
                "ready": 0,
                "blocked": 1,
                "drifted": 0,
            },
            "new_rows": [{"row_id": "docflow:row-1"}],
            "drifted_rows": [],
            "changed_paths": ["in/in-54.md"],
            "out_of_scope_touches": [],
            "unresolved_touched_packets": [],
            "packet_status": [
                {
                    "path": "in/in-54.md",
                    "classification": "metadata_only",
                    "status": "blocked",
                    "row_ids": ["docflow:row-1"],
                }
            ],
        },
    )

    artifact = load_docflow_packet_enforcement_artifact(
        root=tmp_path,
        rel_path="artifacts/out/docflow_packet_enforcement.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert artifact is not None
    assert artifact.identity.artifact_kind is StructuredArtifactKind.DOCFLOW_PACKET_ENFORCEMENT
    assert len(artifact.packets) == 1
    packet = artifact.packets[0]
    row = packet.rows[0]

    assert packet.identity.item_kind == "packet"
    assert row.identity.item_kind == "row"
    assert row.status == "new"
    assert str(row) == "docflow:row-1"
    assert str(row.identity) == "docflow:row-1"
    assert row.identity.wire() != str(row.identity)
    assert {
        item.decomposition_kind for item in row.identity.decompositions
    } >= {
        StructuredArtifactDecompositionKind.CANONICAL,
        StructuredArtifactDecompositionKind.ARTIFACT_KIND,
        StructuredArtifactDecompositionKind.SOURCE_PATH,
        StructuredArtifactDecompositionKind.ITEM_KIND,
        StructuredArtifactDecompositionKind.ITEM_KEY,
    }


# gabion:behavior primary=desired
def test_load_docflow_compliance_artifact_uses_typed_row_and_obligation_identities(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "docflow_compliance.json",
        {
            "version": 2,
            "summary": {
                "compliant": 0,
                "contradicts": 1,
                "excess": 0,
                "proposed": 0,
            },
            "rows": [
                {
                    "row_kind": "docflow_compliance",
                    "invariant": "docflow:missing_explicit_reference",
                    "invariant_kind": "never",
                    "status": "contradicts",
                    "path": "in/in-54.md",
                    "source_row_kind": "doc_requires_ref",
                    "detail": "missing explicit reference",
                }
            ],
            "obligations": {
                "summary": {
                    "total": 1,
                    "triggered": 1,
                    "met": 0,
                    "unmet_fail": 1,
                    "unmet_warn": 0,
                },
                "context": {
                    "changed_paths": ["in/in-54.md"],
                    "sppf_relevant_paths_changed": True,
                    "gh_reference_validated": False,
                    "baseline_write_emitted": False,
                    "delta_guard_checked": False,
                    "doc_status_changed": True,
                    "checklist_influence_consistent": False,
                    "rev_range": "origin/stage..HEAD",
                    "commits": [
                        {
                            "sha": "a" * 40,
                            "subject": "Add PM view boundary renderer",
                        }
                    ],
                    "issue_ids": [],
                    "checklist_impact": [],
                    "issue_lifecycle_fetch_status": "not_applicable",
                    "issue_lifecycles": [],
                    "issue_lifecycle_errors": [],
                },
                "entries": [
                    {
                        "obligation_id": "sppf_gh_reference_validation",
                        "triggered": True,
                        "status": "unmet",
                        "enforcement": "fail",
                        "description": "SPPF-relevant path changes require GH-reference validation.",
                    }
                ],
            },
        },
    )

    artifact = load_docflow_compliance_artifact(
        root=tmp_path,
        rel_path="artifacts/out/docflow_compliance.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert artifact is not None
    assert artifact.identity.artifact_kind is StructuredArtifactKind.DOCFLOW_COMPLIANCE
    assert artifact.contradiction_count == 1
    assert artifact.rev_range == "origin/stage..HEAD"
    assert artifact.changed_paths == ("in/in-54.md",)
    assert artifact.sppf_relevant_paths_changed is True
    assert artifact.gh_reference_validated is False
    assert len(artifact.commits) == 1
    assert artifact.issue_references == ()
    assert artifact.issue_lifecycle_fetch_status == "not_applicable"
    assert artifact.issue_lifecycles == ()
    row = artifact.rows[0]
    obligation = artifact.obligations[0]
    commit = artifact.commits[0]

    assert row.rel_path == "in/in-54.md"
    assert row.identity.item_kind == "row"
    assert str(row).startswith("docflow_compliance:")
    assert row.identity.wire() != str(row.identity)
    assert commit.sha == "a" * 40
    assert commit.identity.item_kind == "commit"
    assert str(commit) == "aaaaaaaaaaaa"
    assert obligation.obligation_id == "sppf_gh_reference_validation"
    assert obligation.identity.item_kind == "obligation"
    assert str(obligation) == "sppf_gh_reference_validation"
    assert obligation.identity.wire() != str(obligation.identity)


# gabion:behavior primary=desired
def test_load_docflow_compliance_artifact_extracts_issue_lifecycle_state(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "docflow_compliance.json",
        {
            "version": 2,
            "summary": {
                "compliant": 1,
                "contradicts": 0,
                "excess": 0,
                "proposed": 0,
            },
            "rows": [],
            "obligations": {
                "summary": {
                    "total": 1,
                    "triggered": 1,
                    "met": 1,
                    "unmet_fail": 0,
                    "unmet_warn": 0,
                },
                "context": {
                    "changed_paths": ["docs/sppf_checklist.md"],
                    "sppf_relevant_paths_changed": True,
                    "gh_reference_validated": True,
                    "baseline_write_emitted": False,
                    "delta_guard_checked": False,
                    "doc_status_changed": True,
                    "checklist_influence_consistent": True,
                    "rev_range": "origin/stage..HEAD",
                    "commits": [],
                    "issue_ids": ["214"],
                    "checklist_impact": [{"issue_id": "214", "commit_count": 1}],
                    "issue_lifecycle_fetch_status": "ok",
                    "issue_lifecycles": [
                        {
                            "issue_id": "214",
                            "state": "open",
                            "labels": ["done-on-stage", "status/pending-release"],
                            "url": "https://example.invalid/214",
                        }
                    ],
                    "issue_lifecycle_errors": [],
                },
                "entries": [
                    {
                        "obligation_id": "sppf_gh_reference_validation",
                        "triggered": True,
                        "status": "met",
                        "enforcement": "fail",
                        "description": "SPPF-relevant path changes require GH-reference validation.",
                    }
                ],
            },
        },
    )

    artifact = load_docflow_compliance_artifact(
        root=tmp_path,
        rel_path="artifacts/out/docflow_compliance.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert artifact is not None
    assert len(artifact.issue_references) == 1
    assert artifact.issue_references[0].issue_id == "214"
    assert artifact.issue_references[0].commit_count == 1
    assert artifact.issue_lifecycle_fetch_status == "ok"
    assert artifact.issue_lifecycle_errors == ()
    assert len(artifact.issue_lifecycles) == 1
    lifecycle = artifact.issue_lifecycles[0]
    assert lifecycle.issue_id == "214"
    assert lifecycle.state == "open"
    assert lifecycle.labels == ("done-on-stage", "status/pending-release")
    assert lifecycle.url == "https://example.invalid/214"
    assert lifecycle.identity.item_kind == "issue_lifecycle"


# gabion:behavior primary=desired
def test_load_kernel_vm_alignment_artifact_uses_typed_binding_and_residue_identities(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "kernel_vm_alignment.json",
        {
            "artifact_kind": "kernel_vm_alignment",
            "schema_version": 1,
            "generated_by": "tests",
            "fragment_id": "ttl_kernel_vm.fragment.augmented_rule_polarity_query_ast",
            "summary": {
                "binding_count": 1,
                "pass_count": 0,
                "partial_count": 1,
                "fail_count": 0,
                "residue_count": 1,
            },
            "bindings": [
                {
                    "binding_id": "kernel_vm.augmented_rule_core",
                    "fragment_id": "ttl_kernel_vm.fragment.augmented_rule_polarity_query_ast",
                    "title": "AugmentedRule core object over semantic-row reflection",
                    "status": "partial",
                    "summary": "Synthetic kernel VM binding",
                    "kernel_terms": ["lg:AugmentedRule"],
                    "runtime_surface_symbols": ["CanonicalWitnessedSemanticRow"],
                    "realizer_symbols": [],
                    "runtime_object_symbols": ["AugmentedRule"],
                    "missing_capability_ids": ["runtime_object_image"],
                    "residue_ids": [
                        "kernel_vm.augmented_rule_core:missing_runtime_object_image"
                    ],
                    "evidence_paths": [
                        "in/lg_kernel_ontology_cut_elim-1.ttl",
                        "src/gabion/analysis/projection/semantic_fragment.py",
                    ],
                    "capabilities": [
                        {
                            "capability_id": "runtime_object_image",
                            "requirement_kind": "runtime_object_image",
                            "status": "fail",
                            "match_mode": "all",
                            "description": "explicit runtime object image for AugmentedRule",
                            "residue_kind": "missing_runtime_object_image",
                            "severity": "warning",
                            "score": 6,
                            "expected_refs": [
                                {
                                    "rel_path": "src/gabion/analysis/projection/semantic_fragment.py",
                                    "evidence_kind": "python_symbol",
                                    "symbol": "AugmentedRule",
                                    "present": False,
                                }
                            ],
                            "matched_refs": [],
                            "missing_refs": [
                                {
                                    "rel_path": "src/gabion/analysis/projection/semantic_fragment.py",
                                    "evidence_kind": "python_symbol",
                                    "symbol": "AugmentedRule",
                                    "present": False,
                                }
                            ],
                        }
                    ],
                }
            ],
            "residues": [
                {
                    "residue_id": "kernel_vm.augmented_rule_core:missing_runtime_object_image",
                    "binding_id": "kernel_vm.augmented_rule_core",
                    "fragment_id": "ttl_kernel_vm.fragment.augmented_rule_polarity_query_ast",
                    "residue_kind": "missing_runtime_object_image",
                    "severity": "warning",
                    "score": 6,
                    "title": "AugmentedRule core object over semantic-row reflection",
                    "message": "Synthetic kernel VM residue",
                    "missing_capability_ids": ["runtime_object_image"],
                    "kernel_terms": ["lg:AugmentedRule"],
                    "runtime_surface_symbols": ["CanonicalWitnessedSemanticRow"],
                    "realizer_symbols": [],
                    "runtime_object_symbols": ["AugmentedRule"],
                    "evidence_paths": [
                        "in/lg_kernel_ontology_cut_elim-1.ttl",
                        "src/gabion/analysis/projection/semantic_fragment.py",
                    ],
                }
            ],
        },
    )

    artifact = load_kernel_vm_alignment_artifact(
        root=tmp_path,
        rel_path="artifacts/out/kernel_vm_alignment.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert artifact is not None
    assert artifact.identity.artifact_kind is StructuredArtifactKind.KERNEL_VM_ALIGNMENT
    assert artifact.binding_count == 1
    assert artifact.residue_count == 1
    binding = artifact.bindings[0]
    capability = binding.capabilities[0]
    residue = artifact.residues[0]

    assert binding.identity.item_kind == "binding"
    assert capability.identity.item_kind == "capability"
    assert residue.identity.item_kind == "residue"
    assert binding.binding_id == "kernel_vm.augmented_rule_core"
    assert capability.capability_id == "runtime_object_image"
    assert capability.missing_refs[0].symbol == "AugmentedRule"
    assert residue.residue_kind == "missing_runtime_object_image"
    assert residue.evidence_paths == (
        "in/lg_kernel_ontology_cut_elim-1.ttl",
        "src/gabion/analysis/projection/semantic_fragment.py",
    )


# gabion:behavior primary=desired
def test_load_identity_grammar_completion_artifact_uses_typed_surface_and_residue_identities(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "identity_grammar_completion.json",
        {
            "artifact_kind": "identity_grammar_completion",
            "schema_version": 1,
            "generated_by": "tests",
            "summary": {
                "surface_count": 2,
                "pass_count": 0,
                "fail_count": 2,
                "residue_count": 2,
                "highest_severity": "high",
            },
            "surfaces": [
                {
                    "surface_id": "identity_grammar.hotspot.raw_string_grouping",
                    "title": "Hotspot queue still groups by raw path strings",
                    "status": "fail",
                    "summary": "Synthetic surface",
                    "evidence_paths": ["scripts/policy/hotspot_neighborhood_queue.py"],
                    "residue_ids": [
                        "identity_grammar.hotspot.raw_string_grouping:raw_string_grouping_in_core_queue_logic"
                    ],
                },
                {
                    "surface_id": "identity_grammar.coherence.two_cell",
                    "title": "Coherence witness carrier exists but is not emitted",
                    "status": "fail",
                    "summary": "Synthetic coherence surface",
                    "evidence_paths": [
                        "src/gabion/tooling/policy_substrate/identity_zone/grammar.py"
                    ],
                    "residue_ids": [
                        "identity_grammar.coherence.two_cell:coherence_witness_emission_missing"
                    ],
                },
            ],
            "residues": [
                {
                    "residue_id": "identity_grammar.hotspot.raw_string_grouping:raw_string_grouping_in_core_queue_logic",
                    "surface_id": "identity_grammar.hotspot.raw_string_grouping",
                    "residue_kind": "raw_string_grouping_in_core_queue_logic",
                    "severity": "high",
                    "score": 9,
                    "title": "Hotspot queue still groups by raw path strings",
                    "message": "Synthetic hotspot residue",
                    "evidence_paths": ["scripts/policy/hotspot_neighborhood_queue.py"],
                },
                {
                    "residue_id": "identity_grammar.coherence.two_cell:coherence_witness_emission_missing",
                    "surface_id": "identity_grammar.coherence.two_cell",
                    "residue_kind": "coherence_witness_emission_missing",
                    "severity": "medium",
                    "score": 6,
                    "title": "Coherence witness carrier exists but is not emitted",
                    "message": "Synthetic coherence residue",
                    "evidence_paths": [
                        "src/gabion/tooling/policy_substrate/identity_zone/grammar.py"
                    ],
                },
            ],
        },
    )

    artifact = load_identity_grammar_completion_artifact(
        root=tmp_path,
        rel_path="artifacts/out/identity_grammar_completion.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert artifact is not None
    assert (
        artifact.identity.artifact_kind
        is StructuredArtifactKind.IDENTITY_GRAMMAR_COMPLETION
    )
    assert artifact.surface_count == 2
    assert artifact.residue_count == 2
    assert artifact.highest_severity == "high"
    assert artifact.surfaces[0].identity.item_kind == "surface"
    assert artifact.residues[0].identity.item_kind == "residue"
    assert artifact.residues[0].residue_kind == "raw_string_grouping_in_core_queue_logic"


# gabion:behavior primary=desired
def test_load_controller_drift_artifact_extracts_markdown_doc_paths(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "controller_drift.json",
        {
            "anchors_scanned": 4,
            "commands_scanned": 2,
            "normative_docs": ["AGENTS.md"],
            "policy": "POLICY_SEED.md",
            "summary": {
                "high_severity_findings": 1,
                "highest_severity": "high",
                "severity_counts": {"critical": 0, "high": 1, "medium": 0, "low": 0},
                "sensors": ["checks_without_normative_anchor"],
                "total_findings": 1,
            },
            "findings": [
                {
                    "sensor": "checks_without_normative_anchor",
                    "severity": "high",
                    "anchor": "CD-999",
                    "detail": "Workflow references `AGENTS.md` and `in/in-54.md` without anchors.",
                }
            ],
        },
    )

    artifact = load_controller_drift_artifact(
        root=tmp_path,
        rel_path="artifacts/out/controller_drift.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert artifact is not None
    assert artifact.total_findings == 1
    finding = artifact.findings[0]

    assert finding.anchor == "CD-999"
    assert finding.doc_paths == ("AGENTS.md", "in/in-54.md")
    assert str(finding.identity) == finding.detail
    assert finding.identity.wire() != str(finding.identity)


# gabion:behavior primary=desired
def test_load_local_repro_closure_ledger_artifact_preserves_validation_statuses(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "local_repro_closure_ledger.json",
        {
            "schema_version": 1,
            "generated_by": "tests",
            "workstream": "full_local_repro_closure",
            "entries": [
                {
                    "cu_id": "CU-R1",
                    "summary": "Close the local repro loop.",
                    "validation": {
                        "policy_workflows": "pass",
                        "policy_ambiguity_contract": "pass",
                    },
                }
            ],
        },
    )

    artifact = load_local_repro_closure_ledger_artifact(
        root=tmp_path,
        rel_path="artifacts/out/local_repro_closure_ledger.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert artifact is not None
    assert artifact.source.schema_version == 1
    entry = artifact.entries[0]
    assert entry.cu_id == "CU-R1"
    assert entry.validation_statuses == ("pass", "pass")
    assert str(entry.identity) == "CU-R1"
    assert entry.identity.wire() != str(entry.identity)


# gabion:behavior primary=desired
def test_load_local_ci_repro_contract_artifact_preserves_surface_and_relation_statuses(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "local_ci_repro_contract.json",
        {
            "schema_version": 2,
            "artifact_kind": "local_ci_repro_contract",
            "generated_by": "tests",
            "summary": "Local CI reproduction topology.",
            "surfaces": [
                {
                    "surface_id": "workflow:ci.yml:checks",
                    "surface_kind": "workflow_job",
                    "title": "CI checks workflow job",
                    "summary": "Strict gates.",
                    "source_ref": ".github/workflows/ci.yml",
                    "mode": "checks",
                    "status": "pass",
                    "required_capabilities": [
                        {
                            "capability_id": "policy_workflows_output",
                            "summary": "Materialize the workflow policy artifact.",
                            "status": "pass",
                            "source_alternative_token_groups": [["policy_check.py", "--workflows"]],
                            "command_alternative_token_groups": [],
                            "matched_source_alternative_index": 0,
                            "matched_command_alternative_index": None,
                        }
                    ],
                    "missing_capability_ids": [],
                    "required_token_groups": [["policy_check.py", "--workflows"]],
                    "missing_token_groups": [],
                    "commands": ["python scripts/policy/policy_check.py --workflows"],
                    "artifacts": ["artifacts/out/policy_check_result.json"],
                },
                {
                    "surface_id": "local_script:scripts/ci_local_repro.sh:checks",
                    "surface_kind": "local_repro_lane",
                    "title": "Local CI reproduction checks lane",
                    "summary": "Local parity lane.",
                    "source_ref": "scripts/ci_local_repro.sh",
                    "mode": "checks-only",
                    "status": "fail",
                    "required_capabilities": [
                        {
                            "capability_id": "policy_workflows_output",
                            "summary": "Materialize the workflow policy artifact.",
                            "status": "fail",
                            "source_alternative_token_groups": [["checks_policy_workflows_output"]],
                            "command_alternative_token_groups": [],
                            "matched_source_alternative_index": None,
                            "matched_command_alternative_index": None,
                        }
                    ],
                    "missing_capability_ids": ["policy_workflows_output"],
                    "required_token_groups": [["checks_policy_workflows_output"]],
                    "missing_token_groups": [["checks_policy_workflows_output"]],
                    "commands": ["scripts/ci_local_repro.sh --checks-only"],
                    "artifacts": ["artifacts/out/policy_check_result.json"],
                },
            ],
            "relations": [
                {
                    "relation_id": "ci-repro:local-checks->workflow-checks",
                    "relation_kind": "reproduces",
                    "source_surface_id": "local_script:scripts/ci_local_repro.sh:checks",
                    "target_surface_id": "workflow:ci.yml:checks",
                    "source_missing_capability_ids": ["policy_workflows_output"],
                    "target_missing_capability_ids": [],
                    "status": "fail",
                    "summary": "Local checks should reproduce workflow checks.",
                }
            ],
        },
    )

    artifact = load_local_ci_repro_contract_artifact(
        root=tmp_path,
        rel_path="artifacts/out/local_ci_repro_contract.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert artifact is not None
    assert artifact.source.schema_version == 2
    assert artifact.summary == "Local CI reproduction topology."
    assert len(artifact.surfaces) == 2
    assert len(artifact.relations) == 1
    failing_surface = artifact.surfaces[1]
    passing_surface = artifact.surfaces[0]
    relation = artifact.relations[0]
    assert passing_surface.required_capabilities[0].capability_id == "policy_workflows_output"
    assert passing_surface.required_capabilities[0].matched_source_alternative_index == 0
    assert failing_surface.surface_id == "local_script:scripts/ci_local_repro.sh:checks"
    assert failing_surface.status == "fail"
    assert failing_surface.missing_capability_ids == ("policy_workflows_output",)
    assert failing_surface.required_capabilities[0].status == "fail"
    assert failing_surface.missing_token_groups == (("checks_policy_workflows_output",),)
    assert failing_surface.artifacts == ("artifacts/out/policy_check_result.json",)
    assert relation.relation_kind == "reproduces"
    assert relation.status == "fail"
    assert relation.source_missing_capability_ids == ("policy_workflows_output",)
    assert str(failing_surface.identity) == "Local CI reproduction checks lane"


# gabion:behavior primary=desired
def test_load_observability_violations_artifact_preserves_violation_metrics(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "audit_reports" / "observability_violations.json",
        {
            "violations": [
                {
                    "ts_utc": "2026-03-20T12:00:00Z",
                    "label": "checks_wrapper",
                    "reason": "max_gap_meaningful_line_exceeded",
                    "command_text": "mise exec -- python -m gabion checks",
                    "wall_seconds": 123.4,
                    "max_gap_seconds": 5.0,
                    "measured_gap_seconds": 18.2,
                    "previous_line": "running check run",
                    "next_line": "finished",
                }
            ]
        },
    )

    artifact = load_observability_violations_artifact(
        root=tmp_path,
        rel_path="artifacts/audit_reports/observability_violations.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert artifact is not None
    assert artifact.identity.artifact_kind is StructuredArtifactKind.OBSERVABILITY_VIOLATIONS
    assert len(artifact.violations) == 1
    violation = artifact.violations[0]
    assert violation.label == "checks_wrapper"
    assert violation.reason == "max_gap_meaningful_line_exceeded"
    assert violation.measured_gap_seconds == 18.2
    assert str(violation.identity) == "checks_wrapper"


# gabion:behavior primary=desired
def test_load_governance_telemetry_history_artifact_preserves_runs_loops_and_timings(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "governance_telemetry_history.json",
        {
            "schema_version": 1,
            "runs": [
                {
                    "run_id": "run-001",
                    "generated_at_utc": "2026-03-19T12:00:00Z",
                    "trend_window_runs": 5,
                    "timings_seconds_by_step": {
                        "checks_wrapper": 24.0,
                        "full_pytest": 135.0,
                    },
                    "suite_red_state": True,
                    "open_blocker_ids": [
                        "test:tests/test_ci.py::test_lane",
                        "surface:local:checks",
                    ],
                    "loops": [
                        {
                            "loop_id": "docflow.contradictions",
                            "domain": "governance",
                            "violation_count": 1,
                            "trend_delta": 1,
                            "recurrence_rate": 0.6,
                            "false_positive_overrides": 0,
                            "time_to_correction_runs": None,
                        }
                    ],
                }
            ],
        },
    )

    artifact = load_governance_telemetry_history_artifact(
        root=tmp_path,
        rel_path="artifacts/out/governance_telemetry_history.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert artifact is not None
    assert artifact.identity.artifact_kind is StructuredArtifactKind.GOVERNANCE_TELEMETRY_HISTORY
    assert len(artifact.runs) == 1
    run = artifact.runs[0]
    assert run.run_id == "run-001"
    assert run.trend_window_runs == 5
    assert run.suite_red_state is True
    assert run.open_blocker_ids == (
        "test:tests/test_ci.py::test_lane",
        "surface:local:checks",
    )
    assert {item.step_label for item in run.timings} == {"checks_wrapper", "full_pytest"}
    assert run.loops[0].loop_id == "docflow.contradictions"
    assert run.loops[0].trend_delta == 1
    assert run.loops[0].recurrence_rate == 0.6


# gabion:behavior primary=desired
def test_load_delivery_flow_summary_artifact_preserves_current_trend_and_history(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "delivery_flow_summary.json",
        {
            "schema_version": 1,
            "artifact_kind": "delivery_flow_summary",
            "generated_at_utc": "2026-03-20T12:00:00Z",
            "generated_by": "gabion governance delivery-flow-emit",
            "history_window_runs": 10,
            "current": {
                "suite_red_state": True,
                "failing_test_case_count": 2,
                "test_failure_count": 2,
                "local_ci_failed_surface_ids": ["local:checks"],
                "local_ci_failed_relation_ids": ["local->workflow"],
                "observability_violation_ids": ["checks_wrapper"],
                "severe_runtime_regression_current_band": True,
                "repeat_blocker_ids": ["test:tests/test_ci.py::test_lane"],
                "stalled_blocker_runs_by_id": {"test:tests/test_ci.py::test_lane": 2},
                "unstable_blocker_ids": ["surface:local:checks"],
            },
            "trend": {
                "latest_total_runtime_seconds": 160.0,
                "baseline_total_runtime_seconds": 100.0,
                "runtime_regression_ratio": 1.6,
                "runtime_delta_seconds": 60.0,
                "red_state_dwell_runs": 2,
                "recurring_loop_ids": ["docflow.contradictions"],
                "closure_lag_loop_ids": ["docflow.contradictions"],
                "max_time_to_correction_runs": 2,
            },
            "history": [
                {
                    "run_id": "run-001",
                    "suite_red_state": True,
                    "open_blocker_ids": ["test:tests/test_ci.py::test_lane"],
                }
            ],
        },
    )

    artifact = load_delivery_flow_summary_artifact(
        root=tmp_path,
        rel_path="artifacts/out/delivery_flow_summary.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert artifact is not None
    assert artifact.identity.artifact_kind is StructuredArtifactKind.DELIVERY_FLOW_SUMMARY
    assert artifact.current.suite_red_state is True
    assert artifact.current.repeat_blocker_ids == ("test:tests/test_ci.py::test_lane",)
    assert artifact.current.stalled_blocker_runs_by_id == {
        "test:tests/test_ci.py::test_lane": 2
    }
    assert artifact.trend.red_state_dwell_runs == 2
    assert artifact.trend.closure_lag_loop_ids == ("docflow.contradictions",)
    assert artifact.history[0].run_id == "run-001"


# gabion:behavior primary=desired
def test_load_git_state_artifact_uses_typed_state_entry_identities(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "git_state.json",
        {
            "schema_version": 1,
            "artifact_kind": "git_state",
            "head_sha": "a" * 40,
            "branch": "main",
            "upstream": "origin/main",
            "is_detached": False,
            "summary": {
                "committed_count": 1,
                "staged_count": 1,
                "unstaged_count": 0,
                "untracked_count": 1,
            },
            "entries": [
                {
                    "state_class": "committed",
                    "change_code": "A",
                    "path": "tracked.txt",
                    "previous_path": "",
                },
                {
                    "state_class": "staged",
                    "change_code": "M",
                    "path": "src/example.py",
                    "previous_path": "",
                    "current_line_spans": [{"start_line": 12, "line_count": 3}],
                },
                {
                    "state_class": "untracked",
                    "change_code": "??",
                    "path": "notes/todo.md",
                    "previous_path": "",
                    "current_line_spans": [{"start_line": 1, "line_count": 8}],
                },
            ],
        },
    )

    artifact = load_git_state_artifact(
        root=tmp_path,
        rel_path="artifacts/out/git_state.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert artifact is not None
    assert artifact.head_sha == "a" * 40
    assert artifact.branch == "main"
    assert len(artifact.entries) == 3
    staged_entry = next(item for item in artifact.entries if item.state_class == "staged")
    assert staged_entry.rel_path == "src/example.py"
    assert staged_entry.identity.item_kind == "staged"
    assert str(staged_entry.identity) == "staged:src/example.py"
    assert staged_entry.identity.wire() != str(staged_entry.identity)
    assert staged_entry.current_line_spans == (
        GitStateLineSpan(start_line=12, line_count=3),
    )


# gabion:behavior primary=desired
def test_load_cross_origin_witness_contract_artifact_uses_typed_row_identities(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "cross_origin_witness_contract.json",
        {
            "schema_version": 1,
            "artifact_kind": "cross_origin_witness_contract",
            "producer": "tests",
            "cases": [
                {
                    "case_key": "analysis_union_path_remap",
                    "case_kind": "cross_origin_path_remap",
                    "title": "analysis witness to union-view path remap",
                    "status": "pass",
                    "summary": "rows=1 mismatches=0",
                    "left_label": "analysis_input_witness",
                    "right_label": "aspf_union_view",
                    "evidence_paths": [
                        "src/gabion/server.py",
                        "src/gabion/tooling/policy_substrate/aspf_union_view.py",
                    ],
                    "row_keys": [
                        "path_remap:src/gabion/sample_alpha.py",
                    ],
                    "field_checks": [
                        {
                            "field_name": "manifest_digest_present",
                            "matches": True,
                            "left_value": "true",
                            "right_value": "true",
                        }
                    ],
                }
            ],
            "witness_rows": [
                {
                    "row_key": "path_remap:src/gabion/sample_alpha.py",
                    "row_kind": "path_remap",
                    "left_origin_kind": "analysis_input_witness.file",
                    "left_origin_key": "src/gabion/sample_alpha.py",
                    "right_origin_kind": "aspf_union_view.module",
                    "right_origin_key": "src/gabion/sample_alpha.py",
                    "remap_key": "src/gabion/sample_alpha.py",
                    "summary": "analysis witness file remapped to union module",
                }
            ],
        },
    )

    artifact = load_cross_origin_witness_contract_artifact(
        root=tmp_path,
        rel_path="artifacts/out/cross_origin_witness_contract.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert artifact is not None
    assert (
        artifact.identity.artifact_kind
        is StructuredArtifactKind.CROSS_ORIGIN_WITNESS_CONTRACT
    )
    assert len(artifact.cases) == 1
    assert len(artifact.witness_rows) == 1
    case = artifact.cases[0]
    row = artifact.witness_rows[0]
    assert case.case_key == "analysis_union_path_remap"
    assert case.row_keys == ("path_remap:src/gabion/sample_alpha.py",)
    assert row.row_kind == "path_remap"
    assert str(row) == "path_remap:src/gabion/sample_alpha.py"
    assert row.identity.wire() != str(row.identity)


# gabion:behavior primary=desired
def test_load_test_evidence_and_junit_failure_artifacts_share_ingress_contract(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "out" / "test_evidence.json",
        {
            "schema_version": 2,
            "tests": [
                {
                    "test_id": "tests/example_test.py::test_value",
                    "file": "tests/example_test.py",
                    "line": 12,
                    "status": "pass",
                    "evidence": [
                        {
                            "key": {
                                "site": {
                                    "path": "src/example.py",
                                    "qual": "example.target",
                                    "span": [10, 0, 14, 0],
                                }
                            }
                        }
                    ],
                }
            ],
        },
    )
    _write(
        tmp_path / "artifacts" / "test_runs" / "junit.xml",
        "\n".join(
            [
                "<testsuite tests='1' failures='1'>",
                "  <testcase classname='tests.example_test.TestExample' name='test_value' file='tests/example_test.py' line='12'>",
                "    <failure message='assert 1 == 2'>Traceback (most recent call last):",
                '  File "src/example.py", line 11, in target',
                "AssertionError",
                "    </failure>",
                "  </testcase>",
                "</testsuite>",
            ]
        ),
    )

    identities = StructuredArtifactIdentitySpace()
    evidence = load_test_evidence_artifact(
        root=tmp_path,
        rel_path="out/test_evidence.json",
        identities=identities,
    )
    junit = load_junit_failure_artifact(
        root=tmp_path,
        rel_path="artifacts/test_runs/junit.xml",
        identities=identities,
    )

    assert evidence is not None
    assert junit is not None
    assert evidence.identity.artifact_kind is StructuredArtifactKind.TEST_EVIDENCE
    assert junit.identity.artifact_kind is StructuredArtifactKind.JUNIT_FAILURES
    assert evidence.cases[0].evidence_sites[0].path == "src/example.py"
    assert {
        item.decomposition_kind for item in evidence.cases[0].identity.decompositions
    } >= {
        StructuredArtifactDecompositionKind.CANONICAL,
        StructuredArtifactDecompositionKind.ARTIFACT_KIND,
        StructuredArtifactDecompositionKind.SOURCE_PATH,
        StructuredArtifactDecompositionKind.ITEM_KIND,
        StructuredArtifactDecompositionKind.ITEM_KEY,
    }
    assert evidence.cases[0].identity.to_payload()["decompositions"]
    assert junit.failures[0].test_id == "tests/example_test.py::TestExample::test_value"
    assert junit.failures[0].traceback_text
    assert evidence.cases[0].identity.canonical.atom_id != junit.failures[0].identity.canonical.atom_id


# gabion:behavior primary=desired
def test_build_and_load_ingress_merge_parity_artifact_uses_typed_case_identities(
    tmp_path: Path,
) -> None:
    fixture_root = Path("tests/fixtures/ingest_adapter")
    for name in (
        "python_raw.json",
        "synthetic_raw.json",
        "python_expected.json",
        "synthetic_expected.json",
    ):
        _write(
            tmp_path / "tests" / "fixtures" / "ingest_adapter" / name,
            (fixture_root / name).read_text(encoding="utf-8"),
        )
    _write(
        tmp_path / "docs" / "policy_rules.yaml",
        "\n".join(
            [
                "rules:",
                "  - rule_id: sample.rule",
                "    domain: ambiguity_contract",
                "    severity: blocking",
                "    predicate:",
                "      op: always",
                "    outcome:",
                "      kind: block",
                "      message: sample",
                "    evidence_contract: none",
            ]
        ),
    )

    identities = StructuredArtifactIdentitySpace()
    artifact = build_ingress_merge_parity_artifact(
        root=tmp_path,
        identities=identities,
    )

    assert artifact.identity.artifact_kind is StructuredArtifactKind.INGRESS_MERGE_PARITY
    assert artifact.source.schema_version == 1
    assert {item.case_key for item in artifact.cases} == {
        "adapter_decision_surface_parity",
        "frontmatter_adapter_projection_parity",
        "policy_source_uniqueness",
        "policy_registry_determinism",
    }
    adapter_case = next(
        item for item in artifact.cases if item.case_key == "adapter_decision_surface_parity"
    )
    assert adapter_case.status == "pass"
    assert adapter_case.identity.item_kind == "case"
    assert str(adapter_case.identity) == "adapter decision surface parity"
    assert adapter_case.identity.wire() != str(adapter_case.identity)

    write_ingress_merge_parity_artifact(
        root=tmp_path,
        rel_path="artifacts/out/ingress_merge_parity.json",
        artifact=artifact,
    )
    loaded = load_ingress_merge_parity_artifact(
        root=tmp_path,
        rel_path="artifacts/out/ingress_merge_parity.json",
        identities=StructuredArtifactIdentitySpace(),
    )

    assert loaded is not None
    assert loaded.source.producer
    assert {item.case_key for item in loaded.cases} == {
        "adapter_decision_surface_parity",
        "frontmatter_adapter_projection_parity",
        "policy_source_uniqueness",
        "policy_registry_determinism",
    }
    loaded_adapter_case = next(
        item for item in loaded.cases if item.case_key == "adapter_decision_surface_parity"
    )
    assert loaded_adapter_case.field_checks[0].field_name == "python_expected_decision_surfaces"
    assert any(
        item.case_key == "frontmatter_adapter_projection_parity"
        for item in loaded.cases
    )
