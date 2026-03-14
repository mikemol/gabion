from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling.policy_substrate.structured_artifact_ingress import (
    GitStateLineSpan,
    StructuredArtifactDecompositionKind,
    StructuredArtifactIdentitySpace,
    StructuredArtifactKind,
    build_ingress_merge_parity_artifact,
    load_cross_origin_witness_contract_artifact,
    load_git_state_artifact,
    load_controller_drift_artifact,
    load_docflow_packet_enforcement_artifact,
    load_ingress_merge_parity_artifact,
    load_junit_failure_artifact,
    load_local_repro_closure_ledger_artifact,
    load_test_evidence_artifact,
    write_ingress_merge_parity_artifact,
)


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
    assert junit.failures[0].test_id == "tests/example_test.py::TestExample::test_value"
    assert junit.failures[0].traceback_text
    assert evidence.cases[0].identity.canonical.atom_id != junit.failures[0].identity.canonical.atom_id


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
