---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: generated_artifact_manifest
doc_role: reference
doc_scope:
  - repo
  - tooling
  - ci
  - governance
doc_authority: informative
doc_requires:
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/governance_control_loops.md#governance_control_loops
doc_reviewed_as_of:
  README.md#repo_contract: 2
  CONTRIBUTING.md#contributing_contract: 2
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  docs/governance_control_loops.md#governance_control_loops: 2
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md#repo_contract rev84/section v2 (artifact-heavy planning/governance outputs remain repo-local tooling surfaces rather than product-facing features)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md#contributing_contract rev120/section v2 (the correction-unit validation stack and evidence-carrier refresh rules define the core normal-course artifact set)."
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md#policy_seed rev57/section v2 (process-relative runtime and the distinction ladder justify documenting planning, governance, timeout, and CI artifact surfaces as real runtime outputs)."
  glossary.md#contract: "Reviewed glossary.md#contract rev47/section v1 (runtime/process scope and queue/workstream semantics remain aligned with the artifact manifest framing)."
  docs/governance_control_loops.md#governance_control_loops: "Reviewed docs/governance_control_loops.md#governance_control_loops rev7/section v2 (state-artifact language and first/second-order control-loop registry remain aligned with the generated artifact inventory)."
doc_sections:
  generated_artifact_manifest: 1
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="generated_artifact_manifest"></a>
# Generated artifact manifest

This document inventories the stable generated artifacts that arise in normal
repo operation.

For this document, “normal repo operation” includes:

- the standard local correction-unit validation stack,
- the normal CI workflow paths,
- supported ordinary runtime/error branches such as timeout, handoff, and
  recovery/reporting paths when those branches occur,
- stable workflow- or command-owned output paths.

It excludes:

- checked-in source inputs and baselines,
- caches and scratch directories,
- one-off investigator outputs,
- arbitrary user-selected override targets that are not the stable path used by
  the repo’s ordinary workflows.

Use this manifest when you need to answer:

- which generated artifact is canonical for a given process,
- which command or workflow step emits it,
- which downstream tools consume it,
- how to regenerate it locally.

To refresh the generated section below:

```bash
mise exec -- python -m scripts.policy.render_generated_artifact_manifest
```

<!-- BEGIN:generated_artifact_manifest -->
_This manifest section is generated from `docs/generated_artifact_manifest.yaml` via `mise exec -- python -m scripts.policy.render_generated_artifact_manifest`._

## Governance, docflow, and controller artifacts

Stable artifacts emitted by docflow, governance audits, packetization, and controller-drift enforcement.

| Artifact ID | Path(s) | Format | Emitted by | Trigger | Regeneration | Primary consumers | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `docflow_suite_graph` | `artifacts/docflow_suite_forest.json`<br>`artifacts/docflow_suite_relation.json` | `json` | command: gabion docflow | always - emitted on normal docflow runs | `mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required` | `gabion docflow`<br>`governance audits` | Suite-site forest and relation carriers feed downstream docflow analysis. |
| `docflow_compliance_bundle` | `artifacts/out/docflow_compliance.json`<br>`artifacts/audit_reports/docflow_compliance.md` | `json+md` | command: gabion docflow | always - emitted on normal docflow runs | `mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required` | `scripts/policy/docflow_packetize.py`<br>`governance_loop_docs` | Core contradiction and warning surface for docflow. |
| `docflow_canonicality_bundle` | `artifacts/out/docflow_canonicality.json`<br>`artifacts/audit_reports/docflow_canonicality.md` | `json+md` | command: gabion docflow | always - emitted on normal docflow runs | `mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required` | `gabion docflow` | Canonicality-specific docflow projection. |
| `docflow_cycles_bundle` | `artifacts/out/docflow_cycles.json`<br>`artifacts/audit_reports/docflow_cycles.md` | `json+md` | command: gabion docflow | always - emitted on normal docflow runs | `mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required` | `gabion docflow` | Dependency-cycle projection for docflow surfaces. |
| `docflow_change_protocol_bundle` | `artifacts/out/docflow_change_protocol.json`<br>`artifacts/audit_reports/docflow_change_protocol.md` | `json+md` | command: gabion docflow | always - emitted on normal docflow runs | `mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required` | `gabion docflow` | Change-protocol projection for normative docs. |
| `docflow_section_reviews_bundle` | `artifacts/out/docflow_section_reviews.json`<br>`artifacts/audit_reports/docflow_section_reviews.md` | `json+md` | command: gabion docflow | always - emitted on normal docflow runs | `mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required` | `scripts/policy/docflow_packetize.py`<br>`scripts/policy/docflow_packet_enforce.py` | Section-review carrier used by the packet loop. |
| `docflow_implication_matrices` | `artifacts/out/docflow_implication_matrices.json` | `json` | command: gabion docflow | always - emitted on normal docflow runs | `mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required` | `gabion docflow` | Machine-readable implication matrix projection. |
| `agent_instruction_drift_bundle` | `artifacts/out/agent_instruction_drift.json`<br>`artifacts/audit_reports/agent_instruction_drift.md` | `json+md` | command: gabion docflow | always - emitted on normal docflow runs | `mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required` | `gabion docflow` | Drift report for agent-instruction graph consistency. |
| `sppf_dependency_graph` | `artifacts/sppf_dependency_graph.json` | `json` | command: gabion docflow | always - emitted on strict docflow runs | `mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required` | `SPPF status checks`<br>`status consistency analysis` | JSON dependency graph used by SPPF governance surfaces. |
| `status_consistency_bundle` | `artifacts/out/status_consistency.json`<br>`artifacts/audit_reports/status_consistency.md` | `json+md` | command: gabion status-consistency | always - emitted when the status-consistency command runs | `mise exec -- python -m gabion status-consistency --fail-on-violations` | `governance review`<br>`SPPF lifecycle checks` | Separate governance command, but part of ordinary repo governance tooling. |
| `docflow_packetization_bundle` | `artifacts/out/docflow_warning_doc_packets.json`<br>`artifacts/out/docflow_warning_doc_packet_summary.json` | `json` | command: scripts/policy/docflow_packetize.py | always - emitted on normal packetization runs | `mise exec -- python -m scripts.policy.docflow_packetize` | `scripts/policy/docflow_packet_enforce.py`<br>`CI audit workflow` | Packetized warning rows and packet summary for the docflow closed loop. |
| `docflow_packet_enforcement_bundle` | `artifacts/out/docflow_packet_enforcement.json`<br>`artifacts/out/docflow_packet_debt_ledger.json` | `json` | command: scripts/policy/docflow_packet_enforce.py | always - emitted on normal packet-enforcement runs | `mise exec -- python -m scripts.policy.docflow_packet_enforce --check` | `CI audit workflow`<br>`governance debt review` | Enforcement result plus age-bounded debt ledger. |
| `controller_drift_audit` | `artifacts/out/controller_drift.json` | `json` | command: scripts/governance/governance_controller_audit.py | always - emitted on controller-drift audit runs | `mise exec -- python scripts/governance/governance_controller_audit.py --out artifacts/out/controller_drift.json` | `scripts/ci/ci_controller_drift_gate.py`<br>`invariant graph ingress` | Second-order controller-loop state artifact. |
| `controller_drift_gate_state` | `artifacts/out/governance_override_record.json`<br>`artifacts/out/controller_drift_gate_history.json` | `json` | workflow_step: .github/workflows/ci.yml audit job controller-drift gate steps | always - emitted on normal CI audit runs | `mise exec -- python scripts/ci/ci_override_record_emit.py --out artifacts/out/governance_override_record.json`<br>`mise exec -- python scripts/ci/ci_controller_drift_gate.py --drift-artifact artifacts/out/controller_drift.json --override-record artifacts/out/governance_override_record.json --history artifacts/out/controller_drift_gate_history.json` | `controller-drift gate`<br>`uploaded CI artifacts` | Override lifecycle record plus branch-local controller-drift history. |
| `governance_telemetry_bundle` | `artifacts/out/governance_telemetry.json`<br>`artifacts/out/governance_telemetry_history.json`<br>`artifacts/audit_reports/governance_telemetry.md`<br>`artifacts/audit_reports/ci_step_timings.json` | `json+md` | workflow_step: .github/workflows/ci.yml audit job governance telemetry emit | always - emitted on normal CI audit runs | `mise exec -- python scripts/governance/governance_telemetry_emit.py --run-id local --timings artifacts/audit_reports/ci_step_timings.json --history artifacts/out/governance_telemetry_history.json --json-out artifacts/out/governance_telemetry.json --md-out artifacts/audit_reports/governance_telemetry.md` | `uploaded CI artifacts`<br>`governance trend review` | Step timings are an input carrier for telemetry and are included with the emitted telemetry bundle because CI treats them as one operational surface. |

## Planning, policy, and projection artifacts

Stable artifacts emitted by workflow policy checks, planning-substrate projection, and queue renderers.

| Artifact ID | Path(s) | Format | Emitted by | Trigger | Regeneration | Primary consumers | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `workflow_policy_quotients` | `out/quotient_governance_report.json`<br>`out/quotient_ratchet_delta.json`<br>`out/quotient_policy_violations.json`<br>`out/quotient_protocol_readiness.json`<br>`out/quotient_promotion_decision.json`<br>`out/quotient_demotion_incidents.json`<br>`artifacts/out/local_ci_repro_contract.json` | `json` | command: scripts/policy/policy_check.py --workflows | always - emitted on normal workflow-policy checks | `mise exec -- python -m scripts.policy.policy_check --workflows` | `governance review`<br>`invariant graph ingress` | Workflow-policy quotients are written even when no explicit policy-check result path is requested. |
| `policy_check_projection_bundle` | `artifacts/out/policy_check_result.json`<br>`artifacts/out/invariant_graph.json`<br>`artifacts/out/invariant_workstreams.json`<br>`artifacts/out/invariant_ledger_projections.json` | `json` | workflow_step: .github/workflows/ci.yml audit job policy_check --workflows --output | conditional - emitted when policy_check is run with the stable CI output path | `mise exec -- python scripts/policy/policy_check.py --workflows --output artifacts/out/policy_check_result.json` | `scripts/policy/policy_scanner_suite.py`<br>`project manager view`<br>`invariant graph tooling` | These are workflow-stable outputs, not generic CLI defaults. |
| `policy_check_auxiliary_artifacts` | `artifacts/out/ingress_merge_parity.json`<br>`artifacts/out/git_state.json`<br>`artifacts/out/cross_origin_witness_contract.json`<br>`artifacts/out/kernel_vm_alignment.json`<br>`artifacts/out/identity_grammar_completion.json` | `json` | workflow_step: .github/workflows/ci.yml audit job policy_check auxiliary artifact writes | conditional - emitted when policy_check is run with the stable CI output path | `mise exec -- python scripts/policy/policy_check.py --workflows --output artifacts/out/policy_check_result.json` | `invariant graph ingress`<br>`hotspot and projection queue tooling` | Auxiliary planning and ingress artifacts written adjacent to the CI policy-check result. |
| `projection_semantic_fragment_queue_bundle` | `artifacts/out/projection_semantic_fragment_queue.json`<br>`artifacts/out/projection_semantic_fragment_queue.md` | `json+md` | workflow_step: .github/workflows/ci.yml audit job policy_check projection queue write | conditional - emitted when policy_check is run with the stable CI output path | `mise exec -- python scripts/policy/policy_check.py --workflows --output artifacts/out/policy_check_result.json` | `projection queue review`<br>`uploaded CI artifacts` | Derived from the policy-check result carrier. |
| `hotspot_neighborhood_queue_bundle` | `artifacts/out/hotspot_neighborhood_queue.json`<br>`artifacts/out/hotspot_neighborhood_queue.md` | `json+md` | command: scripts/policy/policy_scanner_suite.py --out-dir artifacts/out | always - emitted on normal policy-scanner-suite runs | `mise exec -- python scripts/policy/policy_scanner_suite.py --root . --out-dir artifacts/out` | `uploaded CI artifacts`<br>`hotspot review` | Primary queue artifact emitted by the policy scanner suite. |
| `structural_policy_results` | `artifacts/out/structural_hash_result.json`<br>`artifacts/out/deprecated_nonerasability_result.json`<br>`out/deprecated_fibers_current.json` | `json` | workflow_step: .github/workflows/ci.yml audit job structural hash and deprecated-nonerasability steps | always - emitted on normal CI audit runs | `mise exec -- python scripts/policy/structural_hash_policy_check.py --root . --output artifacts/out/structural_hash_result.json`<br>`mise exec -- python scripts/policy/deprecated_nonerasability_policy_check.py --baseline out/deprecated_fibers_baseline.json --current out/deprecated_fibers_current.json --output artifacts/out/deprecated_nonerasability_result.json` | `uploaded CI artifacts`<br>`governance review` | Includes the generated current deprecated-fiber snapshot consumed by the nonerasability gate. |

## Test, evidence, and coverage artifacts

Stable artifacts emitted by the standard evidence-index and pytest coverage flows.

| Artifact ID | Path(s) | Format | Emitted by | Trigger | Regeneration | Primary consumers | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `test_evidence_index` | `out/test_evidence.json` | `json` | command: scripts/misc/extract_test_evidence.py | always - emitted on normal evidence-index refresh runs | `mise exec -- python -m scripts.misc.extract_test_evidence --root . --tests tests --out out/test_evidence.json` | `invariant graph ingress`<br>`impact and obsolescence tooling` | Canonical test-evidence carrier. |
| `test_behavior_index` | `out/test_behavior.json` | `json` | command: scripts/misc/extract_test_behavior.py | always - emitted on normal behavior-index refresh runs | `mise exec -- python -m scripts.misc.extract_test_behavior --root . --tests tests --out out/test_behavior.json` | `behavior-index drift checks` | Canonical test-behavior carrier. |
| `pytest_coverage_bundle` | `artifacts/test_runs/coverage.xml`<br>`artifacts/test_runs/htmlcov/**` | `xml+directory_tree` | workflow_step: .github/workflows/ci.yml audit job pytest coverage step | always - emitted on normal CI pytest runs | `mise exec -- python -m pytest --cov=src/gabion --cov-branch --cov-report=xml:artifacts/test_runs/coverage.xml --cov-report=html:artifacts/test_runs/htmlcov --cov-fail-under=100` | `coverage review`<br>`uploaded CI artifacts` | Coverage XML plus HTML tree emitted by the CI pytest step. |
| `pytest_execution_bundle` | `artifacts/test_runs/junit.xml`<br>`artifacts/test_runs/pytest.log` | `xml+log` | workflow_step: .github/workflows/ci.yml audit job pytest execution step | always - emitted on normal CI pytest runs | `mise exec -- python -m pytest --junitxml artifacts/test_runs/junit.xml --log-file artifacts/test_runs/pytest.log --log-file-level=INFO` | `uploaded CI artifacts`<br>`invariant graph ingress` | JUnit and log surfaces emitted by the CI pytest step. |

## Dataflow, deadline, and handoff artifacts

Stable artifacts emitted by the CI dataflow stage, timeout/deadline summarization, and ASPF handoff flows.

| Artifact ID | Path(s) | Format | Emitted by | Trigger | Regeneration | Primary consumers | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dataflow_report_bundle` | `artifacts/audit_reports/dataflow_report.md`<br>`artifacts/audit_reports/dataflow_report_stage_*.md` | `md` | workflow_step: .github/workflows/ci.yml dataflow-grammar job finalize dataflow outcome | conditional - emitted by the normal dataflow-stage flow when stage/final reports exist | `mise exec -- python -m gabion --carrier direct --timeout 130000000000000ns run-dataflow-stage --debug-dump-interval-seconds 60 --stage-strictness-profile "run=high"` | `uploaded CI artifacts`<br>`audit review` | Stage reports are path-conditional because the runtime may emit zero or more stage snapshots. |
| `deadline_profile_bundle` | `artifacts/out/deadline_profile.json`<br>`artifacts/out/deadline_profile.md`<br>`artifacts/out/deadline_profile_stage_*.json`<br>`artifacts/out/deadline_profile_stage_*.md` | `json+md` | runtime_path: run-dataflow-stage / server orchestrator timeout and profiling paths | conditional - emitted when deadline profiling or stage timeout profiling is exercised | `mise exec -- python -m gabion --carrier direct --timeout 130000000000000ns run-dataflow-stage --debug-dump-interval-seconds 60 --stage-strictness-profile "run=high"` | `scripts/deadline/deadline_profile_ci_summary.py`<br>`uploaded CI artifacts` | Ordinary supported timeout/profiling branch, but not guaranteed on every run. |
| `deadline_profile_ci_summary_bundle` | `artifacts/out/deadline_profile_ci_summary.json`<br>`artifacts/out/deadline_profile_ci_summary.md` | `json+md` | workflow_step: .github/workflows/ci.yml dataflow-grammar job deadline profile summary | conditional - emitted when artifacts/out/deadline_profile.json exists | `mise exec -- python -m scripts.deadline_profile_ci_summary --allow-missing-local --step-summary /tmp/step_summary.md` | `uploaded CI artifacts` | CI summary is explicitly conditional on the base deadline profile being present. |
| `aspf_handoff_state_bundle` | `artifacts/out/aspf_handoff_manifest.json`<br>`artifacts/out/aspf_state/**` | `json+directory_tree` | workflow_step: .github/workflows/ci.yml dataflow-grammar and audit jobs ASPF handoff flow | always - emitted on normal handoff-enabled dataflow and audit runs | `mise exec -- python scripts/misc/aspf_handoff.py run --root . --manifest artifacts/out/aspf_handoff_manifest.json --state-root artifacts/out/aspf_state -- true` | `CI handoff restore paths`<br>`uploaded CI artifacts`<br>`deadline and check runtimes` | Directory tree row documents the stable handoff state root rather than individual handoff files. |

## Conditional runtime analysis artifacts

Stable artifacts emitted on ordinary but path-conditional analysis/reporting branches in the semantic runtime.

| Artifact ID | Path(s) | Format | Emitted by | Trigger | Regeneration | Primary consumers | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `annotation_drift_bundle` | `artifacts/out/test_annotation_drift.json`<br>`out/test_annotation_drift.md`<br>`artifacts/out/test_annotation_drift_delta.json`<br>`out/test_annotation_drift_delta.md` | `json+md` | runtime_path: server_core command orchestrator auxiliary artifact outputs | conditional - emitted when annotation drift reporting or delta emission is requested | `mise exec -- python -m gabion --carrier direct --timeout 130000000000000ns check delta-gates` | `governance telemetry`<br>`delta gates` | Ordinary supported runtime branch; emitted only when the corresponding flags are active. |
| `obsolescence_bundle` | `artifacts/out/test_obsolescence_state.json`<br>`artifacts/out/test_obsolescence_report.json`<br>`out/test_obsolescence_report.md`<br>`artifacts/out/test_obsolescence_delta.json`<br>`out/test_obsolescence_delta.md` | `json+md` | runtime_path: server_core command orchestrator auxiliary artifact outputs | conditional - emitted when obsolescence state/report/delta emission is requested | `mise exec -- python -m gabion --carrier direct --timeout 130000000000000ns check delta-gates` | `governance telemetry`<br>`obsolescence delta gates` | Includes the state carrier plus report and delta projections. |
| `ambiguity_bundle` | `artifacts/out/ambiguity_state.json`<br>`artifacts/out/ambiguity_delta.json`<br>`out/ambiguity_delta.md` | `json+md` | runtime_path: server_core command orchestrator auxiliary artifact outputs | conditional - emitted when ambiguity state or delta emission is requested | `mise exec -- python -m gabion --carrier direct --timeout 130000000000000ns check delta-gates` | `governance telemetry`<br>`ambiguity delta gates` | Ambiguity baseline writes are intentionally excluded because they are explicit baseline-management inputs, not generated defaults. |
| `taint_bundle` | `artifacts/out/taint_state.json`<br>`artifacts/out/taint_delta.json`<br>`out/taint_delta.md` | `json+md` | runtime_path: server_core command orchestrator auxiliary artifact outputs | conditional - emitted when taint state or delta emission is requested | `mise exec -- python -m gabion --carrier direct --timeout 130000000000000ns check delta-gates` | `taint delta review`<br>`governance telemetry` | Readiness/promotion/demotion quotient files are documented under the workflow-policy quotient bundle because they converge on shared stable paths. |
| `test_evidence_suggestions_bundle` | `artifacts/out/test_evidence_suggestions.json`<br>`out/test_evidence_suggestions.md` | `json+md` | runtime_path: server_core command orchestrator auxiliary artifact outputs | conditional - emitted when evidence-suggestion reporting is requested | `mise exec -- python -m gabion --carrier direct --timeout 130000000000000ns check delta-bundle` | `evidence review` | Suggestion surface built from the canonical test-evidence index. |
| `call_clusters_bundle` | `artifacts/out/call_clusters.json`<br>`out/call_clusters.md` | `json+md` | runtime_path: server_core command orchestrator auxiliary artifact outputs | conditional - emitted when call-cluster reporting is requested | `mise exec -- python -m gabion --carrier direct --timeout 130000000000000ns check delta-bundle` | `clustering review` | Cluster summary surface derived from the test-evidence carrier. |
| `call_cluster_consolidation_bundle` | `artifacts/out/call_cluster_consolidation.json`<br>`out/call_cluster_consolidation.md` | `json+md` | runtime_path: server_core command orchestrator auxiliary artifact outputs | conditional - emitted when call-cluster consolidation reporting is requested | `mise exec -- python -m gabion --carrier direct --timeout 130000000000000ns check delta-bundle` | `consolidation review` | Consolidation output sits on the same supported runtime reporting surface as call clusters. |
<!-- END:generated_artifact_manifest -->
