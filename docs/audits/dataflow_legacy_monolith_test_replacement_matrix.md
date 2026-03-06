---
doc_revision: 4
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: legacy_dataflow_monolith_test_replacement_matrix
doc_role: audit
doc_scope:
  - repo
  - tests
  - analysis
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - AGENTS.md#agent_obligations
  - glossary.md#contract
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  AGENTS.md#agent_obligations: 2
  glossary.md#contract: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Matrix tracks marker-kind labeling parity in report/lint replacement surfaces and fallback semantics for historical rows."
  AGENTS.md#agent_obligations: "Rows track runtime-coupled test surfaces for same-CU replacement migration."
  glossary.md#contract: "Capability replacement rows are tracked as explicit commutation evidence placeholders."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="legacy_dataflow_monolith_test_replacement_matrix"></a>
# Dataflow Runtime Test Replacement Matrix

This matrix tracks runtime-coupled test assertions and their replacement targets.
Rows are initialized from current `tests/` imports of `gabion.analysis.legacy_dataflow_monolith`.

| Runtime-coupled test file | Domain | Replacement test surface | Replacement module(s) | Removal CU | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `tests/test_alias_return_propagation.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_ambiguity_helpers.py` | `D3` | `ambiguity helper assertions` | `dataflow_ambiguity_helpers; dataflow_reporting` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_broad_type_lint.py` | `D5` | `broad type lint assertions` | `dataflow_lint_helpers; dataflow_pipeline` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_callsite_evidence_helper.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_config_fields.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_constant_flow_audit.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_constant_flow_varargs_and_knobs.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_dataclass_call_bundles.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_legacy_dataflow_monolith_ambiguity_suite.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_legacy_dataflow_monolith_coverage_gaps.py` | `D2/D3/D6` | `owner-module replacement coverage` | `dataflow_pipeline; dataflow_reporting; dataflow_snapshot_io` | `CU-RT-01` | `replaced` | Pruned alias suite; replaced by owner-surface assertions in active suites. |
| `tests/test_legacy_dataflow_monolith_edges.py` | `D2/D3/D6` | `owner-module replacement coverage` | `dataflow_pipeline; dataflow_reporting; dataflow_snapshot_io` | `CU-RT-01` | `replaced` | Pruned alias suite; replaced by owner-surface assertions in active suites. |
| `tests/test_legacy_dataflow_monolith_flows.py` | `D2/D3/D6` | `owner-module replacement coverage` | `dataflow_pipeline; dataflow_reporting; dataflow_snapshot_io` | `CU-RT-01` | `replaced` | Pruned alias suite; replaced by owner-surface assertions in active suites. |
| `tests/test_legacy_dataflow_monolith_gap_closure.py` | `D2/D3/D6` | `owner-module replacement coverage` | `dataflow_pipeline; dataflow_reporting; dataflow_snapshot_io` | `CU-RT-01` | `replaced` | Pruned alias suite; replaced by owner-surface assertions in active suites. |
| `tests/test_legacy_dataflow_monolith_helpers.py` | `D2/D3/D6` | `owner-module replacement coverage` | `dataflow_pipeline; dataflow_reporting; dataflow_snapshot_io` | `CU-RT-01` | `replaced` | Pruned alias suite; replaced by owner-surface assertions in active suites. |
| `tests/test_legacy_dataflow_monolith_merge_canonicalization.py` | `D2/D3/D6` | `owner-module replacement coverage` | `dataflow_pipeline; dataflow_reporting; dataflow_snapshot_io` | `CU-RT-01` | `replaced` | Pruned alias suite; replaced by owner-surface assertions in active suites. |
| `tests/test_legacy_dataflow_monolith_never_invariants.py` | `D3/D5` | `owner-module replacement coverage` | `dataflow_exception_obligations; dataflow_obligations` | `CU-RT-01` | `replaced` | Pruned alias suite; replaced by owner-surface assertions in active suites. |
| `tests/test_dataflow_class_resolution.py` | `D1` | `class/callee resolution assertions` | `dataflow_callee_resolution; dataflow_evidence_helpers` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_dataflow_dataclass_bundles.py` | `D1/D2` | `dataclass bundle assertions` | `dataflow_bundle_iteration; dataflow_ingest_helpers` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_dataflow_grouping.py` | `D2` | `grouping propagation assertions` | `dataflow_pipeline` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_dataflow_helpers.py` | `D1/D2` | `owner-module replacement coverage` | `dataflow_ingest_helpers; dataflow_analysis_index; dataflow_pipeline` | `CU-RT-01` | `replaced` | Pruned alias suite; replaced by owner-surface assertions in active suites. |
| `tests/test_dataflow_misc_edges.py` | `D2/D3/D5/D6` | `owner-module replacement coverage` | `dataflow_pipeline; dataflow_reporting; dataflow_obligations` | `CU-RT-01` | `replaced` | Pruned alias suite; replaced by owner-surface assertions in active suites. |
| `tests/test_dataflow_report_helpers.py` | `D6` | `report/snapshot helper assertions` | `dataflow_reporting; dataflow_snapshot_io; dataflow_report_rendering` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_dataflow_resolve_callee.py` | `D1` | `callee resolution assertions` | `dataflow_callee_resolution; dataflow_function_index_helpers` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_deadline_coverage.py` | `D5` | `deadline obligation assertions` | `dataflow_obligations; dataflow_deadline_helpers` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_decision_surfaces.py` | `D3` | `decision/value-decision assertions` | `dataflow_decision_surfaces; dataflow_pipeline` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_exception_deadness_helpers.py` | `D5` | `exception/deadness helper assertions` | `dataflow_exception_obligations; dataflow_indexed_file_scan` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_exception_protocol_never.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_fingerprint_soundness.py` | `D4` | `fingerprint soundness assertions` | `dataflow_fingerprint_helpers` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_fingerprint_warnings.py` | `D4` | `fingerprint warning assertions` | `dataflow_fingerprint_helpers` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_forest_only_invariant.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_invariant_emitter_hooks.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_lint_lines.py` | `D5` | `lint line materialization assertions` | `dataflow_lint_helpers; dataflow_decision_surfaces` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_matrix_acceptance_artifacts.py` | `D6` | `raw runtime artifact assertions` | `dataflow_raw_runtime` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_never_invariants.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_property_hook_manifest.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_python_ingest.py` | `D1` | `python ingest assertions` | `dataflow_ingest_helpers; dataflow_analysis_index` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_rewrite_plan_verification.py` | `D3/D4` | `rewrite verification assertions` | `dataflow_decision_surfaces; dataflow_fingerprint_helpers` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_star_import_resolution.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_structure_metrics.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_structure_snapshot.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_structure_snapshot_invariants.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_suite_order_projection_spec.py` | `D3` | `suite order projection assertions` | `dataflow_ambiguity_helpers` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_suite_site_projection_parity.py` | `D2/D6` | `suite-site projection parity assertions` | `dataflow_pipeline; dataflow_snapshot_io` | `CU-RT-01` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_type_fingerprints_sidecar.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_type_flow_callsite_evidence.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_unused_arg_audit.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_visitors_edges.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_visitors_unit.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |
| `tests/test_wildcard_forwarding.py` | `D1/D2/D3/D4/D5/D6` | `owner-module replacement coverage` | `dataflow_indexed_file_scan; owner modules` | `CU-RT-FINAL-2` | `retargeted` | Retargeted to direct owner imports. |

## Marker-kind report/lint parity note

For the report and lint replacement surfaces, never-invariant line labels now preserve each row's `marker_kind` value (`never`, `todo`, `deprecated`) instead of collapsing to `never()` unconditionally.
When a historical row omits `marker_kind` (or provides an empty value), rendering intentionally defaults to `never()` so older artifacts remain interpretable.

