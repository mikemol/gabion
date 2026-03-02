---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: legacy_dataflow_monolith_symbol_migration_index
doc_role: index
doc_scope:
  - repo
  - analysis
  - migration
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
  POLICY_SEED.md#policy_seed: "Migration index follows forward-remediation and compatibility lifecycle obligations."
  AGENTS.md#agent_obligations: "Index and checker align with required correction-unit traceability."
  glossary.md#contract: "Legacy symbol aliases are tracked as explicit commutation anchors."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="legacy_dataflow_monolith_symbol_migration_index"></a>
# Dataflow Audit Symbol Migration Index

This index tracks `in/*.md` anchors that still mention either
`src/gabion/analysis/legacy_dataflow_monolith.py` or
`src/gabion/analysis/legacy_dataflow_monolith.py`.
Audit-era anchors are canonicalized to the runtime path for migration coverage.

Statuses:
- `active_relocated`: active behavior is sourced from replacement module/symbol.
- `historical_anchor`: mention is preserved as historical evidence; runtime source is no longer expected to stay at the legacy symbol.

| Legacy symbol | Replacement module/symbol | Status | Evidence link |
| --- | --- | --- | --- |
| `src/gabion/analysis/legacy_dataflow_monolith.py` | `src/gabion/analysis/dataflow_pipeline.py`, `src/gabion/analysis/dataflow_reporting.py`, `src/gabion/analysis/dataflow_obligations.py`, `src/gabion/analysis/dataflow_run_outputs.py` | `active_relocated` | `docs/audits/functional_core_audit.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_EXECUTION_PATTERN_RULES` | `src/gabion/analysis/pattern_schema_projection.py::EXECUTION_PATTERN_RULES` | `active_relocated` | `src/gabion/analysis/pattern_schema_projection.py` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_accumulate_function_index_for_tree` | `src/gabion/analysis/legacy_dataflow_monolith.py::_accumulate_function_index_for_tree` | `historical_anchor` | `in/in-35.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_analyze_unused_arg_flow_indexed` | `src/gabion/analysis/legacy_dataflow_monolith.py::_analyze_unused_arg_flow_indexed` | `historical_anchor` | `in/in-35.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_bundle_pattern_instances` | `src/gabion/analysis/pattern_schema_projection.py::bundle_pattern_instances` | `active_relocated` | `src/gabion/analysis/pattern_schema_projection.py` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_callsite_evidence_for_bundle` | `src/gabion/analysis/legacy_dataflow_monolith.py::_callsite_evidence_for_bundle` | `historical_anchor` | `in/in-34.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_collect_call_resolution_obligations_from_forest` | `src/gabion/analysis/dataflow_obligations.py::collect_deadline_obligations` | `active_relocated` | `src/gabion/analysis/dataflow_obligations.py` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_collect_closure_lambda_factories` | `src/gabion/analysis/legacy_dataflow_monolith.py::_collect_closure_lambda_factories` | `historical_anchor` | `in/in-34.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_collect_deadline_obligations` | `src/gabion/analysis/dataflow_obligations.py::collect_deadline_obligations` | `active_relocated` | `src/gabion/analysis/dataflow_obligations.py` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_collect_lambda_bindings_by_caller` | `src/gabion/analysis/legacy_dataflow_monolith.py::_collect_lambda_bindings_by_caller` | `historical_anchor` | `in/in-34.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_collect_lambda_function_infos` | `src/gabion/analysis/legacy_dataflow_monolith.py::_collect_lambda_function_infos` | `historical_anchor` | `in/in-34.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_compute_fingerprint_provenance` | `src/gabion/analysis/legacy_dataflow_monolith.py::_compute_fingerprint_provenance` | `historical_anchor` | `in/in-23.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_compute_fingerprint_synth` | `src/gabion/analysis/legacy_dataflow_monolith.py::_compute_fingerprint_synth` | `historical_anchor` | `in/in-23.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_execution_pattern_instances` | `src/gabion/analysis/pattern_schema_projection.py::execution_pattern_instances` | `active_relocated` | `src/gabion/analysis/pattern_schema_projection.py` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_internal_broad_type_lint_lines` | `src/gabion/analysis/legacy_dataflow_monolith.py::_internal_broad_type_lint_lines` | `historical_anchor` | `in/in-31.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_iter_dataclass_call_bundles` | `src/gabion/analysis/dataflow_bundle_iteration.py::iter_dataclass_call_bundle_effects` | `active_relocated` | `src/gabion/analysis/dataflow_bundle_iteration.py` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_materialize_ambiguity_suite_agg_spec` | `src/gabion/analysis/legacy_dataflow_monolith.py::_materialize_ambiguity_suite_agg_spec` | `historical_anchor` | `in/in-30.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_materialize_ambiguity_virtual_set_spec` | `src/gabion/analysis/legacy_dataflow_monolith.py::_materialize_ambiguity_virtual_set_spec` | `historical_anchor` | `in/in-30.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_materialize_call_candidates` | `src/gabion/analysis/dataflow_callee_resolution.py::plan_callee_resolution` | `active_relocated` | `src/gabion/analysis/dataflow_callee_resolution.py` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_materialize_suite_order_spec` | `src/gabion/analysis/legacy_dataflow_monolith.py::_materialize_suite_order_spec` | `historical_anchor` | `in/in-30.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_pattern_schema_snapshot_entries` | `src/gabion/analysis/pattern_schema_projection.py::pattern_schema_snapshot_entries` | `active_relocated` | `src/gabion/analysis/pattern_schema_projection.py` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_resolve_callee` | `src/gabion/analysis/dataflow_callee_resolution.py::resolve_callee_with_effects` | `active_relocated` | `src/gabion/analysis/dataflow_callee_resolution.py` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_resolve_callee_outcome` | `src/gabion/analysis/dataflow_callee_resolution.py::resolve_callee_with_effects` | `active_relocated` | `src/gabion/analysis/dataflow_callee_resolution.py` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_synthetic_lambda_name` | `src/gabion/analysis/legacy_dataflow_monolith.py::_synthetic_lambda_name` | `historical_anchor` | `in/in-34.md` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_unresolved_starred_witness` | `src/gabion/analysis/dataflow_bundle_iteration.py::_unresolved_starred_witness` | `active_relocated` | `src/gabion/analysis/dataflow_bundle_iteration.py` |
| `src/gabion/analysis/legacy_dataflow_monolith.py::_unused_params` | `src/gabion/analysis/legacy_dataflow_monolith.py::_unused_params` | `historical_anchor` | `in/in-35.md` |
