---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: dataflow_runtime_retirement_ledger
doc_role: audit
doc_scope:
  - repo
  - analysis
  - aspf
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
  POLICY_SEED.md#policy_seed: "Ledger rows are projected from existing ASPF carriers; no bespoke proof substrate introduced."
  AGENTS.md#agent_obligations: "Projection output remains evidence-only and does not bypass correction-unit validation obligations."
  glossary.md#contract: "Capability rows expose commutation witness status as proven/intentional_drift/blocked."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="dataflow_runtime_retirement_ledger"></a>
# Dataflow Runtime Retirement Ledger

Projected from ASPF carriers (`trace`, `equivalence`, `state`, `delta_ledger`).

Deterministic probe contract:
- state snapshot: `artifacts/out/runtime_retirement_probe/aspf_state.snapshot.json`
- delta jsonl: `artifacts/out/runtime_retirement_probe/aspf_state.delta.jsonl`
- required surfaces: `groups_by_path`, `decision_surfaces`, `value_decision_surfaces`, `rewrite_plans`, `violation_summary`, `pattern_schema_instances`, `pattern_schema_residue`

Rows: `36`
Status counts: proven=36, intentional_drift=0, blocked=0
Domain status: D1=proven, D2=proven, D3=proven, D4=proven, D5=proven, D6=proven

| Capability ID | Domain | Proof kind | Replacement module | Required surfaces | Status | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| `d1_class_index` | `D1` | `surface_parity` | `src/gabion/analysis/dataflow_evidence_helpers.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
| `d1_function_collection` | `D1` | `surface_parity` | `src/gabion/analysis/dataflow_ingest_helpers.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
| `d1_function_index` | `D1` | `surface_parity` | `src/gabion/analysis/dataflow_function_index_helpers.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
| `d1_indexed_file_scan_cluster` | `D1` | `surface_parity` | `src/gabion/analysis/dataflow_indexed_file_scan.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
| `d1_path_resolution` | `D1` | `surface_parity` | `src/gabion/analysis/dataflow_ingest_helpers.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
| `d1_symbol_table` | `D1` | `surface_parity` | `src/gabion/analysis/dataflow_evidence_helpers.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
| `d2_bundle_forest` | `D2` | `surface_parity` | `src/gabion/analysis/dataflow_bundle_iteration.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
| `d2_bundle_inference` | `D2` | `surface_parity` | `src/gabion/analysis/dataflow_pipeline.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
| `d2_call_resolution_graph` | `D2` | `surface_parity` | `src/gabion/analysis/dataflow_obligations.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
| `d2_resume_payload` | `D2` | `metamorphic_commutation` | `src/gabion/analysis/dataflow_analysis_index.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
| `d3_ambiguity_suite` | `D3` | `surface_parity` | `src/gabion/analysis/dataflow_ambiguity_helpers.py` | `decision_surfaces, value_decision_surfaces` | `proven` | `state+equivalence witnesses satisfied` |
| `d3_decision_surfaces` | `D3` | `surface_parity` | `src/gabion/analysis/dataflow_decision_surfaces.py` | `decision_surfaces` | `proven` | `state+equivalence witnesses satisfied` |
| `d3_pattern_instances` | `D3` | `metamorphic_commutation` | `src/gabion/analysis/pattern_schema_projection.py` | `pattern_schema_instances` | `proven` | `state+equivalence witnesses satisfied` |
| `d3_pattern_residue` | `D3` | `metamorphic_commutation` | `src/gabion/analysis/pattern_schema_projection.py` | `pattern_schema_residue` | `proven` | `state+equivalence witnesses satisfied` |
| `d3_value_decision_surfaces` | `D3` | `surface_parity` | `src/gabion/analysis/dataflow_decision_surfaces.py` | `value_decision_surfaces` | `proven` | `state+equivalence witnesses satisfied` |
| `d4_coherence` | `D4` | `surface_parity` | `src/gabion/analysis/dataflow_fingerprint_helpers.py` | `rewrite_plans` | `proven` | `state+equivalence witnesses satisfied` |
| `d4_fingerprint_matches` | `D4` | `surface_parity` | `src/gabion/analysis/dataflow_fingerprint_helpers.py` | `rewrite_plans` | `proven` | `state+equivalence witnesses satisfied` |
| `d4_fingerprint_provenance` | `D4` | `surface_parity` | `src/gabion/analysis/dataflow_fingerprint_helpers.py` | `rewrite_plans` | `proven` | `state+equivalence witnesses satisfied` |
| `d4_fingerprint_synth` | `D4` | `surface_parity` | `src/gabion/analysis/dataflow_fingerprint_helpers.py` | `rewrite_plans` | `proven` | `state+equivalence witnesses satisfied` |
| `d4_rewrite_plans` | `D4` | `surface_parity` | `src/gabion/analysis/dataflow_fingerprint_helpers.py` | `rewrite_plans` | `proven` | `state+equivalence witnesses satisfied` |
| `d4_structure_reuse` | `D4` | `surface_parity` | `src/gabion/analysis/dataflow_structure_reuse.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
| `d5_deadline_helper_ownership` | `D5` | `surface_parity` | `src/gabion/analysis/dataflow_deadline_helpers.py` | `violation_summary` | `proven` | `state+equivalence witnesses satisfied` |
| `d5_deadline_obligations` | `D5` | `surface_parity` | `src/gabion/analysis/dataflow_obligations.py` | `violation_summary` | `proven` | `state+equivalence witnesses satisfied` |
| `d5_exception_obligations` | `D5` | `surface_parity` | `src/gabion/analysis/dataflow_exception_obligations.py` | `violation_summary` | `proven` | `state+equivalence witnesses satisfied` |
| `d5_handledness_witnesses` | `D5` | `surface_parity` | `src/gabion/analysis/dataflow_exception_obligations.py` | `violation_summary` | `proven` | `state+equivalence witnesses satisfied` |
| `d5_lint_helper_ownership` | `D5` | `surface_parity` | `src/gabion/analysis/dataflow_lint_helpers.py` | `violation_summary` | `proven` | `state+equivalence witnesses satisfied` |
| `d5_lint_lines` | `D5` | `surface_parity` | `src/gabion/analysis/dataflow_decision_surfaces.py` | `violation_summary` | `proven` | `state+equivalence witnesses satisfied` |
| `d5_never_invariants` | `D5` | `surface_parity` | `src/gabion/analysis/dataflow_obligations.py` | `violation_summary` | `proven` | `state+equivalence witnesses satisfied` |
| `d6_baseline_gate` | `D6` | `gate_parity` | `src/gabion/analysis/dataflow_baseline_gates.py` | `violation_summary` | `proven` | `state+equivalence witnesses satisfied` |
| `d6_decision_snapshot` | `D6` | `surface_parity` | `src/gabion/analysis/dataflow_snapshot_io.py` | `decision_surfaces, pattern_schema_instances, pattern_schema_residue` | `proven` | `state+equivalence witnesses satisfied` |
| `d6_dot_render` | `D6` | `surface_parity` | `src/gabion/analysis/dataflow_graph_rendering.py` | `groups_by_path, decision_surfaces` | `proven` | `state+equivalence witnesses satisfied` |
| `d6_raw_runtime_entry` | `D6` | `gate_parity` | `src/gabion/analysis/dataflow_raw_runtime.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
| `d6_report_render` | `D6` | `surface_parity` | `src/gabion/analysis/dataflow_reporting.py` | `violation_summary, rewrite_plans` | `proven` | `state+equivalence witnesses satisfied` |
| `d6_run_outputs` | `D6` | `surface_parity` | `src/gabion/analysis/dataflow_run_outputs.py` | `violation_summary, rewrite_plans` | `proven` | `state+equivalence witnesses satisfied` |
| `d6_structure_snapshot` | `D6` | `surface_parity` | `src/gabion/analysis/dataflow_snapshot_io.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
| `d6_synthesis_refactor` | `D6` | `surface_parity` | `src/gabion/analysis/dataflow_synthesis.py` | `groups_by_path` | `proven` | `state+equivalence witnesses satisfied` |
