---
doc_revision: 18
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: compatibility_layer_debt_register
doc_role: register
doc_scope:
  - repo
  - governance
  - refactor
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - CONTRIBUTING.md#contributing_contract
  - AGENTS.md#agent_obligations
  - docs/normative_clause_index.md#normative_clause_index
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  CONTRIBUTING.md#contributing_contract: 2
  AGENTS.md#agent_obligations: 2
  docs/normative_clause_index.md#normative_clause_index: 2
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev2 and applied compatibility-layer hardening under NCI-SHIFT-AMBIGUITY-LEFT with boundary-only temporary exceptions."
  CONTRIBUTING.md#contributing_contract: "Reviewed contributing contract rev2; this register mirrors correction-unit and lifecycle metadata requirements."
  AGENTS.md#agent_obligations: "Reviewed agent obligations rev2; register aligns with refusal/forward-remediation expectations."
  docs/normative_clause_index.md#normative_clause_index: "Reviewed clause index rev2; register tracks debt against NCI-SHIFT-AMBIGUITY-LEFT."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="compatibility_layer_debt_register"></a>
# Compatibility Layer Debt Register

Canonical clause: [`NCI-SHIFT-AMBIGUITY-LEFT`](./normative_clause_index.md#clause-shift-ambiguity-left).

## Policy status
- Existing compatibility layers are legacy remediation debt.
- Net-new semantic-core compatibility layers are disallowed.
- Sunset deadline for listed debt: `2026-03-31`.

## Required fields
Every debt row must provide:
- owner
- rationale
- removal correction unit
- expiry
- exit criteria

## Debt ledger

| Surface | Owner | Rationale | Removal correction unit | Expiry | Exit criteria | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `src/gabion/schema.py` (`compatibility_shim` union surface) | Maintainer (unassigned) | Legacy request shape still allows compatibility-shim alternation. | `CU-compat-schema-surface` | `2026-03-31` | Replace union compatibility entry with deterministic refactor contract type and remove compatibility shim DTO path. | Open |
| `src/gabion/server.py` (compatibility normalization path) | Maintainer (unassigned) | Server still normalizes compatibility-shim alternation for refactor requests. | `CU-compat-server-normalization` | `2026-03-31` | Remove compatibility normalization branch and require a single validated ingress contract. | Open |
| `src/gabion/cli.py` (`--compat-shim*` flags/payload) | Maintainer (unassigned) | CLI exposes compatibility toggles that preserve dual-path behavior. | `CU-compat-cli-flags` | `2026-03-31` | Remove `--compat-shim*` options and emit only deterministic refactor payloads. | Open |
| `src/gabion/refactor/model.py` (shim normalization helper) | Maintainer (unassigned) | Model layer still lifts generic shim input into compatibility config. | `CU-compat-refactor-model` | `2026-03-31` | Remove `normalize_compatibility_shim` and compatibility config transport from the model contract. | Open |
| `src/gabion/refactor/engine.py` (shim emission path) | Maintainer (unassigned) | Engine still emits compatibility wrappers and optional overload/deprecation surfaces. | `CU-compat-refactor-engine` | `2026-03-31` | Remove compatibility-wrapper emission and keep only deterministic refactor output path. | Open |
| `src/gabion/analysis/dataflow_pipeline.py` (`_bind_runtime_dependencies`) | Maintainer (unassigned) | Temporary boundary adapter replaced broad runtime `__dict__` symbol-copy injection with explicit dependency wiring. | `CU-RT-03` | `2026-03-31` | Remove runtime import and `_bind_runtime_dependencies`; use only extracted/static module dependencies. | Closed |
| `src/gabion/analysis/dataflow_reporting.py` (`_bind_runtime_dependencies`) | Maintainer (unassigned) | Temporary boundary adapter replaced broad runtime `__dict__` symbol-copy injection via explicit imports and extracted/report bridge surfaces. | `CU-RT-03` | `2026-03-31` | Remove runtime import and `_bind_runtime_dependencies`; use only extracted/static module dependencies. | Closed |
| `src/gabion/analysis/dataflow_run_outputs.py` (`_bind_runtime_dependencies`) | Maintainer (unassigned) | Temporary boundary adapter replaced broad runtime `__dict__` symbol-copy injection via explicit imports and extracted/output bridge surfaces. | `CU-RT-03` | `2026-03-31` | Remove runtime import and `_bind_runtime_dependencies`; use only extracted/static module dependencies. | Closed |
| `src/gabion/analysis/dataflow_obligations.py` (`_bind_runtime_dependencies`) | Maintainer (unassigned) | Temporary boundary adapter replaced broad runtime `__dict__` symbol-copy injection via static imports into extracted deadline helper surfaces. | `CU-RT-03` | `2026-03-31` | Remove runtime import and `_bind_runtime_dependencies`; use only extracted/static module dependencies. | Closed |
| `src/gabion/analysis/dataflow_bundle_iteration.py` (`_bind_runtime_dependencies`) | Maintainer (unassigned) | Temporary boundary adapter replaced broad runtime `__dict__` symbol-copy injection after parse-stage/adhoc-guard ownership moved to extracted helpers. | `CU-RT-03` | `2026-03-31` | Remove runtime import and `_bind_runtime_dependencies`; use only extracted/static module dependencies. | Closed |
| `src/gabion/analysis/dataflow_callee_resolution.py` (`_bind_runtime_dependencies`) | Maintainer (unassigned) | Temporary boundary adapter replaced broad runtime `__dict__` symbol-copy injection while extraction completed. | `CU-RT-03` | `2026-03-31` | Remove runtime import and `_bind_runtime_dependencies`; use only extracted/static module dependencies. | Closed |
| `src/gabion/analysis/dataflow_function_index_helpers.py` (`_build_function_index` runtime bridge adapter) | Maintainer (unassigned) | Temporary boundary adapter centralizes runtime function-index delegation while lambda/analyzer ownership extraction proceeds. | `CU-RT-FINAL-1` | `2026-03-31` | Replace runtime bridge delegation with owned function-index implementation and remove runtime adapter path. | Closed |
| `src/gabion/analysis/dataflow_deadline_helpers.py` (runtime delegation adapter) | Maintainer (unassigned) | Temporary boundary adapter centralizes deadline helper delegation while obligations binder is removed and helper ownership extraction proceeds. | `CU-RT-FINAL-1` | `2026-03-31` | Replace runtime-delegating helper calls with owned implementations and remove runtime adapter path. | Closed |
| `src/gabion/analysis/dataflow_reporting_runtime_bridge.py` (runtime helper bridge) | Maintainer (unassigned) | Temporary boundary adapter previously centralized report-helper runtime delegation while reporting helper ownership extraction proceeded. | `CU-RT-01` | `2026-03-31` | Replace runtime bridge wrappers with owned report-helper implementations and remove runtime delegation path. | Closed |
| `src/gabion/analysis/dataflow_run_output_runtime_bridge.py` (runtime helper bridge) | Maintainer (unassigned) | Temporary boundary adapter previously centralized run-output runtime delegation while output/synthesis/refactor ownership extraction proceeded. | `CU-RT-01` | `2026-03-31` | Replace runtime bridge wrappers with owned helper implementations and remove runtime delegation path. | Closed |
| `src/gabion/analysis/dataflow_pipeline_runtime_bridge.py` (runtime helper bridge) | Maintainer (unassigned) | Temporary boundary adapter centralized pipeline runtime delegation after binder removal while extraction progressed. | `CU-RT-PIPE-01` | `2026-03-31` | Delete bridge module and retarget imports to extracted owner surfaces. | Closed |
| `src/gabion/analysis/dataflow_analysis_index.py` + `src/gabion/analysis/dataflow_indexed_file_scan.py` (runtime ownership adapter) | Maintainer (unassigned) | Temporary boundary adapters replace dynamic pipeline bridge loading with static runtime delegation while remaining owner migration is completed by correction unit. | `CU-RT-FINAL-1` | `2026-03-31` | Move delegated implementations into owner modules and remove direct runtime imports from these extracted modules. | Closed |
| `src/gabion/analysis/dataflow_fingerprint_helpers.py` (runtime ownership adapter) | Maintainer (unassigned) | Temporary boundary adapter previously delegated fingerprint surfaces to runtime during early pipeline extraction. | `CU-RT-02` | `2026-03-31` | Replace delegated implementations with owned fingerprint helpers and remove all runtime imports from module. | Closed |
| `src/gabion/analysis/dataflow_lint_helpers.py` (`_compute_lint_lines` delegation) | Maintainer (unassigned) | Temporary boundary adapter previously kept forest-projection lint rendering delegated while bundle-evidence/lint-row projection ownership extraction was pending. | `CU-RT-02` | `2026-03-31` | Move `_compute_lint_lines` off runtime by extracting lint-row projection and bundle-evidence helpers into owner modules; remove final runtime import from lint helpers. | Closed |
| `src/gabion/analysis/dataflow_synthesis.py` + `src/gabion/analysis/dataflow_refactor_planning.py` + `src/gabion/analysis/dataflow_structure_reuse.py` (runtime delegation) | Maintainer (unassigned) | Temporary boundary adapter preserved synthesis/refactor/reuse public surfaces after pipeline bridge removal while ownership extraction proceeded. | `CU-RT-RUNOUT-02` | `2026-03-31` | Replace runtime delegations with owned implementations and remove transitive runtime dependency. | Closed |
| `src/gabion/analysis/dataflow_synthesis_runtime_bridge.py` (synthesis helper bridge) | Maintainer (unassigned) | Temporary boundary adapter centralizes unresolved synthesis-helper runtime calls while transitive helper ownership extraction is completed. | `CU-RT-FINAL-1` | `2026-03-31` | Move bridged synthesis helper implementations into extracted owner modules and remove bridge runtime delegation. | Closed |
| `src/gabion/analysis/dataflow_projection_preview_bridge.py` (report projection preview bridge) | Maintainer (unassigned) | Temporary boundary adapter centralizes report projection preview builders while preview helper ownership extraction is completed. | `CU-RT-FINAL-1` | `2026-03-31` | Move preview section builders into extracted owner modules and remove bridge runtime delegation. | Closed |
| `src/gabion/analysis/dataflow_snapshot_io.py` + `src/gabion/analysis/dataflow_graph_rendering.py` (runtime projection bridge) | Maintainer (unassigned) | Temporary boundary adapter exposed snapshot/graph surfaces outside run-output module while implementation ownership migrated off runtime. | `CU-RT-01` | `2026-03-31` | Replace runtime bridge wrappers with owned snapshot/graph implementations and remove runtime delegation path. | Closed |
| `src/gabion/analysis/dataflow_raw_runtime.py` (raw entry bridge) | Maintainer (unassigned) | Temporary boundary adapter previously exposed raw entrypoints while parser/runtime ownership remained in `legacy_dataflow_monolith.py`; row retained as historical lifecycle evidence. | `CU-RT-05` | `2026-03-31` | Move `_build_parser`, `_run_impl`, `run`, `main` into owned raw-runtime module implementations and remove runtime delegation. | Closed |
| `tests/runtime_surface.py` (owner-first test surface adapter with runtime fallback) | Maintainer (unassigned) | Temporary boundary adapter drains direct `tests` imports of `gabion.analysis.legacy_dataflow_monolith` while owner extraction still has signature/shape drift for a subset of symbols. | `CU-RT-FINAL-2` | `2026-03-31` | Remove runtime fallback path from `tests/runtime_surface.py` after owner modules are signature-commutative; retarget tests to owner modules directly and delete adapter. | Closed |
