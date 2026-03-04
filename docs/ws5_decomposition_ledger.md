---
doc_revision: 4
doc_id: ws5_decomposition_ledger
doc_role: ledger
doc_scope:
  - ws5
  - dataflow
---

# WS-5 Decomposition Ledger

## Current State
- Date: 2026-03-04
- Monolith file: `src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan.py`
- Monolith LOC (current): 4703
- Monolith top-level import statements (current): 105
- Direct monolith imports in `src/`: 0
- Direct monolith imports in `tests/`: 0

## Debt Ledger
- High: monolith still carries large post-phase analysis ownership (type/constant/unused/config/dataclass/decision surfaces).
- High: monolith still carries index/materialization ownership and cache identity carrier definitions.
- Medium: compatibility owner modules still exist (`dataflow_analysis_index_owner.py`, `dataflow_deadline_runtime_owner.py`, `dataflow_facade.py`) and should collapse after canonical ownership lands.
- Medium: monolith top-level import surface remains above target.

## Progress Ledger
- WS-1 through WS-4 completed previously (server/core, CLI/runtime, governance, CST convergence).
- WS-5 prior hard-cuts:
  - Resume-path helpers extracted (`dataflow_resume_paths.py`).
  - Resume serialization ownership extracted to static owner (`dataflow_resume_serialization.py`) and monolith body cutover.
- WS-5 continuation (this CU):
  - Post-phase module converted from importlib facade to static owner + explicit temporary boundary delegates.
  - Static ownership moved for post-phase helper cluster:
    - `_type_from_const_repr`
    - `_split_top_level`
    - `_expand_type_hint`
    - `_combine_type_hints`
    - `_format_call_site`
    - `_format_type_flow_site`
    - `_callsite_evidence_for_bundle`
  - Static ownership moved for property-hook materialization surface:
    - `generate_property_hook_manifest`
    - `_build_property_hook_callable_index`
  - Monolith bodies removed for moved symbols; monolith now aliases from `dataflow_post_phase_analyses.py`.
  - Compatibility return-contract hardening: monolith `SymbolTable` canonicalized to contract owner alias.
  - Validation:
    - policy checks passed
    - targeted pytest passed (`58 passed` for touched suites)
    - evidence refresh/check passed
- WS-5 continuation (next CU):
  - Post-phase analyzer ownership moved to canonical owner module:
    - `analyze_type_flow_repo_with_map`
    - `analyze_type_flow_repo_with_evidence`
    - `analyze_constant_flow_repo`
    - `analyze_deadness_flow_repo`
    - `analyze_unused_arg_flow_repo`
  - Monolith analyzer bodies removed and replaced by owner aliases.
  - ASPF no-change acknowledgement refreshed for this cut.
  - Validation:
    - policy checks passed
    - targeted pytest passed (`66 passed` for touched suites)
    - evidence refresh/check passed
- WS-5 continuation (this CU):
  - Post-phase owner expansion for remaining helper cluster:
    - `_collect_constant_flow_details`
    - `_compute_knob_param_names`
    - `_collect_config_bundles`
    - `_iter_config_fields`
    - `_collect_dataclass_registry`
    - `_iter_dataclass_call_bundles`
  - Monolith bodies removed and replaced with post-phase owner aliases.
  - Boundary compatibility retained by exporting `_ConstantFlowFoldAccumulator` from post-phase and aliasing in monolith for constant-flow runtime deps.
  - ASPF no-change acknowledgement refreshed (`in-65`).
  - Validation:
    - policy checks passed
    - targeted pytest passed (`34 passed` for touched suites)
    - evidence refresh/check passed

## Next Cuts (Queued)
1. Post-phase ownership expansion: decision-surface and invariant/obligation helper owners (remove temporary delegates).
2. Analysis-index ownership expansion: cache identity carriers + indexed pass/build surfaces.
3. Projection/ambiguity ownership expansion and remaining facade contraction.

## Validation Checklist Per CU
- `scripts/policy/policy_check.py --workflows`
- `scripts/policy/policy_check.py --ambiguity-contract`
- Targeted pytest suites for touched surfaces
- `scripts/misc/extract_test_evidence.py --root . --tests tests --out out/test_evidence.json`
- `git diff --exit-code out/test_evidence.json`
