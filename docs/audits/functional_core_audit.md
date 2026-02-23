---
doc_revision: 6
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: functional_core_audit
doc_role: audit
doc_scope:
  - repo
  - server
  - analysis
  - tests
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - AGENTS.md#agent_obligations
  - CONTRIBUTING.md#contributing_contract
  - glossary.md#contract
  - docs/enforceable_rules_cheat_sheet.md#enforceable_rules_cheat_sheet
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 1
  AGENTS.md#agent_obligations: 1
  CONTRIBUTING.md#contributing_contract: 1
  glossary.md#contract: 1
  docs/enforceable_rules_cheat_sheet.md#enforceable_rules_cheat_sheet: 4
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed execution/CI and deadline policy constraints for this audit roadmap."
  AGENTS.md#agent_obligations: "Reviewed LSP-first and boundary-normalization obligations for refactor scope."
  CONTRIBUTING.md#contributing_contract: "Reviewed workflow/test expectations for ratchet-style refactors."
  glossary.md#contract: "Reviewed bundle/tier/protocol semantics to keep functional-core extraction terminology aligned."
  docs/enforceable_rules_cheat_sheet.md#enforceable_rules_cheat_sheet: "Reviewed traceable rule matrix for operational checks referenced by this audit."
doc_sections:
  functional_core_audit: 1
doc_section_requires:
  functional_core_audit:
    - POLICY_SEED.md#policy_seed
    - AGENTS.md#agent_obligations
    - CONTRIBUTING.md#contributing_contract
    - glossary.md#contract
    - docs/enforceable_rules_cheat_sheet.md#enforceable_rules_cheat_sheet
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="functional_core_audit"></a>
# Functional-Core Audit

## Baseline (2026-02-22)

Generated with:

```bash
mise exec -- python scripts/complexity_audit.py --root . --emit artifacts/audit_reports/complexity_baseline.json
```

Snapshot (`artifacts/audit_reports/complexity_baseline.json`):
- `max_function_line_count`: `2337`
- `max_function_branch_count`: `350`
- `top4_test_case_total`: `550`
- private test refs:
  - `server`: `564`
  - `dataflow_audit`: `1582`
  - `cli`: `293`

Top complexity hotspots:
1. `src/gabion/server_core/command_orchestrator.py::execute_command_total` (`2337` lines, `350` branch nodes)
2. `src/gabion/analysis/dataflow_pipeline.py::analyze_paths` (`982` lines, `106` branch nodes)
3. `src/gabion/analysis/dataflow_obligations.py::collect_deadline_obligations` (`834` lines, `172` branch nodes)
4. `src/gabion/analysis/dataflow_reporting.py::emit_report` (`426` lines, `63` branch nodes)

Top concentrated test files:
1. `tests/test_server_execute_command_edges.py` (`179` tests)
2. `tests/test_dataflow_audit_helpers.py` (`147` tests)
3. `tests/test_cli_helpers.py` (`131` tests)
4. `tests/test_dataflow_audit_coverage_gaps.py` (`93` tests)

## Current Ratchet Snapshot (2026-02-23)

Generated with:

```bash
mise exec -- python scripts/complexity_audit.py --root . --emit /tmp/complexity_current.json
```

Snapshot:
- `max_function_line_count`: `564`
- `max_function_branch_count`: `50`
- `top4_test_case_total`: `550`
- top line hotspot: `src/gabion/server_core/command_orchestrator.py::execute_command_total`
- top branch hotspot: `src/gabion/server_core/command_orchestrator.py::_create_progress_emitter`

## Opportunity Register (Ranked)

1. **Server command runtime split**
   - Break `execute_command_total` into reducer/effects/orchestrator layers.
   - Keep thin adapter in `src/gabion/server.py`.
2. **Dataflow audit phase separation**
   - Keep collection, obligations, and report emission as separate cores with typed carriers.
3. **Private API test coupling reduction**
   - Move from private helper assertions to contract/matrix tests over stable public boundary behavior.
4. **CLI/LSP/server boundary unification**
   - Keep one envelope/payload/timeout path and one direct-dispatch registry.
5. **Tooling wrapper collapse**
   - Keep declarative `ToolSpec` wiring and remove pass-through modules.

## Implementation Status (Current Slice)

Completed in this slice:
- added per-slice state orchestration tooling:
  - `scripts/refactor_slice_state.py`
  - `artifacts/audit_reports/refactor_slice_state.json`
- added step-level CI timing capture tooling:
  - `scripts/ci_step_timing_capture.py`
  - `artifacts/audit_reports/ci_step_timings.json`
- wired local CI reproduction step-level KPI collection for selected checks in:
  - `scripts/ci_local_repro.sh`
- added SPPF checklist lane linkage for this roadmap:
  - `docs/sppf_checklist.md` (`Functional-core roadmap lane`)
- decomposed `dataflow_audit` output finalization into functional helper phases:
  - `src/gabion/analysis/dataflow_audit.py::_emit_projection_outputs`
  - `src/gabion/analysis/dataflow_audit.py::_plan_projection_output_effects`
  - `src/gabion/analysis/dataflow_audit.py::_emit_optional_synthesis_outputs`
  - `src/gabion/analysis/dataflow_audit.py::_emit_optional_refactor_outputs`
  - `src/gabion/analysis/dataflow_audit.py::_emit_report_output`
  - `src/gabion/analysis/dataflow_audit.py::_emit_console_output_and_violation_gate`
  - thin orchestration retained in `src/gabion/analysis/dataflow_audit.py::_finalize_run_outputs`
- implemented operation-plan/effect-return decomposition for dataclass call bundles:
  - new functional helper module:
    - `src/gabion/analysis/dataflow_bundle_iteration.py`
  - new pure carrier/effect surfaces:
    - `BundleIterationContext`
    - `BundleIterationOutcome`
    - constructor operation plans and projection outcomes
  - `src/gabion/analysis/dataflow_audit.py::_iter_dataclass_call_bundles` now delegates as a thin boundary adapter:
    - core helper returns witness effects
    - adapter performs side-effect dispatch (`parse_failure_witnesses.extend(...)`) only at the boundary
  - complexity impact from this slice:
    - `max_function_branch_count`: `54 -> 50`
    - top branch hotspot shifted away from dataflow bundle iteration
- completed active dataflow callee-resolution hard-cut (`phase2_dataflow_resolve_callee`):
  - new pure operation/effect module:
    - `src/gabion/analysis/dataflow_callee_resolution.py`
  - boundary adapter parity retained in:
    - `src/gabion/analysis/dataflow_audit.py::_resolve_callee`
    - `src/gabion/analysis/dataflow_audit.py::_resolve_callee_outcome`
  - `_resolve_callee` reduced to thin adapter shape (branch hotspot removed)
- completed output-phase module extraction for dataflow run finalization:
  - new module:
    - `src/gabion/analysis/dataflow_run_outputs.py`
  - new carriers:
    - `DataflowRunOutputContext`
    - `DataflowRunOutputOutcome`
  - operation-sequence dispatcher added:
    - `plan_run_output_ops`
    - `apply_run_output_ops`
  - boundary orchestration remains in `dataflow_audit` via:
    - `src/gabion/analysis/dataflow_audit.py::_run_impl` -> `finalize_run_outputs(...).exit_code`
  - coverage non-regression recovered to lane cap:
    - total misses held at `60`
    - `src/gabion/analysis/dataflow_run_outputs.py` now fully covered
- verification summary for this continuation slice:
  - passed:
    - compileall / order-lifetime / complexity ratchet / policy / docflow
    - targeted dataflow tests (`281 passed`)
    - full suite (`1963 passed`)
    - `scripts/ci_local_repro.sh --dataflow-only --skip-gabion-check-step`
  - expected blockers preserved:
    - strict evidence drift stop (`git diff --exit-code out/test_evidence.json`)
    - coverage threshold stop (`--fail-under=100` at `99%`)

Still open for the next slice:
- dataflow-first hard-cut decomposition:
  - `_run_impl` / `_analyze_file_internal` thin boundary adapters only
- server-core hard-cut decomposition:
  - `execute_command_total` down to boundary orchestration target (`<=250` lines)
  - timeout cleanup / auxiliary outputs / progress runtime split into focused modules
- aggressive test surface collapse toward matrix coverage:
  - top-4 concentration target `<=300`
  - private test refs reduction target `>=40%` for server/dataflow areas

## Ratchet Targets

This roadmap tracks these hard targets:
- no single function above `500` lines in server/dataflow primary modules,
- no command-orchestrator glue above `250` lines in adapter files,
- top-4 test concentration reduced from `550` to `<=300`,
- private test references in server/dataflow areas reduced by at least `40%`.

## Verification Gates

Run after each mutating slice:

```bash
mise exec -- python -m compileall -q src/gabion
mise exec -- python scripts/order_lifetime_check.py --root .
mise exec -- python scripts/complexity_audit.py --root . --fail-on-regression
mise exec -- python scripts/structural_hash_policy_check.py --root .
mise exec -- python scripts/policy_check.py --workflows
mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required
mise exec -- python -m pytest --cov=src/gabion --cov-report=term-missing
mise exec -- python -m coverage report --show-missing --fail-under=100
```

Replay regression check for this cleanup track:

```bash
scripts/ci_local_repro.sh --skip-gabion-check-step
```

## Notes

- This audit document is intentionally informative; canonical enforcement remains in
  `POLICY_SEED.md#policy_seed` and `glossary.md#contract`.
- Complexity baselines are generated artifacts; do not hand-edit them.
- Current cache-identity ratchet: derivation identity paths now require structural
  interning keys (ASPF/forest-backed) with no digest/text key generation.
