---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: dataflow_runtime_debt_ledger
doc_role: audit
doc_scope:
  - repo
  - analysis
  - server
  - tests
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - AGENTS.md#agent_obligations
  - CONTRIBUTING.md#contributing_contract
  - glossary.md#contract
  - docs/audits/dataflow_runtime_retirement_ledger.md#dataflow_runtime_retirement_ledger
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  AGENTS.md#agent_obligations: 2
  CONTRIBUTING.md#contributing_contract: 2
  glossary.md#contract: 1
  docs/audits/dataflow_runtime_retirement_ledger.md#dataflow_runtime_retirement_ledger: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Fix-forward only ledger; no rollback-first or baseline-write remediation entries."
  AGENTS.md#agent_obligations: "Ledger is correction-unit aligned and evidence-linked; no bypass of required validation stack."
  CONTRIBUTING.md#contributing_contract: "Rows map to one blocking surface per correction unit and dual-sensor cadence."
  glossary.md#contract: "Surface/bundle/protocol naming keeps contract semantics explicit."
  docs/audits/dataflow_runtime_retirement_ledger.md#dataflow_runtime_retirement_ledger: "Seed rows align with retirement capability ownership split."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="dataflow_runtime_debt_ledger"></a>
# Dataflow Runtime Debt Ledger

This ledger tracks fix-forward correction opportunities; it is a backlog surface,
not a rollback trigger. Update rows per correction unit.

## Status Key

- `open`: not started.
- `in_progress`: active correction unit.
- `mitigated`: correction merged to `stage`, pending full convergence.
- `closed`: converged and validated at 100% line+branch with policy stack.

## Active Debt Rows

| debt_id | surface | signal_source | blocking? | target_cu | status | evidence_links | owner | expiry | fix_forward_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DFD-001` | invariant emitter type identity divergence (`InvariantProposition`) | `tests/test_invariant_emitter_hooks.py::test_invariant_emitters_are_applied` | yes | `CU-01` | mitigated | `src/gabion/analysis/dataflow_contracts.py`; `src/gabion/analysis/dataflow_indexed_file_scan.py`; `tests/test_invariant_emitter_hooks.py` | codex | 2026-03-09 | keep canonical type ownership in `dataflow_contracts.py`; prohibit duplicate owner classes in indexed monolith boundary |
| `DFD-002` | stale synthesis render signature (`check_deadline` kwarg missing) | `tests/test_cli_commands.py::test_cli_synth_and_synthesis_plan` | yes | `CU-01` | mitigated | `src/gabion/server_core/command_orchestrator.py:2717`; `tests/test_cli_commands.py` | codex | 2026-03-09 | enforce explicit deadline threading at all `render_synthesis_section(...)` callsites |
| `DFD-003` | coverage deficit in boundary wrappers delegating to indexed monolith | strict coverage gate (`--cov-fail-under=100`) | yes | `CU-02` | open | `/tmp/gabion_full_cov_postfix.json`; `src/gabion/analysis/dataflow_exception_obligations.py`; `src/gabion/analysis/dataflow_decision_surfaces.py`; `src/gabion/analysis/dataflow_pipeline.py` | codex | 2026-03-09 | collapse delegating wrappers into owned helpers and remove bridge indirection |
| `DFD-004` | indexed monolith ownership bloat (`dataflow_indexed_file_scan.py`) | strict coverage gate (`82%` in module) | yes | `CU-03` | open | `/tmp/gabion_full_cov_postfix.json`; `src/gabion/analysis/dataflow_indexed_file_scan.py` | codex | 2026-03-09 | deflate indexed module to ingress/runtime glue; relocate pure helpers to extracted owners |
| `DFD-005` | hotspot family `dataflow_evidence_helpers` under-covered | strict coverage gate (`64%`) | yes | `CU-04A` | open | `/tmp/gabion_full_cov_postfix.json`; `src/gabion/analysis/dataflow_evidence_helpers.py` | codex | 2026-03-10 | reify branch families into deterministic helper protocols; add high-signal branch tests |
| `DFD-006` | hotspot family `dataflow_reporting_helpers` under-covered | strict coverage gate (`85%`) | yes | `CU-04B` | open | `/tmp/gabion_full_cov_postfix.json`; `src/gabion/analysis/dataflow_reporting_helpers.py` | codex | 2026-03-10 | normalize decision forks and enforce impossible-by-construction sinks with `never(...)` |
| `DFD-007` | hotspot family `pattern_schema_projection` under-covered | strict coverage gate (`80%`) | yes | `CU-04C` | open | `/tmp/gabion_full_cov_postfix.json`; `src/gabion/analysis/pattern_schema_projection.py` | codex | 2026-03-10 | add tiered execution-pattern tests and tighten parse/error protocol coverage |
| `DFD-008` | hotspot family `dataflow_lint_helpers` under-covered | strict coverage gate (`76%`) | yes | `CU-04D` | open | `/tmp/gabion_full_cov_postfix.json`; `src/gabion/analysis/dataflow_lint_helpers.py` | codex | 2026-03-10 | convert sentinel-like paths to explicit outcomes and test all strictness branches |
| `DFD-009` | baseline conformance drift beyond path-remap (`test_obsolescence`) | `git diff baselines/test_obsolescence_baseline.json` | yes | `CU-05` | open | `baselines/test_obsolescence_baseline.json`; `baselines/branchless_policy_baseline.json`; `baselines/defensive_fallback_policy_baseline.json` | codex | 2026-03-09 | constrain baseline delta to path-field remaps only; reject semantic baseline churn |
| `DFD-010` | path migration/evidence carrier drift (`dataflow_audit.py` to indexed file path) | `git diff --exit-code out/test_evidence.json` | no | `CU-05` | open | `out/test_evidence.json`; `scripts/extract_test_evidence.py` | codex | 2026-03-12 | preserve path remap determinism and eliminate non-remap carrier churn |
| `DFD-011` | ASPF opportunity `opp:reusable-boundary:1fbd708e36e9` | `artifacts/out/aspf_opportunities.json` | no | `CU-06` | open | `artifacts/out/aspf_opportunities.json`; failed obligations: `representative_collision,two_cell_isomorphy_witness` | codex | 2026-03-16 | introduce explicit 2-cell witness pairing for reusable boundary representatives |
| `DFD-012` | ASPF opportunity `opp:reusable-boundary:471137a20ccd` | `artifacts/out/aspf_opportunities.json` | no | `CU-06` | open | `artifacts/out/aspf_opportunities.json`; failed obligations: `representative_collision,two_cell_isomorphy_witness` | codex | 2026-03-16 | same as `DFD-011`; enforce reusable boundary witness emission parity |
| `DFD-013` | ASPF opportunity `opp:reusable-boundary:4f53cda18c2b` | `artifacts/out/aspf_opportunities.json` | no | `CU-06` | open | `artifacts/out/aspf_opportunities.json`; failed obligations: `representative_collision,two_cell_isomorphy_witness` | codex | 2026-03-16 | same as `DFD-011`; close open two-cell isomorphy obligations |
| `DFD-014` | ASPF opportunity `opp:reusable-boundary:5e427cc51979` | `artifacts/out/aspf_opportunities.json` | no | `CU-06` | open | `artifacts/out/aspf_opportunities.json`; failed obligations: `representative_collision,two_cell_isomorphy_witness` | codex | 2026-03-16 | same as `DFD-011`; add representative collision peer linking |
| `DFD-015` | ASPF opportunity `opp:reusable-boundary:eca2fd70c5f5` | `artifacts/out/aspf_opportunities.json` | no | `CU-06` | open | `artifacts/out/aspf_opportunities.json`; failed obligations: `representative_collision,two_cell_isomorphy_witness` | codex | 2026-03-16 | same as `DFD-011`; enforce witnessed isomorphy carrier coverage |
