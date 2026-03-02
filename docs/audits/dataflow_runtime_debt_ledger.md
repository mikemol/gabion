---
doc_revision: 3
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
| `DFD-001` | invariant emitter type identity divergence (`InvariantProposition`) | `tests/test_invariant_emitter_hooks.py::test_invariant_emitters_are_applied` | yes | `CU-01` | mitigated | `src/gabion/analysis/dataflow_contracts.py`; `src/gabion/analysis/dataflow_indexed_file_scan.py`; `tests/test_invariant_emitter_hooks.py` | codex | 2026-03-09 | keep canonical type ownership in `dataflow_contracts.py`; prohibit duplicate owner classes in indexed runtime boundary |
| `DFD-002` | stale synthesis render signature (`check_deadline` kwarg missing) | `tests/test_cli_commands.py::test_cli_synth_and_synthesis_plan` | yes | `CU-01` | mitigated | `src/gabion/server_core/command_orchestrator.py:2717`; `tests/test_cli_commands.py` | codex | 2026-03-09 | enforce explicit deadline threading at all `render_synthesis_section(...)` callsites |
| `DFD-003` | coverage deficit in boundary wrappers delegating to indexed runtime boundary | strict coverage gate (`--cov-fail-under=100`) | yes | `CU-02` | open | `/tmp/gabion_full_cov_postfix.json`; `src/gabion/analysis/dataflow_exception_obligations.py`; `src/gabion/analysis/dataflow_decision_surfaces.py`; `src/gabion/analysis/dataflow_pipeline.py` | codex | 2026-03-09 | collapse delegating wrappers into owned helpers and remove bridge indirection |
| `DFD-004` | indexed runtime-boundary ownership bloat (`dataflow_indexed_file_scan.py`) | strict coverage gate (`82%` -> `87%` in module) | yes | `CU-03` | in_progress | `/tmp/gabion_full_cov_postfix.json`; `src/gabion/analysis/dataflow_indexed_file_scan.py`; `tests/test_dataflow_resume_payloads.py` | codex | 2026-03-09 | deflate indexed module to ingress/runtime glue; relocate pure helpers to extracted owners and close remaining legacy branch families |
| `DFD-005` | hotspot family `dataflow_evidence_helpers` under-covered | strict coverage gate (`64%` -> `91%` in targeted module run) | yes | `CU-04A` | mitigated | `src/gabion/analysis/dataflow_evidence_helpers.py`; `tests/test_dataflow_evidence_helpers.py`; `mise exec -- python -m pytest -q -n0 tests/test_dataflow_evidence_helpers.py --cov=gabion.analysis.dataflow_evidence_helpers --cov-branch --cov-report=term-missing` | codex | 2026-03-10 | keep extracted-owner coverage growth via branch tests while continuing indexed-surface deflation in later CUs |
| `DFD-006` | hotspot family `dataflow_reporting_helpers` under-covered | strict coverage gate (`85%`) | yes | `CU-04B` | open | `/tmp/gabion_full_cov_postfix.json`; `src/gabion/analysis/dataflow_reporting_helpers.py` | codex | 2026-03-10 | normalize decision forks and enforce impossible-by-construction sinks with `never(...)` |
| `DFD-007` | hotspot family `pattern_schema_projection` under-covered | strict coverage gate (`80%`) | yes | `CU-04C` | open | `/tmp/gabion_full_cov_postfix.json`; `src/gabion/analysis/pattern_schema_projection.py` | codex | 2026-03-10 | add tiered execution-pattern tests and tighten parse/error protocol coverage |
| `DFD-008` | hotspot family `dataflow_lint_helpers` under-covered | strict coverage gate (`76%`) | yes | `CU-04D` | open | `/tmp/gabion_full_cov_postfix.json`; `src/gabion/analysis/dataflow_lint_helpers.py` | codex | 2026-03-10 | convert sentinel-like paths to explicit outcomes and test all strictness branches |
| `DFD-009` | baseline conformance drift beyond path-remap (`test_obsolescence`) | `git diff baselines/test_obsolescence_baseline.json` | yes | `CU-05` | open | `baselines/test_obsolescence_baseline.json`; `baselines/branchless_policy_baseline.json`; `baselines/defensive_fallback_policy_baseline.json` | codex | 2026-03-09 | constrain baseline delta to path-field remaps only; reject semantic baseline churn |
| `DFD-010` | path migration/evidence carrier drift (`dataflow_audit.py` to indexed file path) | `git diff --exit-code out/test_evidence.json`; `gh run 22591374131 audit/Test evidence index` | yes | `CU-02B` | mitigated | `out/test_evidence.json`; `scripts/extract_test_evidence.py`; run `22591374131` | codex | 2026-03-12 | preserve path remap determinism and eliminate non-remap carrier churn |
| `DFD-011` | ASPF opportunity `opp:reusable-boundary:1fbd708e36e9` | `artifacts/out/aspf_opportunities.json` | no | `CU-06` | open | `artifacts/out/aspf_opportunities.json`; failed obligations: `representative_collision,two_cell_isomorphy_witness` | codex | 2026-03-16 | introduce explicit 2-cell witness pairing for reusable boundary representatives |
| `DFD-012` | ASPF opportunity `opp:reusable-boundary:471137a20ccd` | `artifacts/out/aspf_opportunities.json` | no | `CU-06` | open | `artifacts/out/aspf_opportunities.json`; failed obligations: `representative_collision,two_cell_isomorphy_witness` | codex | 2026-03-16 | same as `DFD-011`; enforce reusable boundary witness emission parity |
| `DFD-013` | ASPF opportunity `opp:reusable-boundary:4f53cda18c2b` | `artifacts/out/aspf_opportunities.json` | no | `CU-06` | open | `artifacts/out/aspf_opportunities.json`; failed obligations: `representative_collision,two_cell_isomorphy_witness` | codex | 2026-03-16 | same as `DFD-011`; close open two-cell isomorphy obligations |
| `DFD-014` | ASPF opportunity `opp:reusable-boundary:5e427cc51979` | `artifacts/out/aspf_opportunities.json` | no | `CU-06` | open | `artifacts/out/aspf_opportunities.json`; failed obligations: `representative_collision,two_cell_isomorphy_witness` | codex | 2026-03-16 | same as `DFD-011`; add representative collision peer linking |
| `DFD-015` | ASPF opportunity `opp:reusable-boundary:eca2fd70c5f5` | `artifacts/out/aspf_opportunities.json` | no | `CU-06` | open | `artifacts/out/aspf_opportunities.json`; failed obligations: `representative_collision,two_cell_isomorphy_witness` | codex | 2026-03-16 | same as `DFD-011`; enforce witnessed isomorphy carrier coverage |
| `DFD-016` | remote `audit` workflow gate fails ASPF taint crosswalk acknowledgement | `gh run 22591219450 / audit / Policy check (workflows)` | yes | `CU-02A` | mitigated | `docs/aspf_taint_isomorphism_map.yaml`; `docs/aspf_taint_isomorphism_no_change.yaml`; `scripts/policy_check.py --workflows` | codex | 2026-03-09 | commit taint-isomorphism map/no-change acknowledgment updates required by policy check trigger set |
| `DFD-017` | remote `audit` workflow gate fails policy scanner suite (`branchless`) after evidence gate clears | `gh run 22591569227 / audit / Policy scanner suite`; `scripts/policy_scanner_suite.py --root . --out artifacts/out/policy_suite_results.json` | yes | `CU-02C` | mitigated | `src/gabion/tooling/policy_scanner_suite.py`; `tests/test_policy_scanner_suite.py`; `docs/aspf_taint_isomorphism_no_change.yaml` | codex | 2026-03-10 | apply baseline filtering for branchless/defensive scanners in suite runner and annotate extracted analysis owners with decision-protocol module markers (policy-only, no semantic drift) |
| `DFD-018` | indexed resume-state loaders/deserializers under-covered (`_deserialize_*`, `_load_*_resume_*`) | strict coverage hotspot scan over indexed file miss table | yes | `CU-03A` | mitigated | `tests/test_dataflow_resume_payloads.py`; `src/gabion/analysis/dataflow_indexed_file_scan.py`; full run miss table (`96.20%` total) | codex | 2026-03-11 | add high-signal round-trip and malformed-payload tests for resume loaders while continuing structural owner deflation in subsequent CU-03 units |
| `DFD-019` | accidental legacy runtime monolith reintroduction risk (module/import path) | architecture hard-cut constraint + policy scanner gap discovery | yes | `CU-03B` | mitigated | `src/gabion/tooling/policy_scanner_suite.py`; `tests/test_policy_scanner_suite.py`; `scripts/policy_scanner_suite.py` | codex | 2026-03-12 | enforce `no_legacy_monolith_import` scanner rule that fails on retired module file presence and legacy module imports |
| `DFD-020` | indexed helper-branch hotspots under-covered (`_stage_cache_key_aliases`, `_keyword_links_literal`, `_annotation_exception_candidates`, `_type_from_const_repr`, `_split_top_level`, `_collect_module_exports`) | indexed miss table + targeted helper-branch family planning | yes | `CU-03C` | mitigated | `tests/test_dataflow_indexed_helper_branches.py`; `src/gabion/analysis/dataflow_indexed_file_scan.py`; `mise exec -- python -m pytest -q tests/test_dataflow_indexed_helper_branches.py` | codex | 2026-03-12 | maintain owner-first helper branch battery and continue remaining indexed boundary deflation / branch closure until full strict gate reaches 100% |
