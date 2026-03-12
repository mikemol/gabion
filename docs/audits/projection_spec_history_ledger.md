---
doc_revision: 1
doc_id: projection_spec_history_ledger
doc_role: audit
---

# ProjectionSpec Chronology Ledger

## Summary
- file_count: 60
- hit_count: 373
- term_hit_count: 903
- era_count: 6

## Top Hotspots

| rank | path | term_hit_count | category | area_tags |
| ---: | --- | ---: | --- | --- |
| 1 | src/gabion/analysis/dataflow/io/dataflow_projection_helpers.py | 107 | src | dataflow_projection |
| 2 | tests/gabion/analysis/projection/test_projection_spec.py | 103 | tests | projection_core |
| 3 | src/gabion/analysis/projection/projection_registry.py | 101 | src | projection_core |
| 4 | tests/gabion/analysis/dataflow_s1/test_dataflow_projection_helpers.py | 82 | tests | dataflow_projection |
| 5 | src/gabion/analysis/dataflow/engine/dataflow_projection_materialization.py | 33 | src | dataflow_projection |
| 6 | docs/ws5_decomposition_ledger.md | 30 | docs | general |
| 7 | tests/gabion/analysis/projection/test_projection_exec_edges.py | 28 | tests | projection_core |
| 8 | src/gabion_governance/governance_audit_impl.py | 24 | src | general |
| 9 | tests/gabion/analysis/projection/test_suite_order_projection_spec.py | 24 | tests | projection_core |
| 10 | docs/baselines/private_symbol_import_baseline.json | 21 | docs | general |

## PS-ERA-01: ProjectionSpec Core Calculus
- status: `in_progress`
- date_window: `2026-02-09` -> `2026-03-12`

### Intent At The Time
- ProjectionSpec remains unchanged — only the base carrier improves. (in/in-30.md:488)
- 4. **Projection Idempotence** (in/in-30.md:717)
- - **F1:** ProjectionSpec is a quotient morphism that erases evidence: (in/in-31.md:174)

### What Shipped
- `src/gabion/analysis/projection/projection_spec.py`
- `src/gabion/analysis/projection/projection_normalize.py`
- `src/gabion/analysis/projection/projection_exec.py`
- `src/gabion/analysis/projection/projection_registry.py`

### Evidence Commits
- `29906590` `2026-02-09` docflow: align glossary rev29 + include in/ by default
- `79157819` `2026-02-09` policy: mechanized governance default + dispatch guard (GH-83)
- `d77dc91d` `2026-02-09` docflow: enforce SPPF axis tags + GH ref gate (GH-82)
- `1a86872a` `2026-02-10` analysis: projection spec carrier + proof mode
- `358a4207` `2026-02-10` docs: in-30 SuiteSite adoption tracking
- `aece7d9f` `2026-03-12` Promote reflect through typed projection lowering
- `bc006434` `2026-03-12` Extract projection exec ingress adapter
- `c681562f` `2026-03-12` Promote reindex through typed lowering
- `fa174d91` `2026-03-12` Split ProjectionSpec ingress from executor
- `fd9c9782` `2026-03-12` Promote witness synthesis through typed lowering

### What Drifted
- `[PROVISIONAL]` `src/gabion/analysis/projection/projection_registry.py` ( M)

### What Remains
- No explicit completion gap recorded for this era.

### Next Actions
- Preserve deterministic normalization/hash behavior while convergence layers evolve.

## PS-ERA-02: Quotient And Internment Formalization
- status: `implemented`
- date_window: `2026-02-09` -> `2026-03-12`

### Intent At The Time
- ProjectionSpec remains unchanged — only the base carrier improves. (in/in-30.md:488)
- 4. **Projection Idempotence** (in/in-30.md:717)
- - **F1:** ProjectionSpec is a quotient morphism that erases evidence: (in/in-31.md:174)

### What Shipped
- `src/gabion/analysis/dataflow/engine/dataflow_projection_materialization.py`
- `src/gabion/analysis/dataflow/io/dataflow_projection_helpers.py`

### Evidence Commits
- `29906590` `2026-02-09` docflow: align glossary rev29 + include in/ by default
- `79157819` `2026-02-09` policy: mechanized governance default + dispatch guard (GH-83)
- `d77dc91d` `2026-02-09` docflow: enforce SPPF axis tags + GH ref gate (GH-82)
- `1a86872a` `2026-02-10` analysis: projection spec carrier + proof mode
- `358a4207` `2026-02-10` docs: in-30 SuiteSite adoption tracking
- `6f150437` `2026-03-07` CU-PM0-B3 global marker purge batch 3
- `5cf21c86` `2026-03-08` Eliminate *_or_none/_or_empty helper family and streamify timeout progress
- `2de45e25` `2026-03-09` snapshot: stage entire worktree
- `fbfc5801` `2026-03-10` Docs: reconcile ProjectionSpec CF state and SPPF tracking
- `fa174d91` `2026-03-12` Split ProjectionSpec ingress from executor

### What Drifted
- No provisional workspace-only delta for this era.

### What Remains
- No explicit completion gap recorded for this era.

### Next Actions
- Keep quotient/internment invariants tied to canonical projection owner paths.

## PS-ERA-03: WS5 Dataflow Projection Ownerization
- status: `implemented`
- date_window: `2026-03-02` -> `2026-03-12`

### Intent At The Time
- | `DFD-037` | duplicate reporting projection materialization/span rendering helpers still present in indexed boundary module | strict coverage checkpoint (`97.50%`) + reporting hotspot family ownership hard-cut follow-up | yes | `CU-06C` | mitigated | `src/gabion/analysis/dataflow_indexed_file_scan.py`; `src/gabion/analysis/dataflow_reporting_helpers.py`; `tests/gabion/analysis/test_dataflow_report_helpers.py`; `tests/gabion/analysis/test_report_markdown_module.py` | codex | 2026-03-22 | keep monolith as thin boundary wrapper for `_materialize_projection_spec_rows` and `_format_span_fields`, with owner logic centralized in `dataflow_reporting_helpers` | (docs/audits/dataflow_runtime_debt_ledger.md:90)
- | `DFD-038` | redundant pass-through wrapper branch surfaces in `dataflow_ambiguity_helpers` boundary module | strict coverage checkpoint (`97.50%`) + ambiguity helper simplification pass | yes | `CU-06D` | mitigated | `src/gabion/analysis/dataflow_ambiguity_helpers.py`; `tests/gabion/analysis/test_ambiguity_helpers.py`; `tests/gabion/analysis/test_suite_order_projection_spec.py` | codex | 2026-03-22 | replace pass-through wrapper functions with direct alias exports for stable helper API names while preserving ambiguity evidence identity and reducing redundant boundary branch surfaces | (docs/audits/dataflow_runtime_debt_ledger.md:91)
- - `dataflow_projection_materialization` (`_materialize_projection_spec_rows`, `_spec_row_span`) (docs/ws5_decomposition_ledger.md:1183)
- - `dataflow_projection_helpers._topologically_order_report_projection_specs` (docs/ws5_decomposition_ledger.md:1212)
- - `_decode_projection_span`, `_spec_row_span`, `_materialize_projection_spec_rows` (docs/ws5_decomposition_ledger.md:1585)
- - `_topologically_order_report_projection_specs` (docs/ws5_decomposition_ledger.md:1635)

### What Shipped
- `src/gabion/analysis/dataflow/engine/dataflow_projection_materialization.py`
- `src/gabion/analysis/dataflow/io/dataflow_projection_helpers.py`
- `src/gabion/analysis/dataflow/io/dataflow_reporting_helpers.py`

### Evidence Commits
- `0120290e` `2026-03-02` test(dataflow): add indexed helper branch coverage battery (CU-03C)
- `02096aa0` `2026-03-02` test(dataflow): raise evidence-helper branch coverage (CU-04A)
- `03effc7d` `2026-03-02` test(policy): close scanner-suite branch arcs (CU-04E)
- `09d170b5` `2026-03-02` analysis(ambiguity): collapse pass-through wrappers to alias exports (CU-06D)
- `11c88688` `2026-03-02` analysis(indexed-scan): extract resume-state owner and expand owner tests (CU-04)
- `d58caa76` `2026-03-08` CU-D2T batch strict-dispatch payload lift
- `2de45e25` `2026-03-09` snapshot: stage entire worktree
- `e13c54b0` `2026-03-10` Force majeure: continue projection substrate and policy convergence
- `203a3273` `2026-03-11` analysis: tighten normalization decision helpers
- `fa174d91` `2026-03-12` Split ProjectionSpec ingress from executor

### What Drifted
- No provisional workspace-only delta for this era.

### What Remains
- No explicit completion gap recorded for this era.

### Next Actions
- Prevent compatibility wrapper drift by keeping projection/reporting helper ownership centralized.

## PS-ERA-04: Policy DSL Convergence
- status: `implemented`
- date_window: `2026-02-21` -> `2026-03-12`

### Intent At The Time
- Governance gates, ambiguity contract checks, policy scanner decision summaries, and ASPF opportunity classification now share the typed policy DSL in `src/gabion/policy_dsl/` with shared declarations in `docs/policy_rules.yaml` and `docs/aspf_opportunity_rules.yaml`. Scripts remain orchestration wrappers. (docs/enforceable_rules_cheat_sheet.md:220)
- Control-loop gate semantics are authored as DSL sources (`docs/governance_rules.yaml`, `docs/policy_rules.yaml`, `docs/aspf_opportunity_rules.yaml`) and normalized by `src/gabion/policy_dsl/compile.py`, validated in `src/gabion/policy_dsl/typecheck.py`, and executed by `src/gabion/policy_dsl/eval.py`. (docs/governance_control_loops.md:232)
- - removal_condition: `docs/governance_rules.yaml` is rewritten to native DSL schema. (docs/policy_dsl_migration_notes.md:12)
- Temporary boundary adapters kept: (docs/policy_dsl_migration_notes.md:9)

### What Shipped
- `src/gabion/policy_dsl/schema.py`
- `src/gabion/policy_dsl/compile.py`
- `src/gabion/policy_dsl/typecheck.py`
- `src/gabion/policy_dsl/eval.py`
- `src/gabion/policy_dsl/registry.py`
- `src/gabion/analysis/aspf_rule_engine.py`
- `scripts/policy/policy_check.py`

### Evidence Commits
- `4cbde8da` `2026-02-21` docs: add enforceable rules cheat sheet cross-links
- `48f037aa` `2026-02-22` Implement construction-first hard-zero refactor program
- `1e7f3b5e` `2026-02-23` Hard-cut governance audit scripts into gabion CLI subcommands
- `650c67db` `2026-02-23` Add governance control-loop registry and docflow enforcement
- `76c103ae` `2026-02-23` Canonicalize governance loop doc and fix agent drift checks
- `0c516ec4` `2026-03-12` Ratchet suite wrapper prerequisites
- `31f20c3a` `2026-03-12` Move projection policy to semantic-row judgment
- `6ba58c52` `2026-03-12` Keep suite semantics boundary-owned
- `a800876b` `2026-03-12` Make suite runtime surface scan-only
- `dde0d399` `2026-03-12` Retire policy suite result carrier

### What Drifted
- No provisional workspace-only delta for this era.

### What Remains
- No explicit completion gap recorded for this era.

### Next Actions
- Preserve rule_id/witness drift checks so policy decisions stay DSL-owned.

## PS-ERA-05: Fiber-First Lattice Cutover
- status: `implemented`
- date_window: `2026-02-25` -> `2026-03-12`

### Intent At The Time
- ## Policy DSL ownership (docs/aspf_execution_fibration.md:154)
- ## Lattice algebra ownership (docs/aspf_execution_fibration.md:158)

### What Shipped
- `src/gabion/analysis/aspf/aspf_lattice_algebra.py`
- `src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py`
- `src/gabion/tooling/policy_rules/branchless_rule.py`
- `src/gabion/tooling/runtime/policy_scanner_suite.py`

### Evidence Commits
- `4e135831` `2026-02-25` package remaining ASPF cutover and policy surface changes
- `a5cc7be3` `2026-02-27` Add registry-based ASPF opportunity taxonomy
- `d03ba757` `2026-02-27` docs: remove stale ASPF action-plan CLI and artifact surfaces
- `e3f92c4b` `2026-02-27` docs: propagate re-internment metadata across governance dependents
- `776ebde9` `2026-03-03` refactor: split import DAG into submodules and rewrite paths
- `a843a313` `2026-03-12` Drop suite runtime file override
- `aece7d9f` `2026-03-12` Promote reflect through typed projection lowering
- `b0b6735e` `2026-03-12` Require explicit policy suite child inputs
- `dde0d399` `2026-03-12` Retire policy suite result carrier
- `f158353e` `2026-03-12` Move suite changed-path discovery to wrapper

### What Drifted
- No provisional workspace-only delta for this era.

### What Remains
- No explicit completion gap recorded for this era.

### Next Actions
- Keep iterator-first convergence and single-frontier drift tests as required gates.

## PS-ERA-06: Integrated Substrate Completion
- status: `in_progress`
- date_window: `2026-02-12` -> `2026-03-12`

### Intent At The Time
- - removal_condition: `docs/governance_rules.yaml` is rewritten to native DSL schema. (docs/policy_dsl_migration_notes.md:12)
- Temporary boundary adapters kept: (docs/policy_dsl_migration_notes.md:9)

### What Shipped
- `scripts/policy/policy_check.py`
- `src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py`
- `src/gabion/tooling/policy_substrate/dataflow_fibration.py`
- `src/gabion/analysis/aspf/aspf_lattice_algebra.py`

### Evidence Commits
- `58f2eaf1` `2026-02-12` docflow: update in-32 and section-review outputs
- `dd4cb1fd` `2026-02-12` docs: intern in-32 as non-normative hypothesis
- `6b38b878` `2026-02-17` docs(in-32): emphasize non-normative status and implementation boundary
- `d595634e` `2026-02-23` Refresh in/ doc dependency reviews and cadence note
- `dde38f18` `2026-02-23` Merge PR #225: Add governance loop matrix doc and matrix-drift audit check
- `31f20c3a` `2026-03-12` Move projection policy to semantic-row judgment
- `6ba58c52` `2026-03-12` Keep suite semantics boundary-owned
- `a800876b` `2026-03-12` Make suite runtime surface scan-only
- `aece7d9f` `2026-03-12` Promote reflect through typed projection lowering
- `dde0d399` `2026-03-12` Retire policy suite result carrier

### What Drifted
- No provisional workspace-only delta for this era.

### What Remains
- Workflow policy gate stack still reports unresolved workflow/lock-in failures.

### Next Actions
- Close workflow-policy and lock-in source gate failures in a dedicated correction unit.
- Keep strict docflow packet loop green while CF04-CF11 substrate state remains stable.

## Completion Focus Appendix

### Substrate Convergence Criteria

| criterion_id | description | status | evidence |
| --- | --- | --- | --- |
| CF-01 | ProjectionSpec inventory is present and populated | pass | artifacts/out/projection_spec_inventory.json |
| CF-02 | Projection-fiber DSL source is committed and registry-addressable | pass | docs/projection_fiber_rules.yaml |
| CF-03 | Canonical lattice algebra module is committed | pass | src/gabion/analysis/aspf/aspf_lattice_algebra.py |
| CF-04 | Convergence gate is DSL/witness-only semantic evaluation | pass | scripts/policy/policy_check.py |
| CF-05 | Policy DSL migration notes still constrain temporary boundary adapters | pass | docs/policy_dsl_migration_notes.md |
| CF-06 | Policy substrate adapter exports are canonical witness-only | pass | src/gabion/tooling/policy_substrate/__init__.py; src/gabion/tooling/policy_substrate/dataflow_fibration.py |
| CF-07 | Canonical lattice algebra uses a single FrontierWitness contract | pass | src/gabion/analysis/aspf/aspf_lattice_algebra.py |

### Prioritized Closure Sequence
1. Commit projection-fiber rule source and lattice algebra as canonical tracked surfaces.
2. Cut convergence checks to evaluator decisions over semantic lattice witnesses only.
3. Eliminate remaining transitional frontier compatibility branches in policy substrate adapters.
4. Lock deterministic lazy-pull and cache-parity tests as hard convergence gates.
5. Enforce adapter-free substrate exports via drift checks and completion criteria.
6. Enforce single frontier contract in canonical lattice algebra with no recombination bridge symbols.

## Semantic Lowering Appendix

| spec_name | domain | status | semantic | presentation | bridge | quotient_faces |
| --- | --- | --- | ---: | ---: | ---: | --- |
| ambiguity_baseline | ambiguity_baseline | presentation_only | 0 | 0 | 0 |  |
| ambiguity_delta | ambiguity_delta | presentation_only | 0 | 0 | 0 |  |
| ambiguity_state | ambiguity_state | presentation_only | 0 | 0 | 0 |  |
| ambiguity_suite_agg | ambiguity_suite | presentation_only | 0 | 3 | 0 |  |
| ambiguity_virtual_set | ambiguity_suite | presentation_plus_bridge | 0 | 3 | 1 |  |
| ambiguity_summary | ambiguity_witnesses | presentation_only | 0 | 2 | 0 |  |
| call_cluster_summary | call_clusters | presentation_only | 0 | 1 | 0 |  |
| deadline_obligations_summary | deadline_obligations | presentation_only | 0 | 2 | 0 |  |
| lint_findings | lint_findings | presentation_only | 0 | 2 | 0 |  |
| never_invariants_section | never_invariants | presentation_plus_bridge | 0 | 2 | 1 |  |
| projection_fiber_context_wedge | projection_fiber | semantic_promoted | 1 | 0 | 0 |  |
| projection_fiber_existential_image | projection_fiber | semantic_promoted | 1 | 0 | 0 |  |
| projection_fiber_frontier | projection_fiber | mixed | 1 | 1 | 0 | projection_fiber.frontier |
| projection_fiber_reflection | projection_fiber | semantic_promoted | 1 | 0 | 0 |  |
| projection_fiber_reflective_boundary | projection_fiber | mixed | 1 | 1 | 0 | projection_fiber.reflective_boundary |
| projection_fiber_reindex | projection_fiber | semantic_promoted | 1 | 0 | 0 |  |
| projection_fiber_support_reflection | projection_fiber | semantic_promoted | 1 | 0 | 0 |  |
| projection_fiber_witness_synthesis | projection_fiber | semantic_promoted | 1 | 0 | 0 |  |
| quotient_demotion_incidents | quotient_demotion_incidents | presentation_only | 0 | 0 | 0 |  |
| quotient_promotion_decision | quotient_promotion_decision | presentation_only | 0 | 0 | 0 |  |
| quotient_protocol_readiness | quotient_protocol_readiness | presentation_only | 0 | 0 | 0 |  |
| report_section_lines | report_section_lines | presentation_only | 0 | 2 | 0 |  |
| suite_order | suite_order | presentation_only | 0 | 2 | 0 |  |
| taint_baseline | taint_baseline | presentation_only | 0 | 0 | 0 |  |
| taint_delta | taint_delta | presentation_only | 0 | 0 | 0 |  |
| taint_state | taint_state | presentation_only | 0 | 0 | 0 |  |
| test_annotation_drift | test_annotation_drift | presentation_only | 0 | 2 | 0 |  |
| test_annotation_drift_baseline | test_annotation_drift_baseline | presentation_only | 0 | 0 | 0 |  |
| test_annotation_drift_delta | test_annotation_drift_delta | presentation_only | 0 | 0 | 0 |  |
| test_obsolescence_summary | test_obsolescence | presentation_only | 0 | 4 | 0 |  |
| test_obsolescence_baseline | test_obsolescence_baseline | presentation_only | 0 | 0 | 0 |  |
| test_obsolescence_delta | test_obsolescence_delta | presentation_only | 0 | 0 | 0 |  |
| test_obsolescence_state | test_obsolescence_state | presentation_only | 0 | 0 | 0 |  |
| wl_refinement | wl_refinement | presentation_only | 0 | 0 | 0 |  |
