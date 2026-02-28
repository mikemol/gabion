---
doc_revision: 6
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: aspf_execution_fibration
doc_role: contract
doc_scope:
  - repo
  - analysis
  - aspf
  - dataflow
doc_authority: informative
doc_requires:
  - README.md#repo_contract
  - docs/user_workflows.md#user_workflows
  - glossary.md#contract
doc_reviewed_as_of:
  README.md#repo_contract: 2
  docs/user_workflows.md#user_workflows: 2
  glossary.md#contract: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md rev2 (removed stale ASPF action-plan CLI/examples; continuation docs now state/delta only)."
  docs/user_workflows.md#user_workflows: "Reviewed user_workflows rev2 (state/delta-only ASPF examples and ci_watch failure-bundle workflow language)."
  glossary.md#contract: "Reviewed semantic contract terms used by witness/drift classification text."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_sections:
  aspf_execution_fibration: 2
doc_section_requires:
  aspf_execution_fibration:
    - README.md#repo_contract
    - docs/user_workflows.md#user_workflows
    - glossary.md#contract
doc_section_reviews:
  aspf_execution_fibration:
    README.md#repo_contract:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Repo contract rev2 reviewed; command and artifact guidance remains aligned."
    docs/user_workflows.md#user_workflows:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "User workflows rev2 reviewed; operational examples remain aligned."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 2
      outcome: no_change
      note: "Witness and drift terms remain consistent with glossary contract."
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="aspf_execution_fibration"></a>
# ASPF Execution Fibration (Phase 1)

Phase-1 execution tracing is ASPF-native:
- In-memory carriers and artifact carriers are modeled as `BasisZeroCell`.
- Serialization/materialization/loading boundaries are modeled as `AspfOneCell`.
- Cross-path semantic equivalence is represented with `AspfTwoCellWitness`.
- Domain-to-ASPF boundary embeddings are tracked as `DomainToAspfCofibration`.

## Semantic Surface Projection

Phase-1 default semantic surfaces:
- `groups_by_path`
- `decision_surfaces`
- `rewrite_plans`
- `synthesis_plan`
- `delta_state`
- `delta_payload`
- `violation_summary`

Each surface is projected to a deterministic representative and emitted as a
1-cell projection event.

## Equivalence Semantics

Equivalence compares current surface representatives against baseline traces:
1. Select a baseline representative (deterministically).
2. Link representatives with an available 2-cell witness when present.
3. Classify each surface as `non_drift` or `drift` via homotopy-compatible logic.

Aggregate verdict is `non_drift` only when all surfaced classifications are
`non_drift`.

## Opportunity Semantics

Phase-1 opportunities are emitted via a registry keyed by explicit algebraic
predicates over one-cell / two-cell / cofibration structure. The default class
set is:
- `materialize_load_observed` (one-cell predicate: at least one `resume_load`)
- `materialize_load_fusion` (one-cell predicate: `resume_load` + `resume_write`
  over the same resume reference)
- `reusable_boundary_artifact` (two-cell predicate: representative fan-out over
  projected semantic surfaces)
- `fungible_execution_path_substitution` (two-cell predicate: `non_drift`
  classification with witness-carrying equivalence rows)
- `cofibration_prime_embedding_reuse` (cofibration predicate: at least one
  validated domain→ASPF basis embedding)

Evidence requirements match the class semantics:
- `none` for ingress-only observations (`materialize_load_observed`,
  `materialize_load_fusion`)
- `representative_pair` for representative-confluence reuse
- `two_cell_witness` for fungible substitution opportunities
- `cofibration_witness` for cofibration-prime embedding opportunities

Adding a new opportunity kind now requires adding a normalized observation shape
plus a taxonomy registration; no ad-hoc branch expansion is required inside
`OpportunityPayloadEmitter`.

## Cross-Script Handoff

Phase-1 cross-script reuse is ASPF-state-native and file-based:
- ASPF serialized state snapshots live in
  `artifacts/out/aspf_state/<session>/<seq>_<step>.snapshot.json`.
- ASPF mutation ledgers live in
  `artifacts/out/aspf_state/<session>/<seq>_<step>.delta.jsonl`.
- handoff sequencing/import chains are tracked in
  `artifacts/out/aspf_handoff_manifest.json`.
- manifest path fields are repo-relative for cross-job portability (absolute
  paths remain readable for backward compatibility).

Default cumulative import rule:
1. for a manifest session, each new entry imports all prior entries with
   `status: success`;
2. failed entries are kept for history but excluded from default import chains.

Repo scripts (`checks.sh`, `ci_local_repro.sh`, `refresh_baselines.py`,
`audit_snapshot.sh`, and `run-dataflow-stage`) apply this handoff by default
and expose `--no-aspf-handoff` escape hatches.

## Artifacts and Controls

Phase-1 artifacts:
- `artifacts/out/aspf_trace.json`
- `artifacts/out/aspf_equivalence.json`
- `artifacts/out/aspf_opportunities.json`
- `artifacts/out/aspf_state/<session>/<seq>_<step>.snapshot.json`
- `artifacts/out/aspf_state/<session>/<seq>_<step>.delta.jsonl`
- `artifacts/out/aspf_handoff_manifest.json`

CLI controls:
- `--aspf-trace-json`
- `--aspf-import-trace`
- `--aspf-equivalence-against`
- `--aspf-opportunities-json`
- `--aspf-state-json`
- `--aspf-import-state`
- `--aspf-delta-jsonl`
- `--aspf-semantic-surface`
