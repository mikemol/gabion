---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: planning_chart_architecture
doc_role: architecture
doc_scope:
  - repo
  - planning
  - tooling
doc_authority: informative
doc_requires:
  - docs/governance_control_loops.md#governance_control_loops
  - docs/ttl_kernel_semantics.md#ttl_kernel_semantics
  - glossary.md#contract
doc_reviewed_as_of:
  docs/governance_control_loops.md#governance_control_loops: 6
  docs/ttl_kernel_semantics.md#ttl_kernel_semantics: 1
  glossary.md#contract: 1
doc_review_notes:
  docs/governance_control_loops.md#governance_control_loops: "Reviewed governance control-loop registry so planning phases align with existing scanner/judge/witness loops."
  docs/ttl_kernel_semantics.md#ttl_kernel_semantics: "Reviewed TTL-kernel explainer so the planning chart can treat kernel alignment residues as scan-phase inputs."
  glossary.md#contract: "Reviewed glossary contract so phase naming remains carrier-first rather than presentation-first."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

# Planning chart architecture

The planning substrate now carries a domain-neutral planning chart with three explicit phases:

- `scan`
- `predict`
- `complete`

This is intentionally Earley-like in structure without persisting `earley_*` names into repo artifacts.

## Why the model is Earley-like

The planner already behaved like a chart parser over repo state:

- scanner-like work ingests live surfaces and control-loop evidence,
- predictor-like work expands declared dependencies and counterfactual next moves,
- completer-like work selects recommended cuts and followup lanes.

The new chart layer makes that shape explicit so the graph can expose the planning protocol structurally instead of only through final recommendations.

## Why persisted names stay domain-neutral

The artifact schema persists:

- `planning_chart_report`
- `planning_phase`
- `planning_chart_item`

and phase values:

- `scan`
- `predict`
- `complete`

This keeps the stored carrier readable to non-parser consumers while preserving the structural isomorphism in documentation and implementation.

## Current phase mapping

### Scan

The scan phase currently materializes planner-significant live inputs, including:

- declared touchsites that represent active planning surfaces,
- control-loop evidence that changes planner state,
- kernel-VM alignment report, binding, and residue nodes.

### Predict

The predict phase currently materializes declared future-expansion surfaces:

- blocking dependencies between work items,
- declared counterfactual actions.

### Complete

The complete phase currently materializes the planner outputs already selected on the current tree:

- recommended cuts by readiness class,
- repo followups and followup lanes.

## Current scope

This first slice is descriptive only.

- It does not change cut readiness semantics.
- It does not change cut ranking semantics.
- It does not change PM-view output.

The chart is an overlay that exposes the current planning protocol in one graph-native carrier so later slices can reason about phase-local leverage and counterfactual legality directly.
