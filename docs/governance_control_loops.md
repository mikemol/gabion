---
doc_revision: 1
doc_id: governance_control_loops
doc_role: governance
doc_scope:
  - repo
  - policy
  - tooling
doc_authority: normative
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

# Governance control loops

This document defines correction modes and transition criteria for baseline and delta governance loops.

## Correction modes

- `advisory`: emit diagnostics and summary only; do not block execution.
- `ratchet`: allow bounded movement only toward tighter policy; block regressions.
- `hard-fail`: block any blocking-threshold regression immediately.

## Transition criteria

Each loop transitions using `warning_threshold` and `blocking_threshold` values from `docs/governance_rules.yaml`:

1. `advisory -> ratchet` when warning-threshold events recur and no override token is recorded.
2. `ratchet -> hard-fail` when blocking-threshold events recur under ratchet mode.
3. `hard-fail -> ratchet|advisory` only with an explicit policy override token and an annotated rationale in the change artifact.

## Bounded-step correction rules

Baseline updates are bounded by these invariants:

1. Baseline writes require explicit `--write-*` flags.
2. Strictness cannot be reduced unless both are present:
   - an explicit policy override token (`GABION_POLICY_OVERRIDE_TOKEN`), and
   - an annotated rationale (`GABION_POLICY_OVERRIDE_RATIONALE`).
3. Delta gates use shared severity mapping from `docs/governance_rules.yaml`; ad-hoc threshold branching is prohibited.

## Loop table

| Loop | Correction mode | Warning threshold | Blocking threshold |
| --- | --- | --- | --- |
| obsolescence opaque | hard-fail | 0 | 1 |
| obsolescence unmapped | ratchet | 0 | 1 |
| annotation orphaned | ratchet | 0 | 1 |
| ambiguity total | hard-fail | 0 | 1 |
| docflow contradictions | advisory | 0 | 1 |

The source of truth for loop thresholds, mode defaults, and transitions is `docs/governance_rules.yaml`.
