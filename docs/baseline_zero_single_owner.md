---
doc_revision: 3
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: baseline_zero_single_owner
doc_role: implementation_playbook
doc_scope:
  - repo
  - tooling
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - CONTRIBUTING.md#contributing_contract
doc_relations:
  refines:
    - CONTRIBUTING.md#contributing_contract
  informs:
    - docs/governance_control_loops.md#governance_control_loops
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---
# Baseline Zero Playbook (Single Owner)

This playbook is for the case where baseline burn-down ownership is one person.

## Objective

Reach and hold `baseline == 0` quickly, without allowing debt regrowth during the campaign.

## Single-owner operating model

- **Owner:** one person (you).
- **Concurrency:** process by lane, not by assignee.
- **Priority order:**
  1. high-frequency baseline keys,
  2. semantically critical evidence classes,
  3. remaining long-tail keys.

## Required controls during burn-down

1. **No-growth gate**
   - Fail checks on net-new baseline entries.
   - Allow temporary waivers only with explicit expiry.
2. **Ledger completeness**
   - Every baseline key must have a disposition:
     - `fix_in_code`,
     - `fix_in_analysis`,
     - `obsolete_remove`.
3. **Removal proof**
   - Baseline removals must ship with deterministic tests reproducing the prior failure mode and asserting the corrected output.

## Daily loop (single-owner cadence)

1. Snapshot baseline inventory and sort by count/frequency.
2. Select a bounded batch (for example 10-30 keys).
3. Fix + add/adjust tests.
4. Re-run checks and confirm net decrease.
5. Commit immediately when batch reaches zero regressions.

## Definition of done

Baseline campaign is complete only when all are true:

- Baseline artifacts are empty.
- Required checks are green.
- No active waivers remain.
- No-growth gate is still enabled.

## Hold-the-line policy after zero

- Keep baseline files empty.
- Treat any new entry as release-blocking until fixed or explicitly waived with expiry.
- Prefer immediate fix over baseline reintroduction.
