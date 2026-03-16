---
doc_revision: 4
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: grade_monotonicity_policy_rules
doc_role: policy
doc_scope:
  - repo
  - governance
  - policy
  - grade
doc_authority: normative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 57
  glossary.md#contract: 46
  docs/shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol: 3
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED rev57; grade monotonicity remains a blocking governance surface for downstream contract regressions while frontmatter-backed summary playbooks move to generated rendering."
  glossary.md#contract: "Reviewed glossary rev46; contract, bundle, and decision-protocol terminology remain aligned while the summary playbook section renders from canonical frontmatter guidance."
  docs/shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol: "Reviewed the protocol card rev3; these grade playbooks specialize the same upstream-normalization bias."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
playbook_rendering:
  references:
    grade_monotonicity.new_violations:
      - label: Shift-Ambiguity-Left Protocol
        href: ../shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol
rules:
  - rule_id: grade_monotonicity.new_violations
    domain: grade_monotonicity
    severity: blocking
    predicate:
      op: int_gte
      path: [new_violations]
      value: 1
    outcome:
      kind: block
      message: graded determinism/cost monotonicity violations
      guidance:
        why: a downstream call edge has widened structure, decision work, or complexity beyond its caller contract
        prefer: move normalization and alternation to the earliest lawful seam, then keep downstream edges monotone in structure and work
        avoid:
          - do not reintroduce Optional, sentinel, or multi-shape carriers after a stricter caller has already normalized them
          - do not hide higher complexity behind an ordinary core edge without an explicit named boundary
    evidence_contract: none
    playbook_anchor: grade-monotonicity-new-violations

  - rule_id: grade_monotonicity.ok
    domain: grade_monotonicity
    severity: info
    predicate:
      op: always
    outcome:
      kind: pass
      message: graded determinism/cost monotonicity check passed
    evidence_contract: none
---

<a id="grade_monotonicity_policy_rules"></a>
# Grade Monotonicity Policy Rules

This document is the authoritative playbook body for grade-monotonicity policy
guidance and per-violation remediation references.

<!-- BEGIN:generated_policy_rule_playbooks -->
_The playbook sections below are generated from this document's `rules:` frontmatter via `mise exec -- python -m scripts.policy.render_policy_rule_playbooks`._

<a id="grade-monotonicity-new-violations"></a>
## `grade_monotonicity.new_violations`

Meaning: a downstream call edge has widened structure, decision work, or complexity beyond its caller contract

Preferred response:
- move normalization and alternation to the earliest lawful seam, then keep downstream edges monotone in structure and work

Avoid:
- do not reintroduce Optional, sentinel, or multi-shape carriers after a stricter caller has already normalized them
- do not hide higher complexity behind an ordinary core edge without an explicit named boundary
Reference: [Shift-Ambiguity-Left Protocol](../shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol).
<!-- END:generated_policy_rule_playbooks -->

The `GMP-*` sections below remain the canonical per-violation playbooks consumed
by `src/gabion/tooling/policy_substrate/grade_monotonicity_semantic.py`.

<a id="gmp-001"></a>
## `GMP-001`

Meaning: a callee accepts nullable or sentinel-bearing carriers after a stricter caller.
Priority: 10

Preferred response:
- normalize nullability once at ingress
- keep downstream callees strict

Avoid:
- do not reintroduce Optional or sentinel-bearing contracts downstream

<a id="gmp-002"></a>
## `GMP-002`

Meaning: a callee widens the runtime type domain beyond the caller contract.
Priority: 20

Preferred response:
- make type alternation explicit at the boundary
- pass one accepted internal variant downstream

Avoid:
- do not widen a strict caller contract back to `Any`, `object`, or new unions

<a id="gmp-003"></a>
## `GMP-003`

Meaning: a callee widens structural payload shape after normalization.
Priority: 30

Preferred response:
- keep one internal DTO/carrier shape
- convert legacy shapes only once

Avoid:
- do not accept `dict`/`list`/`tuple` shape alternation after normalization

<a id="gmp-004"></a>
## `GMP-004`

Meaning: a callee reintroduces imperative runtime classification work.
Priority: 40

Preferred response:
- move the branch to ingress or an explicit decision protocol

Avoid:
- do not add probe-then-recover locals or deeper classification cascades

<a id="gmp-005"></a>
## `GMP-005`

Meaning: a callee regresses protocol discharge level.
Priority: 50

Preferred response:
- keep invariant discharge and decision-protocol explicitness monotone downstream

Avoid:
- do not call raw-ingress-style helpers from a stricter decision or invariant-discharged caller

<a id="gmp-006"></a>
## `GMP-006`

Meaning: a callee expands output cardinality without an explicit named boundary.
Priority: 60

Preferred response:
- make the fan-out/materialization boundary explicit
- keep ordinary core edges cardinality-stable

Avoid:
- do not introduce unmarked fan-out or materialization in ordinary call chains

<a id="gmp-007"></a>
## `GMP-007`

Meaning: a callee expands work growth without an explicit named boundary.
Priority: 70

Preferred response:
- concentrate budgeted complexity at named boundaries with a stated reason
- keep ordinary core edges from hiding asymptotic growth

Avoid:
- do not hide higher-complexity helpers behind ordinary core edges
