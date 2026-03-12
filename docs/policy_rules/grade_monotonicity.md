---
doc_revision: 1
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
  POLICY_SEED.md#policy_seed: 55
  glossary.md#contract: 44
  docs/shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol: 3
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED rev55; grade monotonicity remains a blocking governance surface for downstream contract regressions."
  glossary.md#contract: "Reviewed glossary rev44; contract, bundle, and decision-protocol terminology remain aligned with these grade playbooks."
  docs/shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol: "Reviewed the protocol card rev3; these grade playbooks specialize the same upstream-normalization bias."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
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

<a id="grade-monotonicity-new-violations"></a>
## `grade_monotonicity.new_violations`

Meaning: one or more call edges regressed the caller-to-callee grade contract.

Preferred response:
- locate the earliest fiber seam where the widening first appears
- discharge the alternation there
- keep downstream edges monotone in structure, cardinality, and work

Avoid:
- pushing the same ambiguity or complexity into a deeper helper
- compensating with more branches instead of a stricter carrier

Reference: [Shift-Ambiguity-Left Protocol](../shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol).

<a id="gmp-001"></a>
## `GMP-001`

Meaning: a callee accepts nullable or sentinel-bearing carriers after a stricter caller.

Preferred response:
- normalize nullability once at ingress
- keep downstream callees strict

<a id="gmp-002"></a>
## `GMP-002`

Meaning: a callee widens the runtime type domain beyond the caller contract.

Preferred response:
- make type alternation explicit at the boundary
- pass one accepted internal variant downstream

<a id="gmp-003"></a>
## `GMP-003`

Meaning: a callee widens structural payload shape after normalization.

Preferred response:
- keep one internal DTO/carrier shape
- convert legacy shapes only once

<a id="gmp-004"></a>
## `GMP-004`

Meaning: a callee reintroduces imperative runtime classification work.

Preferred response:
- move the branch to ingress or an explicit decision protocol

<a id="gmp-005"></a>
## `GMP-005`

Meaning: a callee regresses protocol discharge level.

Preferred response:
- keep invariant discharge and decision-protocol explicitness monotone downstream

<a id="gmp-006"></a>
## `GMP-006`

Meaning: a callee expands output cardinality without an explicit named boundary.

Preferred response:
- make the fan-out/materialization boundary explicit
- keep ordinary core edges cardinality-stable

<a id="gmp-007"></a>
## `GMP-007`

Meaning: a callee expands work growth without an explicit named boundary.

Preferred response:
- concentrate budgeted complexity at named boundaries with a stated reason
- keep ordinary core edges from hiding asymptotic growth
