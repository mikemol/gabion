---
doc_revision: 2
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: shift_ambiguity_left_protocol
doc_role: playbook
doc_scope:
  - repo
  - governance
  - agents
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - AGENTS.md#agent_obligations
  - CONTRIBUTING.md#contributing_contract
  - docs/normative_clause_index.md#normative_clause_index
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  AGENTS.md#agent_obligations: 2
  CONTRIBUTING.md#contributing_contract: 2
  docs/normative_clause_index.md#normative_clause_index: 2
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev2 (forward-remediation order, ci_watch failure-bundle durability, and enforced execution-coverage policy wording)."
  AGENTS.md#agent_obligations: "Reviewed AGENTS.md rev2 (required validation stack, forward-remediation preference, and ci_watch failure-bundle triage guidance)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev2 (two-stage dual-sensor cadence, correction-unit validation stack, and strict-coverage trigger guidance)."
  docs/normative_clause_index.md#normative_clause_index: "Reviewed normative_clause_index rev2 (extended existing dual-sensor/shift-ambiguity/deadline clauses without introducing new clause IDs)."
---

<a id="shift_ambiguity_left_protocol"></a>
# Shift-Ambiguity-Left Protocol Card

Canonical rule: [`NCI-SHIFT-AMBIGUITY-LEFT`](./normative_clause_index.md#clause-shift-ambiguity-left).

Trigger this protocol whenever a change would otherwise add local `isinstance`,
`Optional`/`Union`/`Any`/`|`, sentinel returns, branch ladders, or compatibility
wrappers/legacy bridges in semantic core.

1. **Classify the ambiguity**: input shape vs decision predicate vs cross-boundary bundle.
2. **Reify the contract**: introduce/extend Protocol, dataclass, or Decision Protocol.
3. **Normalize once at boundary**: discharge alternation before core execution.
4. **Remove downstream ambiguity guards**: delete repeated local checks in core suites.
5. **Collapse to one deterministic path**: remove compatibility wrappers/dual-shape bridges in core; keep temporary adapters at boundary ingress only with lifecycle metadata.
6. **Verify signatures**: run policy checks and confirm no new ambiguity-contract findings.

Reviewer instruction: reject patches that add ambiguity signatures in deterministic
core zones without boundary-level reification evidence, and reject semantic-core
"legacy bridge" patches that lack explicit boundary-lifecycle evidence.
