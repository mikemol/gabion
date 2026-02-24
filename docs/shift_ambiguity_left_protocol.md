---
doc_revision: 1
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
  POLICY_SEED.md#policy_seed: 1
  AGENTS.md#agent_obligations: 1
  CONTRIBUTING.md#contributing_contract: 1
  docs/normative_clause_index.md#normative_clause_index: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Protocol card operationalizes §4.8 for low-entropy execution under refactor pressure."
  AGENTS.md#agent_obligations: "Examples align with boundary-normalization and Decision Protocol obligations."
  CONTRIBUTING.md#contributing_contract: "Card mirrors mandatory refactor sequence used by reviewers."
  docs/normative_clause_index.md#normative_clause_index: "Card references canonical clause IDs instead of duplicating governance prose."
---

<a id="shift_ambiguity_left_protocol"></a>
# Shift-Ambiguity-Left Protocol Card

Canonical rule: [`NCI-SHIFT-AMBIGUITY-LEFT`](./normative_clause_index.md#clause-shift-ambiguity-left).

Trigger this protocol whenever a change would otherwise add local `isinstance`,
`Optional`/`Union`/`Any`/`|`, sentinel returns, or branch ladders in semantic core.

1. **Classify the ambiguity**: input shape vs decision predicate vs cross-boundary bundle.
2. **Reify the contract**: introduce/extend Protocol, dataclass, or Decision Protocol.
3. **Normalize once at boundary**: discharge alternation before core execution.
4. **Remove downstream ambiguity guards**: delete repeated local checks in core suites.
5. **Verify signatures**: run policy checks and confirm no new ambiguity-contract findings.

Reviewer instruction: reject patches that add ambiguity signatures in deterministic
core zones without boundary-level reification evidence.
