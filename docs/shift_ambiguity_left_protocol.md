---
doc_revision: 4
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
   Pattern:
   when identity/render/report code is probing raw payload maps, normalize the
   payload once into a small carrier/dataclass at the boundary, then make the
   downstream logic consume only that strict carrier.
4. **Remove downstream ambiguity guards**: delete repeated local checks in core suites.
5. **Collapse to one deterministic path**: remove compatibility wrappers/dual-shape bridges in core; keep temporary adapters at boundary ingress only with lifecycle metadata.
6. **Verify signatures**: run policy checks and confirm no new ambiguity-contract findings.
7. **Reject out-and-back relocation**: moving logic out of a prohibited zone and then re-entering it on the same fiber is non-remediation; shift the boundary upstream instead.

Reviewer instruction: reject patches that add ambiguity signatures in deterministic
core zones without boundary-level reification evidence, and reject semantic-core
"legacy bridge" patches that lack explicit boundary-lifecycle evidence. Also reject
out-and-back relocations that hide prohibited behavior without moving the boundary
upstream on the same affected fiber.

## Named Adapter Pattern

**Normalize once into a small carrier, then derive downstream structure from the
strict carrier.**

Use this when a boundary currently does work like:
- repeated `payload.get(...)` probing,
- nested `mapping_optional(...)` lookups,
- per-call identity-token assembly from raw maps,
- report/render logic that keeps rediscovering the same payload alternation.

Preferred sequence:
1. normalize the raw boundary payload once into a small dataclass/Protocol with
   explicit fields,
2. validate required anchors there,
3. compute any digests or normalized lists there,
4. pass that strict carrier to identity/render/report functions,
5. delete the downstream raw-map probing.

This keeps the boundary as the only place that knows about legacy or
alternating payload shape, and it prevents helper-local ambiguity from
reappearing in code that should be deterministic.
