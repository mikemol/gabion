---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: shift_ambiguity_left_single_task
doc_role: implementation_task
doc_scope:
  - repo
  - governance
  - agents
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - CONTRIBUTING.md#contributing_contract
  - AGENTS.md#agent_obligations
  - docs/normative_clause_index.md#normative_clause_index
  - docs/enforceable_rules_cheat_sheet.md#enforceable_rules_cheat_sheet
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 1
  CONTRIBUTING.md#contributing_contract: 1
  AGENTS.md#agent_obligations: 1
  docs/normative_clause_index.md#normative_clause_index: 1
  docs/enforceable_rules_cheat_sheet.md#enforceable_rules_cheat_sheet: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Task composes §4.8 shift-ambiguity-left requirements into one implementation slice."
  CONTRIBUTING.md#contributing_contract: "Task preserves refactor-under-ambiguity sequence while collapsing rollout into a single PR objective."
  AGENTS.md#agent_obligations: "Task aligns with boundary normalization, Decision Protocol, and no-sentinel-core constraints."
  docs/normative_clause_index.md#normative_clause_index: "Task extends canonical clause usage for stable cross-doc references."
  docs/enforceable_rules_cheat_sheet.md#enforceable_rules_cheat_sheet: "Task keeps enforceable mappings explicit and machine-checkable."
---

<a id="shift_ambiguity_left_single_task"></a>
# Single task: Shift-Ambiguity-Left policy consolidation

Use this as one cohesive implementation task so policy, contributor guidance,
and automation can land together in one PR.

## Objective
Ship one integrated policy slice that prevents reintroduction of local ambiguity
shortcuts in semantic core code.

## Scope (single PR)
1. **Canonical clause promotion**
   - Add a stable clause ID for Shift-Ambiguity-Left in
     `docs/normative_clause_index.md#normative_clause_index`.
   - Update references in `POLICY_SEED.md#policy_seed`,
     `CONTRIBUTING.md#contributing_contract`, and `AGENTS.md#agent_obligations`
     to point at that clause ID.
2. **Zone-scoped policy expression**
   - Codify allowed-at-boundary vs forbidden-in-core ambiguity handling in
     architecture/governance docs.
   - Explicitly call out runtime narrowing (`isinstance`), type alternation
     (`Optional`/`Union`/`|`), and sentinel control outcomes as boundary-only
     normalization tools, not semantic-core control strategies.
3. **Unified enforcement hook**
   - Add or extend one policy gate that emits stable violation IDs for
     ambiguity reintroduction patterns.
   - Ensure it is wired into existing policy-check/CI orchestration as a
     first-class policy failure signal.
4. **Agent + reviewer execution protocol**
   - Publish one compact protocol card/checklist that states the mandatory
     transformation order when ambiguity pressure appears:
     classify -> reify decision surface -> normalize at boundary -> remove
     downstream ambiguity guards -> verify no new signatures.

## Acceptance criteria
- A single clause ID is the canonical anchor for Shift-Ambiguity-Left and is
  used consistently across governance docs.
- Boundary/core zone semantics are explicit enough that reviewers can reject
  ambiguity shortcuts deterministically.
- Automation reports ambiguity-policy violations with stable rule IDs and
  actionable locations.
- Contributor/agent instructions include one concise playbook that can be
  copied into prompts/checklists without reinterpretation.

## Out of scope
- Broad repository-wide deletion of all historical ambiguity constructs.
- Large semantic-core rewrites unrelated to introducing the policy/control
  surface above.

## Suggested implementation order
1. Add canonical clause ID and references.
2. Add zone-scoped wording.
3. Wire the enforcement gate.
4. Add the compact protocol card and checklist references.
5. Run policy checks and include outputs in PR evidence.
