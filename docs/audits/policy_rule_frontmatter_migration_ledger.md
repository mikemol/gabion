---
doc_revision: 5
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: policy_rule_frontmatter_migration_ledger
doc_role: audit
doc_scope:
  - repo
  - policy
  - governance
doc_authority: advisory
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/policy_rules/ambiguity_contract.md
  - docs/policy_rules/grade_monotonicity.md
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 57
  glossary.md#contract: 46
  docs/policy_rules/ambiguity_contract.md: 2
  docs/policy_rules/grade_monotonicity.md: 4
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED rev57; process-relative runtime now applies to policy/governance workflows, so this ledger continues to treat documentation mechanization as an active runtime planning surface."
  glossary.md#contract: "Reviewed glossary rev46; runtime scope and admissibility-ladder terminology remain aligned with this ledger's policy-document mechanization queue."
  docs/policy_rules/ambiguity_contract.md: "Reviewed rev2; frontmatter-backed ambiguity playbook sections now render from canonical markdown guidance while the document remains the authoritative DSL rule source."
  docs/policy_rules/grade_monotonicity.md: "Reviewed rev4; the summary rule playbook now renders from markdown frontmatter while the `GMP-*` sections remain the canonical per-violation guidance consumed by runtime policy tooling."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="policy_rule_frontmatter_migration_ledger"></a>
# Policy Rule Frontmatter Migration Ledger

This ledger tracks the markdown-frontmatter policy-rule migration and the
follow-up corrections discovered during implementation audit.

## Queue

| Queue ID | Scope | Status | Notes |
| --- | --- | --- | --- |
| `PRF-001` | Reject non-object `rules:` entries during policy document compilation | `landed` | Compiler now fails closed instead of silently dropping malformed rule entries. |
| `PRF-002` | Treat malformed YAML frontmatter as a strict compiler failure | `landed` | Markdown rule docs now emit `invalid_frontmatter` instead of degrading to generic missing-rules failures. |
| `PRF-003` | Reject blank `playbook_anchor` values | `landed` | Markdown rule docs now treat blank anchors as invalid rather than silently absent. |
| `PRF-004` | Remove duplicated `GMP-*` guidance text from Python so grade playbooks are fully markdown-authoritative | `landed` | Per-violation grade guidance is now parsed from `docs/policy_rules/grade_monotonicity.md` and emitted from that markdown source instead of duplicated runtime strings. |
| `PRF-005` | Bootstrap a machine-readable catalog + renderer for `docs/enforceable_rules_cheat_sheet.md` Rule Matrix | `landed` | The first mechanically-owned subset is now the Rule Matrix: `docs/enforceable_rules_catalog.yaml` owns the rows and `scripts/policy/render_enforceable_rules_cheat_sheet.py` renders the generated block, while the rest of the cheat sheet remains hand-authored. |
| `PRF-006` | Normalize governance loop registry data and render `docs/governance_control_loops.md` plus `docs/governance_loop_matrix.md` from a shared catalog | `landed` | `docs/governance_control_loops.yaml` now owns the shared registry data and `scripts/policy/render_governance_loop_docs.py` renders the generated blocks in both governance loop docs. |
| `PRF-007` | Render policy-rule playbooks from markdown frontmatter for `docs/policy_rules/ambiguity_contract.md` and `docs/policy_rules/grade_monotonicity.md` | `in_progress` | The new renderer owns frontmatter-backed playbook sections; `grade_monotonicity.md` keeps its `GMP-*` sections manual because they remain the runtime-consumed canonical violation playbooks. |
| `PRF-008` | Autodenormalize clause-backed obligation decks for `AGENTS.md` and `CONTRIBUTING.md` from a small audience-specific clause catalog | `queued` | Restrict generation to repetitive clause bullet decks and keep explanatory workflow prose hand-authored. |
| `PRF-009` | Extend cheat-sheet mechanization to generate guardrail/validation sections from governance catalogs | `queued` | Use the loop registry and clause catalogs as inputs once PRF-006 and PRF-008 converge. |

## Notes

- This ledger is manual for now. It records the same kind of corrective state as
  the repo's queue/ledger maintainer tools, but it does not yet have a
  dedicated machine-generated queue artifact.
- A future generalization pass could unify this style of migration ledger with
  the existing queue/ledger maintainer tooling once more than one policy-doc
  migration stream needs the same lifecycle.
- `PRF-005` was the first follow-on stream beyond markdown frontmatter itself:
  it treated the cheat-sheet Rule Matrix as a process-relative runtime surface
  for governance planning, so the matrix moved through structured catalog
  ownership before any broader full-document generation pass.
- The current follow-on order is deliberate:
  1. shared governance loop registry feeding both control-loop prose and loop matrix (`PRF-006`)
  2. policy-rule playbook rendering from canonical markdown frontmatter (`PRF-007`)
  3. clause-backed obligation-deck generation for agent/contributor docs (`PRF-008`)
  4. second-phase cheat-sheet section generation from those stabilized catalogs (`PRF-009`)
