---
doc_revision: 2
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
  POLICY_SEED.md#policy_seed: 55
  glossary.md#contract: 44
  docs/policy_rules/ambiguity_contract.md: 1
  docs/policy_rules/grade_monotonicity.md: 2
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED rev55; this ledger remains a fix-forward continuation surface for the markdown-frontmatter policy-rule migration."
  glossary.md#contract: "Reviewed glossary rev44; carrier/boundary terminology remains aligned with the findings tracked here."
  docs/policy_rules/ambiguity_contract.md: "Reviewed rev1; the ambiguity-contract markdown rule doc is now an authoritative source of truth for DSL-evaluated rule guidance."
  docs/policy_rules/grade_monotonicity.md: "Reviewed rev2; per-violation `GMP-*` guidance now lives in the markdown playbook body and is emitted from that source."
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

## Notes

- This ledger is manual for now. It records the same kind of corrective state as
  the repo's queue/ledger maintainer tools, but it does not yet have a
  dedicated machine-generated queue artifact.
- A future generalization pass could unify this style of migration ledger with
  the existing queue/ledger maintainer tooling once more than one policy-doc
  migration stream needs the same lifecycle.
