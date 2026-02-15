---
doc_revision: 18
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: agents
doc_role: agent
doc_scope:
  - repo
  - agents
  - tooling
doc_authority: normative
doc_requires:
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
doc_reviewed_as_of:
  README.md#repo_contract: 1
  CONTRIBUTING.md#contributing_contract: 1
  POLICY_SEED.md#policy_seed: 1
  glossary.md#contract: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md rev1 (docflow audit now scans in/ by default); no conflicts with this document's scope."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev1 (docflow now fails on missing GH references for SPPF-relevant changes); no conflicts with this document's scope."
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev1 (mechanized governance default; branch/tag CAS + check-before-use constraints); no conflicts with this document's scope."
  glossary.md#contract: "Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline)."
doc_sections:
  agent_obligations: 1
doc_section_requires:
  agent_obligations:
    - README.md#repo_contract
    - CONTRIBUTING.md#contributing_contract
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
doc_section_reviews:
  agent_obligations:
    README.md#repo_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Repo contract reviewed; agent obligations unchanged."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Contributor contract reviewed; agent obligations unchanged."
    POLICY_SEED.md#policy_seed:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed reviewed; agent obligations unchanged."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Glossary contract reviewed; agent obligations unchanged."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_invariants:
  - read_policy_glossary_first
  - refuse_on_conflict
  - tier2_reification
  - tier3_documentation
  - lsp_first_invariant
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="agent_obligations"></a>
# AGENTS.md#agent_obligations

This repository is governed by `POLICY_SEED.md#policy_seed`. Treat it as authoritative.
Semantic correctness is governed by `[glossary.md#contract](glossary.md#contract)` (co-equal contract).

## Cross-references (normative pointers)
- `README.md#repo_contract` defines project scope, status, and entry points.
- `CONTRIBUTING.md#contributing_contract` defines human+machine workflow guardrails.
- `POLICY_SEED.md#policy_seed` defines execution and CI safety constraints.
- `[glossary.md#contract](glossary.md#contract)` defines semantic meanings, axes, and commutation obligations.

## Required behavior
- Read `POLICY_SEED.md#policy_seed` and `[glossary.md#contract](glossary.md#contract)` before proposing or applying changes.
- If a request conflicts with `POLICY_SEED.md#policy_seed`, stop and ask for guidance.
- Do not weaken or bypass self-hosted runner protections.
- Keep workflow actions pinned to full commit SHAs and allow-listed.
- When changing workflows, run the policy checks (once the scripts exist) and
  surface any violations explicitly.
- Preserve the LSP-first invariant: the server is the semantic core and the
  CLI remains a thin LSP client.
- Use `mise exec -- python` for repo-local tooling to ensure the pinned
  interpreter and dependencies are used.
- Treat docflow as repo-local convenience only; do not project it as a
  general Gabion feature without explicit policy change.
- Do not mechanistically bump `doc_reviewed_as_of`; update only with explicit
  `doc_review_notes` based on a real content review.

## Dataflow grammar invariant
- Recurring parameter bundles are type-level obligations.
- Any bundle that crosses function boundaries must be promoted to a Protocol
  (dataclass config/local bundle) or explicitly documented with a
  `# dataflow-bundle:` marker.
- Tier-2 bundles must be reified before merge (see `[glossary.md#contract](glossary.md#contract)`).
- Tier-3 bundles must be documented or reified (see `[glossary.md#contract](glossary.md#contract)`).

## Doc hygiene
- Markdown docs include a YAML front-matter block with `doc_revision`.
- Bump `doc_revision` for conceptual changes.
- Record convergence in `doc_reviewed_as_of` (must match dependency revisions).

If unsure, prefer refusal over unsafe compliance.
