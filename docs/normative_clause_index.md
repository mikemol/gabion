---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: normative_clause_index
doc_role: normative_index
doc_scope:
  - repo
  - ci
  - agents
  - governance
doc_authority: normative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - AGENTS.md#agent_obligations
  - glossary.md#contract
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 1
  README.md#repo_contract: 1
  CONTRIBUTING.md#contributing_contract: 1
  AGENTS.md#agent_obligations: 1
  glossary.md#contract: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Clause index derived from policy invariants to reduce duplicated prose drift."
  README.md#repo_contract: "README obligation references consolidated to stable clause IDs."
  CONTRIBUTING.md#contributing_contract: "Contributor-facing obligations consolidated behind stable clause IDs."
  AGENTS.md#agent_obligations: "Agent obligations mapped to canonical clause anchors."
  glossary.md#contract: "Dataflow tier references remain governed by glossary contract."
doc_sections:
  normative_clause_index: 1
doc_section_requires:
  normative_clause_index:
    - POLICY_SEED.md#policy_seed
    - README.md#repo_contract
    - CONTRIBUTING.md#contributing_contract
    - AGENTS.md#agent_obligations
    - glossary.md#contract
doc_section_reviews:
  normative_clause_index:
    POLICY_SEED.md#policy_seed:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Policy clauses indexed without changing normative meaning."
    README.md#repo_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "README summary references verified against canonical clause IDs."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Contributor obligations reduced to clause references."
    AGENTS.md#agent_obligations:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Agent obligations reduced to clause references."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Dataflow tier clauses stay glossary-aligned."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="normative_clause_index"></a>
# Normative Clause Index

This document provides stable clause IDs for high-signal obligations that are
repeated across governance-facing documents. Other docs should summarize and
link to clause IDs instead of duplicating long-form normative prose.

## Canonical clauses

<a id="clause-lsp-first"></a>
### `NCI-LSP-FIRST` — LSP-first semantic core
- The language server is the semantic core.
- The CLI must remain a thin LSP client and must not duplicate core analysis.
- Canonical sources: `POLICY_SEED.md#policy_seed`, `CONTRIBUTING.md#contributing_contract`.

<a id="clause-actions-pinned"></a>
### `NCI-ACTIONS-PINNED` — Workflow action pinning
- Workflow actions must be pinned to full commit SHAs.
- Canonical sources: `POLICY_SEED.md#policy_seed`, `README.md#repo_contract`.

<a id="clause-actions-allowlist"></a>
### `NCI-ACTIONS-ALLOWLIST` — Workflow action allow-list
- Workflow actions must be allow-listed.
- Canonical source of allow-listed entries: `docs/allowed_actions.txt`.
- Canonical policy source: `POLICY_SEED.md#policy_seed`.

<a id="clause-dataflow-bundle-tiers"></a>
### `NCI-DATAFLOW-BUNDLE-TIERS` — Dataflow bundle tier obligations
- Recurring parameter bundles crossing function boundaries are type-level obligations.
- Tier-2 bundles must be reified before merge.
- Tier-3 bundles must be reified or documented with `# dataflow-bundle:`.
- Canonical sources: `[glossary.md#contract](../glossary.md#contract)`, `AGENTS.md#agent_obligations`, `CONTRIBUTING.md#contributing_contract`.

<a id="clause-baseline-ratchet"></a>
### `NCI-BASELINE-RATCHET` — Baseline ratchet integrity
- Baselines are ratchet checkpoints, not bypass levers.
- Do not refresh baselines to bypass positive deltas while gates are enabled.
- Canonical source: `CONTRIBUTING.md#contributing_contract`.

## Usage rule

When referencing one of these obligations in `README.md`, `CONTRIBUTING.md`,
or `AGENTS.md`, use a short summary with direct clause links, for example:

- `NCI-LSP-FIRST` (`docs/normative_clause_index.md#clause-lsp-first`)
- `NCI-DATAFLOW-BUNDLE-TIERS` (`docs/normative_clause_index.md#clause-dataflow-bundle-tiers`)

