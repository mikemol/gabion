---
doc_revision: 13
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: agents
doc_role: agent
doc_scope:
  - repo
  - agents
  - tooling
doc_authority: normative
doc_requires:
  - README.md
  - CONTRIBUTING.md
  - POLICY_SEED.md
  - glossary.md
doc_reviewed_as_of:
  README.md: 59
  CONTRIBUTING.md: 78
  POLICY_SEED.md: 33
  glossary.md: 29
doc_review_notes:
  README.md: "Reviewed README.md rev59 (docflow audit now scans in/ by default); no conflicts with this document's scope."
  CONTRIBUTING.md: "Reviewed CONTRIBUTING.md rev77 (docflow now fails on missing GH references for SPPF-relevant changes); no conflicts with this document's scope."
  POLICY_SEED.md: "Reviewed POLICY_SEED.md rev33 (mechanized governance default; branch/tag CAS + check-before-use constraints); no conflicts with this document's scope."
  glossary.md: "Reviewed glossary rev29 (obsolescence projection path + self-review/mirror definitions); agent obligations unchanged."
doc_change_protocol: "POLICY_SEED.md §6"
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

# AGENTS.md

This repository is governed by `POLICY_SEED.md`. Treat it as authoritative.
Semantic correctness is governed by `glossary.md` (co-equal contract).

## Cross-references (normative pointers)
- `README.md` defines project scope, status, and entry points.
- `CONTRIBUTING.md` defines human+machine workflow guardrails.
- `POLICY_SEED.md` defines execution and CI safety constraints.
- `glossary.md` defines semantic meanings, axes, and commutation obligations.

## Required behavior
- Read `POLICY_SEED.md` and `glossary.md` before proposing or applying changes.
- If a request conflicts with `POLICY_SEED.md`, stop and ask for guidance.
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
- Tier-2 bundles must be reified before merge (see `glossary.md`).
- Tier-3 bundles must be documented or reified (see `glossary.md`).

## Doc hygiene
- Markdown docs include a YAML front-matter block with `doc_revision`.
- Bump `doc_revision` for conceptual changes.
- Record convergence in `doc_reviewed_as_of` (must match dependency revisions).

If unsure, prefer refusal over unsafe compliance.
