---
doc_revision: 3
reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.
doc_id: doer_judge_witness
doc_role: protocol
doc_scope:
  - repo
  - governance
  - workflow
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - AGENTS.md#agent_obligations
  - CONTRIBUTING.md#contributing_contract
  - README.md#repo_contract
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  AGENTS.md#agent_obligations: 2
  CONTRIBUTING.md#contributing_contract: 2
  README.md#repo_contract: 2
doc_review_notes:
  POLICY_SEED.md#policy_seed: Reviewed POLICY_SEED.md rev1 (mechanized governance default; branch/tag CAS + check-before-use constraints); no conflicts with this document's scope.
  glossary.md#contract: Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline).
  AGENTS.md#agent_obligations: Agent obligations align with doer/judge/witness separation.
  CONTRIBUTING.md#contributing_contract: Reviewed CONTRIBUTING.md rev1 (docflow now fails on missing GH references for SPPF-relevant changes); no conflicts with this document's scope.
  README.md#repo_contract: Reviewed README.md rev1 (docflow audit now scans in/ by default); no conflicts with this document's scope.
doc_change_protocol: POLICY_SEED.md#change_protocol
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
doc_sections:
  doer_judge_witness: 1
doc_section_requires:
  doer_judge_witness:
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
    - AGENTS.md#agent_obligations
    - CONTRIBUTING.md#contributing_contract
    - README.md#repo_contract
doc_section_reviews:
  doer_judge_witness:
    POLICY_SEED.md#policy_seed:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed rev2 reviewed; governance obligations remain aligned."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline).
    AGENTS.md#agent_obligations:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Agent obligations rev2 reviewed; clause and cadence links remain aligned."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Contributor contract rev2 reviewed; dual-sensor cadence and correction gates remain aligned."
    README.md#repo_contract:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Repo contract rev2 reviewed; command and artifact guidance remains aligned."
---

<a id="doer_judge_witness"></a>

# Doer / Judge / Witness (Optional Framing)

This framing is an optional workflow lens. It separates responsibility for
changes into three roles so the governance layer remains explicit and testable.

## Roles

- **Doer:** produces candidate changes (code, docs, scripts).
- **Judge:** evaluates changes against `POLICY_SEED.md#policy_seed`, `[glossary.md#contract](glossary.md#contract)`, and CI.
- **Witness:** records outcomes and ensures the governance record stays coherent.

## Why this exists

The governance layer is a self-stabilizing system. Separating these roles helps
avoid conflating implementation with validation, and helps keep policy checks
repeatable under automation.

## How to use it (lightweight)

1. **Doer** proposes a change on `stage`.
2. **Judge** runs `scripts/checks.sh --no-docflow` and reviews any policy or
   dataflow violations.
3. **Witness** updates governance docs or the checklist if the change shifts
   the repo's obligations.

If you do not need this framing, ignore it.