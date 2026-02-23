---
doc_revision: 4
reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.
doc_id: synthesis_payload
doc_role: schema
doc_scope:
  - repo
  - synthesis
  - tooling
doc_authority: informative
doc_requires:
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - glossary.md#contract
  - POLICY_SEED.md#policy_seed
  - AGENTS.md#agent_obligations
doc_reviewed_as_of:
  README.md#repo_contract: 1
  CONTRIBUTING.md#contributing_contract: 1
  glossary.md#contract: 1
  POLICY_SEED.md#policy_seed: 1
  AGENTS.md#agent_obligations: 1
doc_review_notes:
  README.md#repo_contract: Reviewed README.md rev1 (docflow audit now scans in/ by default); no conflicts with this document's scope.
  CONTRIBUTING.md#contributing_contract: Reviewed CONTRIBUTING.md rev1 (docflow now fails on missing GH references for SPPF-relevant changes); no conflicts with this document's scope.
  glossary.md#contract: Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline).
  POLICY_SEED.md#policy_seed: Reviewed POLICY_SEED.md rev1 (mechanized governance default; branch/tag CAS + check-before-use constraints); no conflicts with this document's scope.
  AGENTS.md#agent_obligations: Agent obligations unchanged; payload remains tool-facing.
doc_change_protocol: POLICY_SEED.md#change_protocol
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
doc_sections:
  synthesis_payload: 1
doc_section_requires:
  synthesis_payload:
    - README.md#repo_contract
    - CONTRIBUTING.md#contributing_contract
    - glossary.md#contract
    - POLICY_SEED.md#policy_seed
    - AGENTS.md#agent_obligations
doc_section_reviews:
  synthesis_payload:
    README.md#repo_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed README.md rev1 (docflow audit now scans in/ by default); no conflicts with this document's scope.
    CONTRIBUTING.md#contributing_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed CONTRIBUTING.md rev1 (docflow now fails on missing GH references for SPPF-relevant changes); no conflicts with this document's scope.
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline).
    POLICY_SEED.md#policy_seed:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed POLICY_SEED.md rev1 (mechanized governance default; branch/tag CAS + check-before-use constraints); no conflicts with this document's scope.
    AGENTS.md#agent_obligations:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Agent obligations unchanged; payload remains tool-facing.
---

<a id="synthesis_payload"></a>

# Synthesis Plan Payload (Prototype)

This document describes the JSON payload accepted by `gabion synthesis-plan`
and the LSP command `gabion.synthesisPlan`.

## Payload shape

```json
{
  "bundles": [
    { "bundle": ["ctx", "config"], "tier": 2 }
  ],
  "field_types": {
    "ctx": "Context",
    "config": "Config"
  },
  "existing_names": ["CtxBundle"],
  "frequency": {
    "ctx": 3,
    "config": 1
  },
  "fallback_prefix": "Bundle",
  "max_tier": 2,
  "min_bundle_size": 2,
  "allow_singletons": false,
  "merge_overlap_threshold": 0.75
}
```

### Fields

- `bundles` (required): list of bundle entries.
  - `bundle`: list of parameter names.
  - `tier`: integer tier (1 = explicit, 2 = implicit strong, 3 = implicit weak).
- `field_types` (optional): map from parameter name to type hint string.
- `existing_names` (optional): list of names to avoid in naming heuristics.
- `frequency` (optional): occurrence counts used by naming heuristics.
- `fallback_prefix` (optional): base prefix used when no field hints exist.
- `max_tier` (optional): highest tier to synthesize (default: 2).
- `min_bundle_size` (optional): minimum bundle size to synthesize (default: 2).
- `allow_singletons` (optional): allow 1-field bundles when true (default: false).
- `merge_overlap_threshold` (optional): Jaccard overlap threshold for merging bundles
  before synthesis (default: 0.75).

## Response shape (summary)

The synthesis plan response includes:

- `protocols`: list of generated protocol specs with `name`, `fields`, `bundle`, `tier`.
- `warnings`: list of warning messages (e.g., no eligible bundles).
- `errors`: list of error messages (empty for prototype).