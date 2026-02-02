---
doc_revision: 2
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: synthesis_payload
doc_role: schema
doc_scope:
  - repo
  - synthesis
  - tooling
doc_authority: informative
doc_requires:
  - README.md
  - CONTRIBUTING.md
  - glossary.md
  - POLICY_SEED.md
  - AGENTS.md
doc_change_protocol: "POLICY_SEED.md §6"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

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
