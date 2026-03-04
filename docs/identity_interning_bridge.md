---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: identity_interning_bridge
doc_role: architecture
doc_scope:
  - repo
  - analysis
  - identity
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - AGENTS.md#agent_obligations
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  AGENTS.md#agent_obligations: 2
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed policy execution constraints; bridge is additive and does not alter execution surfaces."
  glossary.md#contract: "Reviewed glossary terms for structural fingerprint, ASPF, forest, and hash-consing alignment."
  AGENTS.md#agent_obligations: "Reviewed agent obligations; bridge preserves deterministic contracts and does not bypass policy checks."
doc_sections:
  bridge: 1
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="bridge"></a>
# Identity Interning Bridge

Phase 1 introduces a shared identity substrate seam that aligns three existing
identity idioms without changing current semantic outputs:

1. Transcript-style stream interning (`name -> int -> ordered path tuple`).
2. Prime-based structural identity (`PrimeRegistry`, dimensional fingerprints).
3. ASPF/Forest internment and stage-cache identity surfaces.

## Canonical Carrier

- Canonical internal identity: **ordered basis path** (tuple of interned atom IDs).
- Derived aliases:
  - commutative scalar projection (`prime_product`),
  - stable digest alias (`digest_alias`).
- Witness metadata records that path order is canonical while scalar projection
  intentionally erases order.

## Reified Seams

1. [identity_namespace.py](/home/mikemol/github/gabion/src/gabion/analysis/core/identity_namespace.py):
- canonical namespace/prefix mapping and raw-key codec
- shared by fingerprint registry and identity facade

2. [identity_space.py](/home/mikemol/github/gabion/src/gabion/analysis/foundation/identity_space.py):
- namespaced atom/path interning
- projection helpers (`basis_path`, `prime_product`, `digest_alias`, witness)
- allocation ledger (`seq`, `namespace`, `token`, `atom_id`) for replay/audit

3. [prime_identity_adapter.py](/home/mikemol/github/gabion/src/gabion/analysis/core/prime_identity_adapter.py):
- adapter from `IdentityAllocator` protocol to `PrimeRegistry`
- preserves seed/load and seeded-vs-learned assignment policy behavior

## Reuse vs Deferred Cleanup

Reusable immediately:
- `PrimeRegistry` deterministic assignment and seed payload model.
- Existing namespace key shapes (`type_base`, `ctor:`, `evidence:`, `site:`, `synth:`).
- Existing cache-identity and ASPF witness infrastructure.

Deferred to later phases:
- Unified event envelope across transcript/ASPF/indexed progress streams.
- Stream-wide migration of payload carriers to identity facade outputs.
- Pruning duplicate legacy adapters once envelope and identity surfaces converge.

## Compatibility Notes

- No CLI behavior changes.
- No report schema changes.
- No ASPF semantic model replacement.
- Phase 1 is additive: it exposes a consolidation seam while preserving current
  runtime contracts.
