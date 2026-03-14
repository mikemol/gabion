---
doc_revision: 1
doc_id: identity_zone_grammar
doc_role: architecture_note
doc_scope:
  - tooling
  - policy_substrate
doc_authority: informative
doc_requires:
  - docs/ttl_kernel_semantics.md#ttl_kernel_semantics
  - in/lg_kernel_ontology_cut_elim-1.ttl
---

# Identity Zone Grammar

The hierarchical identity grammar treats scanner, hotspot, and planning-chart
surfaces as origin-local zones whose identities remain immutable once interned.

The categorical reading is:

- `lg:KernelCongruence` corresponds to the erased-vs-retained structure recorded
  on an identity-zone quotient.
- `lg:QuotientProjection` corresponds to the explicit scanner-to-hotspot face
  projection.
- `lg:ReflectionFunctor` corresponds to the deterministic representative chooser
  back into the richer source fiber when that chooser is lawful.
- `lg:AdjointPair` corresponds to the reified quotient/section pairing when the
  implementation can exhibit the relevant law checks.
- Fiber/comma-style provenance corresponds to the retained source-member fiber
  over one target identity.

The implementation keeps lower-zone primes immutable and interns cross-zone
structure only in the higher-order grammar. This avoids string-identified joins
and avoids rewriting existing local prime assignments.

In v1:

- scanner is the upstream rich zone
- hotspot is the first explicit quotient zone
- planning-chart is interned into the same grammar internally
- planning/ranking behavior does not yet consume identity-grammar evidence

This gives the repo a basis for a hierarchical identity grammar without
changing current planning semantics.
