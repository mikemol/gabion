---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: structural_ambiguity_gradient_note
doc_role: note
doc_scope:
  - repo
  - projection
  - policy
  - dataflow
doc_authority: informative
doc_requires:
  - glossary.md#contract
  - docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc
doc_reviewed_as_of:
  glossary.md#contract: 44
  docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc: 111
doc_review_notes:
  glossary.md#contract: "Reviewed glossary rev44; ambiguity, witness, and carrier terminology remain aligned with this note."
  docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc: "Reviewed the implementation RFC rev111; this note records an observed future-query shape without changing the RFC's current cutover commitments."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="structural_ambiguity_gradient_note"></a>
# Structural Ambiguity Gradient Note

This note records an observed challenge from the ongoing projection/cutover
work. It does not propose an implementation in this correction unit.

## Observation

Data structure transformation is its own ambiguity gradient.

The important pattern is that structural ambiguity can appear before any true
whole-Gabion I/O boundary exists. In those cases, the local pressure often
looks like:

- one data fiber carries multiple partially-factored internal shapes
- a downstream consumer wants only one of those shapes
- a tempting local fix is to loosen the carrier or normalize into a generic
  mapping

That local fix is usually wrong. It hides the ambiguity instead of locating the
earliest lawful seam where the ambiguity should be discharged.

## Working Hypothesis

Structural ambiguity should be pushed upstream rather than normalized away
mid-fiber.

The likely long-term mechanism is a seam-meet analysis expressed as a policy
query over structural identifiers. The intended question is:

> given one data fiber and several competing structural factorizations, what is
> the greatest common denominator shape that can be made explicit earlier?

That search is recursive. It should walk the structural factorization of the
fiber, compare the available seams, and find the meet where the shared carrier
can be reified without widening compatibility surfaces downstream.

## Why This Matters

This is one of the first concrete higher-value uses for structural identifiers
beyond grouping and reporting:

- identify repeated shape divergence along the same fiber
- compare candidate upstream seams for common structure
- ask the policy layer where ambiguity should move, rather than where it is
  currently tolerated

In other words, the policy layer could eventually answer not just "is this
boundary valid?" but also "where is the earliest valid structural seam for this
fiber?"

## Non-goals

This note does not:

- implement seam-meet analysis
- change the current ambiguity gate
- authorize generic JSON bridges or loose internal payload carriers

The current rule remains: JSON belongs at RPC/file I/O boundaries, and strict
internal carriers remain the default inside Gabion.
