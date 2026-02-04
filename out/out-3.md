---
doc_revision: 2
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: out_3
"doc_role": "hypothesis"
doc_scope:
  - repo
  - tooling
  - research
  - semantics
doc_authority: informative
doc_requires:
  - POLICY_SEED.md
  - glossary.md
  - CONTRIBUTING.md
  - README.md
doc_reviewed_as_of:
  POLICY_SEED.md: 28
  glossary.md: 16
  CONTRIBUTING.md: 71
  README.md: 58
doc_change_protocol: "POLICY_SEED.md §6"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

# Hypothesis: ASPF as an SPPF-Equivalent Semantic Carrier

## 0. Thesis
**Algebraic Structural Prime Fingerprints (ASPF)** are not hashes; they are
**packed‑forest labels**. When dimensionalized (base, constructor, provenance,
synth), ASPF provides a compact but exact representation of structural meaning
equivalent to a Shared Packed Parse Forest (SPPF).

The value proposition is twofold:
1) **Compression** of structural space without erasing semantic detail.
2) **Conserved semantics** across refactorings via algebraic invariants.

This is stronger than metaphor: it is a mapping between a semantic derivation
structure (SPPF) and a conserved algebraic carrier (ASPF).

## 1. Core Construction (ASPF in Practice)
ASPF is a **multiset algebra** over canonicalized type atoms:
- Canonical type keys map to unique primes.
- Composite structures map to prime products.
- Multiplicity is preserved (multiset, not set).

Dimensionalization adds orthogonal carriers:
- **Base dimension**: canonical atomic semantics.
- **Constructor dimension**: structural operators (`list`, `dict`, `union`, etc.).
- **Provenance dimension**: derivational origin (how the composite arose).
- **Synth dimension**: entropy‑control for repeated composites (`synth@k`).

The hybrid representation (prime product + bitmask) accelerates membership
tests without sacrificing exactness.

## 2. Determinism as a Semantic Invariant
ASPF only becomes a durable carrier when its registry is **deterministically
seeded**:
- Canonical keys are collected, sorted, and interned before use.
- Prime and bit assignments are stable across runs.

This is not just performance: determinism is **semantic stability** and enables
baseline diffs, audit trails, and reproducible invariants.

## 3. SPPF ↔ ASPF Correspondence

| SPPF Concept     | ASPF Construct                                |
|------------------|-----------------------------------------------|
| Terminal         | Base prime                                    |
| Nonterminal      | Synthesized prime (`synth@k`)                 |
| Packed node      | Composite prime with provenance carrier       |
| Derivation path  | Provenance dimension                           |
| Shared subtree   | Reused prime factors across products           |

Thus, ASPF encodes **packed derivations**; it is not a lossy hash. Distinct
derivations can share a base product while retaining provenance differences.

## 4. Invariants (Design Obligations)
These define the “exactness” guarantees of the ASPF carrier:
1) **Base conservation**: valid refactors preserve base product.
2) **Constructor coherence**: constructor dimension commutes with base.
3) **Provenance alignment**: provenance differences must be explainable by
   accepted derivation paths.
4) **Synth tail soundness**: synthesized primes must store reversible tails.
5) **Carrier soundness**: bitmask and prime products must agree (gcd/mask checks).

If any invariant fails, we do not “round.” We emit a warning and treat the
fingerprint as semantically unstable.

## 5. Where This Lives in Gabion Today
The implementation already exposes the core of this hypothesis:
- canonicalization + prime registry
- constructor registry
- dimensional fingerprints
- synth registry payload + reversible tails
- soundness checks + match reporting

What remains is to **formalize the carrier semantics** and make provenance a
first‑class reporting artifact (SPPF-equivalent derivation evidence).

## 6. Proposed Next Actions (High Leverage)
1) **Glossary entries for ASPF carriers**  
   Add terms for: ASPF, fingerprint dimension, provenance carrier, synth tail,
   packed derivation. (This anchors the semantics.)
2) **Report‑level provenance view**  
   Emit a compact SPPF‑style derivation summary in audit reports.
3) **Determinism tests**  
   Explicit CI tests for stable prime/bit assignment and reversible tails.

These three steps operationalize the equivalence without expanding the engine.

## 7. Minimal Success Criteria
- Provenance dimension is surfaced and human‑readable in reports.
- Glossary explicitly defines ASPF semantics + invariants.
- Deterministic seeding is enforced and regression‑tested.

---

If the above holds, we can treat ASPF as **a semantic carrier equivalent to an
SPPF node labeling scheme**, not just a heuristic compression trick.
