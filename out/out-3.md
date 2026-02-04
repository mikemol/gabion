---
doc_revision: 3
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: out_3
doc_role: hypothesis
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
  glossary.md: 17
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

**Algebraic Structural Prime Fingerprints (ASPF)** are not hashes.
They are **packed-forest labels**.

When dimensionalized (base, constructor, provenance, synth), ASPF provides a
**compact, exact semantic carrier** equivalent in expressive power to a
**Shared Packed Parse Forest (SPPF)**.

The value proposition is twofold:

1. **Compression** of structural space without erasing semantic information.
2. **Semantic conservation** across refactorings, enforced by algebraic
   invariants rather than syntactic normalization.

This is not metaphorical: ASPF establishes a correspondence between
**derivation structure** (SPPF) and a **conserved algebraic carrier**.

---

## 1. Core Construction (ASPF in Practice)

ASPF is a **multiset algebra** over canonicalized semantic atoms:

* Canonical type keys are mapped to unique primes.
* Composite structures are represented as products of primes.
* Multiplicity is preserved; the carrier is multiset-exact, not set-based.

Semantic structure is decomposed into orthogonal dimensions:

* **Base dimension**
  Canonical atomic semantics.
* **Constructor dimension**
  Structural operators (`list`, `dict`, `union`, etc.).
* **Provenance dimension**
  Derivational origin (how a composite was constructed).
* **Synth dimension (`synth@k`)**
  Deterministic interning of repeated composites to bound entropy.

Each dimension carries a dual representation:

* an **exact prime product**, and
* a **bitmask carrier** for fast overlap tests.

The bitmask is a filter; the prime product is authoritative.

---

## 2. Determinism as a Semantic Invariant

ASPF is only a durable semantic carrier if its registries are
**deterministically seeded**:

* Canonical keys are collected, sorted, and interned prior to assignment.
* Prime and bit positions are stable across runs and environments.
* Synth registries are versioned, serializable, and reloadable.

Determinism is not an optimization.
It is a **semantic invariant** enabling:

* reproducible audits,
* stable baselines,
* meaningful diffs,
* and conserved interpretation of fingerprints over time.

---

## 3. SPPF ↔ ASPF Correspondence

| SPPF Concept    | ASPF Construct                            |
| --------------- | ----------------------------------------- |
| Terminal        | Base prime                                |
| Nonterminal     | Synthesized prime (`synth@k`)             |
| Packed node     | Composite carrier with provenance         |
| Derivation path | Provenance dimension                      |
| Shared subtree  | Reused prime factors / shared synth atoms |

ASPF therefore encodes **packed derivations**, not flattened syntax.
Distinct derivations may collapse to the same base carrier while remaining
distinguishable via provenance.

---

## 4. Invariants (Design Obligations)

These invariants define the **exactness contract** of the ASPF carrier:

1. **Base conservation**
   All valid refactorings preserve the base-dimension product.
2. **Constructor coherence**
   Constructor carriers commute with base semantics and are not silently erased.
3. **Provenance alignment**
   Distinct provenance must correspond to admissible derivation paths.
4. **Synth tail soundness**
   Each synthesized prime stores a reversible, canonical tail.
5. **Carrier soundness**
   Bitmask overlap is a necessary (not sufficient) condition for algebraic
   overlap; inconsistencies are detected via gcd/mask checks.

Violation of any invariant is **not approximated away**.
It is surfaced explicitly as semantic instability.

---

## 5. Current Status in Gabion

The implementation already realizes the core of this hypothesis:

* canonicalization and prime registry
* constructor registry
* dimensional fingerprints
* synth registry with reversible tails
* loadable/persisted synth bases
* provenance artifacts and soundness checks

What remains is not capability, but **formalization**:
treating the carrier semantics and provenance artifacts as first-class,
SPPF-equivalent evidence.

---

## 6. Proposed Next Actions (High Leverage)

1. **Glossary formalization**
   Add glossary entries for:
   ASPF, fingerprint dimension, provenance carrier, synth tail,
   packed derivation.
2. **Report-level provenance view**
   Emit a compact, structured derivation summary equivalent to an SPPF node view.
3. **Determinism enforcement tests**
   CI tests asserting stable prime/bit assignment and tail reversibility across
   runs and reloads.

These steps operationalize the equivalence without expanding the engine.

---

## 7. Minimal Success Criteria

* Provenance is surfaced as a structured, human-readable artifact.
* ASPF semantics and invariants are explicitly defined in the glossary.
* Deterministic seeding and reversible synthesis are enforced and tested.

---

**If these criteria hold, ASPF may be treated as a semantic carrier
equivalent to an SPPF node-labeling scheme**, rather than as a heuristic or
lossy compression mechanism.
