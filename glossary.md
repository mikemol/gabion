---
doc_revision: 9
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: glossary
doc_role: glossary
doc_scope:
  - repo
  - semantics
  - tooling
doc_authority: normative
doc_requires:
  - README.md
  - CONTRIBUTING.md
  - AGENTS.md
  - POLICY_SEED.md
doc_reviewed_as_of:
  README.md: 58
  CONTRIBUTING.md: 68
  AGENTS.md: 12
  POLICY_SEED.md: 22
doc_commutes_with:
  - POLICY_SEED.md
doc_change_protocol: "POLICY_SEED.md §6"
doc_invariants:
  - rule_of_polysemy
  - tier2_reification
  - tier3_documentation
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

# Glossary (Normative)

> **Glossary Contract (Normative):**
> This glossary defines the semantic typing discipline for the project.
> Any term reused in code, tests, or documentation must conform to exactly one
> glossary entry, declare its axis, state its commutation law, and identify
> what is erased by aliasing or projection.
>
> **Security Contract (Normative Pointer):**
> Execution and CI safety are governed by `POLICY_SEED.md`.
> The semantic obligations in this glossary are enforced only when execution
> complies with that policy. Both contracts are required for validity.
>
> **Dataflow Grammar Invariant (Normative Pointer):**
> The dataflow grammar audit in `POLICY_SEED.md` treats recurring parameter
> bundles as type-level obligations. Any bundle that crosses function
> boundaries must be promoted to a Protocol (dataclass config/local bundle)
> or explicitly documented with a `# dataflow-bundle:` marker. Enforcement is
> via `gabion check`.
>
> **Repository Cross-References (Normative Pointers):**
> `README.md` defines project scope and status.
> `CONTRIBUTING.md` defines workflow guardrails and required checks.
> `AGENTS.md` defines LLM/agent obligations and refusal rules.
>
> **Reserved Notation (Normative):**
> `alias(·)` denotes a bijective renaming of symbols within a signature scope.
> `bundle_id(·)` denotes the canonical identity derived from a signature's
> co-occurrence structure.
> `tier(·, scope)` denotes the evidence-based classification of a bundle in a
> declared observation scope.

## 0. Rule of Polysemy

Polysemy is permitted only when:

1. the meanings lie on orthogonal axes, and
2. any interaction is declared to commute (or declared non-interacting), and
3. there is a test or enforcement obligation for the commutation claim.

If any of (1-3) are absent, the reuse is invalid.

Goal: engineer convergence, not avoid ambiguity.

## 1. Bundle

### Meaning

**Definition:** A set of symbols that appear together in a call signature.

### Axis

**Axis:** Structural (signature grouping / co-occurrence).

### Desired Commutation (Alias Invariance)

Let `B` be a bundle and `alias(·)` a bijective renaming of symbols.

```
bundle_id(alias(B)) = bundle_id(B)
```

### Why We Want This

- bundles express structure, not names
- renames should not create new bundle identity

### Failure Modes

- local aliasing changes bundle identity
- symbol-level renames are treated as structural changes
- bundles cross boundaries without reification or explicit marking

### Normative Rule

> Bundle identity is defined by co-occurrence in a call signature.
> Local aliasing and variable renames must not create new bundle identity.
> Any bundle that crosses a function boundary must be reified into a Protocol
> (dataclass config/local bundle) or explicitly documented with
> `# dataflow-bundle:`. Tier-2 bundles must be promoted before merge (see Tier).

### Erasure

Renaming variables erases the symbol but preserves the bundle identity.

### Test Obligations (to be mapped)

- `gabion check` dataflow grammar audit
- aliasing invariance of `bundle_id`
- stability of `bundle_id` under local renames

---

## 2. Tier

### Meaning

**Definition:** The confidence level of a bundle's existence.

### Axis

**Axis:** Probability (evidence / occurrence-based confidence).

### Inputs

**Scope:** The declared observation window used to count occurrences.

### Mappings (Normative)

- **Tier-1 (explicit):** a Protocol/config dataclass exists for the bundle.
- **Tier-2 (implicit strong):** repeated more than once within scope.
- **Tier-3 (implicit weak):** single occurrence within scope.

### Desired Commutation (Rename Invariance)

Let `B` be a bundle observed in `scope`.

```
tier(alias(B), scope) = tier(B, scope)
```

### Failure Modes

- tier inferred from mixed or undeclared scopes
- "Explicit" applied without a config object
- evidence counts depend on symbol renames
- Tier-2 bundles introduced without Protocol reification

### Normative Rule

> Tier reflects evidence only. It must be derived from config presence and
> occurrence counts within a declared scope.
> Tier-2 bundles must be reified into Protocols before merge.
> Tier-3 bundles must be either documented with `# dataflow-bundle:` or
> reified before merge.

### Erasure

Renaming symbols and observation ordering are erased; only config presence and
occurrence counts within scope contribute to tier.

### Test Obligations (to be mapped)

- `gabion check` Tier-2 enforcement
- tier classification is stable under symbol renames
- tier thresholds match declared scope counts

### Audit Outputs (Normative)

- Graph outputs in Mermaid and/or DOT format.
- Component summaries that list observed bundles by tier.
- A violations list for undocumented or unreified bundles.

---

## 3. Protocol

### Meanings (must be qualified)

- **Protocol (dataflow):** a reified bundle used across function boundaries,
  implemented as a dataclass (config/local bundle) or an explicit
  `# dataflow-bundle:` marker.
- **Protocol (change):** an ordered, enforceable procedure for modifying
  policy (e.g. the change protocol in `POLICY_SEED.md`).

### Axes

- Dataflow typing vs Governance/process control.

### Desired Commutation (Orthogonality)

Statements about Protocol (dataflow) must not imply anything about Protocol
(change), and vice versa. Any coupling must be explicit and justified.

### Failure Modes

- "Protocol" used without qualification
- dataflow Protocol claims used to justify policy changes
- change Protocol steps treated as dataflow typing rules

### Normative Rule

> Qualify the meaning of Protocol in every normative statement.
> In `POLICY_SEED.md` §1.1, "Protocols" refers to Protocol (dataflow).

### Erasure

Protocol naming is erased by aliasing; Protocol identity is determined by the
underlying bundle or the defined change steps.

### Test Obligations (to be mapped)

- `gabion check` enforces Protocol (dataflow) reification of Tier-2 bundles
