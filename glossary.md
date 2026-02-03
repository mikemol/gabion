---
doc_revision: 13
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
  CONTRIBUTING.md: 70
  AGENTS.md: 12
  POLICY_SEED.md: 28
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

**Decision-flow analogs (control-flow axis):**
- **Decision Protocol** ↔ Tier-1 (explicit control schema).
- **Decision Bundle** ↔ Tier-2 (recurring guard structure).
- **Decision Table** ↔ Tier-3 (documented, weak evidence).

These analogs live on the control-flow axis; see §§12–14.

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

---

## 4. Semantic Transport (LSP Boundary)

### Meaning

**Definition:** The semantic payloads exchanged between the CLI and server
(commands, diagnostics, edits) are the meaning-bearing objects. The transport
mechanism used to carry them (serialization, framing, IO) is an implementation
detail that must commute with those meanings.

### Axis

**Axis:** Interface (protocol semantics vs transport mechanics).

### Desired Commutation (Transport Transparency)

Let `M` be a semantic message payload.

```
deserialize(serialize(M)) = M
```

Meaning must be preserved regardless of whether `M` is carried over a pipe,
socket, or passed as an in-memory structure.

### Failure Modes

- tests assert JSON-RPC framing instead of semantic payloads
- CLI logic reimplements analysis instead of delegating via semantic messages
- transport changes appear as semantic changes

### Normative Rule

> The LSP layer is a **semantic boundary**. CLI code constructs payloads and
> delegates execution; server code consumes payloads and returns results.
> Transport mechanics (serialization, framing, IO) are opaque and must not be
> relied on by core logic or tests.

### Erasure

Transport details (headers, framing, stdio vs socket) are erased; only the
message payloads are semantically relevant.

### Test Obligations (to be mapped)

- Unit tests exercise payload construction and server command handlers directly.
- A minimal end-to-end smoke test may cover transport, but must not be required
  for semantic correctness.

---

## 5. Execution Substrate (Interpreter / Env)

### Meaning

**Definition:** The interpreter and environment (mise/venv/system) provide
execution capability but do not define meaning. Given equivalent interpreter
version and declared dependencies, semantic outcomes must be invariant.

### Axis

**Axis:** Execution (environment / substrate).

### Desired Commutation (Substrate Transparency)

Let `P` be the project and `E1`, `E2` be equivalent environments.

```
analysis(P, E1) = analysis(P, E2)
```

### Failure Modes

- tests depend on activation method or venv path
- analysis results change solely due to interpreter wrapper
- policy checks rely on non-repo environment state

### Normative Rule

> Environment selection is a substrate concern. Semantic logic must not
> depend on activation mechanics or venv paths. Repo tooling uses `mise exec`
> to select the interpreter, but meaning is defined by code + declared deps.

### Erasure

Activation method, venv location, and shell wrappers are erased.

### Test Obligations (to be mapped)

- Tests use dependency injection or fixtures, not environment coupling.
- CI and local runs are expected to agree when inputs match.

---

## 6. Dependency Resolution (Supply Chain)

### Meaning

**Definition:** Dependency resolution supplies third-party code; it must not
change semantic outcomes beyond the declared dependency set.

### Axis

**Axis:** Supply chain (resolution / build inputs).

### Desired Commutation (Resolution Determinism)

Let `D` be the declared dependency set.

```
resolve(D) = resolve(D)  (deterministic given the same declarations)
```

### Failure Modes

- build outcomes change due to resolver nondeterminism
- semantic behavior depends on transient upstream changes
- trusted workflows execute with undeclared dependencies

### Normative Rule

> Semantic outcomes must depend only on declared dependencies. If resolver
> nondeterminism can affect meaning, it is a policy failure and must be
> addressed by tightening declarations or pinning inputs.

### Erasure

Resolver ordering and network timing are erased.

### Test Obligations (to be mapped)

- Build/test workflows use the declared dependency set.

---

## 7. Repository Control Plane (Git / Hosting)

### Meaning

**Definition:** Git and hosting settings enforce workflow constraints but do
not define semantic meaning. Meaning is derived from the repo content and its
governance documents.

### Axis

**Axis:** Governance (control plane / history).

### Desired Commutation (Control-Plane Independence)

```
meaning(tree, docs) is independent of hosting UI mechanics
```

### Failure Modes

- semantics encoded only in hosting settings
- history mechanics treated as semantic change without doc updates
- policy drift hidden in untracked settings

### Normative Rule

> Governance meaning must be reified in tracked documents. Hosting settings
> enforce but do not define meaning.

### Erasure

PR numbers, UI labels, and hosting metadata are erased.

### Test Obligations (to be mapped)

- Policy checks validate workflow constraints in-repo.

---

## 8. CI Runner Substrate

### Meaning

**Definition:** CI runners provide execution for checks; they witness outcomes
but do not define semantics.

### Axis

**Axis:** Validation (execution surface).

### Desired Commutation (Runner Equivalence)

```
verdict(P, runner_a) = verdict(P, runner_b)  (given identical inputs)
```

### Failure Modes

- CI-only logic that is not reproducible locally
- runner-specific behavior changes semantic outcomes
- self-hosted constraints violated by untrusted inputs

### Normative Rule

> CI is a witness. It must not introduce semantic divergence from local runs.
> Runner choice changes risk posture, not meaning.

### Erasure

Runner labels and host identity are erased.

### Test Obligations (to be mapped)

- CI and local checks must agree on the same inputs.

---

## 9. Parsing Layer (AST / CST)

### Meaning

**Definition:** Parsing produces structural representations of code. Analysis
meaning must be invariant under formatting and trivia changes.

### Axis

**Axis:** Syntax (representation / meaning).

### Desired Commutation (Trivia Erasure)

```
analysis(parse(code)) = analysis(parse(format(code)))
```

### Failure Modes

- analysis depends on whitespace or comment placement
- different parsers yield divergent semantics for equivalent code

### Normative Rule

> Formatting and trivia are erased. Semantic analysis depends on structure,
> not text layout.

### Erasure

Whitespace, comments (unless explicitly semantically marked) are erased.

### Test Obligations (to be mapped)

- Formatting changes must not alter analysis outcomes.

---

## 10. Docflow Metadata (Front Matter)

### Meaning

**Definition:** Front-matter metadata encodes governance state; formatting and
ordering do not affect its meaning.

### Axis

**Axis:** Documentation (metadata / governance).

### Desired Commutation (Metadata Canonicalization)

```
canonicalize(frontmatter) is invariant under key order and whitespace
```

### Failure Modes

- governance semantics depend on YAML formatting
- doc-revision meaning inferred from prose instead of front matter

### Normative Rule

> Front-matter is the canonical metadata. Formatting and ordering are erased.
> Review chains are enforced via `doc_reviewed_as_of`.

### Erasure

Key order, spacing, and YAML style are erased.

### Test Obligations (to be mapped)

- Docflow audits treat front-matter canonically.

---

## 11. Visualization Layer (Mermaid / DOT)

### Meaning

**Definition:** Visualizations are projections of analysis results; they do
not define semantics.

### Axis

**Axis:** Representation (projection).

### Desired Commutation (Projection Erasure)

```
analysis(P) = analysis(P)  (independent of rendering)
```

### Failure Modes

- analysis depends on rendering output
- visualization format changes are treated as semantic change

### Normative Rule

> Visualization is a projection. Rendering changes must not alter analysis
> meaning or policy decisions.

### Erasure

Layout choices, rendering engines, and styling are erased.

### Test Obligations (to be mapped)

- Graph outputs are validated as projections, not inputs.

---

## 12. Decision Table (Control-Flow Documentation)

### Meaning

**Definition:** A tabular mapping from guard predicates to outcomes for a
control-flow decision space.

**Tier analog:** Tier-3 (implicit weak). Documented, not enforced.

### Axis

**Axis:** Control-flow (documentation / presentation).

### Desired Commutation (Row Order + Predicate Renaming)

Reordering rows or renaming predicates **without changing semantics** must not
change the decision table’s meaning.

### Failure Modes

- Decision table diverges from code.
- Overlapping rows without declared precedence.
- Missing rows for reachable outcomes.

### Normative Rule

> Decision tables are Tier-3 evidence only. They must declare scope and be
> re-reviewed when the corresponding code changes.

### Erasure

Row order, formatting, and whitespace are erased.

### Test Obligations (to be mapped)

- Table ↔ code alignment check (spot-test).
- At least one negative case (false positive avoidance).

---

## 13. Decision Bundle (Control-Flow Structural)

### Meaning

**Definition:** A recurring cluster of guard predicates or branch patterns that
can be centralized into a single decision map or dispatcher without changing
behavior.

**Tier analog:** Tier-2 (implicit strong). Repeated, centralized, but not
formally declared.

### Axis

**Axis:** Control-flow (structural grouping).

### Desired Commutation (Guard Renaming)

Renaming guard predicates or reordering equivalent checks must not change the
bundle identity.

### Failure Modes

- Same decision logic duplicated in multiple functions.
- Bundles are implied but never centralized.
- Centralization changes behavior due to hidden precedence.

### Normative Rule

> Repeated decision logic must be centralized into a decision bundle before
> merge, or explicitly documented as Tier-3 (decision table).

### Erasure

Predicate names and order are erased if semantics are preserved.

### Test Obligations (to be mapped)

- Positive/negative cases for a centralized guard map.
- Metamorphic test showing invariant behavior under guard reordering.

---

## 14. Decision Protocol (Control-Flow Explicit)

### Meaning

**Definition:** A declared schema for decision logic whose execution is derived
or validated against the schema.

**Tier analog:** Tier-1 (explicit). Enforced and validated.

### Axis

**Axis:** Control-flow (governance / explicit schema).

### Desired Commutation (Spec ↔ Implementation)

Implementations must commute with the declared decision schema:
changes to the schema must be reflected in execution, and execution must not
exceed the schema.

### Failure Modes

- Ad-hoc branches outside the declared protocol.
- Schema exists but is not validated or enforced.
- Protocol references ambiguous or overlapping guard cases.

### Normative Rule

> Critical decision logic must be expressed as a Decision Protocol when it
> crosses module boundaries or affects external behavior.

### Erasure

Serialization format, transport, and layout are erased if the schema is
equivalent.

### Test Obligations (to be mapped)

- Schema validation (reject invalid decision states).
- Positive/negative/edge cases derived from the protocol.
