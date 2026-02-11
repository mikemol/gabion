---
doc_revision: 42
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: glossary
doc_role: glossary
doc_scope:
  - repo
  - semantics
  - tooling
doc_authority: normative
doc_dependency_projection: glossary_root
doc_requires: []
doc_reviewed_as_of:
doc_review_notes:
doc_sections:
  contract: 1
  rule_of_polysemy: 1
  bundle: 1
  tier: 1
  decision_table: 1
  decision_bundle: 1
  decision_protocol: 1
  decision_surface: 1
  value_encoded_decision: 1
  aspf: 1
  forest: 1
  suite_site: 1
  hash_consing: 1
  never_throw_exception_protocol: 1
  deadness_witness: 1
  coherence_witness: 1
  rewrite_plan: 1
  exception_path: 1
  handledness_witness: 1
  exception_obligation: 1
  attribute_carrier: 1
  attribute_transport: 1
  evidence_id: 1
  witness: 1
  evidence_surface: 1
  evidence_dominance: 1
  equivalent_witness: 1
  test_obsolescence_projection: 1
  test_evidence_suggestions_projection: 1
  evidence_key: 1
  ambiguity_set: 1
  partition_witness: 1
  annotation_drift: 1
  grothendieck_analysis: 1
  self_review: 1
doc_commutes_with:
  - POLICY_SEED.md#policy_seed
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_invariants:
  - rule_of_polysemy
  - tier2_reification
  - tier3_documentation
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="contract"></a>
# Glossary (Normative)

> **Glossary Contract (Normative):**
> This glossary defines the semantic typing discipline for the project.
> Any term reused in code, tests, or documentation must conform to exactly one
> glossary entry, declare its axis, state its commutation law, and identify
> what is erased by aliasing or projection.
>
> **Security Contract (Normative Pointer):**
> Execution and CI safety are governed by `POLICY_SEED.md#policy_seed`.
> The semantic obligations in this glossary are enforced only when execution
> complies with that policy. Both contracts are required for validity.
>
> **Dataflow Grammar Invariant (Normative Pointer):**
> The dataflow grammar audit in `POLICY_SEED.md#policy_seed` treats recurring parameter
> bundles as type-level obligations. Any bundle that crosses function
> boundaries must be promoted to a Protocol (dataclass config/local bundle)
> or explicitly documented with a `# dataflow-bundle:` marker. Enforcement is
> via `gabion check`.
>
> **Repository Cross-References (Normative Pointers):**
> `README.md#repo_contract` defines project scope and status.
> `CONTRIBUTING.md#contributing_contract` defines workflow guardrails and required checks.
> `AGENTS.md#agent_obligations` defines LLM/agent obligations and refusal rules.
>
> **Reserved Notation (Normative):**
> `alias(·)` denotes a bijective renaming of symbols within a signature scope.
> `bundle_id(·)` denotes the canonical identity derived from a signature's
> co-occurrence structure.
> `tier(·, scope)` denotes the evidence-based classification of a bundle in a
> declared observation scope.

<a id="rule_of_polysemy"></a>
## 0. Rule of Polysemy

Polysemy is permitted only when:

1. the meanings lie on orthogonal axes, and
2. any interaction is declared to commute (or declared non-interacting), and
3. there is a test or enforcement obligation for the commutation claim.

If any of (1-3) are absent, the reuse is invalid.

Goal: engineer convergence, not avoid ambiguity.

<a id="bundle"></a>
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

<a id="tier"></a>
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

## 3. Subsystem Interface (Template)

### Meaning

**Definition:** A named, minimal boundary (module + exported functions/Protocol)
that reifies a recurring code-flow structure. It is the code-flow analogue of a
bundle: when the same parameter clusters and decision surfaces recur across
functions, a subsystem interface captures that common structure as a reusable
unit.

### Axis

**Axis:** Structural (code-flow boundary / interface contract).

### Desired Commutation (Refactor Invariance)

Let `S` be a subsystem interface and `refactor_internal(·)` be any refactor that
preserves the interface contract (signatures, bundles, and documented decision
surface).

```
interface_id(refactor_internal(S)) = interface_id(S)
```

### Failure Modes

- recurring param clusters remain scattered across modules
- interface identity changes with internal refactors
- decision surfaces are not reflected at the interface boundary

### Normative Rule

> Recurring parameter clusters and decision surfaces that cross module
> boundaries must be candidates for a subsystem interface template
> (module + exported functions/Protocol). If adopted, the interface must
> be the sole entry point for those bundles in the affected scope.

### Erasure

Internal function placement, helper naming, and file layout are erased; only the
interface contract and its declared bundles remain.

### Test Obligations (to be mapped)

- Interface identity stable under internal refactors.
- Bundles and decision surfaces remain visible at the interface boundary.
- All consumers route through the interface (no parallel entry points).

---

## 3.1 Higher‑Order Bundle (2‑Path)

### Meaning

**Definition:** A parameter set that recurs across multiple functions within a
declared scope (module or subsystem), forming a second‑order co‑occurrence
signal. It is a bundle‑of‑bundles: repeated co‑occurrence of the *same* param
set across multiple functions.

### Axis

**Axis:** Structural (multi‑function co‑occurrence / module cohesion).

### Desired Commutation (Order & Alias Invariance)

Let `H` be a higher‑order bundle and `alias(·)` a bijective renaming of symbols.

```
hbundle_id(alias(H), scope) = hbundle_id(H, scope)
```

Function ordering and naming must not change the higher‑order identity.

### Failure Modes

- repeated param sets remain scattered across modules
- higher‑order bundles are detected but not reified as interfaces
- bundle identity changes due to function renames or ordering

### Normative Rule

> Higher‑order bundles observed above declared thresholds must be treated as
> candidates for a **Subsystem Interface (Template)**. If adopted, the
> interface becomes the canonical boundary for those bundles in the scope.

### Erasure

Function names, ordering, and file layout are erased; only the repeated param
set and declared scope counts remain.

### Test Obligations (to be mapped)

- Consolidation audit surfaces higher‑order bundles at configured thresholds.
- Higher‑order bundle identity stable under function renames and reordering.

---

## 3. Protocol

### Meanings (must be qualified)

- **Protocol (dataflow):** a reified bundle used across function boundaries,
  implemented as a dataclass (config/local bundle) or an explicit
  `# dataflow-bundle:` marker.
- **Protocol (change):** an ordered, enforceable procedure for modifying
  policy (e.g. the change protocol in `POLICY_SEED.md#policy_seed`).

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
> In `POLICY_SEED.md#policy_seed` §1.1, "Protocols" refers to Protocol (dataflow).

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

<a id="decision_table"></a>
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

<a id="decision_bundle"></a>
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

<a id="decision_protocol"></a>
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

---

<a id="decision_surface"></a>
## 15. Decision Surface (Control-Flow Boundary)

### Meaning

**Definition:** An input, predicate, or derived value that bifurcates behavior
and therefore defines a control boundary (e.g., API surface decisions).

### Axis

**Axis:** Control-flow (semantic boundary).

### Desired Commutation (Surface Equivalence)

Equivalent decision expressions (e.g., `if x` vs `if x is True`) must not change
the identity of the decision surface.

### Failure Modes

- Decision surfaces appear deep in internal helpers instead of boundary layers.
- Tier-2/Tier-3 decision variables used below permitted scope.
- Decision logic hidden by refactors (loss of surface visibility).

### Normative Rule

> Decision surfaces must be documented or elevated to the appropriate boundary.
> Tier rules for decision surfaces follow the Decision Table/Bundle/Protocol
> ladder (Tier-3/2/1 respectively).

### Erasure

Predicate naming and formatting are erased if semantics are preserved.

### Test Obligations (to be mapped)

- Boundary placement checks for declared decision surfaces.

---

<a id="value_encoded_decision"></a>
## 16. Value-Encoded Decision (Branchless Control)

### Meaning

**Definition:** Control-flow encoded in arithmetic or bitwise expressions rather
than explicit branches (e.g., `min/max`, masks, boolean arithmetic).

### Axis

**Axis:** Control-flow (algebraic encoding).

### Desired Commutation (Branchless ↔ Branched)

Replacing value-encoded control with an equivalent `if`/`else` must not change
semantic classification or tier enforcement.

### Failure Modes

- Decision audits miss algebraic control paths.
- Tier violations bypassed due to branchless encoding.
- Algebraic rewrites change behavior (non-equivalence).

### Normative Rule

> Value-encoded decisions are decision surfaces and inherit all tier rules.
> Detection must be semantics-driven, not syntax-only.

### Erasure

Choice of algebraic encoding is erased if behavior is equivalent.

### Test Obligations (to be mapped)

- Spot checks that branchless forms are detected as decision surfaces.

---

## 17. Structural Snapshot (Audit Artifact)

### Meaning

**Definition:** A canonical, serialized representation of an analysis result
(e.g., factorization tree, bundle/tier listings) suitable for diffing.

### Axis

**Axis:** Evidence (audit artifact).

### Desired Commutation (Canonicalization)

Snapshot contents must be invariant to ordering, formatting, and stable renames.

### Failure Modes

- Non-canonical ordering causing noisy diffs.
- Snapshot treated as semantic source of truth (instead of audit artifact).
- Missing glossary/tier context in snapshots.

### Normative Rule

> Snapshots must be canonical and treated as evidence only. Baselines may
> allowlist existing violations but must be explicit and reviewed.

### Erasure

Formatting and serialization choices are erased.

### Test Obligations (to be mapped)

- Snapshot stability under deterministic re-runs.

---

## 18. Structural Diff (Audit Delta)

### Meaning

**Definition:** A comparison between two structural snapshots to identify new
bundles, tier shifts, or violations.

### Axis

**Axis:** Evidence (change detection).

### Desired Commutation (Order Independence)

Diff results must depend only on snapshot contents, not creation order.

### Failure Modes

- Diffs driven by non-canonical snapshot noise.
- Baseline updates performed implicitly in CI.

### Normative Rule

> Structural diffs are used to detect regressions; baseline updates are manual
> and must be reviewed.

### Erasure

Diff presentation is erased; only semantic deltas matter.

### Test Obligations (to be mapped)

- Diff detects new bundles and tier changes in fixtures.

---

## 19. Structural Metrics (Audit Summary)

### Meaning

**Definition:** Aggregate statistics derived from structural snapshots
(bundle counts, tier counts, violations).

### Axis

**Axis:** Evidence (summary reporting).

### Desired Commutation (Derivation Invariance)

Metrics must be derived deterministically from snapshots.

### Failure Modes

- Metrics computed from non-canonical inputs.
- Metrics treated as semantic truth rather than indicators.

### Normative Rule

> Metrics are advisory summaries and must be traceable to snapshots.

### Erasure

Presentation format and dashboard tooling are erased.

### Test Obligations (to be mapped)

- Metric totals reconcile with snapshot contents.

---

## 20. Structural Fingerprint (Algebraic Encoding)

### Meaning

**Definition:** A canonical algebraic encoding of a bundle or type structure
(e.g., prime products or bitmask fingerprints).

**Multiplicity note (normative):** By default, fingerprints are **multiset**
encodings—duplicate type keys multiply the same prime multiple times. Any
set‑like projection that erases multiplicity must be explicit and treated as
an erasure (not the default identity).

### Axis

**Axis:** Structural (algebraic identity).

### Desired Commutation (Order + Alias Invariance)

Permuting fields or renaming aliases must not change the fingerprint.

### Failure Modes

- Collisions or lossy encodings without declaration.
- Fingerprints diverge from glossary-defined bundle identity.

### Normative Rule

> Fingerprints must be declared invertible or explicitly marked as lossy.
> Glossary mappings must state whether multiplicity is preserved.
> Set‑like fingerprints are permitted only as explicit projections that erase
> multiplicity; multiset fingerprints remain the canonical encoding.

### Erasure

Ordering and superficial naming are erased.

### Test Obligations (to be mapped)

- Fingerprint equality for equivalent structures.

---

## 21. Invariant Proposition (Dependent Constraint)

### Meaning

**Definition:** A declared relationship between bundle fields or tree nodes
(e.g., length equality, ordering, alignment).

### Axis

**Axis:** Semantic (constraint).

### Desired Commutation (Refactor Preservation)

Refactors must preserve declared invariants.

### Failure Modes

- Invariants lost during synthesis.
- Constraints inferred but not recorded.

### Normative Rule

> Extracted invariants must be preserved by refactors and surfaced in
> synthesis outputs when available.

### Erasure

Constraint serialization and syntax are erased if meaning is preserved.

### Test Obligations (to be mapped)

- Property tests or proofs aligned with invariant propositions.

---

## 22. Structural Lemma (Reuse Candidate)

### Meaning

**Definition:** A reusable subtree or structural fragment extracted from the
factorization tree to reduce duplication.

### Axis

**Axis:** Synthesis (reuse).

### Desired Commutation (Factor ↔ Expand)

Factoring a lemma out and expanding it inline must commute.

### Failure Modes

- Lemma extraction changes semantics.
- Reuse suggested without glossary alignment where appropriate.

### Normative Rule

> Lemma suggestions are advisory unless promoted to explicit refactor plans.

### Erasure

Lemma naming and placement are erased if semantics are preserved.

### Test Obligations (to be mapped)

- Equivalence checks between factored and unfactored forms.

---

<a id="aspf"></a>
## 23. ASPF (Algebraic Structural Prime Fingerprint)

### Meaning

**Definition:** A dimensional fingerprint system where structural meaning is
encoded as prime products across orthogonal carriers (base, constructor,
provenance, synth). ASPF is treated as a **packed-forest label**, not a hash.

### Axis

**Axis:** Structural + semantic (packed‑derivation carrier).

### Desired Commutation (Derivation Equivalence)

Equivalent derivations may commute in **base/constructor** space while
remaining distinct in **provenance** space. Commutation is permitted only when
the provenance carrier records the equivalence class explicitly.

### Failure Modes

- ASPF used as a lossy hash without provenance semantics.
- Non‑deterministic prime assignment causing unstable meanings.

### Normative Rule

> ASPF must be deterministic and reversible at the base/constructor layer.
> Provenance must remain inspectable and must not be erased by default.

### Erasure

Formatting and ordering of input types are erased; provenance is **not**.

**Bitmask note (normative):** bitmask carriers are filters only; prime products
remain authoritative for equivalence.

### Test Obligations (to be mapped)

- Deterministic seeding yields stable fingerprints across runs.
- Provenance carrier is emitted when requested.

---

## 24. Fingerprint Dimension (Carrier)

### Meaning

**Definition:** One orthogonal carrier within ASPF (e.g., base, constructor,
provenance, synth). Dimensions are algebraically independent but jointly
constrained.

### Axis

**Axis:** Structural (orthogonal carriers).

### Desired Commutation (Carrier Independence)

Operations in one dimension must not mutate another dimension, except via
explicit synthesis rules (e.g., synth tail attachment).

### Failure Modes

- Cross‑dimension contamination (e.g., base keys leaking into provenance).
- Carrier soundness violated (mask/product mismatch).

### Normative Rule

> Each dimension must preserve its own algebraic identity and pass carrier
> soundness checks against its peers.

### Erasure

Internal bit positions are erased; carrier identity is retained.

### Test Obligations (to be mapped)

- Carrier soundness checks pass for non‑empty dimensions.

---

## 25. Provenance Carrier

### Meaning

**Definition:** The ASPF dimension that encodes **how** a composite structure
arose (packed derivation path).

### Axis

**Axis:** Semantic (derivation).

### Desired Commutation (Derivation Transparency)

Semantically equivalent derivations may commute **only if** provenance is
explicitly recorded and preserved.

### Failure Modes

- Provenance dropped while treating ASPF as a complete semantic carrier.
- Packed derivations conflated without evidence.

### Normative Rule

> Provenance is mandatory when ASPF is used for equivalence claims.
> Loss of provenance must be declared as an erasure.

### Erasure

Provenance may be erased only in explicitly lossy projections.

### Test Obligations (to be mapped)

- Provenance summary is emitted in audit reports when enabled.

---

## 26. Synth Tail (Entropy Control)

### Meaning

**Definition:** A reversible mapping from a synthesized prime to the composite
tail it represents (the “definition” of a synthesized carrier).

### Axis

**Axis:** Structural (compression).

### Desired Commutation (Fold ↔ Unfold)

Synth tails must commute with expansion: unfolding a synthesized prime yields
the original composite tail.

### Failure Modes

- Synth primes without tails (non‑reversible compression).
- Tail mismatch across runs (non‑deterministic seeding).

### Normative Rule

> Every synthesized prime must carry a tail that is reversible and stable.

### Erasure

Synth tail ordering is erased; tail identity is preserved.

### Test Obligations (to be mapped)

- Tail round‑trip tests for synth registry payloads.

---

## 27. Packed Derivation (SPPF Node)

### Meaning

**Definition:** A shared derivation node in an SPPF; in Gabion, represented by
an ASPF fingerprint plus provenance metadata.

### Axis

**Axis:** Structural + semantic (packed derivation).

### Desired Commutation (Share ↔ Expand)

Sharing a derivation node must commute with expanding it into its derivations
without semantic loss.

### Failure Modes

- Packed nodes treated as opaque hashes.
- Loss of derivation identity during refactor.

### Normative Rule

> Packed derivations must be inspectable and traceable to their components.

### Erasure

Syntactic formatting is erased; derivation identity is not.

### Test Obligations (to be mapped)

- Packed derivations are visible in reports or JSON artifacts.

---

<a id="never_throw_exception_protocol"></a>
## 28. Never‑Throw Exception Protocol

### Meaning

**Definition:** A named exception type (e.g., `gabion.exceptions.NeverRaise`)
used to mark a branch as *semantically unreachable*. Raising it is a static
analysis signal, not a runtime control path.

### Axis

**Axis:** Control‑flow + error boundary (proof obligation).

### Desired Commutation (Unreachable ↔ Assertion)

Replacing a never‑throw raise with an equivalent unreachable assertion
(e.g., `assert False`) must commute **only** when the branch is proven
unreachable by analysis.

### Failure Modes

- Never‑throw exceptions are caught/handled as ordinary runtime errors.
- Never‑throw exceptions are used for normal control flow.
- Reachability proof is missing but no violation is emitted.

### Normative Rule

> Any raise of a Never‑Throw exception must be proven unreachable (DEAD) by
> Gabion analysis; otherwise it is a violation.

### Erasure

Exception message text and subclassing details are erased. The protocol
identity and reachability obligation are not.

### Test Obligations (to be mapped)

- Reports include never‑throw callsite evidence with spans.
- Violations are emitted when reachability is not proven.

---

<a id="deadness_witness"></a>
## 29. Deadness Witness

### Meaning

**Definition:** A structured proof object that a branch is unreachable under an
explicit environment (inherited attribute constraints).

### Axis

**Axis:** Control‑flow + evidence (negative).

### Desired Commutation (Environment ↔ Witness)

Serializing and reloading the environment must commute with the deadness claim:
the witness remains valid for the same environment assumptions.

### Failure Modes

- DEAD asserted without a witness artifact.
- Environment omitted or implicit.
- UNKNOWN coerced to DEAD.

### Normative Rule

> Any DEAD claim must reference a Deadness Witness that includes environment,
> predicate, and reduced core evidence.

### Erasure

Formatting and ordering are erased; environment assumptions are **not**.

### Test Obligations (to be mapped)

- Deadness artifacts are deterministic and schema‑complete.

---

<a id="coherence_witness"></a>
## 30. Coherence Witness

### Meaning

**Definition:** A structured proof that multiple admissible derivations share
the same surface boundary while remaining distinct internally.

### Axis

**Axis:** Semantic + evidence (positive higher).

### Desired Commutation (Alternative Order)

Reordering alternative derivations must commute with the witness identity and
boundary equivalence claim.

### Failure Modes

- COHERENT asserted without a witness artifact.
- Fork signature or frack path omitted.
- UNKNOWN coerced to COHERENT.

### Normative Rule

> Any COHERENT claim must reference a Coherence Witness with explicit boundary,
> fork signature, frack path, and alternatives.

### Erasure

Ordering of alternatives is erased; boundary and fork identity are **not**.

### Test Obligations (to be mapped)

- Coherence artifacts are deterministic and schema‑complete.

---

<a id="rewrite_plan"></a>
## 31. Rewrite Plan (Proof‑Carrying Refactor)

### Meaning

**Definition:** A refactor proposal that carries explicit evidence links
(provenance/deadness/coherence) and a verification predicate.

### Axis

**Axis:** Refactoring + verification.

### Desired Commutation (Rewrite ↔ Re‑audit)

Applying a rewrite and re‑auditing must commute with the stated boundary
invariants (base/ctor conservation and obligation non‑regression).

### Failure Modes

- Plan emitted without evidence links.
- Verification predicate missing or non‑executable.
- UNKNOWN evidence treated as verified.

### Normative Rule

> A rewrite plan is admissible only if its verification predicate is executable
> and passes on the post‑state.

### Erasure

Formatting is erased; plan identity and evidence links are **not**.

### Test Obligations (to be mapped)

- Plan artifacts include verification predicates that fail on counterexamples.

---

<a id="exception_path"></a>
## 32. Exception Path

### Meaning

**Definition:** A potential runtime exception path enumerated by the audit
(explicit raise, known thrower, or declared contract).

### Axis

**Axis:** Control‑flow + error boundary.

### Desired Commutation (Enumeration Stability)

Syntactic refactoring and formatting changes must not alter which explicit
exception paths are enumerated.

### Failure Modes

- Explicit raise sites are not enumerated.
- Enumeration is non‑deterministic across runs.

### Normative Rule

> All explicit raise sites (E0) must be enumerated as Exception Paths.

### Erasure

Exception message text is erased; path identity is **not**.

### Test Obligations (to be mapped)

- Exception path enumeration is deterministic for E0 sites.

---

<a id="handledness_witness"></a>
## 33. Handledness Witness

### Meaning

**Definition:** Structured evidence that an exception path is dominated by a
handler or converted into a declared outcome.

### Axis

**Axis:** Control‑flow + error boundary.

### Desired Commutation (Handler Dominance)

Refactoring that preserves handler dominance must commute with handledness
claims.

### Failure Modes

- HANDLED asserted without a witness artifact.
- Handler boundary unspecified.

### Normative Rule

> Any HANDLED claim must reference a Handledness Witness with explicit handler
> kind and boundary outcome.

### Erasure

Formatting is erased; handler boundary is **not**.

### Test Obligations (to be mapped)

- Handledness witnesses are schema‑complete when emitted.

---

<a id="exception_obligation"></a>
## 34. Exception Obligation

### Meaning

**Definition:** A status binding of an Exception Path to its discharge outcome
(DEAD, HANDLED, or UNKNOWN) with an evidence reference.

### Axis

**Axis:** Governance + control‑flow correctness.

### Desired Commutation (Status ↔ Evidence)

Status updates must commute with evidence references; UNKNOWN must never be
coerced into a discharged state without a witness.

### Failure Modes

- UNKNOWN treated as discharged.
- Status set without evidence reference.

### Normative Rule

> Exception obligations must be explicit and block acceptance when configured.

### Erasure

Ordering is erased; obligation identity is **not**.

### Test Obligations (to be mapped)

- Exception obligation artifacts are deterministic and evidence‑linked.

---

<a id="attribute_carrier"></a>
## 35. Attribute Carrier

### Meaning

**Definition:** The explicit, structured representation of attribute values
in an attribute‑grammar view (e.g., environment frames, evidence bundles,
invariant bindings).

### Axis

**Axis:** Semantic (attribute grammar / evidence structure).

### Desired Commutation (Transport Transparency)

Attribute meaning must commute with transport mechanism:

```
carrier(param‑threaded) = carrier(context‑transported)
```

### Failure Modes

- Attribute values exist only as implicit globals or ad‑hoc locals.
- Multiple competing carrier shapes for the same attribute family.
- Transport mechanism changes semantics (hidden mutation or aliasing).

### Normative Rule

> Attributes must be reified as explicit carriers (dataclass/DTO). The default
> is explicit parameter threading. Context‑based transport is allowed only at
> declared edge adapters and must preserve carrier shape.

### Erasure

Transport mechanism is erased; carrier shape and values are **not**.

### Test Obligations (to be mapped)

- Carrier schema is stable and serialized in artifacts when applicable.

---

<a id="attribute_transport"></a>
## 36. Attribute Transport (ContextVar)

### Meaning

**Definition:** A transport mechanism for inherited attributes that avoids
explicit parameter threading by using a scoped ambient context.

### Axis

**Axis:** Operational (transport / wiring).

### Desired Commutation (Carrier Equivalence)

Attribute transport must commute with explicit threading:

```
transport(carrier) ⇔ pass(carrier)
```

### Failure Modes

- Hidden dependencies (ambient reads without declared ownership).
- Context leakage across async/task boundaries.
- Multiple modules mutating the same attribute context.

### Normative Rule

> ContextVar transport is permitted only through a single owning module with
> typed accessors and a context manager. Core logic must treat carriers as
> explicit data; transport remains an edge concern.

### Erasure

ContextVar identity is erased; carrier content is **not**.

### Test Obligations (to be mapped)

- Access is centralized (single module); no direct reads/writes elsewhere.

---

<a id="evidence_id"></a>
## 37. Evidence ID

### Meaning

**Definition:** A canonical identifier naming a specific obligation, invariant,
or graph-anchored entity that a test discharges (e.g., `E:bundle/alias_invariance`).
Evidence IDs have a structured **Evidence Key** as their canonical identity;
string forms are renderings for humans.

### Axis

**Axis:** Semantic (evidence labeling).

### Desired Commutation (Rename/Refactor Stability)

Evidence IDs must commute with unrelated renames or file moves:

```
rename(test) ⟹ same Evidence ID
```

### Failure Modes

- IDs are ad-hoc strings without stable structure.
- Evidence IDs change when tests are renamed.
- Multiple string forms refer to the same obligation.

### Normative Rule

> Evidence IDs must be stable, deterministic, and machine-readable. The
> **Evidence Key** is canonical; string IDs are presentation only. Evidence
> tags use the `# gabion:evidence ...` marker in tests.

### Erasure

Tag placement and formatting are erased; only the ID values persist.

### Test Obligations (to be mapped)

- Evidence extraction is deterministic and stable across re-runs.

---

<a id="witness"></a>
## 38. Witness

### Meaning

**Definition:** A test instance that discharges one or more Evidence IDs.

### Axis

**Axis:** Evidence (test obligation).

### Desired Commutation (Order Independence)

Witness sets are invariant to test ordering.

### Failure Modes

- Tests execute without evidence linkage (silent witnesses).
- Multiple tests collapse into a single witness due to ID collisions.

### Normative Rule

> Each test must be a witness to at least one Evidence ID or be explicitly
> recorded as unmapped.

### Erasure

Test ordering is erased; witness identity is not.

### Test Obligations (to be mapped)

- Unmapped tests are explicitly surfaced in evidence artifacts.

---

<a id="evidence_surface"></a>
## 39. Evidence Surface

### Meaning

**Definition:** The set of all Evidence IDs discharged by the test suite.

### Axis

**Axis:** Evidence (coverage surface).

### Desired Commutation (Idempotence)

Duplicate witnesses do not change the surface:

```
E ∪ E = E
```

### Failure Modes

- Surface depends on duplicate counts rather than set membership.
- Evidence surface is inferred from execution alone.

### Normative Rule

> The evidence surface is defined by explicit Evidence IDs, not coverage
> percentages.

### Erasure

Multiplicity of witnesses is erased; set membership remains.

### Test Obligations (to be mapped)

- Evidence surface remains stable under deterministic re-extraction.

---

<a id="evidence_dominance"></a>
## 40. Evidence Dominance (Strict)

### Meaning

**Definition:** A strict partial order over tests where `A` dominates `B` iff
`E(B) ⊂ E(A)` (proper subset).

### Axis

**Axis:** Evidence (ordering).

### Desired Commutation (Equivalence Preservation)

If `E(A) = E(B)`, neither dominates the other.

### Failure Modes

- Treating equal evidence sets as redundant.
- Allowing dominance to depend on non-evidence signals by default.

### Normative Rule

> Redundancy requires strict dominance. Equal evidence sets are classified as
> equivalent witnesses, not redundant.

### Erasure

Ordering of Evidence IDs is erased; set inclusion is not.

### Test Obligations (to be mapped)

- Dominance results are stable under deterministic ordering.

---

<a id="equivalent_witness"></a>
## 41. Equivalent Witness

### Meaning

**Definition:** A group of tests that share identical evidence sets.

### Axis

**Axis:** Evidence (equivalence class).

### Desired Commutation (Permutation Invariance)

Permutation of equivalent witnesses does not change classification.

### Failure Modes

- Equivalent witnesses are treated as redundant by default.
- Equivalence classes are hidden or unstable.

### Normative Rule

> Equivalent witnesses are a distinct class and require secondary signals
> before removal.

### Erasure

Within-class ordering is erased.

### Test Obligations (to be mapped)

- Equivalent witness classes are reported deterministically.

---

<a id="test_obsolescence_projection"></a>
## 42. Test Obsolescence Projection

### Meaning

**Definition:** A deterministic projection that classifies tests using the
evidence carrier and guardrails (e.g., `artifacts/out/test_obsolescence_report.json`,
with Markdown projections under `out/test_obsolescence_report.md`).

### Axis

**Axis:** Evidence (projection).

### Desired Commutation (Determinism)

Projection output must be invariant to ordering and run time.

### Failure Modes

- Non-deterministic report ordering.
- Obsolescence classification bypasses guardrails.

### Normative Rule

> Obsolescence is computed from the evidence carrier with strict dominance and
> risk guardrails. The projection is advisory unless explicitly gated.

### Erasure

Formatting is erased; class assignments are not.

### Test Obligations (to be mapped)

- Obsolescence reports are stable across re-runs.

---

<a id="test_evidence_suggestions_projection"></a>
## 43. Test Evidence Suggestions Projection

### Meaning

**Definition:** A deterministic projection that proposes evidence tags for
tests based on graph resolution (with heuristic fallback as needed).

### Axis

**Axis:** Evidence (projection).

### Desired Commutation (Resolution Stability)

Graph-resolved suggestions must be stable under reordering and equivalent
call-graph construction.

### Failure Modes

- Suggestions derived only from filename heuristics.
- Heuristics override graph-resolved mappings.

### Normative Rule

> Suggestions must prefer graph-derived evidence and only fall back to
> heuristics when graph resolution fails.

### Erasure

Suggestion ordering is erased; evidence IDs are not.

### Test Obligations (to be mapped)

- Suggestions include provenance (graph vs heuristic).

---

<a id="evidence_key"></a>
## 44. Evidence Key

### Meaning

**Definition:** The canonical, structured identity for evidence items, derived
from the graph carrier. Display strings are renderings of the key.
Examples of key kinds include `paramset`, `decision_surface`, `never_sink`,
`function_site`, `call_footprint`, and `call_cluster`.

### Axis

**Axis:** Evidence (identity).

### Desired Commutation (Graph Derivation)

Evidence keys must commute with presentation changes:

```
render(key) ⇒ key
```

### Failure Modes

- Keys authored manually without graph derivation.
- Display strings treated as canonical identity.

### Normative Rule

> Evidence keys are graph-derived and stable; display strings are presentation
> only and must round-trip to the same key when possible.

### Erasure

Display formatting is erased; key identity is not.

### Test Obligations (to be mapped)

- Key rendering/parsing is deterministic and stable.

---

<a id="ambiguity_set"></a>
## 45. Ambiguity Set

### Meaning

**Definition:** A canonical, order-independent set of candidate carriers when
resolution is not unique. Ambiguity sets are derived from the graph carrier
and recorded as first-class nodes.

### Axis

**Axis:** Resolution (ambiguity carrier).

### Desired Commutation (Candidate Order)

Let `A` be an ambiguity set with candidate list `C`.

```
permute(C) ⇒ same Ambiguity Set
```

### Failure Modes

- Ambiguity erased or silently resolved without recording candidates.
- Candidate ordering treated as identity.

### Normative Rule

> Ambiguity Sets must be derived from the graph carrier and recorded with
> canonical candidate ordering. Resolution must not drop candidates.

### Erasure

Candidate ordering is erased; candidate identity is not.

### Test Obligations (to be mapped)

- Ambiguity sets are deterministic and order-independent.

---

<a id="partition_witness"></a>
## 46. Partition Witness

### Meaning

**Definition:** A structured certificate explaining why an ambiguity exists
and what would collapse it, anchored to a specific Ambiguity Set.

### Axis

**Axis:** Resolution (witness).

### Desired Commutation (Witness Stability)

```
permute(candidates) ⇒ same Partition Witness
```

### Failure Modes

- Ambiguity recorded without a witness.
- Witness depends on non-semantic presentation details.

### Normative Rule

> Each Ambiguity Set must have at least one Partition Witness that records
> the resolution phase and a minimal collapse hint.

### Erasure

Formatting of witness details is erased; witness identity is not.

### Test Obligations (to be mapped)

- Witness emission is deterministic and stable.

---

<a id="annotation_drift"></a>
## 47. Annotation Drift

### Meaning

**Definition:** Evidence tags that no longer resolve to the current evidence
universe (or fail to parse), indicating stale or orphaned annotations.

### Axis

**Axis:** Evidence (hygiene).

### Desired Commutation (Presentation)

```
render(key) changes ⇒ no drift
```

### Failure Modes

- Orphaned tags remain undetected.
- Drift is computed from display strings rather than keys.

### Normative Rule

> Annotation drift is defined by Evidence Key identity, not display. Orphaned
> tags must be reported via an advisory projection before any ratchet.

### Erasure

Display formatting is erased; key identity is not.

### Test Obligations (to be mapped)

- Drift audit detects orphaned tags deterministically.

---

<a id="grothendieck_analysis"></a>
## 48. Grothendieck Analysis (Doc Review Cofibration)

### Meaning

**Definition:** A structured self-audit that co‑fibrates a document against
itself (normalize and align internal structure), deduplicates observations, and
contrasts the result against the document’s semantics and completeness.

### Axis

**Axis:** Documentation (review discipline).

### Desired Commutation (Idempotence)

```
analyze(doc) = analyze(normalize(doc))
```

### Failure Modes

- Mechanical review stamps without structural alignment.
- Deduplication removed without validating semantic completeness.

### Normative Rule

> Self‑review is valid only when a Grothendieck analysis is performed and the
> result is recorded in `doc_review_notes`.

### Erasure

Formatting and ordering of notes are erased; semantic contrasts are not.

### Test Obligations (to be mapped)

- Docflow audit rejects self‑review entries without explicit analysis notes.

---

<a id="self_review"></a>
## 49. Self‑Review (Docflow Exception)

### Meaning

**Definition:** A document review performed by the author of the document,
permitted only with a Grothendieck analysis.

### Axis

**Axis:** Documentation (review discipline).

### Desired Commutation (Reviewer Identity)

Self‑review does not commute with normal review; it requires explicit evidence.

### Failure Modes

- Self‑review used to bypass review discipline.
- `doc_reviewed_as_of` bumped without substantive notes.

### Normative Rule

> Self‑review must cite a Grothendieck analysis in `doc_review_notes`. Absent
> that evidence, the review is invalid.

### Erasure

Formatting of notes is erased; evidence of analysis is not.

### Test Obligations (to be mapped)

- Self‑review without analysis evidence is rejected.

---

## 50. Mirror Branch (Fast‑Forward Equivalence)

### Meaning

**Definition:** Branch `A` mirrors branch `B` iff `A` has no unique commits
relative to `B` (i.e., `A` is an ancestor of `B`), and mirror updates are
fast‑forward only. After a mirror update, `A` and `B` are equal by commit SHA.

### Axis

**Axis:** Governance (branch integrity).

### Desired Commutation (Fast‑Forward Stability)

```
mirror(A, B) and fast_forward(A ← B) ⇒ mirror(A, B)
```

### Failure Modes

- Mirror branch diverges (unique commits).
- Check‑before‑use race between verification and update.

### Normative Rule

> Mirror updates must verify ancestry, use explicit commit SHAs, and employ
> compare‑and‑swap (`--force-with-lease`) to prevent TOCTOU races.

### Erasure

Remote alias names are erased; commit‑graph relations are not.

### Test Obligations (to be mapped)

- Policy checker enforces ancestor checks and explicit SHA updates.

---

<a id="forest"></a>
## 50. Forest (Interned Carrier Graph)

### Meaning

**Definition:** The Forest is the materialized, interned carrier graph for ASPF.
It stores nodes, alternatives, and metadata such that ASPF semantics are
queryable and auditable. Forest does not add semantics; it **materializes**
ASPF structure for ProjectionSpec and evidence emission.

### Axis

**Axis:** Structural (carrier graph / provenance materialization).

### Desired Commutation (Internment ↔ Queryability)

```
intern(x) ⇒ node(x) with stable identity and queryable provenance
```

### Failure Modes

- Forest treated as an ephemeral runtime cache (identity not preserved).
- Forest nodes emitted without stable span/qual keys (non‑semantic edits break identity).
- Facets attached to functions rather than SuiteSites (locality lost).

### Normative Rule

> Forest nodes are interned into ASPF and must preserve identity across
> re‑audits. SuiteSites are Forest nodes; locality facets attach to SuiteSites.
> Function‑level evidence is an aggregation over SuiteSites.

### Erasure

Ordering of node emission is erased; canonical keys are not.

### Test Obligations (to be mapped)

- Forest emission is deterministic across re‑audits.
- Forest identity is stable under non‑semantic edits.

---

<a id="suite_site"></a>
## 51. SuiteSite

### Meaning

**Definition:** A SuiteSite is the canonical locality carrier: a contiguous
executable block (code suite) or logical suite (docflow / issue metadata) with
stable identity keyed by `(domain, kind, path, qual, span)`.

### Axis

**Axis:** Structural (locality carrier / suite containment).

### Desired Commutation (Projection Stability)

Let `suite_id` be derived from the canonical key.

```
alias(suite_id) with identical key ⇒ projections invariant
```

### Failure Modes

- Suite identity changes under non‑semantic edits (span instability).
- Facets attach to function scope instead of the enclosing suite.
- Mixed domains (`docflow` vs `github`) without explicit links.

### Normative Rule

> All locality‑bound facets attach to SuiteSite.
> FunctionSite is a projection over contained SuiteSites.
> Suite identity must remain stable under non‑semantic edits; spans and domain
> are not erased. Docflow and GitHub suites follow `in/in-30.md`.

### Erasure

Suite alias names may be erased; canonical key fields are not.

### Test Obligations (to be mapped)

- Suite emission is deterministic across re‑audits.
- Suite→function aggregation matches legacy function‑level reports.

---

<a id="hash_consing"></a>
## 52. Hash‑Consing (Internment)

### Meaning

**Definition:** Internment is hash‑consing: `hash(x)` *is* normalization and
returns the canonical normalized representative. Hashes are addresses of
normalized forms only; raw objects are not separately addressable.

### Axis

**Axis:** Identity (canonicalization / ledger address).

### Desired Commutation (Normalization ↔ Hash)

```
hash(normalize(x)) == normalize(hash(x))
hash(hash(x)) == hash(x)
```

### Failure Modes

- Hash used as a lossy digest (information loss).
- Hash computed on raw/un‑normalized forms.
- Multiple hashes for equivalent normalized structures.

### Normative Rule

> `hash()` must not lose information. It returns the normalized form and is
> idempotent. Any intrinsic computation (e.g., β‑reduction or substitution)
> must be defined as a normalization rule prior to internment.
> Because internment is hash‑consing, β‑reduction is intrinsic and automatic
> to internment whenever it is part of the normalization system.

### Erasure

Formatting differences are erased; semantic structure is not.

### Test Obligations (to be mapped)

- Hash idempotence and normalization commutation.
