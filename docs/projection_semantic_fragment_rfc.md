---
doc_revision: 77
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: projection_semantic_fragment_rfc
doc_role: playbook
doc_scope:
  - repo
  - projection
  - policy
  - aspf
  - ttl
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/ttl_kernel_semantics.md#ttl_kernel_semantics
  - docs/aspf_execution_fibration.md#aspf_execution_fibration
  - docs/audits/projection_spec_history_ledger.md
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 55
  glossary.md#contract: 44
  docs/ttl_kernel_semantics.md#ttl_kernel_semantics: 1
  docs/aspf_execution_fibration.md#aspf_execution_fibration: 7
  docs/audits/projection_spec_history_ledger.md: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev55 (correctness-by-construction and drift-control framing remain aligned with this RFC's implementation scope)."
  glossary.md#contract: "Reviewed glossary.md rev44 (witness, evidence, and projection terminology remain aligned with this RFC's carrier/judgment language)."
  docs/ttl_kernel_semantics.md#ttl_kernel_semantics: "Reviewed the TTL explainer rev1 and carried its semantic layering forward into an implementation-targeted RFC."
  docs/aspf_execution_fibration.md#aspf_execution_fibration: "Reviewed ASPF execution fibration rev7 and reused its identity, witness, and fibration substrate as the bootstrap layer for this RFC."
  docs/audits/projection_spec_history_ledger.md: "Reviewed projection-spec history ledger rev1; current ProjectionSpec remains a quotient/erasure runtime and is treated here as a temporary adapter surface."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_sections:
  projection_semantic_fragment_rfc: 1
doc_section_requires:
  projection_semantic_fragment_rfc:
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
    - docs/ttl_kernel_semantics.md#ttl_kernel_semantics
    - docs/aspf_execution_fibration.md#aspf_execution_fibration
    - docs/audits/projection_spec_history_ledger.md
doc_section_reviews:
  projection_semantic_fragment_rfc:
    POLICY_SEED.md#policy_seed:
      dep_version: 55
      self_version_at_review: 9
      outcome: no_change
      note: "Policy seed rev55 reviewed; this RFC remains implementation-driving rather than normative."
    glossary.md#contract:
      dep_version: 44
      self_version_at_review: 9
      outcome: no_change
      note: "Glossary rev44 reviewed; witness/evidence terminology remains aligned."
    docs/ttl_kernel_semantics.md#ttl_kernel_semantics:
      dep_version: 1
      self_version_at_review: 9
      outcome: no_change
      note: "TTL explainer rev1 reviewed; semantic layering and TTL-source interpretation remain aligned."
    docs/aspf_execution_fibration.md#aspf_execution_fibration:
      dep_version: 7
      self_version_at_review: 9
      outcome: no_change
      note: "ASPF execution fibration rev7 reviewed; identity/fibration reuse remains aligned."
    docs/audits/projection_spec_history_ledger.md:
      dep_version: 1
      self_version_at_review: 9
      outcome: no_change
      note: "ProjectionSpec history ledger rev1 reviewed; quotient/adapter framing remains aligned."
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="projection_semantic_fragment_rfc"></a>
# Projection Semantic Fragment RFC

This document is an implementation RFC, not a normative policy document. Its
purpose is to turn the semantic direction described in
[ttl_kernel_semantics.md#ttl_kernel_semantics](./ttl_kernel_semantics.md#ttl_kernel_semantics)
into a decision-complete migration spec for repository implementation.

The default choices in this RFC are fixed:

- bootstrap from the existing ASPF/global identity and fibration substrate
- treat the TTL kernel as semantic foundation, not as a one-to-one codegen spec
- keep the semantic core narrow in v1
- land carrier-first before compiler-first

## 1. Problem statement and target model

Today the repository has a useful but split architecture:

- [`projection_spec.py`](../src/gabion/analysis/projection/projection_spec.py)
  and [`projection_exec.py`](../src/gabion/analysis/projection/projection_exec.py)
  shape JSON-like rows
- the policy DSL judges witness rows and aggregate artifacts
- the projection-fiber substrate checks convergence over witness carriers
- the TTL kernel under [`in/`](../in/) describes a stronger unified semantic stack

That split is serviceable, but it is weaker than the TTL model in one decisive
way: semantic construction and semantic judgment are not yet centered on the
same canonical carrier.

This RFC chooses the following target layering:

1. **Canonical semantic core**
   One witnessed carrier with lawful semantic transforms and explicit
   obligations.
2. **Executable SHACL/SPARQL substrate**
   Compiled realizations of semantic obligations and graph/query semantics.
3. **Friendly authoring layer**
   `ProjectionSpec`, policy DSL sources, and higher-level repo syntax.
4. **Presentation/render layer**
   Rendering/report/report-order transforms derived from semantic normal form.

The key invariant is fixed:

> semantic construction creates closed, witness-complete objects; DSL judgment
> reads and classifies those objects, but does not create them.

This keeps the existing repo-level insight from
[`docs/projection_fiber_rules.yaml`](./projection_fiber_rules.yaml) and
[`lattice_convergence_semantic.py`](../src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py),
but moves semantic ownership into a canonical carrier rather than leaving it
split between projection shaping and witness judgment.

## 2. Canonical semantic fragment

### 2.1 Bootstrap substrate

The canonical semantic fragment must reuse the existing support structure:

- structural identity hashing from
  [`aspf_lattice_algebra.py`](../src/gabion/analysis/aspf/aspf_lattice_algebra.py)
- diagnostic/provenance site identities from the same module
- existing fibration carriers exported through
  [`dataflow_fibration.py`](../src/gabion/tooling/policy_substrate/dataflow_fibration.py)
- current obligation and boundary witness families already present in the ASPF
  lattice/fibration substrate
- trace continuity and execution-carrier expectations from
  [`aspf_execution_fibration_impl.py`](../src/gabion/analysis/foundation/aspf_execution_fibration_impl.py)

This RFC does **not** allow a second identity universe. Any new semantic
carrier must intern into the existing ASPF/global identity space.

### 2.2 Planned carrier and interfaces

Implementation names remain provisional, but v1 must expose the following
planned interfaces.

#### Canonical witnessed semantic carrier

`CanonicalWitnessedSemanticRow` is the default planned name for the core row
envelope. It must carry at least:

- `structural_identity`
  primary stable identity, reused for baselines and equivalence
- `site_identity`
  diagnostic/provenance identity, not the primary semantic key
- `surface`
  declared semantic surface or quotient face
- `payload`
  semantic object payload after reflection
- `input_witnesses`
  ingress/trace witnesses carried into the row
- `synthesized_witnesses`
  witnesses added under explicit synthesis laws
- `obligations`
  unresolved/discharged obligations attached to the object
- `boundary_trace`
  boundary or lowering/reflection crossings relevant to the row
- `transform_trace`
  ordered semantic transform history

The carrier is the default judgment input for future semantic projection and
projection-fiber policy work.

During the carrier-first migration window, adapter modules/functions that only
materialize or normalize canonical semantic carriers may be marked as
`semantic_carrier_adapter` grade boundaries. That boundary class is temporary
and exists to keep semantic-carrier materialization out of ordinary
deterministic-core monotonicity enforcement while the full semantic op layer is
still landing.

#### Semantic op schema

`SemanticOp` is the default planned name for the semantic transform calculus.
The v1 operation vocabulary is fixed:

- `reflect`
- `quotient_face`
- `synthesize_witness`
- `wedge`
- `reindex`
- `existential_image`
- `support_reflect`
- `negate`

No other operator may be treated as semantic in v1 without a follow-up RFC.

#### Judgment boundary

The policy DSL consumes `CanonicalWitnessedSemanticRow` collections and their
attached obligations. It may:

- classify
- compare
- summarize
- block/warn/pass

It may not:

- mutate payload
- invent witnesses
- alter quotient choice
- repair unresolved obligations

### 2.3 Core laws

The semantic fragment must preserve the following laws.

#### Reflect/realize closure law

Each semantic surface has a reflect/realize pair:

- `alpha_P`: carrier -> spec-visible semantic face
- `gamma_P`: semantic face -> canonical carrier

The induced closure `cl_P = gamma_P ∘ alpha_P` must be:

- monotone
- extensive
- idempotent

Implementation acceptance criterion: repeated reflection converges to a fixed
point on canonical form.

#### Quotient-face law

`quotient_face` is only lawful when backed by an explicit kernel/congruence.
It must:

- name the congruence class or kernel basis
- preserve structural identity continuity into the face
- record factorization/uniqueness obligations when the quotient is semantic

There is no generic “projection as arbitrary field drop” in the semantic layer.

#### Witness synthesis admissibility law

`synthesize_witness` is the only generative semantic operation.

Its inputs are restricted to:

- input carrier payload
- prior canonical witnesses
- declared transform trace
- spec-local synthesis laws

It must be:

- inflationary
- bounded
- entailed

It may not use ad hoc runtime state outside those bases. If a witness cannot be
explained from those bases, it is not a lawful synthesized witness.

#### Wedge/context law

`wedge` is the canonical support/history/proof context composition operator.
It composes:

- proof/query context
- history/support context
- boundary stamp or equivalent carrier

It must preserve structural identity continuity of the participating inputs and
must record the derived support context explicitly in transform/witness traces.

#### Judgment-separation law

DSL judgment is orthogonal to semantic construction:

- semantic transforms change carrier state
- DSL rules observe carrier state

Violation of this law is any design where policy evaluation itself creates or
repairs witnesses.

## 3. SHACL/SPARQL compilation contract

The TTL kernel embeds SHACL and SPARQL as executable realizers of the semantic
layer. This RFC fixes the compilation direction:

> friendly DSLs compile into the semantic fragment, and from there into
> SHACL/SPARQL obligations.

SHACL/SPARQL are realizers, not the source of the laws.

### 3.1 SHACL compilation targets

The SHACL compiler target is `CompiledShaclPlan`.

V1 SHACL compilation owns:

- node-shape and property-shape validation for semantic carrier projections
- reflective-boundary obligations
- lowering soundness checks
- surjectivity and conservative-projection obligations
- denotational compatibility checks where the target is a shape/constraint
  object rather than a query-evaluation path

SHACL compilation must preserve:

- originating `structural_identity`
- originating `site_identity`
- transform trace reference
- obligation-to-witness linkage

### 3.2 SPARQL compilation targets

The SPARQL compiler target is `CompiledSparqlPlan`.

V1 SPARQL compilation owns:

- graph-native query AST lowering
- quotient-face extraction/query realization over graph data
- reindexing/existential image realization
- support-reflection lookup queries
- anti-join and `NOT EXISTS` realizations
- witness/support query plans tied to history-stamped or boundary-stamped
  semantic context

SPARQL compilation must preserve:

- originating `structural_identity`
- originating `site_identity`
- quotient/kernel annotation when relevant
- support/history/proof context trace derived via `wedge`

### 3.3 Compilation mapping by semantic op

| Semantic op | Primary realization |
| --- | --- |
| `reflect` | semantic core only; may emit SHACL/SPARQL obligations but is not itself a lowered executable |
| `quotient_face` | SHACL when validating quotient-face constraints; SPARQL when extracting/querying the face |
| `synthesize_witness` | semantic core only; lowered plans may validate prerequisites but do not own witness invention |
| `wedge` | semantic core plus SPARQL context composition/query planning |
| `reindex` | SPARQL/query algebra target |
| `existential_image` | SPARQL/query algebra target |
| `support_reflect` | SPARQL plus semantic truth/support interpretation boundary |
| `negate` | SPARQL anti-join/`NOT EXISTS` realization plus policy/semantic truth-structure linkage |

Any compilation path that loses ASPF/global identity continuity or breaks
witness traceability is out of contract.

## 4. ProjectionSpec split and current-op classification

Current `ProjectionSpec` remains a compatibility surface until the semantic
carrier and compiler path exist.

### 4.1 Stratified model

The stratified model is fixed:

- semantic transforms are canonical
- presentation transforms are derived
- presentation must factor through semantic normal form

This means the current `projection_exec.py` runtime remains available during
migration, but it is not the owner of future semantic projection laws.

### 4.2 Current operation classification

The current operation surface is classified as follows.

| Current op | Classification in v1 | Result |
| --- | --- | --- |
| `sort` | presentation only | stays in render/report layer |
| `limit` | presentation only | stays in render/report layer |
| `count_by` | presentation/aggregation only | not semantic in v1; may consume closed semantic objects |
| generic `select` | adapter only | not semantic in current form; replace later with declared semantic predicates if promoted |
| generic `project` | presentation by default | semantic only when tied to an explicit declared quotient face |
| `traverse` | bridge/helper only | not semantic until reified as a lawful operator |

### 4.3 Planned future friendly surface

The future `ProjectionSpec` authoring layer should be split into:

- a semantic authoring surface over `SemanticOp`
- a presentation authoring surface over rendering/report operations

The compatibility rule is:

- current JSON pipeline remains supported through adapters
- no new semantic behavior may be added directly to the legacy row-pipeline
  layer after this RFC lands

## 5. Migration plan

### Phase 1. Bootstrap on existing ASPF/fibration substrate

Deliverable:
- one canonical carrier spec rooted in existing ASPF/global identity and fiber
  substrate

Required reuse:
- `canonical_structural_identity`
- `canonical_site_identity`
- current fiber-bundle carriers
- current obligation/boundary witness patterns
- current execution-fibration trace continuity rules

Out of scope:
- any second identity graph
- any new standalone witness store detached from ASPF/fibration carriers

Acceptance gate:
- the RFC implementation slice can point to one planned carrier that reuses the
  existing substrate rather than replacing it

### Phase 2. Carrier-first semantic skeleton

Deliverable:
- minimal `CanonicalWitnessedSemanticRow`
- minimal `SemanticOp` schema
- adapter path from current relation/witness rows into the canonical carrier

Implementation rule:
- do not cut over `ProjectionSpec` behavior yet
- do not require SHACL/SPARQL compilers yet
- establish carrier and laws first

Acceptance gate:
- at least one end-to-end path can materialize canonical semantic rows from
  existing substrate rows without dual-shape ambiguity in core

### Phase 3. Compiler bridge

Deliverable:
- `CompiledShaclPlan`
- `CompiledSparqlPlan`
- deterministic lowerings from v1 semantic ops to those plans

Current implementation status:
- projection-fiber `reflect` now lowers into typed `CompiledShaclPlan` and
  `CompiledSparqlPlan` artifacts derived from the canonical semantic row
- this compiler bridge is intentionally narrow and exists to prove
  deterministic lowering, trace continuity, and carrier/plan agreement before
  broader semantic-op lowering lands
- one authoring-layer path now continues through that bridge: a lowered
  `ProjectionSpec` `project` op with declared quotient-face metadata compiles
  into typed SHACL/SPARQL plans over the existing projection-fiber semantic
  rows
- that bridge now has two declared authoring-layer faces over the same carrier:
  `projection_fiber.frontier` and
  `projection_fiber.reflective_boundary`
- those faces now have a real substrate consumer: registered
  `projection_fiber_frontier` and
  `projection_fiber_reflective_boundary` `ProjectionSpec`s are lowered and
  compiled inside the lattice-convergence policy artifact, so the
  authoring-layer bridge is no longer test-only

Implementation rule:
- all compilation is explicit and typed
- no freehand query or shape authoring in semantic core
- graph-native AST ownership is preserved even when lowered strings/plans exist

Acceptance gate:
- the same semantic path from Phase 2 lowers deterministically into SHACL/SPARQL
  artifacts with preserved identity and witness trace continuity

### Phase 4. Friendly-surface convergence

Deliverable:
- split authoring model for semantic vs presentation transforms
- adapter-backed `ProjectionSpec` compatibility story
- direct policy-DSL consumption path for canonical carrier rows

Current implementation status:
- `ProjectionSpec` lowering now has an explicit typed stratification pass that
  classifies current pipeline ops into semantic, presentation, and bridge
  buckets before any execution-path cutover
- `project` ops with declared `quotient_face` metadata and the explicit
  `reflect(surface=projection_fiber)` op now promote into semantic ops in v1;
  `sort`, `limit`, and `count_by` remain presentation, while `select`,
  `traverse`, and unknown ops remain bridge/compatibility surfaces
- the `projection_fiber.frontier` and
  `projection_fiber.reflective_boundary` faces now compile through to typed
  SHACL/SPARQL plans against canonical semantic rows, proving the
  friendly-surface-to-executable bridge for more than one declared face without
  widening generic row transforms
- the direct reflect-plan compilation branch has been removed from the
  lattice-convergence substrate; reflect now compiles through the same typed
  lowering/compiler path via a registered `projection_fiber_reflection`
  `ProjectionSpec`, and the substrate report flattens its top-level reflect
  plans from those compiled bundles rather than bypassing lowering
- the first real consumer of that bridge is the lattice-convergence substrate
  report, which now emits compiled semantic-plan bundles for the registered
  frontier, reflection, and reflective-boundary specs alongside the raw
  semantic rows and top-level reflect plans derived from the same typed bundle
  output
- that substrate output now crosses a runtime-facing boundary: `policy_check
  --output` carries a `projection_fiber_semantics` payload with the lattice
  decision, semantic report, and compiled projection-semantic bundles, so the
  first outward-facing consumer is no longer test-only
- that outward-facing payload is now consumed by the policy suite through
  `policy_results.policy_check`, and the hotspot-neighborhood queue still
  receives the same derived semantic summary without the policy-suite artifact
  carrying its own duplicate `projection_fiber_semantics_summary` field
- that hotspot-neighborhood consumer can now source its semantic summary
  directly from `policy_check_result.json` when that faster artifact is
  available, rather than requiring the slower policy-suite payload to remain
  the semantic-summary carrier
- that same faster `policy_check_result.json` path now emits the
  `projection_semantic_fragment_queue` JSON/Markdown artifacts directly, so
  semantic-fragment continuation state is offloaded before the slower
  policy-suite scan runs
- the projection-semantic-fragment queue tool itself now names its ingress as a
  generic source artifact rather than a `policy_suite` path, matching the
  fast-path ownership model and removing stale suite-era interface language
- the policy-suite wrapper has now dropped both its earlier pre-scan duplicate
  semantic-queue emission step and its later queue backfill step, so
  `projection_semantic_fragment_queue` ownership is now fully fast-path-only:
  `policy_check` emits it, the suite wrapper preserves it if present, and
  missing queue artifacts remain visible rather than being regenerated by the
  slower wrapper
- the same wrapper now treats child-owned policy artifacts generically: if a
  valid canonical result already exists for `policy_check`, `structural_hash`,
  or `deprecated_nonerasability`, the wrapper consumes it directly instead of
  rerunning the owning child check
- hotspot-neighborhood queue generation is now payload-first when invoked from
  the wrapper: the suite runner passes the in-memory suite payload directly to
  the queue builder instead of treating `policy_suite_results.json` as a disk
  rendezvous surface just to re-read the same data
- the hotspot-neighborhood queue tool itself now exposes a generic
  `source_artifact` ingress rather than a `policy-suite`-named one, matching
  the real ownership model and removing another stale wrapper-era interface
- the hotspot-neighborhood queue artifact itself now serializes generic
  `source_generated_at_utc` and `source_counts` metadata rather than
  `policy_suite_*` fields, so emitted queue payloads no longer project
  suite-era ownership language after ingress has already been generalized
- the hotspot-neighborhood queue artifact no longer projects
  policy-scanner-suite cache hashes (`inventory_hash`, `rule_set_hash`,
  `policy_results_hash`, `changed_scope_hash`) into its own source metadata;
  those wrapper cache internals are now treated as non-contractual provenance
  and no longer widen the queue/reporting surface
- hotspot-neighborhood queue analysis now consumes one normalized source
  payload only; the separate `policy_check` supplement path was removed once
  the canonical source payload already carried `policy_results.policy_check`
- hotspot-neighborhood queue CLI ingress now requires an explicit
  `--source-artifact` instead of defaulting to `policy_suite_results.json`,
  so the generic queue tool no longer projects a suite-era rendezvous default
- policy-scanner-suite runtime and CLI ingress now require an explicit output
  artifact path instead of baking in `policy_suite_results.json`, so the
  compatibility wrapper no longer projects that artifact name as an implicit
  ownership contract
- the runtime policy-scanner-suite cache loader now reuses the generic
  canonical `policy_results` mapping normalizer instead of a fixed three-child
  projection helper, removing another suite-specific assumption from cached
  artifact reads
- the hotspot-neighborhood queue no longer materializes a
  `projection_fiber_semantics_summary` blob inside its own artifact; it now
  carries only the direct semantic fields it actually uses
- the shared projection-fiber summary helper no longer exposes an explicit
  parser for retired materialized-summary payloads; it now accepts only live
  canonical carriers (`projection_fiber_semantics`)
- the shared projection-fiber summary helper no longer exposes a separate
  `...from_policy_results(...)` entrypoint; canonical decoding now lives behind
  a single payload decoder that follows the live carrier shape directly
- the shared projection-fiber summary decoder no longer accepts a top-level
  `policy_check` wrapper payload as a generic ingress shape; canonical summary
  decoding is now restricted to direct `policy_check` payloads,
  direct `projection_fiber_semantics`, or explicit summary payload parsing
- the shared projection-fiber summary decoder no longer accepts the retired
  top-level `projection_fiber_semantics_summary` suite embedding as a generic
  input shape; canonical payload decoding now goes through
  `projection_fiber_semantics` or an explicit
  summary-payload parser where a materialized summary is already owned locally
- queue/report consumers no longer depend on a `ProjectionFiberSemanticsSummary`
  bridge object just to read canonical semantic fields; hotspot and semantic
  fragment queues now consume direct helper projections over the live
  `projection_fiber_semantics` carrier
- the suite wrapper no longer peels `projection_fiber_semantics` out of
  `policy_check_result.json` just to hand it to the hotspot queue; the queue
  now owns that last child-artifact read when invoked from canonical
  wrapper/runtime inputs
- the runtime scanner no longer returns a `PolicySuiteResult` wrapper carrier
  at all; `scan_policy_suite()` now returns the canonical `violations_by_rule`
  mapping directly, and suite-decision evaluation is an explicit helper over
  that mapping rather than a method on a compatibility result object
- policy-scanner-suite runtime payloads now surface semantic context, when
  needed, as a direct top-level `projection_fiber_semantics` carrier rather
  than nested child `policy_results`, and queue/report consumers no longer
  follow the wrapper-only `policy_results.policy_check` semantic path
- the runtime `PolicySuiteResult` carrier no longer retains full child-owned
  `policy_results` payloads in memory, and it no longer carries the wrapper's
  pass-through `projection_fiber_semantics` either; the runtime result is now
  limited to the violations it actually computes
- the runtime `PolicySuiteResult` carrier no longer retains wrapper child
  statuses at all, and wrapper orchestration no longer depends on any
  child-status carrier once child-owned artifacts are validated
- the runtime policy-scanner-suite load/scan boundary no longer accepts raw
  child `policy_results` mappings; wrapper ingress now normalizes child-owned
  artifacts once into a direct `projection_fiber_semantics` boundary value
- the policy-scanner-suite wrapper itself no longer traffics raw child result
  mappings after ingress validation; external child checks now resolve
  directly to `projection_fiber_semantics` before any wrapper orchestration
  continues
- the policy-scanner-suite wrapper no longer materializes a temporary
  `dict[rule_id, payload]` child-result rendezvous during ingress resolution;
  preserved or newly emitted child artifacts are normalized directly into
  `projection_fiber_semantics` at the wrapper boundary
- the runtime policy-scanner-suite module no longer exposes a raw child-result
  parser at all; raw child payload normalization lives only in the wrapper
  boundary helper, and runtime no longer consumes boundary-owned semantic
  payloads once normalized
- the policy-scanner-suite wrapper no longer exposes separate raw child-payload
  peelers for `status` and `projection_fiber_semantics`; child artifact ingress
  is now normalized through one boundary loader before wrapper
  orchestration continues
- the policy-scanner-suite wrapper no longer carries its own closed
  policy-rule family registry just to print violations; console rendering now
  follows the runtime result's own `violations_by_rule` families directly
- the outward-facing `PolicySuiteResult` carrier and payload no longer project
  cache identity hashes (`inventory_hash`, `rule_set_hash`); those no longer
  appear on the public carrier at all
- the runtime policy-scanner-suite module no longer exposes a cache/load API;
  `load_or_scan_policy_suite()`, `PolicySuiteLoadOutcome`, and the
  cache-normalization helpers are gone, and direct `scan_policy_suite()` is the
  only remaining runtime orchestration surface
- the runtime `PolicySuiteResult` carrier no longer owns or projects `root`;
  repository-root provenance remains a wrapper/cache concern rather than part
  of the outward suite semantic/reporting carrier
- the outward `PolicySuiteResult` payload no longer projects a redundant
  `counts` summary; downstream reporting derives family totals directly from
  canonical `violations`, so one more wrapper-era summary field is removed
- the retired `policy_suite_results.json` compatibility artifact is gone
  entirely; the wrapper now publishes only its hotspot-neighborhood queue
  artifacts, and no wrapper-owned mirror of suite violations remains on disk
- the outward `PolicySuiteResult` payload no longer emits a redundant derived
  `decision`; callers that need the suite decision compute it from
  `PolicySuiteResult.decision()`, keeping the serialized reporting carrier
  closer to canonical `violations`
- the outward `PolicySuiteResult` payload no longer emits `generated_at_utc`,
  and the hotspot-neighborhood queue no longer copies a
  `source_generated_at_utc` field from that payload; queue artifacts now keep
  only their own generation timestamp plus direct source-derived content
- the runtime `PolicySuiteResult` carrier no longer serializes itself via
  `to_payload()`; boundary payload shaping for hotspot/report consumers now
  lives at the wrapper edge, and the runtime carrier remains a typed in-memory
  result only
- the runtime policy-scanner-suite module no longer exposes a redundant
  `violations_for_rule(...)` accessor; wrapper/report consumers now read
  `violations_by_rule` directly from the typed result carrier
- the runtime `PolicySuiteResult` carrier no longer exposes a trivial
  `total_violations()` accessor, and the wrapper no longer carries a dedicated
  `_hotspot_source_payload(...)` helper; both totals and hotspot payloads are
  now derived directly from canonical `violations`
- the policy-scanner-suite wrapper no longer invokes `policy_check.py`
  itself; `policy_check_result.json` is now a preserve-only child-owned
  artifact on this path, and the wrapper fails closed if that fast-path result
  is missing
- the policy-scanner-suite wrapper no longer invokes any child policy check at
  all; the remaining child checks stay owned by their own workflow steps, but
  the wrapper now consumes only `policy_check_result.json` because that is the
  only child artifact it still semantically reads
- the wrapper no longer preserves a generic child-artifact loader for that
  seam; it now loads only the canonical `policy_check_result.json` payload
  directly, so the wrapper boundary matches its single remaining child-owned
  prerequisite instead of keeping a dead generic ingress shape
- the wrapper no longer preserves a whole-payload policy-check helper or
  queue-preservation test contract for projection-semantic-fragment artifacts;
  it now loads `projection_fiber_semantics` directly from
  `policy_check_result.json`, and wrapper coverage is hotspot-only because the
  wrapper never reads or writes projection-semantic-fragment queue files
- the hotspot-neighborhood queue now owns the last minimal payload-assembly
  step for this seam too; the policy-scanner-suite wrapper passes canonical
  `violations_by_rule` plus optional `projection_fiber_semantics` directly to
  the queue helper instead of hand-assembling another wrapper-local payload map
- the wrapper no longer prints status for the projection-semantic-fragment
  queue it does not emit or own; its console surface is now limited to the
  hotspot-neighborhood queue plus the actual scan decision/violations it still
  produces
- workflow and local-repro orchestration now ratchet that narrower contract:
  policy-scanner-suite entrypoints must materialize
  `artifacts/out/policy_check_result.json` before invoking the wrapper, and
  wrapper-oriented PR/local-repro paths no longer stage unrelated child-owned
  artifacts just to satisfy a retired suite-era prerequisite bundle
- the active policy-scanner-suite wrapper path no longer depends on
  the retired runtime cache/load surface; it now calls `scan_policy_suite()`
  directly, consumes only child-owned result artifacts, and writes only the
  hotspot-neighborhood queue at the boundary instead of routing through
  runtime cache orchestration or publishing a suite-results compatibility file
- the runtime `scan_policy_suite()` surface no longer accepts
  `projection_fiber_semantics` at all; the wrapper resolves that child-owned
  semantic carrier at ingress and passes it directly to downstream reporting,
  so runtime scanning stays limited to the policy violations it computes
- wrapper-owned policy-result synthesis has now been removed from the
  policy-suite path entirely: the deprecated-nonerasability child check emits
  its own canonical `skip` result when baseline/current inputs are absent, and
  all child checks now own their own artifact contract while the wrapper fails
  closed if an expected artifact is missing
- that reporting-layer consumer now joins compiled bundle bindings back to
  canonical semantic rows and emits deterministic semantic previews
  (`spec_name`, `quotient_face`, `path`, `qualname`, `structural_path`) rather
  than only counts/spec names, so downstream reporting is reading actual
  semantic-plan structure
- lattice witness serialization for branch/frontier carriers is now owned by
  `FrontierWitness.as_payload()` in the canonical ASPF algebra, so the
  lattice-convergence substrate no longer imports the runtime policy-suite
  wrapper merely to validate or reuse a serializer contract
- the runtime policy-scanner-suite module is now scan-only: it no longer
  exports a suite-decision helper, and callers evaluate policy directly from
  the canonical `violations_by_rule` map instead of routing that boundary
  judgment back through the runtime scan surface
- boundary coverage now follows actual ownership on the queue seam:
  hotspot-queue helper contracts are pinned in hotspot-queue tests rather than
  indirectly through policy-scanner-suite wrapper tests
- `projection_exec.py` still executes the legacy row pipeline unchanged; the
  lowering layer exists to let the authoring surface converge on the semantic
  fragment without widening semantic behavior in the legacy runtime

Implementation rule:
- policy DSL must consume canonical carrier rows rather than infer semantics
  from ad hoc row shape
- current row-based projections continue via adapters during the compatibility
  window

Acceptance gate:
- one path uses canonical carrier rows for policy judgment while presentation
  output remains compatible with the current surface

### Phase 5. Cutover and ratchet

Cutover criteria:

- canonical carrier exists and is used by at least one end-to-end semantic path
- SHACL/SPARQL lowering is deterministic for that path
- DSL judgment consumes canonical carrier rows for that path
- intended presentation output remains stable for that path

Ratchet rules:

- no new semantic behavior may land directly in the legacy row-pipeline layer
- new semantic features must land as semantic ops first, presentation second
- legacy `ProjectionSpec` shaping remains adapter-only after cutover begins
- slower wrappers must consume valid preexisting child-owned canonical
  artifacts rather than rerunning the child checks that already own those
  artifacts
- wrappers that already hold a canonical payload in memory must pass that
  payload directly to downstream reporting consumers instead of re-reading
  wrapper-owned artifacts from disk
- queue/report tools must not retain suite-era ingress names once their actual
  contract is a generic source artifact or canonical payload
- queue/report artifacts must not retain suite-era field names once the same
  metadata is sourced from a generic artifact or canonical payload
- queue/report consumers must not preserve sidecar semantic-supplement inputs
  once the canonical source payload already carries the same semantic carrier
- queue/report CLIs must not preserve suite-era default artifact paths once the
  input contract is intentionally generic and explicit
- runtime/wrapper APIs must not preserve suite-era default artifact paths once
  every in-repo caller already passes an explicit artifact owner
- wrapper cache artifacts must not persist child-owned canonical policy
  results once those results are already explicit boundary inputs; cache only
  the wrapper-owned scan output plus the hashes needed to validate reuse
- runtime aggregate payloads must not preserve child-result nesting for
  semantic carriers once the same semantic data can be surfaced directly on
  the payload boundary
- queue/report artifacts must not materialize queue-owned semantic-summary
  blobs when direct semantic fields and preview rows already discharge the same
  reporting obligation
- shared semantic-summary helpers must not preserve dedicated parsers for
  retired materialized-summary payloads once no live consumer emits that shape
- shared semantic-summary helpers must not preserve redundant alias entrypoints
  once only one canonical decode surface remains
- shared semantic-summary decoders must not preserve wrapper-only ingress
  branches once the canonical artifact paths are explicit
- shared decoders must not retain retired suite-local embedding fallbacks once
  those embeddings are no longer emitted on real artifact paths; explicit local
  summary parsing should use a dedicated boundary, not a generic compatibility
  branch
- wrappers must not synthesize surrogate policy-result carriers from child
  process return codes once the child check owns canonical artifact emission;
  any admissible skip protocol must be emitted by the owning child check rather
  than synthesized by the aggregate wrapper

## 6. Public interfaces and planned types

The following planned surfaces must be named explicitly in implementation work,
even if final symbol names vary:

- `CanonicalWitnessedSemanticRow`
- `SemanticOp`
- `SemanticKernelCongruence`
- `SemanticWedge`
- `CompiledShaclPlan`
- `CompiledSparqlPlan`
- `PresentationPlan`

The following current surfaces are temporary adapters:

- current `ProjectionSpec` JSON pipeline
- current `projection_exec.py` runtime execution path
- current projection-fiber DSL rules when they operate as judgment-only without
  a canonical semantic carrier underneath

The policy DSL remains a judgment surface. It is not promoted to semantic
construction ownership by this RFC.

## 7. Test and acceptance plan

### 7.1 Semantic-law tests

Required semantic-law tests for the first implementation slice:

- reflection/closure reaches fixed point
- repeated semantic application is idempotent
- quotient faces preserve declared congruence semantics
- witness synthesis is bounded and does not invent unsupported evidence
- wedge composition preserves support/history/proof context identity

### 7.2 Compiler tests

Required compiler tests once Phase 3 lands:

- semantic fragment lowers deterministically to SHACL/SPARQL
- compiled SHACL/SPARQL preserve carrier identity and witness trace continuity
- lowered executables agree with the semantic fragment on representative TTL
  examples already documented in
  [`ttl_kernel_semantics.md#ttl_kernel_semantics`](./ttl_kernel_semantics.md#ttl_kernel_semantics)

### 7.3 Migration tests

Required migration tests:

- current user-facing presentation output remains stable where only rendering is
  intended
- semantic paths can be bridged from existing projection/policy surfaces
  without introducing dual-shape core ambiguity
- the policy DSL remains judgment-only over canonical carrier rows

### 7.4 Document validation

Required validation for this RFC and its follow-on doc updates:

- `scripts/policy/docflow_packetize.py`
- `scripts/policy/docflow_packet_enforce.py --check`
- `python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required`
- explicit cross-links to the TTL explainer and current ASPF/fibration docs

## 8. Defaults and non-goals

This RFC fixes the following defaults:

- existing ASPF/global identity and fibration substrate is the bootstrap layer
- TTL kernel is semantic foundation, not direct implementation blueprint
- document authority remains informative/implementation-driving
- semantic core stays narrow in v1
- carrier-first slice lands before compiler-first slice

The following are explicit non-goals for v1:

- replacing ASPF/global identity with a parallel semantic identity system
- treating every existing projection operator as semantic
- embedding full raw category-theory syntax as the public executable DSL
- letting SHACL/SPARQL become the source of semantic law rather than compiled
  realizers of that law

## 9. Immediate next implementation slice

The next implementation slice after this RFC is fixed:

1. introduce the canonical witnessed semantic carrier over the existing
   ASPF/fibration substrate
2. adapt one existing projection/policy path into that carrier
3. prove fixed-point/idempotence and witness-boundedness on that path
4. only then add SHACL/SPARQL compilation for that path
5. only then split friendly semantic vs presentation authoring more broadly

That order is the repo's default anti-drift path for projection semantic
convergence.
