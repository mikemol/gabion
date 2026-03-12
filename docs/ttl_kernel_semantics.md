---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: ttl_kernel_semantics
doc_role: architecture
doc_scope:
  - repo
  - semantics
  - projection
  - policy
  - ttl
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/audits/projection_spec_history_ledger.md
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 55
  glossary.md#contract: 44
  docs/audits/projection_spec_history_ledger.md: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev55 (correctness-by-construction and drift-control framing remain aligned with this explainer's scope)."
  glossary.md#contract: "Reviewed glossary.md rev44 (witness, evidence, and projection terminology remain aligned with this explainer's usage)."
  docs/audits/projection_spec_history_ledger.md: "Reviewed projection-spec history ledger rev1 (current ProjectionSpec remains a quotient/erasure carrier, which this explainer positions relative to the TTL model)."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_sections:
  ttl_kernel_semantics: 1
doc_section_requires:
  ttl_kernel_semantics:
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
    - docs/audits/projection_spec_history_ledger.md
doc_section_reviews:
  ttl_kernel_semantics:
    POLICY_SEED.md#policy_seed:
      dep_version: 55
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed rev55 reviewed; informative framing remains aligned with repo governance."
    glossary.md#contract:
      dep_version: 44
      self_version_at_review: 1
      outcome: no_change
      note: "Glossary rev44 reviewed; witness/evidence language remains aligned."
    docs/audits/projection_spec_history_ledger.md:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "ProjectionSpec history ledger rev1 reviewed; current quotient framing remains an accurate starting point."
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="ttl_kernel_semantics"></a>
# TTL Kernel Semantics

This document explains the TTL kernel materials under [`in/`](../in/) and
positions them relative to the current repository surfaces:

- [`src/gabion/analysis/projection/projection_spec.py`](../src/gabion/analysis/projection/projection_spec.py)
- [`src/gabion/analysis/projection/projection_exec.py`](../src/gabion/analysis/projection/projection_exec.py)
- [`docs/projection_fiber_rules.yaml`](./projection_fiber_rules.yaml)
- [`src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py`](../src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py)

The shortest reading is:

- the TTL files define a semantic core,
- SHACL and SPARQL are embedded as executable realizers of that core,
- friendly repo DSLs sit above that layer,
- presentation should be derived from semantic normal form, not treated as a peer.

## 1. What the input artifacts are

The kernel is split into four artifacts.

| Artifact | Role |
| --- | --- |
| [`in/lg_kernel_ontology_cut_elim-1.ttl`](../in/lg_kernel_ontology_cut_elim-1.ttl) | Defines the ontology vocabulary: categories, rule objects, query AST objects, polarity/closure objects, quotient objects, history/proof objects, and policy-indexed proof objects. |
| [`in/lg_kernel_shapes_cut_elim-1.ttl`](../in/lg_kernel_shapes_cut_elim-1.ttl) | Adds SHACL node shapes plus graph-native SPARQL query structures and lowered `sh:select` renderings. |
| [`in/lg_kernel_example_cut_elim-1.ttl`](../in/lg_kernel_example_cut_elim-1.ttl) | Instantiates one end-to-end example that walks from an augmented rule, through polarity and quotient recovery, into reflective SHACL boundaries, history-stamped queries, adjoint `EXISTS`/`NOT EXISTS`, and policy-indexed proofs. |
| [`in/TTL_Derived_Homology_Report.md`](../in/TTL_Derived_Homology_Report.md) | A mechanically derived report over the three TTL graphs. It is a digest of the graphs, not the source of truth. |

The report is useful because it exposes the staged shape of the example:

- rule and query objects
- Galois / polarity package
- quotient recovery and concept lattice
- theorem / proof-obligation layer
- reflective SHACL boundary
- proof-carrying query with immutable history
- adjoint `EXISTS` / `NOT EXISTS` over history-stamped wedges
- policy-indexed proof calculus

## 2. What the ontology is actually saying

The ontology is not only a data model. It is already a semantic stack.

### 2.1 Kernel rule object

The base kernel centers an `lg:AugmentedRule` that binds:

- a syntax clause,
- a typing clause indexed over syntax,
- a categorical clause indexed over typing,
- and any introduced identifiers.

The initial example in [`in/lg_kernel_example_cut_elim-1.ttl`](../in/lg_kernel_example_cut_elim-1.ttl)
shows this with `lg:EatRule`, `lg:eatSyntaxClause`, `lg:eatTypingClause`,
and `lg:eatCategoricalClause`.

This makes the rule itself the first-class object. Syntax, typing, and
categorical semantics are recoverable faces of that object, not unrelated
artifacts.

### 2.2 Query AST and SHACL lowering are both first-class

The ontology does not treat SPARQL as opaque text. It models:

- `lg:SelectQuery`
- `lg:TriplePattern`
- `lg:JoinPattern`
- `lg:AntiJoinPattern`
- `lg:QueryRendering`
- `lg:SPARQLConstraint`

The shapes file then connects the graph-native AST to a standard lowered string
via `sh:select`.

That matters because the query has two forms at once:

- an intensional graph-native form,
- an interoperable serialization form.

This is already a reflective boundary pattern in small.

### 2.3 Galois / polarity is the semantic heart

The ontology then enriches `AugmentedRule` with a polarity package:

- `lg:WitnessDomain`
- `lg:PredicateDomain`
- `lg:IncidenceRelation`
- `lg:DerivationOperator`
- `lg:ClosureOperator`
- `lg:SupportReflection`
- `lg:RulePolarity`
- `lg:ClosedExtent`
- `lg:ClosedIntent`
- `lg:ClosedRuleCell`

The example names these concretely as `lg:eatWitnessDomain`,
`lg:eatPredicateDomain`, `lg:eatIncidence`, `lg:eatWitnessDerivation`,
`lg:eatPredicateDerivation`, `lg:eatExtentClosure`, and `lg:eatIntentClosure`.

The important point is structural:

- a rule has a witness side and a predicate side,
- derivation runs both ways,
- closure on each side picks fixed points,
- a `ClosedRuleCell` is the resulting bifacial object.

This is why closure and fixed-point language is not ornamental here. The
ontology already encodes it as the way rule-cells are formed.

### 2.4 Quotient recovery is explicit, not implicit

The example then defines kernel congruences and quotient projections:

- syntax projection
- typing projection
- semantic projection
- extent projection
- intent projection

These are not generic views. They are stated as `lg:QuotientProjection`
instances with explicit `lg:KernelCongruence` witnesses.

The concept lattice is then generated from the rule polarity, and proof
obligations are attached to lattice operations and quotient universality.

That gives a much stronger reading of projection than the current repo surface:
projection is a recoverable face of a closed object under an explicit kernel,
not merely a row-shaping pipeline.

## 3. What the shapes file adds

The shapes file is the executable boundary layer.

It does two jobs at once:

1. It constrains the ontology objects themselves with SHACL node shapes.
2. It embeds graph-native SPARQL/anti-join structure and also lowers it into
   standard `sh:select` text.

This means SHACL and SPARQL are not post-hoc utilities. They are already
embedded as realizers of the semantic model.

That is the right architectural reading:

- category-theoretic objects give the denotational layer,
- SHACL/SPARQL provide executable realization,
- friendly repo DSLs should compile downward into that fragment.

The ontology even says this directly for `sh:select`: the primary intensional
representation is the query AST, while `sh:select` remains the lowered
interoperable serialization.

## 4. What the example file demonstrates end-to-end

The example file is the most important piece because it shows the intended
composition, not just the vocabulary.

### 4.1 Closed rule-cell recovery

`lg:EatRule` becomes a `lg:ClosedRuleCell` with:

- a polarity,
- a closed extent,
- a closed intent.

From that closed cell, the example recovers syntax, typing, semantic, extent,
and intent faces through explicit quotient projections.

### 4.2 Proof obligations are attached to the structure

The example does not stop at construction. It adds:

- meet/join associativity obligations,
- commutativity obligations,
- idempotence obligations,
- absorption obligations,
- boundedness obligations,
- order-compatibility obligations,
- quotient universality obligations.

This is correctness-by-construction in graph form: the semantic object carries
its obligations instead of leaving them implicit in prose.

### 4.3 Reflective SHACL boundary

The example then defines:

- an internal proof calculus,
- a SHACL denotation category,
- a lowering functor,
- a reflection functor,
- a denotational quotient by observational equivalence,
- obligations for soundness, surjectivity, reflective quotient, and
  conservative projection.

This is the cleanest place to see how SHACL belongs in the model. It is not the
source of truth; it is the executable denotational boundary of an internal
proof calculus.

### 4.4 History, wedges, and constructive empty results

The next stage introduces:

- immutable history lineage,
- history states and historical assertions,
- history stamps,
- proof-carrying queries,
- query wedge products,
- existential images,
- empty-result proof queries.

The wedge is the important new operator for the repo-level discussion. The
example uses a `QueryWedgeProduct` to compose:

- a proof-carrying query,
- a history stamp,
- and supporting historical assertions.

That makes absence and emptiness constructive: an empty result is not just
"nothing returned", it is a traced object with support context.

### 4.5 Adjoint `EXISTS` / `NOT EXISTS`

The example then makes the adjoint structure explicit:

- context projection,
- reindexing,
- existential image,
- adjoint pair,
- support reflection,
- negated existential image.

This is why the TTL material matters for `ProjectionSpec`: it does not model
`EXISTS` / `NOT EXISTS` as ad hoc query syntax. It models them as operators over
history-stamped witness contexts with explicit proof obligations.

### 4.6 Policy-indexed proof calculus

Finally, the example adds:

- semantic policy bundles,
- truth-value structures,
- policy-scoped judgments,
- policy certificates,
- policy-indexed derivations,
- policy-preserving lowering obligations,
- policy-reflective compatibility obligations.

This is the strongest statement in the stack:

- semantic construction happens first,
- policy selection indexes the resulting proof objects,
- judgments are made over policy-scoped, witness-carrying objects.

That matches the desired repo direction for the DSL: judgment should be a layer
over closed, witness-complete objects, not the place where those objects are
invented.

## 5. Positioning relative to current repo surfaces

The current repo does not yet implement the full TTL model.

### 5.1 What exists today

Today the repository already has:

- a projection carrier and runtime in
  [`src/gabion/analysis/projection/projection_spec.py`](../src/gabion/analysis/projection/projection_spec.py)
  and
  [`src/gabion/analysis/projection/projection_exec.py`](../src/gabion/analysis/projection/projection_exec.py)
- a policy DSL and registry in `src/gabion/policy_dsl/`
- projection-fiber judgment rules in
  [`docs/projection_fiber_rules.yaml`](./projection_fiber_rules.yaml)
- a semantic collector for projection-fiber convergence in
  [`src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py`](../src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py)

That means the repo already has judgment over witness rows, but it does not yet
have the TTL kernel as its canonical transform carrier.

### 5.2 What `ProjectionSpec` currently is

Current `ProjectionSpec` is a small row-pipeline DSL over plain JSON-like
relations. It supports operations such as:

- `select`
- `project`
- `count_by`
- `traverse`
- `sort`
- `limit`

This is useful, but it is weaker than the TTL model in two ways:

1. it does not operate over a canonical witness-carrying closed object,
2. it mixes semantic and presentation concerns.

The history ledger currently describes `ProjectionSpec` as a quotient morphism
that erases evidence. The TTL stack suggests a stricter future reading:

- projection should act over the same witnessed carrier it emits,
- quotient should be explicit,
- witness synthesis should be bounded and lawful,
- presentation should factor through semantic normal form.

### 5.3 What the policy DSL currently is

The policy DSL is closer to the TTL direction than `ProjectionSpec` is, because
it already judges over witness carriers instead of directly shaping payloads.

But it is still only part of the picture. The TTL model suggests that the full
stack should be:

1. semantic construction over a canonical witnessed carrier,
2. executable realization via SHACL/SPARQL,
3. DSL judgment over the resulting closed objects,
4. presentation as a derived image.

## 6. Working interpretation for future repo design

The TTL materials support the following interpretation.

### 6.1 Semantic core

The semantic core should define:

- the witnessed carrier,
- Galois / closure structure,
- quotient relations and quotient faces,
- wedge/context composition,
- witness-synthesis admissibility rules,
- judgment obligations.

This is the layer that should be law-governed.

### 6.2 Executable substrate

SHACL and SPARQL should realize the semantic core, not replace it.

- SHACL validates shapes and obligations.
- SPARQL realizes extraction, support images, quotient-face checks, and witness
  queries.

The ontology and shapes files already model this relationship.

### 6.3 Friendly authoring layers

Friendly repo surfaces should sit on top:

- `ProjectionSpec`
- policy DSL sources
- any higher-level repo-native declarative syntax

These layers should compile downward into the semantic fragment and then into
SHACL/SPARQL realizations.

### 6.4 Presentation layer

Presentation should not be a peer transform algebra. It should be the image of
closed semantic objects under rendering/reporting functors.

Practically, that means operations like `sort` and `limit` should be treated as
presentation-layer concerns unless they are explicitly part of a semantic law.

## 7. What the TTL files strongly support, and what they do not yet settle

### Strongly supported by the TTL materials

- projection as quotient recovery from a closed object
- witness/predicate duality with Galois-style closure
- correctness obligations attached to semantic objects
- reflective SHACL boundaries with lowering/reflection obligations
- history-stamped wedge composition for support-sensitive queries
- policy-indexed proof objects over truth-value structures

### Not yet settled for the current repo

- the exact canonical carrier type for repo witness rows
- the exact admissibility law for witness synthesis in the Python/runtime layer
- how much of current `ProjectionSpec` survives as semantic calculus versus
  presentation algebra
- whether wedges should appear directly as first-class runtime objects or first
  as compiled/query-level realizations

Those are design questions. The TTL files do not remove them, but they do
constrain the answer space sharply.

## 8. Recommended reading order

When reading the kernel for design work, use this order:

1. [`in/lg_kernel_ontology_cut_elim-1.ttl`](../in/lg_kernel_ontology_cut_elim-1.ttl)
   to learn the vocabulary.
2. [`in/lg_kernel_shapes_cut_elim-1.ttl`](../in/lg_kernel_shapes_cut_elim-1.ttl)
   to see how SHACL/SPARQL are embedded as executable realizations.
3. [`in/lg_kernel_example_cut_elim-1.ttl`](../in/lg_kernel_example_cut_elim-1.ttl)
   to see the semantic composition end-to-end.
4. [`in/TTL_Derived_Homology_Report.md`](../in/TTL_Derived_Homology_Report.md)
   as a mechanical digest and lookup aid.

For repo design discussion, the most relevant staged blocks in the derived
report are:

- Galois / polarity example
- quotient recovery and concept lattice example
- reflective SHACL boundary example
- proof-carrying query / immutable history example
- adjoint `EXISTS` / `NOT EXISTS` over stamped-history wedges example
- policy-indexed proof calculus example

## 9. Practical takeaway

The TTL kernel should be read as the semantic foundation that current repo
projection/policy surfaces only partially expose.

In that reading:

- category theory provides the denotational laws,
- SHACL and SPARQL provide executable realizations,
- `ProjectionSpec` and the policy DSL are friendly authoring/judgment layers,
- presentation should derive from semantic normal form,
- and drift control comes from keeping those layers separate and law-governed.

That is the main reason these TTL files matter to the repo. They do not just
offer a richer notation. They define the shape of a correctness-by-construction
semantic stack that future projection and policy work can converge toward.
