---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: kernel_vm
doc_role: contract
doc_scope:
  - repo
  - analysis
  - tooling
  - semantics
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 57
  glossary.md#contract: 47
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev57; the module operates at module-import time within an explicit deadline_clock_scope (30s budget), consistent with the process-relative runtime invariant. The isolated prime registry is noted as a CSA-RGC gap and not a policy violation per se. No execution on untrusted surfaces."
  glossary.md#contract: "Reviewed glossary.md rev47; prime identity, ASPF, and GlobalIdentitySpace are used per glossary definitions. The module creates an isolated PrimeRegistry under namespace ttl_kernel_vm rather than participating in a shared GlobalIdentitySpace — this isolation is the documented architectural gap tracked under CSA-RGC."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_sections:
  kernel_vm: 1
doc_section_requires:
  kernel_vm:
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
doc_section_reviews:
  kernel_vm:
    POLICY_SEED.md#policy_seed:
      dep_version: 57
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed rev57 reviewed; import-time initialization with deadline scope and isolated prime namespace remain consistent with policy."
    glossary.md#contract:
      dep_version: 47
      self_version_at_review: 1
      outcome: no_change
      note: "Glossary rev47 reviewed; prime identity and ASPF terms used correctly; isolation gap noted."
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="kernel_vm"></a>
# Kernel VM Object Images

The `gabion.analysis.kernel_vm` package loads the repository's RDF/TTL semantic
ontology at module import time and produces ten identity-backed
`KernelVmObjectImage` instances — one per core semantic concept. These images
are the canonical reflection surface through which analysis code refers to
kernel semantic concepts by stable prime-backed integer identifiers rather than
by string names.

## Source ontologies

Three TTL files (under `in/`) are parsed at import:

| File | Role |
| --- | --- |
| `in/lg_kernel_ontology_cut_elim-1.ttl` | Class hierarchy, property domain/range constraints, `cat:` and `lg:` vocabulary |
| `in/lg_kernel_shapes_cut_elim-1.ttl` | SHACL shapes for query patterns (`lg:SelectQuery`, join/anti-join structure) |
| `in/lg_kernel_example_cut_elim-1.ttl` | Concrete instances grounding the class hierarchy |

The `cat:` prefix expresses category-theory primitives (`cat:Category`,
`cat:Morphism`, `cat:Functor`). The `lg:` prefix expresses language-kernel
concepts: rule objects, grammar states, execution transitions, and query
patterns.

## Exported object images

Ten `KernelVmObjectImage` instances are produced and exported as module-level
names:

| Name | Concept |
| --- | --- |
| `AugmentedRule` | A grammar rule augmented with semantic annotations |
| `ClosedRuleCell` | A rule cell with all open positions closed |
| `RulePolarity` | Polarity marker for a rule (positive/negative) |
| `WitnessDomain` | Domain of witness terms for a derivation |
| `PredicateDomain` | Domain of predicates admissible in a rule |
| `SupportReflection` | Reflection of supporting term sets |
| `SelectQuery` | Top-level query pattern |
| `TriplePattern` | Atomic RDF triple match |
| `JoinPattern` | Conjunction of patterns |
| `AntiJoinPattern` | Negation/anti-join of patterns |

Each `KernelVmObjectImage` carries:

```
object_id          — prime-backed integer identity for the object image itself
zone_id            — prime-backed integer for the ttl_kernel_vm zone
source_path_ids    — primes for the source TTL files that mention this object
class_term_id      — prime for the lg:<Name> class term
supporting_term_ids — primes for all terms across ontology + shape + example
ontology_term_ids  — primes for terms from the ontology file only
shape_term_ids     — primes for terms from the shapes file only
example_term_ids   — primes for terms from the examples file only
label              — the object name string
```

## Identity substrate

All prime assignments are made at import time using a `PrimeIdentityAdapter`
backed by a `PrimeRegistry`. Four namespaces are used within the isolated
registry:

| Namespace | Contents |
| --- | --- |
| `ttl_kernel_vm.zone` | Single zone token `"ttl_kernel_vm"` |
| `ttl_kernel_vm.path` | One entry per source TTL file path |
| `ttl_kernel_vm.term` | One entry per unique `lg:`/`cat:`/`sh:`/`rdf:`/`rdfs:`/`xsd:` term |
| `ttl_kernel_vm.object_image` | One entry per exported object name |

**Isolation gap (CSA-RGC):** This registry is a fresh `PrimeRegistry()` that
does not participate in the shared `GlobalIdentitySpace` used by policy rules,
planning chart identity, or the ASPF forest. Unifying these registries is
tracked as a blocker under the `CSA-RGC` workstream. Until that work lands,
consumers must treat kernel VM term IDs as opaque integers valid only within
the `ttl_kernel_vm.*` namespace family.

## Import-time initialization

All parsing and prime assignment happens at module import, wrapped in:

```python
with deadline_scope(Deadline.from_timeout_ms(30_000)):
    with deadline_clock_scope(MonotonicClock()):
        ...
```

Callers that import this module must ensure a `MonotonicClock` is active (or
use `deadline_scope` around the import) to avoid a missing-clock error. The
30-second budget is intended to be ample for local TTL parsing; if the budget
is exceeded the import raises.

The module also validates at parse time that each exported class term
(`lg:AugmentedRule`, etc.) is present in the ontology file and that at least
one supporting term is found. A missing term raises `RuntimeError` at import.

## Consumers

The object images are imported by:

- `src/gabion/analysis/projection/semantic_fragment.py`
- `src/gabion/analysis/projection/semantic_fragment_compile.py`
- `src/gabion/analysis/projection/projection_semantic_lowering.py`
- `src/gabion/analysis/aspf/aspf_lattice_algebra.py`

These consumers reference object images by name (e.g. `AugmentedRule`) and use
their integer IDs as stable keys into analysis data structures.
