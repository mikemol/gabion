---
doc_revision: 53
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: influence_index
doc_role: index
doc_scope:
  - repo
  - governance
  - planning
  - documentation
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - glossary.md#exception_obligation
  - glossary.md#handledness_witness
  - CONTRIBUTING.md#contributing_contract
  - README.md#repo_contract
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  glossary.md#exception_obligation: 1
  glossary.md#handledness_witness: 1
  CONTRIBUTING.md#contributing_contract: 2
  README.md#repo_contract: 2
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev2 (forward-remediation order, ci_watch failure-bundle durability, and enforced execution-coverage policy wording)."
  glossary.md#contract: "Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline)."
  glossary.md#exception_obligation: "Reviewed glossary.md#exception_obligation rev1 (exception obligation status + evidence linkage)."
  glossary.md#handledness_witness: "Reviewed glossary.md#handledness_witness rev1 (handledness witness requirements + handler boundary)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev2 (two-stage dual-sensor cadence, correction-unit validation stack, and strict-coverage trigger guidance)."
  README.md#repo_contract: "Reviewed README.md rev2 (removed stale ASPF action-plan CLI/examples; continuation docs now state/delta only)."
doc_sections:
  influence_index: 3
doc_section_requires:
  influence_index:
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
    - glossary.md#exception_obligation
    - glossary.md#handledness_witness
    - CONTRIBUTING.md#contributing_contract
    - README.md#repo_contract
doc_section_reviews:
  influence_index:
    POLICY_SEED.md#policy_seed:
      dep_version: 2
      self_version_at_review: 3
      outcome: no_change
      note: "Policy seed rev2 reviewed; governance obligations remain aligned."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 3
      outcome: no_change
      note: "Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline)."
    glossary.md#exception_obligation:
      dep_version: 1
      self_version_at_review: 3
      outcome: no_change
      note: "Reviewed glossary.md#exception_obligation rev1 (exception obligation status + evidence linkage)."
    glossary.md#handledness_witness:
      dep_version: 1
      self_version_at_review: 3
      outcome: no_change
      note: "Reviewed glossary.md#handledness_witness rev1 (handledness witness requirements + handler boundary)."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 2
      self_version_at_review: 3
      outcome: no_change
      note: "Contributor contract rev2 reviewed; dual-sensor cadence and correction gates remain aligned."
    README.md#repo_contract:
      dep_version: 2
      self_version_at_review: 3
      outcome: no_change
      note: "Repo contract rev2 reviewed; command and artifact guidance remains aligned."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="influence_index"></a>

# Influence Index (`in/` → `out/`)

This index records which inbound documents (`in/`) have been reviewed, and how
(or whether) they have been reflected in `out/`, `docs/`, or the checklist. It
is a lightweight bridge between the inbox and the rest of the repo.

Normative anchors: [POLICY_SEED.md#policy_seed](POLICY_SEED.md#policy_seed), [glossary.md#contract](glossary.md#contract), [glossary.md#exception_obligation](glossary.md#exception_obligation), [glossary.md#handledness_witness](glossary.md#handledness_witness), [CONTRIBUTING.md#contributing_contract](CONTRIBUTING.md#contributing_contract), [README.md#repo_contract](README.md#repo_contract).

Status legend:
- **untriaged**: not yet reviewed.
- **queued**: reviewed; awaiting adoption.
- **partial**: partially adopted; remaining items tracked in checklist.
- **adopted**: reflected in `out/`/`docs/`/code.
- **rejected**: explicitly out of scope.

## Inbox entries

- in/in-1.md — **adopted** (core dataflow audit + type/constant audits + tiered reporting implemented; see `docs/sppf_checklist.md`.)
- in/in-2.md — **partial** (import/alias/class hierarchy resolved; dynamic dispatch + decorator transparency remain limited.)
- in/in-3.md — **adopted** (alias‑aware forwarding / identity tracking implemented.)
- in/in-4.md — **adopted** (identity vs. symbol and chain‑of‑custody logic implemented.)
- in/in-5.md — **partial** (wildcard forwarding, import resolution, synthesis, schedule done; partial‑application merging still open.)
- in/in-6.md — **adopted** (deterministic import resolution implemented.)
- in/in-7.md — **adopted** (symbol table + aliasing + propagation implemented in dataflow audit.)
- in/in-8.md — **adopted** (protocol/dataclass synthesis + type aggregation implemented.)
- in/in-9.md — **adopted** (topological refactor scheduling + SCC detection implemented.)
- in/in-10.md — **adopted** (root anchoring, excludes, ignore params, external filter, strictness, naming stubs implemented.)
- in/in-11.md — **adopted** (duplicate of in-10; same adoption status.)
- in/in-12.md — **partial** (LSP integration + LibCST refactor engine present; analysis still AST‑based.)
- in/in-13.md — **adopted** (LSP‑first architecture + CLI as thin client implemented.)
- in/in-14.md — **adopted** (repo scaffold + CLI/LSP split aligned with current structure.)
- in/in-15.md — **partial** (decision surfaces + boundary classification + tier enforcement; context suggestions; remaining rewrite work; SPPF/GH-60, GH-61.)
- in/in-16.md — **adopted** (structural snapshots/diffing/metrics + baseline ratchet implemented; SPPF/GH-62/63/64/23.)
- in/in-17.md — **partial** (subtree reuse detection + hashing in structure snapshot; remaining lemma synthesis hooks; SPPF/GH-65.)
- in/in-18.md — **partial** (value‑encoded decision detection + reports + glossary warnings + decision snapshot/diff + audit snapshot capture + rebranch suggestions; remaining rewrite work; SPPF/GH-66.)
- in/in-19.md — **partial** (proposition model + assert-based invariants + emitter hooks implemented; dependent‑type synthesis remains; SPPF/GH-67.)
- in/in-20.md — **partial** (prime registry + canonical type keys + arithmetic ops + glossary warnings + reverse mapping + hybrid bitmask + nested ctor registry implemented; remaining rewrite work; SPPF/GH-68.)
- in/in-21.md — **partial** (longform expansion of in-20; nested ctor registry + hybrid bitmask + reverse mapping + glossary warnings + arithmetic ops implemented; remaining rewrite work; SPPF/GH-68.)
- in/in-22.md — **partial** (ASPF/SPPF equivalence framing; dimensional fingerprints incl. provenance/synth dimensions; entropy-controlled synthesis tracked in SPPF GH-70/GH-71/GH-72.)
- in/in-23.md — [**adopted**](docs/sppf_checklist.md#in-23-aspf-carrier-formalization) (completed landing: deterministic synth/provenance artifacts + reloadable basis + snapshot selectors, with module/test anchors in the checklist entry; SPPF/GH-73.)
- in/in-24.md — **adopted** (deadness evidence artifacts emitted from constant-flow analysis + JSON/report/LSP/snapshots + determinism/schema tests; SPPF/GH-74.)
- in/in-25.md — **adopted** (coherence evidence artifacts emitted from glossary ambiguity + JSON/report/LSP/snapshots + determinism/schema tests; SPPF/GH-75.)
- in/in-26.md — **partial** (proof-carrying rewrite plans emitted + verified incl. exception-obligation non-regression; remaining rewrite kinds tracked; SPPF/GH-76, GH-78, GH-79.)
- in/in-27.md — **partial** (exception obligations emitted with deadness/handledness witnesses; exception-aware rewrite acceptance predicates implemented; remaining handledness refinement tracked; SPPF/GH-77, GH-80.)
- in/in-28.md — **adopted** (in_step template discipline enforced; docflow structure and review requirements codified.)
- in/in-29.md — **partial** (test evidence carrier + dominance/equivalence + obsolescence/suggestions projections implemented; dominance deltas pending.)
- in/in-30.md — **partial** (implemented: suite ambiguity projections + suite-order SpecFacet path + tick-budget deadline propagation; open: SuiteSite-native loop obligation enforcement; deferred: phase-3/4 decision-surface migration; SPPF/GH-85, GH-87, GH-88.)
- in/in-31.md — **partial** (implemented: suite-order ProjectionSpec/SpecFacet quotient path; open: explicit quotient/internment regression harness; deferred: internal broad-type lint tightening impact; SPPF/GH-85, GH-89.)
- in/in-32.md — **queued** (hypothetical/non-normative Gödel-numbering exploration; acknowledged, but not a controlling contract for implementation or CI at this time.)
- in/in-33.md — [**partial**](docs/sppf_checklist.md#in-33-pattern-schema-unification) (implemented: PatternSchema/PatternInstance/PatternResidue carriers + unified schema suggestion/residue pipeline + Tier-2 residue ratchet/metafactory reification gate; open: execution-rule coverage breadth.)
- in/in-34.md — [**partial**](docs/sppf_checklist.md#in-34-lambda-callable-sites) (synthetic lambda function sites are indexed and used for direct/bound lambda call resolution, while broader closure/alias cases still fall back conservatively.)
- in/in-35.md — [**partial**](docs/sppf_checklist.md#in-35-dict-key-carrier-tracking) (dict key normalization now supports name-bound constants and records explicit unknown-key carrier evidence for non-recoverable keys; supported key grammar remains deliberately conservative.)
- in/in-36.md — [**adopted**](docs/sppf_checklist.md#in-36-starred-dataclass-call-bundles) (dataclass call-bundle extraction now decodes deterministic starred literals for `*` and `**` and emits unresolved-starred witnesses for dynamic payloads.)
- in/in-37.md — [**adopted**](docs/sppf_checklist.md#in-37-dynamic-dispatch-uncertainty) (callee resolution now distinguishes `unresolved_dynamic` from unresolved internal/external states and emits a dedicated `unresolved_dynamic_callee` obligation kind.)
- in/in-38.md — **partial** (formalized ASPF mutation log-structured persistence design with partial implementation now landed: protobuf payload records, protobuf-defined filesystem envelope projection, and tar-packaged archive transport with migration compatibility behavior; checklist tracks remaining completion work; SPPF/GH-196.)
- in/in-46.md — **queued** (fiber-coloring model proposal for taint-permitted versus strict regions and explicit crossing laws.)
- in/in-47.md — **queued** (erasure morphism witness schema proposal with deterministic witness identity contract.)
- in/in-48.md — **queued** (boundary registry proposal defining legal erasure loci and expiry-managed temporary boundaries.)
- in/in-49.md — **queued** (unified taint ledger proposal across state, baseline, and delta carriers.)
- in/in-50.md — **queued** (boolean ambiguity taint/erasure proposal for strict-core ingress control.)
- in/in-51.md — **queued** (type ambiguity erasure proposal for ingress normalization into strict contracts.)
- in/in-52.md — **queued** (control ambiguity budget and essential-branch justification proposal with ratchet framing.)
- in/in-53.md — **queued** (promotion protocol proposal for experimental to boundary to strict-core readiness.)
- in/in-54.md — **queued** (quotiented scenario algebra contract for profile-indexed test optimization.)
- in/in-55.md — **queued** (corpus projection and quotient migration protocol for deterministic class assignment.)
- in/in-56.md — **queued** (graded execution strategy for quotient classes with deterministic routing and budgets.)
- in/in-57.md — **queued** (governance ratchets for quotient integrity, assignment debt, and routing quality floors.)
- in/in-58.md — **queued** (convergence and promotion lifecycle protocol for quotient plus graded optimization rollout.)
