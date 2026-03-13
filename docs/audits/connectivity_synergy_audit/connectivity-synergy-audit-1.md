---
doc_revision: 5
doc_id: connectivity_synergy_audit
doc_role: audit
doc_scope:
  - repo
  - architecture
  - refactor
doc_authority: informative
doc_requires:
  - docs/architecture_zones.md#architecture_zones
  - README.md#repo_contract
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
doc_reviewed_as_of:
  docs/architecture_zones.md#architecture_zones: 1
  README.md#repo_contract: 2
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
doc_review_notes:
  docs/architecture_zones.md#architecture_zones: "Used boundary/core zoning to classify weak couplings and consolidation candidates."
  README.md#repo_contract: "Validated status/model notes so recommendations stay in prototype-safe zones."
  POLICY_SEED.md#policy_seed: "Checked policy constraints so consolidation suggestions preserve core/boundary and command-carrier obligations."
  glossary.md#contract: "Applied tier/bundle contract to prioritize reification and reusable protocol extraction."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

# Connectivity + Synergy Audit

## Scope and method

This audit looked for low-connectivity and low-synergy surfaces across `src/gabion/**`, `src/gabion_governance/**`, `scripts/**`, and governance markdown metadata, with a specific focus on:

1. import-graph isolation (where modules are minimally linked),
2. wrapper duplication (same behavior exposed in multiple places),
3. high-line-count/high-branch hotspots likely carrying merged concerns, and
4. policy logic that is still code-hardcoded instead of frontmatter-driven.

The analysis used repository-local Python (`mise exec -- python`) with AST import extraction, plus `scripts/complexity_audit.py`.


## Audit rationale (expanded)

This audit is not only a hotspot inventory; it is intended to be a **sequencing instrument** for convergence.

- **Why connectivity first:** low-connectivity surfaces often hide implicit duplicate semantics. If those surfaces are mapped early, later consolidation can preserve behavior while reducing line count through explicit shared carriers.
- **Why frontmatter migration is structural, not cosmetic:** policy constants split across Python and markdown create dual sources of truth. Converging policy intent into declarative carriers reduces drift, and gives ASPF ingestion a stable contract to project over.
- **Why ASPF-first is the leverage center:** when analysis reduces to ASPF enrichments + queries, every new substrate adapter increases capability without multiplying bespoke policy logic.
- **Why merge-safe multi-registry interning matters:** parallel ingestion is practical for heterogeneous origins, but only if remap/overlap traces are deterministic and auditable. This converts ingestion diversity from a governance risk into a reusable integration primitive.

The roadmap below is therefore ordered by **structural leverage propagation**: each step should produce contracts/artifacts that make subsequent steps smaller, safer, and more automatable; each later step should feed refinements back into earlier contracts via explicit deltas.

## Where connectivity is weakest

### 1) Governance implementation is structurally separated from runtime core

`src/gabion_governance/governance_audit_impl.py` is effectively a standalone governance engine, while `src/gabion_governance/__init__.py` contributes no integration surface of its own.

**Synergy gap:** normalization, sorting, and projection helpers overlap with semantic-core patterns but are not uniformly abstracted into shared tier-1 protocol carriers.

### 2) Runtime/tooling islands retain low fan-in leaves

Import-graph sampling flagged several modules with near-zero graph degree (package `__init__` modules and some utility leaves). Low fan-in can be valid, but repeated low-fan-in helpers indicate opportunities to collapse one-off carriers into fewer boundary-normalization hubs.

**Synergy gap:** runtime helpers and command/tooling helpers still form partly parallel tracks instead of one explicit contract bundle per concern.

### 3) Script/tooling dual-surface has dependency inversion

Multiple `scripts/*.py` files are thin wrappers over `src/gabion/tooling/*`; at the same time, `src/gabion/tooling/policy_scanner_suite.py` dynamically imports modules from `scripts/*.py`.

**Synergy gap:** this inverts the desired direction (`scripts` as wrappers only), increases drift risk, and leaves line-count reuse unrealized.

## Frontmatter migration gaps

The repository has begun moving policy semantics into markdown frontmatter, but key checks remain partially code-local.

### Gap A: governance document inventory is hardcoded in Python

`governance_audit_impl.py` still hardcodes governance doc lists and required frontmatter field sets (`CORE_GOVERNANCE_DOCS`, `GOVERNANCE_DOCS`, `REQUIRED_FIELDS`, `LIST_FIELDS`, `MAP_FIELDS`).

**Merge/subsume opportunity:** move this registry into a single markdown/YAML policy carrier and have `governance_audit_impl.py` consume that source. This centralizes policy intent and reduces code+doc divergence.

### Gap B: policy scanner rule registry is code-fixed and script-coupled

`policy_scanner_suite.py` statically encodes rule IDs and dynamically loads script modules.

**Merge/subsume opportunity:** represent scanner rule metadata as frontmatter/declarative config and resolve package-native implementations from that registry. Keep scripts as launch shims only.

### Gap C: wrapper pattern is repeated, not generated

Several script entrypoints share the same import/export/`main()` bridge shape.

**Merge/subsume opportunity:** define a tiny declarative launcher manifest (module + symbol) and generate wrappers, or implement a single generic wrapper command that dispatches by name.

## ASPF-first convergence target (requested direction)

The long-term unification path should be: **analysis == ASPF enrichments + ASPF queries**.

### Target model

- Treat every analysis source as an ingress adapter that emits ASPF-normalized events/artifacts.
- Keep semantic logic in reusable ASPF query/projection layers rather than per-surface bespoke scanners.
- Treat pyast/libCST/markdown/yaml/frontmatter/html as parser front-ends feeding one semantic substrate.

### Immediate implications

1. **Ingestion normalization:** add/extend adapters that map markdown+frontmatter and YAML governance docs into ASPF carriers.
2. **Rule execution unification:** express policy/gov checks as ASPF queries (or projections over ASPF-enriched state) instead of hardcoded file lists + ad-hoc walkers.
3. **Artifact parity:** require governance/docflow checks to publish ASPF snapshots/deltas with the same continuation semantics used in dataflow analysis.
4. **Wrapper collapse:** script entrypoints should invoke package commands that all resolve to ASPF enrichment + query pipelines.


### Multi-registry ASPF interning strategy (prime-space merge)

A single global ingress interning point is idealized but may be impractical once heterogeneous substrate adapters run in parallel. A viable architecture is:

1. **Origin-local registries:** each adapter lane (pyast/libCST/markdown/yaml/frontmatter/html) maintains its own ASPF node + prime registry with deterministic local assignment.
2. **Canonical identity witnesses:** each local node emits a stable identity witness tuple (for example: normalized kind, canonical payload hash, provenance class, and scope key).
3. **Merge remap pass:** a post-hoc merger computes overlap classes from witnesses and builds remap tables `origin_node_id -> merged_node_id` plus `origin_prime -> merged_prime`.
4. **Traceable overlap ledger:** persist merge evidence showing exact identity matches, conflicts, and unresolved residues so downstream queries can distinguish exact merges from approximate or deferred joins.

This keeps ingestion parallel and origin-friendly while still converging to one queryable ASPF substrate.

### Merge safety invariants for parallel registries

To keep prime remapping sound and auditable:

- **No silent coalescence:** merged identities require explicit witness agreement; otherwise preserve distinct nodes and log a non-merge witness.
- **Deterministic remap materialization:** given the same origin registries and witness sets, remap tables must be byte-for-byte stable.
- **Bidirectional provenance:** every merged node must retain links back to all origin node IDs/primes that contributed.
- **Conflict stratification:** contradictory payloads with partial witness overlap should produce structured conflict nodes, not fallback overwrite behavior.

These invariants make post-hoc convergence compatible with policy-grade forensics and continuation deltas.

### Pandoc exploration note

Pandoc can be evaluated as a boundary parser/normalizer for markdown/html conversion into a single intermediate representation, but should remain an ingress adapter only. Deterministic semantics must still be encoded in Gabion-native ASPF carriers and query contracts.

## Highest-yield consolidation targets

### A) Command orchestration core (very high)

`src/gabion/server_core/command_orchestrator.py` contains the largest functions in the repo (e.g., `execute_command_total` at 608 lines, plus additional 300-480 line functions).

**Merge/subsume opportunity:** split orchestration into a protocolized phase pipeline where emitters, success/failure projection, and progress adaptation are tier-1 dataclass bundles consumed by smaller reducers.

### B) Dataflow analysis pipeline/reporting core (very high)

`src/gabion/analysis/dataflow_pipeline.py` and `src/gabion/analysis/dataflow_audit.py` contain very large, branch-heavy functions (e.g., `analyze_paths` at 578 lines; several 250-340 line helper blocks in `dataflow_audit`).

**Merge/subsume opportunity:** subsume repeated phase-transition/report-tail logic into explicit decision protocols reused across `dataflow_pipeline`, `dataflow_reporting`, and `dataflow_audit`.

### C) Policy scanning surface (high)

`src/gabion/tooling/policy_scanner_suite.py` currently loads rule implementations from `scripts/*.py` dynamically.

**Merge/subsume opportunity:** migrate rule visitors into package-native modules and keep scripts as minimal CLI shims.

### D) Frontmatter registry convergence (high)

Governance checks are partially frontmatter-aware, but enforcement registries are still split across markdown and Python constants.

**Merge/subsume opportunity:** canonicalize registry-like policy data in markdown frontmatter + one loader contract, then simplify enforcement code to projections over that carrier.

## Suggested reduction roadmap (ordered, leverage-coupled)

### Step 1 — Define cross-origin ASPF identity contract + merge witnesses

**Primary objective:** establish the canonical identity/witness schema used by every substrate adapter and remap merger.

**Produces leverage for next steps:**
- gives governance/frontmatter migration a typed target carrier instead of ad-hoc document parsing contracts,
- gives scanner/rule migration a stable query key space,
- gives wrapper consolidation a single semantic backend contract.

**Refines previous/following steps:**
- back-propagates constraints into existing pyast/libCST adapters (identity completeness checks),
- forward-defines required fields for registry carriers introduced in Step 2.

### Step 2 — Converge governance/policy registries into declarative carriers

**Primary objective:** move hardcoded governance doc/rule inventories into frontmatter/YAML registries aligned to Step-1 identity/witness schemas.

**Produces leverage for next steps:**
- scanner migration can bind rule execution to declarative registry rows,
- policy checks become projection queries over one registry substrate,
- wrapper generation can be driven by the same declarative manifest style.

**Refines previous/following steps:**
- validates Step-1 identity schema against real governance metadata diversity,
- constrains Step-3 migration scope to explicit registry-backed rules only.

### Step 3 — Rebase policy scanner execution onto package-native ASPF projections

**Primary objective:** remove package->scripts inversion; execute rule logic from package-native modules that consume Step-2 registries and Step-1 ASPF identities.

**Produces leverage for next steps:**
- converts scanner logic into reusable projections that can be reused by orchestration/reporting refactors,
- reduces boundary duplication before deeper core extraction, minimizing moving parts.

**Refines previous/following steps:**
- supplies empirical query patterns to tighten Step-1 witness fields,
- supplies reusable decision-protocol fragments used in Steps 4–5 extraction work.

### Step 4 — Extract orchestration protocol bundles from command core

**Primary objective:** carve `command_orchestrator` into explicit phase/progress/result protocol bundles and reducers, driven by Step-3 projection patterns.

**Produces leverage for next steps:**
- provides reusable orchestration carriers for analysis-core phase transitions,
- stabilizes artifact routing/progress semantics before analysis decomposition.

**Refines previous/following steps:**
- pushes real runtime constraints back into Step-1/Step-2 schemas (missing witness/registry facets),
- gives Step 5 a tested pattern for phase decomposition and reducer boundaries.

### Step 5 — Extract analysis-core decision protocols and post-phase reducers

**Primary objective:** decompose `dataflow_pipeline`/`dataflow_audit` branch-heavy tails into explicit decision protocols and shared reducers, aligned to orchestration carriers from Step 4.

**Produces leverage for next steps:**
- completes core-side reuse so wrappers can collapse without behavior forks,
- increases parity between governance and analysis projections over ASPF artifacts.

**Refines previous/following steps:**
- feeds ambiguity/decision evidence back to refine Step-3 scanner projections,
- confirms whether Step-1 identity witnesses are sufficient for cross-core joins.

### Step 6 — Collapse script wrappers to declarative launcher manifests

**Primary objective:** replace repeated script bridge patterns with one launcher contract generated from declarative manifests tied to package-native commands.

**Produces leverage for next steps:**
- operationalizes all earlier convergence into minimal boundary wrappers,
- makes future command additions additive (manifest row) rather than duplicative (new bespoke wrapper).

**Refines previous/following steps:**
- creates a durable feedback channel: wrapper manifests can assert required Step-1/Step-2 contracts and fail fast when drift appears,
- closes the loop by making policy checks enforce structural assumptions introduced in Steps 1–5.

### Step 7 — Institutionalize bidirectional refinement loop (delta governance cadence)

**Primary objective:** formalize a recurring cadence where artifacts from Steps 3–6 are diffed against Step-1/Step-2 contracts and used to publish targeted corrections.

**Structural leverage outcome:**
- keeps the roadmap self-stabilizing rather than one-off,
- ensures each new adapter/rule/refactor incrementally improves both upstream contracts and downstream execution surfaces.

## Expected outcome

If executed in this order, the codebase should gain:

- tighter cross-zone alignment (tooling/governance/runtime),
- one analysis substrate across code and governance docs,
- less hardcoded policy logic in Python,
- lower branch density in core orchestration paths,
- fewer duplicate entrypoint lines,
- and better reuse leverage through explicit tier-1 bundles and ASPF-driven queries.
