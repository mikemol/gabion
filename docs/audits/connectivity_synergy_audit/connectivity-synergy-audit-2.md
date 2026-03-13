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

This audit is intentionally framed as a **structural leverage map**, not a static backlog. The purpose is to identify changes that:

1. reduce repeated semantic work across analysis/governance/tooling surfaces,
2. shift ambiguity to boundary adapters while keeping deterministic core semantics queryable,
3. maximize reuse by turning ad-hoc branches into explicit protocol carriers, and
4. improve correction-unit velocity by making each future change operate on fewer, more canonical control surfaces.

The key leverage principle is: **normalize once at ingress, intern once per origin lane, merge once via explicit identity witnesses, and query everywhere through shared ASPF projections**.

Under this principle, line-count reduction is a secondary effect of semantic consolidation:

- duplicated wrapper logic disappears when wrappers become declarative launch metadata,
- duplicated policy registry constants disappear when governance/policy inventories become frontmatter carriers,
- duplicated walker/visitor logic disappears when checks compile to ASPF queries over common enriched state.

This also aligns with forward-remediation policy: regressions should be fixed by strengthening canonical carriers and merge invariants, not by introducing compatibility branches.

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

Each step is designed to increase the structural leverage of the next step while constraining rework on prior steps.

1. **Define the multi-origin ASPF identity contract (foundational constraint layer).**
   - Specify canonical identity witness fields, origin-lane provenance schema, and merge/remap artifact formats.
   - Establish deterministic merge semantics (exact match, conflict class, deferred join).
   - **Leverage forward:** all later adapters/queries target one identity contract.
   - **Leverage backward:** forces earlier assumptions about "same node" to become explicit witness logic.

2. **Introduce non-Python adapter lanes that emit contract-compliant ASPF enrichments.**
   - Add markdown/frontmatter/yaml adapters first; keep html/pandoc as optional ingress adapters behind the same contract.
   - Publish per-origin snapshots/deltas plus overlap ledgers ready for merge.
   - **Leverage forward:** creates real multi-origin evidence for registry and rule migration.
   - **Leverage backward:** validates and stress-tests the identity contract from step 1.

3. **Externalize governance/policy registries into declarative carriers and ingest them via ASPF.**
   - Move hardcoded inventories/rule registries into frontmatter/YAML canonical sources.
   - Make loaders produce ASPF facts instead of direct constant maps.
   - **Leverage forward:** policy scanners and docflow checks can run as projections over shared facts.
   - **Leverage backward:** registry schemas become another proving ground for adapter fidelity from step 2.

4. **Recast policy/docflow enforcement as ASPF query bundles.**
   - Replace bespoke walkers/constant-coupled checks with query specs over merged ASPF state.
   - Treat query bundles as tier-1 decision protocols with explicit evidence outputs.
   - **Leverage forward:** unifies enforcement semantics before touching orchestration hotspots.
   - **Leverage backward:** exposes missing witness fields or registry semantics that steps 1-3 can tighten.

5. **Collapse script/tooling entrypoints to declarative launch manifests over package-native commands.**
   - Remove package->scripts inversion; scripts become thin/generated orchestration shims.
   - Route wrappers to ASPF-backed commands and query bundles from step 4.
   - **Leverage forward:** reduces surface area before large orchestration refactors.
   - **Leverage backward:** wrapper metadata becomes an additional declarative substrate validated by step 3/4 machinery.

6. **Extract protocol bundles from `command_orchestrator` against stabilized ASPF/query boundaries.**
   - Split progress emission, response projection, artifact routing into dataclass/protocol carriers.
   - Bind orchestration phases to explicit query/evidence contracts instead of implicit branches.
   - **Leverage forward:** simplifies downstream analysis/refactor integrations.
   - **Leverage backward:** confirms wrapper and enforcement unification really reduced orchestration complexity.

7. **Refactor analysis hotspots (`dataflow_pipeline`, `dataflow_audit`) to shared decision/query protocols.**
   - Move repeated phase/report logic into reusable query/projection bundles.
   - Standardize continuation artifacts so governance and analysis share merge semantics.
   - **Leverage forward:** enables sustained reduction in branch density and duplicate helpers.
   - **Leverage backward:** final convergence test for all prior steps (identity, adapters, registries, wrappers, orchestration).

8. **Close-loop hardening: enforce invariants and drift checks as first-class correction-unit gates.**
   - Add/ratchet checks for remap determinism, provenance completeness, and registry/query parity.
   - Require evidence of ASPF overlap quality and unresolved residue accounting in correction artifacts.
   - **Leverage forward:** keeps the unified substrate stable as new adapters are added.
   - **Leverage backward:** continuously validates that each earlier step remains true under repository evolution.

## Expected outcome

If executed in this order, the codebase should gain:

- tighter cross-zone alignment (tooling/governance/runtime),
- one analysis substrate across code and governance docs,
- less hardcoded policy logic in Python,
- lower branch density in core orchestration paths,
- fewer duplicate entrypoint lines,
- and better reuse leverage through explicit tier-1 bundles and ASPF-driven queries.
