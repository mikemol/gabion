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

This audit is not just a catalog of hotspots; it is intended as a **sequencing instrument**.

The core rationale is that line-count and branch-count reduction only hold over time when structural reuse is paired with substrate unification. In this repository, that means:

- reducing duplicated orchestration surfaces by converging on explicit protocol carriers,
- reducing duplicated policy semantics by converging on declarative/frontmatter registries, and
- reducing duplicated analysis engines by converging on ASPF enrichment + query execution.

These three vectors are interdependent. If any one vector lags, simplifications in the other two tend to regress into adapter drift, wrapper proliferation, or policy/code split-brain.

So the roadmap is intentionally staged to maximize **structural leverage**:

1. early steps create canonical carriers and identity witnesses,
2. mid steps collapse duplicated execution paths onto those carriers,
3. later steps refactor high-complexity cores after substrate constraints are in place,
4. final steps close the loop with enforcement so gains remain stable.

This ordering is designed to make each step both a consumer of prior structure and a producer of constraints for the next step.

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

## Suggested reduction roadmap (ordered, leverage-maximizing)

Each step is written to amplify the next step while retroactively hardening prior steps.

### Step 1 — Canonical identity witness contract for multi-registry ASPF merge

Define and ratify one witness schema for cross-origin identity (node kind, canonical payload hash, provenance class, scope key, plus conflict annotations).

**Leverage produced:**
- gives every later adapter and registry migration a shared merge target,
- provides deterministic remap primitives required for governance-grade forensics,
- establishes the data contract that later enforcement can validate.

**Refinement loop:**
- later steps feed new substrate examples back into witness schema tests, tightening Step 1 rather than bypassing it.

### Step 2 — Non-Python ASPF ingress adapters (markdown/frontmatter/yaml first)

Implement adapters that emit the Step-1 witness envelope and origin-local registry artifacts for governance docs and policy carriers.

**Leverage produced:**
- converts policy/document analysis from bespoke walkers into ASPF-producible streams,
- creates immediate substrate parity with pyast/libCST lanes,
- produces merge-ready evidence to drive declarative registry convergence.

**Refinement loop:**
- merge conflicts discovered here sharpen Step-1 witness rules and expose missing normalization obligations before core refactors begin.

### Step 3 — Declarative registry convergence (frontmatter/YAML as source of truth)

Move governance/policy inventories and rule registries out of Python constants and into declarative carriers consumed through ASPF-aware loaders.

**Leverage produced:**
- collapses code/doc split-brain on policy intent,
- turns registry evolution into data changes instead of control-flow edits,
- gives scanner/orchestrator refactors stable inputs independent of file-local constants.

**Refinement loop:**
- Step-2 adapters validate whether declarative carriers remain ASPF-normalizable; failures feed backward to adapter and witness constraints.

### Step 4 — Policy scanner dependency inversion cleanup

Remove package->scripts dynamic imports; require scripts to be launch shims over package-native scanner logic driven by Step-3 registries.

**Leverage produced:**
- eliminates dual authority surfaces for rule semantics,
- reduces wrapper drift and unlocks reusable scanner pipelines,
- simplifies test topology for downstream orchestrator extraction.

**Refinement loop:**
- scanner pipeline outputs become regression fixtures for Step-3 registry semantics and Step-2 adapter completeness.

### Step 5 — Command orchestration phase-protocol extraction

Refactor `command_orchestrator` around tier-1 dataclass/protocol phase bundles (progress emit, result projection, artifact routing), consuming Step-4 scanner/runtime contracts.

**Leverage produced:**
- reduces large branch surfaces where policy and execution concerns are currently fused,
- creates reusable orchestration primitives for analysis and governance commands,
- localizes decision surfaces so ASPF projections can observe them uniformly.

**Refinement loop:**
- extracted phase protocols provide concrete hooks for Step-6 analysis refactors and can be validated against Step-1/2 merge traces.

### Step 6 — Analysis-core phase and report-surface extraction

Refactor `dataflow_pipeline`/`dataflow_audit` to consume the same phase/dependency protocol style and to emit ASPF-aligned decision/report artifacts.

**Leverage produced:**
- aligns analysis core with orchestrator semantics, reducing bespoke control branches,
- enables shared query paths across governance and analysis outputs,
- prepares a single substrate for comprehensive enforcement checks.

**Refinement loop:**
- analysis edge cases stress-test Step-5 phase contracts and Step-1 witness identity assumptions, feeding improvements backward.

### Step 7 — Wrapper manifest + generation/validation gate

Replace repeated script wrappers with one manifest-driven launcher pattern and policy validation that enforces wrapper minimalism.

**Leverage produced:**
- prevents reintroduction of wrapper duplication/inversion,
- makes script surface area a predictable projection of package command topology,
- reduces maintenance noise so complexity budgets focus on semantic core only.

**Refinement loop:**
- wrapper manifest diffs provide a compact signal when Step-4/5/6 APIs drift, creating early correction triggers.

### Step 8 — Closed-loop enforcement and drift budgets

Add or tighten checks that assert: witness determinism, remap stability, registry↔ASPF parity, wrapper conformance, and branch/line budgets on targeted core functions.

**Leverage produced:**
- converts refactor outcomes into durable guardrails,
- closes the loop so each prior step is continuously revalidated,
- keeps future simplifications on the same substrate-first path instead of reintroducing parallel stacks.

**Refinement loop:**
- enforcement failures become correction-unit signals that can be localized to the exact upstream step contract that regressed.

### Dependency spine (why this order)

- Steps 1-2 establish identity + ingestion substrate.
- Steps 3-4 migrate authority surfaces onto that substrate.
- Steps 5-6 simplify the highest-complexity cores using stabilized upstream contracts.
- Steps 7-8 freeze gains with generation discipline and continuous drift detection.

## Expected outcome

If executed in this order, the codebase should gain:

- tighter cross-zone alignment (tooling/governance/runtime),
- one analysis substrate across code and governance docs,
- less hardcoded policy logic in Python,
- lower branch density in core orchestration paths,
- fewer duplicate entrypoint lines,
- and better reuse leverage through explicit tier-1 bundles and ASPF-driven queries.
