---
doc_revision: 9
doc_id: connectivity_synergy_audit
doc_role: audit
doc_scope:
  - repo
  - architecture
  - refactor
doc_authority: informative
doc_targets:
  - gabion.analysis.semantics.impact_index.build_impact_index
  - gabion.analysis.semantics.impact_index._links_from_doc
  - gabion.analysis.semantics.impact_index._collect_symbol_universe
  - scripts.policy.policy_check.collect_aspf_lattice_convergence_result
  - gabion.tooling.policy_substrate.lattice_convergence_semantic.materialize_semantic_lattice_convergence
  - gabion.tooling.policy_substrate.lattice_convergence_semantic.iter_semantic_lattice_convergence._events
  - gabion.analysis.aspf.aspf_lattice_algebra.build_dataflow_fiber_bundle_for_qualname
  - gabion.analysis.aspf.aspf_lattice_algebra.build_fiber_bundle_for_qualname
  - gabion.analysis.aspf.aspf_lattice_algebra.iter_lattice_witnesses._query
  - gabion.analysis.aspf.aspf_lattice_algebra._module_bound_symbols
  - gabion.analysis.aspf.aspf_lattice_algebra.compute_lattice_witness
  - gabion.tooling.runtime.invariant_graph._resolve_perf_dsl_overlay
  - gabion.tooling.runtime.invariant_graph._load_profile_observations
  - gabion.tooling.runtime.invariant_graph._print_perf_heat_map
  - gabion.tooling.runtime.perf_artifact.build_cprofile_perf_artifact_payload
  - gabion_governance.governance_audit_impl._sppf_sync_check
  - gabion_governance.governance_audit_impl._evaluate_docflow_obligations
  - gabion.tooling.sppf.sync_core._collect_commits
  - gabion.tooling.sppf.sync_core._issue_ids_from_commits
  - gabion.tooling.sppf.sync_core._validate_issue_lifecycle
  - gabion.tooling.sppf.sync_core._run_validate_mode
  - gabion.analysis.semantics.obligation_registry.evaluate_obligations
  - gabion.execution_plan.ExecutionPlan.with_issue_link
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

This audit is intentionally framed as a **control-surface convergence exercise**, not only a code-size reduction pass. The core rationale is:

1. **Carrier convergence over local optimization:** reducing lines is valuable only when it also reduces semantic drift between code, docs, and governance enforcement carriers.
2. **ASPF as the semantic join substrate:** heterogeneous ingress is expected; the system should optimize for a single post-ingress query substrate rather than a single pre-ingress parser path.
3. **Forward-remediation economics:** each consolidation should lower future correction-unit cost by turning ad-hoc branches and duplicated wrappers into reusable protocolized bundles.
4. **Policy/audit commutation:** governance rules, scanner rules, and semantic analyses should commute over shared identity witnesses so one correction does not create hidden divergence in another surface.

Operationally, this means the highest-priority work is not whichever file is biggest, but whichever change increases the reuse and determinism available to the *next* change set.

### Why leverage-first sequencing matters

The repo currently has multiple partially overlapping enforcement surfaces (analysis runtime, governance/docflow checks, scanner wrappers). If large-function refactors happen before carrier/registry convergence, they tend to re-encode local assumptions that must later be unwound. A leverage-first sequence prevents that by first creating common substrate contracts, then forcing all later refactors to consume them.

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

### Gap D: the doc-target selector substrate is split across convergence and runtime surfaces

The repo already has a frontmatter-driven doc-to-symbol selector in `impact_index`, but the runtime perf path currently reconstructs a local overlay in `tooling/runtime/invariant_graph.py` instead of consuming one shared selector/interner carrier.

**Synergy gap:** this leaves doc-target selection, symbol-universe interning, and profiler-query rooting adjacent but not identical, which weakens the quotient-algebra story exactly where cross-surface ranking should be strongest.

**Merge/subsume opportunity:** converge `impact_index`, frontmatter ingestion, and perf-query rooting onto one declarative selector substrate, then treat runtime perf heat as a consumer of that carrier rather than a parallel selector implementation.

### Gap E: git/GH provenance and docflow obligation state are still second-order sidecars

The repo can now graph SPPF checklist nodes, inbox governance actions, planner artifacts, and control-loop sidecars, but the state that governs whether those same artifacts are lawfully attached to the planning loop still remains out-of-band:

- correction-unit rev-range provenance,
- commit-to-issue GH linkage,
- issue lifecycle validation state,
- docflow obligation evaluation,
- execution-plan issue-link facets.

**Synergy gap:** the planner can see first-order governance objects, but not yet the second-order control state that says whether the current correction unit is correctly linked into that governance forest. This means the graph still consumes planning outputs without fully consuming the policy gate that governs planning-for-planning integration.

**Merge/subsume opportunity:** reify git-range provenance, GH-reference validation, obligation summaries, and issue-link facets as graph-native carriers so the planner can reason over governance attachment debt instead of discovering it only as an out-of-band gate failure.

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
5. **Selector convergence:** the frontmatter-driven doc target DSL, symbol-universe interning, and profiler-query rooting should converge into one selector/interner substrate instead of being reconstituted independently at runtime.


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

The roadmap below is designed so each step both **depends on** and **improves** adjacent steps.

1. **Define ASPF cross-origin identity witness contract and remap ledger schema.**
   - Deliverables: witness tuple schema, merge-decision taxonomy, remap table format, overlap/conflict ledger artifacts.
   - Leverage: creates the canonical identity surface required by every later registry migration and scanner unification.
   - Feedback edge: unresolved identity classes discovered later must back-populate witness schema refinements.

2. **Implement non-Python adapter lanes against the step-1 contract (markdown/frontmatter/yaml first).**
   - Deliverables: ingress adapters that emit ASPF-normalized nodes + local prime registries + witness payloads.
   - Leverage: converts governance/docflow carriers into first-class ASPF inputs, enabling policy checks to stop depending on hardcoded document sets.
   - Feedback edge: adapter friction reveals missing witness dimensions; feed that into step 1 and re-run deterministic remap checks.

3. **Converge governance/policy registries into declarative carriers consumed via ASPF queries.**
   - Deliverables: markdown/YAML registry source of truth, loader contract, and query projections replacing Python constant lists.
   - Leverage: removes duplication between docs and code, and gives scanner/rule orchestration one registry substrate.
   - Feedback edge: registry ambiguities discovered here should tighten adapter normalization rules (step 2).

4. **Reify git-range/GH-reference provenance and docflow obligation state into the governance graph.**
   - Deliverables: typed rev-range carriers, commit-to-issue linkage carriers, issue lifecycle state, obligation summaries, and execution-plan issue-link projections.
   - Leverage: upgrades the graph from planning over first-order artifacts to planning over the legality and attachment state of those artifacts, which is the required second-order cybernetic loop.
   - Feedback edge: unmet provenance obligations and ambiguous issue-link state should feed back into both registry schemas (step 3) and adapter witness design (steps 1-2).

5. **Unify policy scanner execution around package-native rule modules + declarative registry resolution.**
   - Deliverables: remove package->scripts dynamic imports, make scripts thin launch shims, resolve active rules from declarative registry.
   - Leverage: establishes a single rule-execution topology that orchestration refactors can call without wrapper branching.
   - Feedback edge: scanner execution metrics should inform registry pruning/normalization in step 3.

6. **Extract orchestration/analysis protocol bundles using the stabilized substrate contracts.**
   - Deliverables: tier-1 dataclass/protocol bundles for progress emission, result projection, phase transitions, and report-tail assembly.
   - Leverage: this is where the largest line/branch reductions land, now constrained by stable carriers from steps 1-4.
   - Feedback edge: any new recurring parameter bundles discovered here should be promoted into shared registry/query vocabulary upstream.

7. **Normalize wrapper surfaces and continuation artifacts to enforce substrate parity.**
   - Deliverables: launcher manifest/template, wrapper policy checks, ASPF snapshot/delta parity across governance + analysis commands.
   - Leverage: hardens previous steps into ongoing invariants so future features inherit convergence by default.
   - Feedback edge: parity/audit drift signals become inputs for incremental protocol tightening in steps 3-5.

### Step coupling matrix (previous/next enrichment)

- **1 ↔ 2:** witness contracts guide adapters; adapter anomalies refine witness contracts.
- **2 ↔ 3:** normalized non-Python ingress enables declarative registry execution; registry ambiguities refine ingress normalization.
- **3 ↔ 4:** declarative registries expose what provenance and obligation state must become graph-native; second-order provenance carriers force clearer registry schemas.
- **4 ↔ 5:** graph-native provenance lets scanner execution become governance-aware instead of only registry-aware; scanner telemetry improves provenance coverage.
- **5 ↔ 6:** unified execution topology enables safe protocol extraction; extracted protocols reveal additional rule orchestration reuse points.
- **6 ↔ 7:** protocolized core simplifies wrapper unification; wrapper parity checks enforce protocol usage and expose residual divergence.

## Expected outcome

If executed in this order, the codebase should gain:

- tighter cross-zone alignment (tooling/governance/runtime),
- one analysis substrate across code and governance docs,
- graph-visible second-order provenance over planning attachment and obligation state,
- less hardcoded policy logic in Python,
- lower branch density in core orchestration paths,
- fewer duplicate entrypoint lines,
- and better reuse leverage through explicit tier-1 bundles and ASPF-driven queries.
