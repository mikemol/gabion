---
doc_revision: 5
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: planning_substrate
doc_role: architecture
doc_scope:
  - repo
  - planning
  - tooling
doc_authority: informative
doc_requires:
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
doc_reviewed_as_of:
  README.md#repo_contract: 2
  CONTRIBUTING.md#contributing_contract: 2
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md#repo_contract rev84/section v2 (planning artifacts and the repo-local project-manager view remain tooling surfaces, not user-facing product features)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md#contributing_contract rev120/section v2 (correction-unit validation, git drainage, and workflow checks govern planning-substrate changes)."
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md#policy_seed rev57/section v2 (planning workflows are process-relative runtime and planning distinctions must remain constructible, reachable, observable, and coverable)."
  glossary.md#contract: "Reviewed glossary.md#contract rev47/section v1 (queue/root/subqueue/workstream_registry semantics remain canonical for the planner overlay)."
doc_sections:
  planning_substrate: 1
  runtime_model: 1
  registry_authoring: 1
  graph_pipeline: 1
  artifacts_and_commands: 1
  operating_rules: 1
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="planning_substrate"></a>
# Planning substrate

The planning substrate is the repo-local planning runtime that turns declared
workstream metadata, invariant markers, repo-state artifacts, and policy/test
signals into machine-readable queue state.

At a high level it does five jobs:

1. declares planning units as typed registry packets (`root`, `subqueue`,
   `touchpoint`, `touchsite`, counterfactual action),
2. lifts those declarations into the invariant graph alongside ordinary marker
   and artifact nodes,
3. computes planner-facing projections such as queue health, ranked cuts,
   blocker chains, followups, and ledger views,
4. overlays a phase-local planning chart (`scan`, `predict`, `complete`) on top
   of the graph,
5. emits repo-local artifacts that drive the project-manager view and planning
   diagnostics.

The substrate is not just one file. It is the coordinated system centered on:

- [`src/gabion/tooling/policy_substrate/workstream_registry.py`](../src/gabion/tooling/policy_substrate/workstream_registry.py)
- [`src/gabion/tooling/policy_substrate/invariant_graph.py`](../src/gabion/tooling/policy_substrate/invariant_graph.py)
- [`src/gabion/tooling/policy_substrate/planning_chart.py`](../src/gabion/tooling/policy_substrate/planning_chart.py)
- [`src/gabion/tooling/runtime/invariant_graph.py`](../src/gabion/tooling/runtime/invariant_graph.py)
- the workstream registry modules under `src/gabion/tooling/policy_substrate/`

This document covers the whole substrate. The narrower planning-chart phase
model is documented separately in
[`docs/planning_chart_architecture.md#planning_chart_architecture`](planning_chart_architecture.md#planning_chart_architecture).

<a id="runtime_model"></a>
## Runtime model

The substrate treats planning as a real runtime, not as static prose. That
follows [POLICY_SEED.md#policy_seed](../POLICY_SEED.md#policy_seed) and
[glossary.md#contract](../glossary.md#contract): runtime is process-relative
and a runtime distinction is valid only if it is constructible, reachable,
observable, and coverable.

In the planning substrate, the main runtime distinctions are:

- **Root:** the top-level declared work item in a registry packet.
- **Subqueue:** a first-order structural child of a root.
- **Touchpoint:** a concrete planning unit inside a subqueue.
- **Touchsite:** a declared boundary or module/document surface touched by a
  touchpoint.
- **Counterfactual action:** a declared follow-on action the planner may rank
  when direct completion is blocked.
- **Queue:** a planner-side overlay membership distinction derived from graph
  state. A queue is not the same thing as the `WorkstreamRegistry` object that
  declared some of its structure.

### Structural model versus overlay model

One registry packet declares exactly one root:

- `WorkstreamRegistry.root` holds the root metadata.
- `WorkstreamRegistry.subqueues` holds the declared structural children.
- `WorkstreamRegistry.touchpoints` holds declared executable planning units.

That structural tree is not itself the planner. The planner is an overlay built
from:

- registry declarations,
- invariant markers,
- artifact ingress,
- diagnostics and coverage signals,
- queue-health and ranking logic.

That is why the glossary distinguishes `queue` from `root`, `subqueue`, and
`workstream_registry`.

### Marker lifecycle versus planner status

The substrate uses two related but distinct state channels:

- **Marker lifecycle** lives in marker payloads and comes from the decorator.
  Open work uses `todo_decorator(...)`; closed recorded work uses
  `landed_todo_decorator(...)`.
- **Planner status hints** live on registry definitions and are values such as
  `queued`, `in_progress`, and `landed`.

These must agree, but they are not identical:

- lifecycle answers “what kind of marker is this?”
- status hint answers “how should the planner treat this work item right now?”

Closure consistency is enforced centrally by
`validate_workstream_closure_consistency(...)`.

### Identity model

The substrate is heavily identity-driven. Important identity channels include:

- **object IDs** such as `SCC`, `SCC-SQ-001`, `SCC-TP-008`,
- **site identities** for concrete source locations,
- **structural identities** for stable role-in-structure identity,
- **marker identities** for the underlying invariant markers.

This lets the graph connect declared queue structure, source locations, and
artifact evidence without collapsing them into one overloaded identifier.

<a id="registry_authoring"></a>
## Registry authoring model

Registry authoring is intentionally explicit. A planning unit is not declared by
YAML alone or by inferred file naming; it is declared by decorated Python
symbols plus typed registry definitions.

### Declaring a registry

The usual pattern in a registry module is:

1. declare root/subqueue/touchpoint carrier symbols with `@todo_decorator(...)`
   or `@landed_todo_decorator(...)`,
2. derive marker metadata with `registry_marker_metadata(...)`,
3. construct `RegisteredRootDefinition`,
   `RegisteredSubqueueDefinition`, and `RegisteredTouchpointDefinition`,
4. return one `WorkstreamRegistry` from the module entrypoint.

The carrier symbols are intentionally thin. Their main jobs are:

- to hold reviewable invariant metadata,
- to provide stable symbol identity for scanning,
- to make queue state visible in normal code review and AST-based tooling.

The `WorkstreamRegistry` object is then the typed packet used by the planner.

### Why there is one root per registry

`WorkstreamRegistry` is deliberately single-root. That keeps each declaration
packet reviewable and mechanically bounded. Cross-root planning is produced by
aggregation in the invariant graph, not by authoring giant multi-root registry
blobs.

Examples of root families currently aggregated include:

- `PRF`
- `PSF-007`
- `SCC`
- `RCI`
- `BIC`
- `CSA-*`
- synthetic current/trend overlays such as `UTR`, `DFR`, and `DFM`

### Synthetic operational roots

Some roots are synthetic planner overlays rather than owner roots for a product
or semantic subsystem. They summarize operational state from repo-local
artifacts and keep that state visible in the same root/subqueue/touchpoint
grammar as ordinary work.

Current examples include:

- `UTR` for unit-test readiness and repo-drain red-state indicators
- `DFR` for current delivery-flow reliability blockers across the dev + CI loop
- `DFM` for historical delivery-flow momentum and trend drag across recent runs

`DFR` and `DFM` are intentionally separate:

- `DFR` is a current-indicator root driven by current blockers such as red-state,
  local-vs-CI parity drift, observability gaps, and severe runtime regressions
- `DFM` is a trend root driven by historical telemetry such as runtime trend,
  recurrence rate, and correction-lag drift

This keeps observability as an ingress signal rather than inventing a separate
observability-root family inside the planning substrate.

### Touchsites

Touchsites are the declared planner-facing boundary surfaces of a touchpoint.
They are how the planning substrate makes “this work item touches these concrete
surfaces” observable.

Touchsites may be:

- module touchsites,
- document touchsites,
- scanned touchsites derived from source under a declared boundary,
- collapsed helper surfaces when a queue intentionally hides private-helper fanout.

Touchsites matter because most planner health metrics are touchsite-centric:

- how many declared surfaces exist,
- how many survive collapse,
- how many are uncovered,
- how many are blocked by policy or diagnostics.

### Counterfactual actions

Touchpoints can declare counterfactual actions that the planner may rank when a
direct cut is blocked. These actions are metadata, not imperative code. They
allow the planner to expose “next best move” options without smearing that logic
through ad hoc recommendation branches.

<a id="graph_pipeline"></a>
## Graph and projection pipeline

The planning substrate’s runtime pipeline is centered on the invariant graph.

### 1. Marker scan

`invariant_marker_scan.py` walks the repo scan roots and extracts invariant
decorators into `InvariantMarkerScanNode` values. This is where the substrate
recovers:

- marker kind,
- lifecycle state,
- object/doc/policy/invariant links,
- reasoning summaries and controls,
- source location and identities.

The scan understands both `todo_decorator` and `landed_todo_decorator`.

### 2. Registry aggregation

`invariant_graph.py` imports the declared registry entrypoints and aggregates
them with ordinary invariant-marker nodes and artifact ingress. This is where
the planning substrate stops being just “a pile of registry files” and becomes a
unified graph.

Registry aggregation contributes:

- root/subqueue/touchpoint nodes,
- declared touchsite nodes,
- structural `contains` edges,
- overlay membership and dependency relationships,
- declared counterfactual actions,
- status hints and closure metadata.

### 3. Artifact ingress

The graph also ingests repo-local artifact signals such as:

- docflow compliance,
- controller drift,
- local repro closure,
- git-state reports,
- kernel/VM alignment artifacts,
- identity-grammar completion artifacts,
- test evidence,
- ingress merge parity,
- JUnit failures.

These artifacts are what make the planning graph operational rather than merely
declarative. They are how the substrate sees real blockers, coverage gaps, and
evidence drift.

### 4. Workstream projection

Once the graph is built, the substrate computes workstream projections. Those
projections summarize queue health and planner actions in queue-centric terms,
including:

- touchsite counts,
- surviving versus collapsible touchsites,
- coverage gaps,
- policy and diagnostic blockers,
- ranked touchpoint cuts,
- ranked subqueue cuts,
- recommended ready cuts,
- recommended blocked cuts,
- recommended followups and lanes,
- blocker chains and ledger projections.

This is the layer that turns graph structure into actionable planning output.

### Synthetic readiness roots

Not every root is an ownership queue. Some roots are synthetic planner
overlays that summarize cross-cutting operational state from artifact evidence.

`UTR` (`unit_test_readiness`) is the current example. It does not own semantic
surfaces the way `SCC`, `RCI`, or `BIC` do. Instead it tracks repo-drain
readiness for the full unit-test suite by coupling junit-derived `test_case`
and `test_failure` nodes to declared touchpoints through
`RegisteredTouchpointDefinition.test_path_prefixes`.

That coupling has a few important consequences:

- the root-level failing-test count is the count of unique failing test cases
  currently matched into `UTR`,
- touchpoint counts can overlap when one failing file intentionally belongs to
  more than one readiness bucket,
- a touchpoint may be recorded as landed while the root and sibling touchpoints
  remain active,
- a targeted green run is not enough to claim movement; the junit feed must be
  refreshed so the current indicators disappear from the projection.

`UTR` is therefore a planner-visible current-indicator root, not a replacement
for owner roots or a prose-only incident register. The operational loop for
seeding, refreshing, and closing it is documented in
[`docs/unit_test_readiness_playbook.md`](unit_test_readiness_playbook.md).

### 5. Planning-chart overlay

The planning chart is a further overlay computed from the graph and workstream
projection. It groups planner items into the domain-neutral phases:

- `scan`
- `predict`
- `complete`

The chart is not a second independent planner. It is a phase-local view over
the same underlying planning runtime:

- `scan` covers live evidence and observed planning surfaces,
- `predict` covers dependencies and counterfactual futures,
- `complete` covers selected cuts and followups.

### 6. Project-manager view

The project-manager view is the presentation layer generated from planning
artifacts. It is what feeds the auto-generated section in
[`README.md#repo_contract`](../README.md#repo_contract).

The important dependency direction is:

`registry declarations -> invariant graph -> workstream projection -> planning chart -> project manager view`

The README view is downstream of the substrate; it is not the substrate itself.

<a id="artifacts_and_commands"></a>
## Artifacts and commands

The main runtime entrypoint is
[`src/gabion/tooling/runtime/invariant_graph.py`](../src/gabion/tooling/runtime/invariant_graph.py).

Its default build command emits:

- `artifacts/out/invariant_graph.json`
- `artifacts/out/invariant_workstreams.json`
- `artifacts/out/invariant_ledger_projections.json`

Additional commands expose:

- summaries,
- node traces,
- blocker chains,
- per-object workstream projections,
- ledger deltas and alignments,
- blast-radius overlays,
- perf-heat overlays,
- before/after workstream comparison.

Related downstream artifacts include:

- `artifacts/out/project_manager_view.json`
- `artifacts/out/project_manager_view.md`
- `artifacts/out/project_manager_view.mmd`

### Practical command loop

Common repo-local commands are:

```bash
mise exec -- python -m gabion.tooling.runtime.invariant_graph build
mise exec -- python -m gabion.tooling.runtime.invariant_graph summary
mise exec -- python -m gabion.tooling.runtime.invariant_graph workstream --object-id SCC
mise exec -- python -m gabion.tooling.runtime.invariant_graph blockers --object-id CSA-RGC
mise exec -- python -m scripts.policy.project_manager_view
```

For `UTR`, the canonical feed/inspection loop is:

```bash
mise exec -- python -m pytest --junitxml artifacts/test_runs/junit.xml --log-file artifacts/test_runs/pytest.log --log-file-level=INFO
mise exec -- python -m gabion.tooling.runtime.invariant_graph workstream --object-id UTR
```

When the stable projection artifacts are being written through the normal
runtime path, `UTR` is also visible in
`artifacts/out/invariant_workstreams.json`.

The policy gate also consumes planning-substrate state through:

```bash
mise exec -- python -m scripts.policy.policy_check --workflows
```

That gate is the mechanized enforcement point for workstream-definition and
closure-consistency drift.

<a id="operating_rules"></a>
## Operating rules for editing the planning substrate

### Treat queue structure as code

Registry state is semantic runtime state. Do not treat it as incidental
metadata. If queue membership, status, touchsites, or closure semantics change,
the registry change is part of the real behavior change.

### Prefer explicit registry updates over patching tests

If a planner-visible distinction is real, declare it in the substrate:

- add or update a root/subqueue/touchpoint,
- declare touchsites explicitly,
- add counterfactual actions explicitly,
- update status hints and marker lifecycle coherently.

Do not hide planner drift behind test monkeypatches or by broadening assertions
until they stop noticing the mismatch.

### Keep closure honest

Closed work must be structurally closed, not merely narrated as closed. In
practice that means:

- landed nodes use landed marker lifecycle,
- landed nodes keep no blocking dependencies,
- landed nodes use closed/recorded language,
- landed parents do not retain open descendants.

If those conditions are not true, reopen the queue or add a follow-on touchpoint
instead of pretending the work is done.

### Use synthetic roots for semantic tests

The substrate has dedicated DI seams so semantic tests can build graphs from
synthetic registries and synthetic roots. Use live-repo signal tests only when
the purpose of the test is explicitly to observe repo state rather than code
semantics.

### Keep planning-chart and queue docs separate

The planning chart is a view of the planning runtime, not the whole system.
When documenting phase behavior, use
[`docs/planning_chart_architecture.md`](planning_chart_architecture.md). When
documenting registries, graph build, artifacts, and queue semantics, use this
document.

## Source map

If you need to extend the planning substrate, these files are the usual entry
points:

| Surface | Purpose |
| --- | --- |
| `src/gabion/tooling/policy_substrate/workstream_registry.py` | Core registry dataclasses and closure validation. |
| `src/gabion/tooling/policy_substrate/invariant_marker_scan.py` | AST marker scan and marker-payload recovery. |
| `src/gabion/tooling/policy_substrate/invariant_graph.py` | Graph construction, registry aggregation, and planner projections. |
| `src/gabion/tooling/policy_substrate/unit_test_readiness_registry.py` | Synthetic full-suite readiness root driven by junit failure selectors. |
| `src/gabion/tooling/policy_substrate/planning_chart.py` | Phase-local planning-chart overlay. |
| `src/gabion/tooling/runtime/invariant_graph.py` | CLI/runtime entrypoint for building and querying planning artifacts. |
| `src/gabion/tooling/policy_substrate/project_manager_view.py` | Downstream rendering of planning artifacts into the README-facing PM view. |
| `tests/gabion/tooling/runtime_policy/test_workstream_registry_definitions.py` | Registry-shape regression coverage. |
| `tests/gabion/tooling/runtime_policy/test_workstream_closure_consistency.py` | Closure-model regression coverage. |
| `tests/gabion/tooling/runtime_policy/test_invariant_graph.py` | Synthetic/injected invariant-graph semantics. |
| `tests/gabion/tooling/runtime_policy/test_invariant_graph_live_repo.py` | Live repo-state sentinel coverage. |

## Related documents

- [`docs/planning_chart_architecture.md`](planning_chart_architecture.md)
- [`docs/unit_test_readiness_playbook.md`](unit_test_readiness_playbook.md)
- [`docs/governance_control_loops.md`](governance_control_loops.md)
- [`README.md#repo_contract`](../README.md#repo_contract)
- [`CONTRIBUTING.md#contributing_contract`](../CONTRIBUTING.md#contributing_contract)
- [`glossary.md#queue`](../glossary.md#queue)
- [`glossary.md#root`](../glossary.md#root)
- [`glossary.md#subqueue`](../glossary.md#subqueue)
- [`glossary.md#workstream_registry`](../glossary.md#workstream_registry)
