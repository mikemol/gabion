---
doc_revision: 2
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: planning_substrate_formal_spec
doc_role: design
doc_scope:
  - repo
  - planning
  - tooling
  - governance
doc_authority: informative
doc_requires:
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/planning_substrate.md#planning_substrate
  - docs/planning_chart_architecture.md#planning_chart_architecture
  - docs/governance_control_loops.md#governance_control_loops
  - docs/progress_transition_contract.md#progress_transition_contract
  - docs/aspf_execution_fibration.md#aspf_execution_fibration
  - docs/aspf_fingerprint_contract.md#aspf_fingerprint_contract
doc_reviewed_as_of:
  README.md#repo_contract: 2
  CONTRIBUTING.md#contributing_contract: 2
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  docs/planning_substrate.md#planning_substrate: 1
  docs/planning_chart_architecture.md#planning_chart_architecture: 2
  docs/governance_control_loops.md#governance_control_loops: 2
  docs/progress_transition_contract.md#progress_transition_contract: 7
  docs/aspf_execution_fibration.md#aspf_execution_fibration: 2
  docs/aspf_fingerprint_contract.md#aspf_fingerprint_contract: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md#repo_contract rev84/section v2 so the planning substrate stays framed as repo-local runtime tooling rather than product-facing milestone prose."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md#contributing_contract rev120/section v2 so correction-unit validation and git-drain obligations remain part of planning-substrate acceptance semantics."
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md#policy_seed rev57/section v2 so process-relative runtime, distinction admissibility, and governance control-loop obligations remain part of the planning substrate's operating model."
  glossary.md#contract: "Reviewed glossary.md#contract rev47/section v1 so queue/root/subqueue/workstream_registry semantics and the commutation vocabulary remain canonical for the formal model below."
  docs/planning_substrate.md#planning_substrate: "Reviewed planning_substrate section v1 as the canonical explainer for the runtime pipeline, synthetic roots, queue semantics, and operating rules."
  docs/planning_chart_architecture.md#planning_chart_architecture: "Reviewed planning_chart_architecture rev2 so the scan/predict/complete overlay is described as Earley-like and descriptive rather than as a second independent planner."
  docs/governance_control_loops.md#governance_control_loops: "Reviewed governance_control_loops section v2 so first-order and second-order correction loops remain part of planning acceptance and escalation semantics."
  docs/progress_transition_contract.md#progress_transition_contract: "Reviewed progress_transition_contract rev7 so progress-v2 normalization is treated as adjacent telemetry law rather than as a planning-substrate phase machine."
  docs/aspf_execution_fibration.md#aspf_execution_fibration: "Reviewed aspf_execution_fibration section v2 so ASPF remains the adjacent witness/fibration layer rather than the planning runtime itself."
  docs/aspf_fingerprint_contract.md#aspf_fingerprint_contract: "Reviewed aspf_fingerprint_contract rev1 so fibration/cofibration and identity-strata language is used only where the planning substrate actually consumes ASPF-adjacent identities."
doc_sections:
  planning_substrate_formal_spec: 1
  artifact_taxonomy: 1
  object_model: 1
  runtime_pipeline_and_ordering: 1
  transition_and_acceptance_semantics: 1
  invariant_preservation_and_aspf_relationship: 1
  gap_analysis: 1
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="planning_substrate_formal_spec"></a>
# Planning Substrate Formal Specification

This document formalizes the current planning substrate as it exists in the
repository today. It is a synthesis document: the normative force still comes
from [`POLICY_SEED.md#policy_seed`](../POLICY_SEED.md#policy_seed),
[`glossary.md#contract`](../glossary.md#contract), and the specific planning
and governance contracts cited below.

The core claim is precise:

> The planning substrate is a typed family of single-root `WorkstreamRegistry`
> packets lifted into an invariant graph, joined with repo-local artifact
> evidence, projected into queue/workstream state, and overlaid with a
> phase-local planning chart. It is not a milestone checklist.

<a id="artifact_taxonomy"></a>
## Artifact taxonomy

### Normative and semantic contracts

| Artifact | Object kind | Governs | Relationships | Invariants / commutation laws | Test / enforcement status |
| --- | --- | --- | --- | --- | --- |
| [`POLICY_SEED.md#policy_seed`](../POLICY_SEED.md#policy_seed) | policy contract | Execution safety, process-relative runtime, CI/governance operating rules | Upstream of planning docs, runtime tooling, and policy gates | runtime distinctions must be constructible, reachable, observable, and coverable; governed workflows must close through explicit checks | `mechanized but indirect` via policy checks, docflow, and runtime gates |
| [`glossary.md#contract`](../glossary.md#contract) | semantic contract | Meaning of `Queue`, `Root`, `Subqueue`, `WorkstreamRegistry`, and commutation vocabulary | Co-equal with `POLICY_SEED`; upstream of planning docs and code interpretations | `Queue` is overlay, not ownership; root closure is invariant under planner overlays; `WorkstreamRegistry` remains single-root | `declared_to_be_mapped` for the queue/root/subqueue/workstream_registry glossary obligations, because those entries still say “Test Obligations (to be mapped)” |
| [`docs/governance_control_loops.md#governance_control_loops`](governance_control_loops.md#governance_control_loops) | governance contract | First-order and second-order correction loops for workflows, docflow, LSP, dataflow grammar, baseline ratchets, execution coverage, and controller drift | Upstream of policy gates and downstream of README/CONTRIBUTING/POLICY_SEED | normalized loop schema; advisory/ratchet/hard-fail modes; one correction step per blocking surface | `exists` via governance/docflow tests and policy-check execution |
| [`docs/progress_transition_contract.md#progress_transition_contract`](progress_transition_contract.md#progress_transition_contract) | telemetry contract | `progress_transition_v2` normalization and validation for progress notifications | Adjacent to planning; consumed by command/orchestrator progress handling, not by the planning chart | recursive progress tree, forbidden transitions, terminal replay rules | `exists` via direct command/progress transition tests outside the planning slice |

### Planning-runtime documents

| Artifact | Object kind | Governs | Relationships | Invariants / commutation laws | Test / enforcement status |
| --- | --- | --- | --- | --- | --- |
| [`docs/planning_substrate.md#planning_substrate`](planning_substrate.md#planning_substrate) | architecture explainer | Whole planning runtime: registries, graph build, projections, chart overlay, artifacts, operating rules | Canonical explainer upstream of this spec; downstream of policy/glossary | queue is overlay, not ownership; closure consistency is central; README PM view is downstream of planning artifacts | `mechanized but indirect` because the runtime behaviors it describes are covered in code/tests, but the doc itself is not a separate gate |
| [`docs/planning_chart_architecture.md#planning_chart_architecture`](planning_chart_architecture.md#planning_chart_architecture) | architecture explainer | Phase-local chart overlay only | Downstream of planning substrate, upstream of chart vocabulary | `scan` / `predict` / `complete` are descriptive phases; Earley-like shape without persisted parser naming | `mechanized but indirect`; `planning_chart.py` is directly tested, but the doc itself explicitly says the current slice is descriptive only |

### ASPF-adjacent contracts

| Artifact | Object kind | Governs | Relationships | Invariants / commutation laws | Test / enforcement status |
| --- | --- | --- | --- | --- | --- |
| [`docs/aspf_execution_fibration.md#aspf_execution_fibration`](aspf_execution_fibration.md#aspf_execution_fibration) | contract/explainer | ASPF-native execution tracing, witnesses, handoff, semantic surface projections | Adjacent to planning; planning consumes ASPF-adjacent identities and artifacts where needed | 0-cell/1-cell/2-cell/cofibration distinctions; drift classified through witnesses; canonical lattice algebra ownership | `mechanized but indirect` via ASPF and runtime-policy tests |
| [`docs/aspf_fingerprint_contract.md`](aspf_fingerprint_contract.md) | design contract | Fingerprint fibration/cofibration and canonical identity strata | Adjacent to planning identity channels, not part of planning ownership grammar | canonical identity strata; representative choice as explicit decision protocol | `mechanized but indirect` via ASPF/fingerprint tests, not via planning-specific gates |

### Runtime and registry implementation surfaces

| Artifact | Object kind | Governs | Relationships | Invariants / commutation laws | Test / enforcement status |
| --- | --- | --- | --- | --- | --- |
| [`src/gabion/tooling/policy_substrate/workstream_registry.py`](../src/gabion/tooling/policy_substrate/workstream_registry.py) | typed registry substrate | `WorkstreamRegistry`, root/subqueue/touchpoint/touchsite definitions, closure validation | Upstream of all declared registry modules; aggregated by `invariant_graph.py` | single-root registry packet; landed lifecycle/language/dependency/descendant closure rules | `exists` via direct registry and closure-consistency tests |
| [`src/gabion/tooling/policy_substrate/declared_workstream_registry_catalog.py`](../src/gabion/tooling/policy_substrate/declared_workstream_registry_catalog.py) | registry assembly seam | Canonical catalog and provider ordering for registry loaders | Upstream of invariant-graph registry aggregation | stable provider assembly without widening registry type | `mechanized but indirect` through invariant-graph and registry-definition tests |
| [`src/gabion/tooling/policy_substrate/structured_artifact_ingress.py`](../src/gabion/tooling/policy_substrate/structured_artifact_ingress.py) | artifact-ingress substrate | Typed loading/identity/decomposition for repo-local artifacts | Upstream of invariant-graph joins; downstream of emitted artifacts | artifact identities are canonicalized and decomposable; artifact ingress is normalized before graph joins | `mechanized but indirect` via invariant-graph tests and runtime-policy projections |
| [`src/gabion/tooling/policy_substrate/invariant_graph.py`](../src/gabion/tooling/policy_substrate/invariant_graph.py) | graph and projection runtime | marker scan aggregation, registry lift, artifact joins, workstream projection, planning chart summary | Central runtime seam; downstream of registries and artifacts; upstream of project-manager view | identities remain separate (`object`, `site`, `structural`, `marker`); structural ownership is not collapsed into queue overlays | `exists` via direct invariant-graph tests |
| [`src/gabion/tooling/policy_substrate/planning_chart.py`](../src/gabion/tooling/policy_substrate/planning_chart.py) | chart overlay runtime | `scan`, `predict`, `complete` itemization and summaries | Downstream of invariant graph + workstream projection; upstream of PM view | chart phases are overlay groupings; completion items are recommendations, not ownership mutations | `exists` via direct planning-chart tests |
| [`src/gabion/tooling/policy_substrate/project_manager_view.py`](../src/gabion/tooling/policy_substrate/project_manager_view.py) | presentation/runtime view | README-facing PM summary, markdown/JSON/Mermaid render | Downstream of invariant-workstreams and planning-chart summary | PM view is projection/render only; rendering must not redefine planning semantics | `exists` via direct project-manager-view tests |

### Representative registry artifacts

| Artifact | Object kind | Governs | Relationships | Invariants / commutation laws | Test / enforcement status |
| --- | --- | --- | --- | --- | --- |
| [`unit_test_readiness_registry.py`](../src/gabion/tooling/policy_substrate/unit_test_readiness_registry.py) | synthetic root registry | repo-drain readiness from full-suite junit state | Downstream of junit/test evidence; aggregated into invariant graph | synthetic roots summarize operational state without becoming owner roots | `exists` via direct registry tests and invariant-graph tests |
| [`delivery_flow_reliability_registry.py`](../src/gabion/tooling/policy_substrate/delivery_flow_reliability_registry.py) | synthetic root registry | current delivery blockers across red-state, parity, observability, current-band runtime, and blocker-pattern pressure | Downstream of the canonical delivery-flow summary artifact; bounded delivery-flow run history is owned upstream by governance telemetry history while the summary remains a pure derivation from junit, local-CI repro, observability, and telemetry inputs | `DFR` is current-indicator only; current blockers stay separate from trend debt | `exists` via direct registry tests and invariant-graph tests |
| [`delivery_flow_momentum_registry.py`](../src/gabion/tooling/policy_substrate/delivery_flow_momentum_registry.py) | synthetic root registry | historical runtime, recurrence, red-state dwell, and closure-lag drag | Downstream of the canonical delivery-flow summary artifact built from governance telemetry history plus current-signal inputs | `DFM` is trend-only and should not absorb instantaneous blockers | `exists` via direct registry tests and invariant-graph tests |
| [`connectivity_synergy_registry.py`](../src/gabion/tooling/policy_substrate/connectivity_synergy_registry.py) | multi-root registry family | identity/rendering, ingress/merge, registry convergence, and impact velocity work families | Loaded as a registry family through the catalog; intersects many owner roots | connectivity families remain expressed as explicit roots/subqueues/touchpoints rather than prose-only audit notes | `exists` via direct registry-shape tests and invariant-graph tests |

### Validation surfaces

| Artifact | Object kind | Governs | Relationships | Invariants / commutation laws | Test / enforcement status |
| --- | --- | --- | --- | --- | --- |
| [`test_workstream_registry_definitions.py`](../tests/gabion/tooling/runtime_policy/test_workstream_registry_definitions.py) | regression test suite | registry shape, ordering, status hints, touchsite declarations | Directly validates representative registry modules and the declared registry catalog | single-root queue declarations stay explicit and stable | `exists` |
| [`test_workstream_closure_consistency.py`](../tests/gabion/tooling/runtime_policy/test_workstream_closure_consistency.py) | regression test suite | closure semantics for landed/open registry state | Directly validates `validate_workstream_closure_consistency(...)` and policy-check emission of closure drift | landed lifecycle/language/dependency/descendant rules | `exists` |
| [`test_planning_chart.py`](../tests/gabion/tooling/runtime_policy/test_planning_chart.py) | regression test suite | chart-rule projection and phase summary behavior | Directly validates `planning_chart.py` | `scan` / `predict` / `complete` overlay mapping and selected completion items | `exists` |
| [`test_invariant_graph.py`](../tests/gabion/tooling/runtime_policy/test_invariant_graph.py) | regression test suite | graph construction, artifact joins, workstream projection | Directly validates `invariant_graph.py` | registry aggregation, diagnostics, planning-chart summary persistence | `exists` |
| [`test_project_manager_view.py`](../tests/gabion/tooling/policy/test_project_manager_view.py) | regression test suite | PM summary, markdown, Mermaid, README section render | Downstream view validation for planning artifacts | downstream rendering remains projection-only and pressure-ordered | `exists` |
| [`test_docflow_governance_control_loops.py`](../tests/gabion/tooling/docflow/test_docflow_governance_control_loops.py) | docflow regression test suite | governance-loop declaration coverage and governance-loop matrix drift | Directly validates mechanized governance-loop registry expectations | governance loop domains and loop-matrix rows remain synchronized | `exists` |

<a id="object_model"></a>
## Planning substrate object model

### Core objects

The planning substrate is built from four structural declaration objects and
three overlay objects:

1. `WorkstreamRegistry`
2. `Root`
3. `Subqueue`
4. `Touchpoint`
5. `Touchsite` as the planner-facing declared boundary surface of a touchpoint
6. `counterfactual action` as planner-ranked metadata for blocked direct cuts
7. `Queue` as the planner-side scheduling envelope

`WorkstreamRegistry` is the core planning-substrate carrier. It is deliberately
single-root. The code-level contract in
[`workstream_registry.py`](../src/gabion/tooling/policy_substrate/workstream_registry.py)
and the glossary entry
[`glossary.md#workstream_registry`](../glossary.md#workstream_registry) agree on
this point: multi-root planner grouping must not widen the registry packet.

### Rooted containment grammar

The rooted ownership grammar is:

`root -> subqueue -> touchpoint -> touchsite`

This grammar is structural, not merely presentational:

- `root` is the top-level declared work item
- `subqueue` is the first-order structural child cut of a root
- `touchpoint` is the planner-visible actionable unit under a subqueue
- `touchsite` is the concrete boundary surface a touchpoint declares

In the aggregated substrate, the full registry catalog therefore forms a finite
family of rooted trees. Equivalently, the ownership layer is a finite rooted
forest of single-root packets.

### Overlay/runtime distinctions

The planning substrate also carries overlay distinctions that must remain
separate from rooted ownership:

- `Queue` is a planner envelope over the graph, not a root/subqueue synonym.
- dependency edges and blocker chains are overlay relations, not containment.
- counterfactual actions are predictive metadata, not ownership nodes.
- planning-chart phases are overlay groupings, not lifecycle states.

The glossary commutation law for `Queue` is the key formal statement here:

> projecting a queue back onto `contains` must preserve root/subqueue ownership.

In other words, queue overlays may intersect roots, but they must not rewrite
the rooted ownership forest.

### Synthetic operational roots

`UTR`, `DFR`, and `DFM` are synthetic roots. They are planner-visible and real,
but they are not owner roots for semantic-core subsystems.

- `UTR` summarizes unit-test readiness and repo-drain red-state.
- `DFR` summarizes current delivery-flow blockers from the canonical
  `delivery_flow_summary` ingress.
- `DFM` summarizes historical delivery-flow drag from that same canonical
  summary/history ingress.

These roots share the same registry grammar as ordinary roots, but they are
operational overlays expressed as roots so the planner can reason about them in
the same machine-readable shape as ownership queues.

<a id="runtime_pipeline_and_ordering"></a>
## Runtime pipeline and ordering principle

### Dependency direction

The canonical dependency direction is:

`registry declarations -> invariant marker scan -> invariant graph aggregation -> artifact ingress joins -> workstream projection -> planning chart -> project manager view`

Each layer is downstream of the previous one:

- registry declarations make planning structure explicit
- marker scan recovers decorated-symbol metadata
- invariant graph lifts declarations and joins artifacts
- workstream projection computes queue health and recommended cuts
- planning chart groups current planner items into `scan` / `predict` / `complete`
- project-manager view renders the resulting planning state

### Top-level ordering principle

The top-level ordering principle is **not** a linear milestone sequence.

The most faithful formal description is:

- a **finite rooted forest** of single-root declaration packets for ownership
- plus an **overlay multigraph** of dependencies, blockers, artifact evidence,
  and queue membership
- plus a **phase-local chart-parser-like projection** that groups planning
  items into `scan`, `predict`, and `complete`

This means the planning substrate is closer to a typed forest-with-overlays than
to a waterfall phase plan.

### Consequences

1. There is no single global milestone order that every work item must follow.
2. `scan`, `predict`, and `complete` are current chart views over the same
   runtime, not ownership stages.
3. Multi-root planning happens by graph/projection overlays, not by widening
   `WorkstreamRegistry`.

<a id="transition_and_acceptance_semantics"></a>
## Transition and acceptance semantics

The repository declares three distinct transition families. They must not be
collapsed into one vague “phase progression” story.

### 1. Registry lifecycle and planner status

Registry symbols carry lifecycle state through `todo_decorator` or
`landed_todo_decorator`. The typed registry definitions separately carry a
`status_hint`.

These are related but not identical:

- lifecycle answers what marker kind/state the symbol claims
- `status_hint` answers how the planner should currently treat the object

This separation is explicit in
[`docs/planning_substrate.md#runtime_model`](planning_substrate.md#runtime_model).

### 2. Workstream closure transitions

Closure acceptance is centralized in
`validate_workstream_closure_consistency(...)`.

Closure acceptance criteria are:

- landed lifecycle must match landed status
- landed nodes must not retain blocking dependencies
- landed reasoning text must use closed/recorded language
- landed parents must not retain open descendants
- landed roots must not reference missing descendants

This is the actual acceptance surface for “done” at the registry-structure
level. A prose claim of completion is insufficient if closure consistency fails.

### 3. Planning-chart phase grouping

The planning chart is explicitly domain-neutral and currently descriptive:

- `scan` groups live evidence and current planning surfaces
- `predict` groups dependencies and counterfactual futures
- `complete` groups selected cuts, followups, and lanes

These are not ownership transitions. A `complete` chart item is a downstream
recommendation, not a claim that the underlying root/subqueue/touchpoint is
structurally closed.

### Progress transitions are adjacent, not identical

[`docs/progress_transition_contract.md#progress_transition_contract`](progress_transition_contract.md#progress_transition_contract)
declares a separate progress/telemetry transition law over recursive progress
trees and `progress_transition_v2`.

That contract matters to repo runtime and telemetry, but it is not the planning
substrate’s ownership or chart-phase state machine. It constrains progress
normalization, not `WorkstreamRegistry` semantics.

<a id="invariant_preservation_and_aspf_relationship"></a>
## Invariant preservation and ASPF relationship

### How semantic invariants are preserved across transitions

The planning substrate preserves semantic invariants through five cooperating
mechanisms:

1. **Glossary commutation laws**
   - queue overlay must preserve root/subqueue ownership
   - root closure must remain invariant under planner overlays
   - `WorkstreamRegistry` must remain single-root
2. **Governance control loops**
   - policy/docflow/runtime gates turn declared obligations into sensors,
     actuators, verification commands, and escalation thresholds
3. **Identity separation**
   - object identity
   - site identity
   - structural identity
   - marker identity
4. **Closure-consistency validation**
   - landed/open transitions are mechanically checked
5. **Artifact-ingress normalization**
   - repo-local evidence is normalized into typed ingress carriers before
     graph-level planning joins

The net effect is that phase-local projections and queue overlays can change
without rewriting structural ownership semantics.

### Relationship to ASPF

ASPF is the adjacent self-analysis and witness layer. It provides:

- semantic surface projections
- 0-cell / 1-cell / 2-cell witness structure
- domain-to-ASPF cofibrations
- fiber/lattice algebra and witness-based drift classification

The planning substrate is related to ASPF, but it is not identical to it.

The clean boundary is:

- ASPF owns witness/fibration semantics and execution-trace algebra
- the planning substrate owns registry declarations, invariant-graph joins,
  workstream projection, and planning-chart overlays

Planning consumes ASPF-adjacent artifacts and identity infrastructure where
useful, but the planning runtime itself is registry/graph/projection/chart
oriented, not an ASPF fiber object by default.

<a id="gap_analysis"></a>
## Gap analysis

### Declared but not yet directly gated

1. The glossary entries for `Queue`, `Root`, `Subqueue`, and
   `WorkstreamRegistry` still say “Test Obligations (to be mapped).”
   The semantics are real, but the glossary-level obligation mapping is not yet
   closed as an explicit test matrix.
2. The planning-chart document explicitly says the current slice is
   descriptive only. The chart exists and is tested, but phase-local leverage
   and legality are not yet a dedicated direct gate family.

### Declared and indirectly tested

1. `POLICY_SEED.md`, `planning_substrate.md`, and ASPF-adjacent docs are mostly
   enforced through downstream policy/runtime tests rather than document-specific
   semantic tests.
2. `declared_workstream_registry_catalog.py` and
   `structured_artifact_ingress.py` are validated primarily through invariant
   graph behavior rather than through standalone artifact-specific suites.

### Structures used by the planning substrate but not yet glossary-raised

The current planning runtime uses `touchpoint`, `touchsite`, and
`counterfactual action` as first-class terms in code and docs, but the glossary
entries reviewed here are explicit for `Queue`, `Root`, `Subqueue`, and
`WorkstreamRegistry`. The absence of parallel glossary-raised entries for those
other planning nouns is a real vocabulary gap.

### Terms intentionally absent from the current planning formalism

The planning-substrate corpus reviewed here does **not** bind:

- `BSPᵗ`
- `BSPˢ`
- a planning-specific `q`
- a planning-specific `strata` formalism

Those terms appear elsewhere in the repository or in adjacent mathematical
surfaces, but they are not the canonical formal vocabulary of the planning
substrate as currently implemented. They should not be imported speculatively
into planning documentation.

## Locked conclusions

The following conclusions are fixed by the current repository state:

1. The planning substrate’s core formal object is `WorkstreamRegistry`, not a
   milestone file.
2. `WorkstreamRegistry` is intentionally single-root; multi-root views are
   downstream overlays.
3. `Queue` is an overlay/runtime projection concept, not a synonym for `root`
   or `subqueue`.
4. The phase system is the planning-chart overlay with `scan`, `predict`, and
   `complete`; it is descriptive/projectional, not an ownership-state machine.
5. Acceptance of “done” at the planning-structure level is governed by closure
   consistency plus policy/verification loops, not by prose milestone
   completion.
6. ASPF is related through witness/fibration and identity infrastructure, but
   the planning substrate remains a distinct runtime layer.
