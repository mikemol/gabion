---
doc_revision: 25
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: friction
doc_role: audit
doc_scope:
  - repo
  - tooling
  - dx
  - agents
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - AGENTS.md#agent_obligations
  - CONTRIBUTING.md#contributing_contract
  - glossary.md#contract
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  AGENTS.md#agent_obligations: 2
  CONTRIBUTING.md#contributing_contract: 2
  glossary.md#contract: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Friction funnel stays informative, but the constructability axiom and process-relative runtime framing remain the governing semantic backdrop for what counts as a real friction."
  AGENTS.md#agent_obligations: "Reviewed AGENTS.md#agent_obligations rev37/section v2; this funnel is aligned with the agent instruction surface that now points here for raw navigation/operational friction capture."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md#contributing_contract rev120/section v2; contributor workflow now points here as the canonical friction-capture funnel."
  glossary.md#contract: "Reviewed glossary.md#contract rev47/section v1; the runtime distinction ladder still justifies treating observable-but-uncovered navigation/operational frictions as valid runtime distinctions."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="friction_funnel"></a>
# Repo Navigation and Operational Friction Funnel

This document is the canonical funnel for recording navigation, operational, and
agent-orientation friction encountered while working in this repository.

The document itself is an informative audit/log surface. The workflow
expectation to use it is carried by `AGENTS.md#agent_obligations` and
`CONTRIBUTING.md#contributing_contract`.

> **Constructability axiom:** If a distinction is real, it must be
> constructible; if constructible, it must be behaviorally reachable; if
> reachable, it must be observable; if observable, it must be coverable; if
> not, it is not a valid runtime distinction.

LLM agents are treated as first-class developers for impact-assessment
purposes.

This funnel is for:
- repo navigation friction
- repo operation and workflow friction
- agent-orientation friction
- discoverability gaps that materially slow or mislead real work

This funnel is not:
- a generic feature-request backlog
- a design-proposal document
- a substitute for correction-unit ledgers or workstream registries

<a id="logging_protocol"></a>
## Logging Protocol

Use this protocol while planning or executing work:

1. Capture friction at the moment it is encountered.
2. Prefer raw observations over fixes or solutions.
3. Use the minimum schema needed to make the friction reusable by the next
   reader or agent.
4. Group repeated frictions only when they are clearly the same phenomenon.
5. Keep the main log append-only; move outcomes to `Resolution Notes` rather
   than rewriting history.

Entry lifecycle:
- Raw observations are append-only.
- Clarifications may be added beneath the same entry if they preserve the
  original observation.
- Resolutions belong in the separate resolution section and should reference
  the original friction ID.

<a id="entry_schema"></a>
## Entry Schema

### Required Fields

- `Trigger`: what action or moment exposed the friction
- `Friction`: the concrete difficulty, ambiguity, or hidden coupling
- `Impact`: who or what is affected, and how
- `Hypothesis`: the best current explanation of why this friction exists

### Optional Fields

- `Evidence`: paths, commands, logs, or artifacts that ground the observation
- `Workstream/Context`: the active root, correction unit, or task context
- `Higher-order synthesis`: optional deeper AA/AB/BA/BB analysis for unusually
  important frictions

### Template

```md
## FN-###

**Trigger:** ...

**Friction:** ...

**Impact:** ...

**Hypothesis:** ...

**Evidence:** ...

**Workstream/Context:** ...
```

### Logging Style

- Keep each field concise and factual.
- Prefer local, observable symptoms over abstract complaints.
- Do not turn raw entries into fix proposals unless the observation is
  inseparable from the mechanism.
- If a friction later proves illusory, record that in `Resolution Notes`
  instead of erasing the original note.

<a id="higher_order_synthesis"></a>
## Higher-Order Synthesis (Optional)

Two depths are allowed:

- `Raw log entry`: the default and expected mode
- `Higher-order synthesis`: optional deeper analysis for especially important
  or recurring frictions

If higher-order synthesis is used, keep it subordinate to the raw observation.
The raw `Trigger`/`Friction`/`Impact`/`Hypothesis` entry remains the canonical
surface; the synthesis is a supplement, not a replacement.

Permitted synthesis structure:
- `AA constructs`
- `AB critiques`
- `Convergence (Mind A)`
- optional `Mind B wedge product`

The deeper AA/AB/BA/BB or implication-tree style is intentionally optional.
Making it mandatory for every note would make the funnel too expensive to use
and would defeat its operational purpose.

<a id="resolution_notes"></a>
## Resolution Notes

Use this section for fix-forward outcomes, not for raw first observations.

Resolution format:
- reference the original `FN-###`
- state what changed
- state where the resolution landed
- keep the original friction entry untouched

### Resolution Template

```md
### Resolution: FN-###

Short outcome summary.

Evidence:
- path or commit
- validation or runtime confirmation
```

### Resolution: FN-000

This file previously existed as a long-form essay-style friction analysis.
That format has been superseded by this structured funnel so friction can be
captured incrementally during real work instead of only through large
post-hoc analyses. The highest-signal legacy notions have now been normalized
into the raw-observation format below, with optional synthesis retained only
where it materially preserves the original argument.

## Raw Observations

The first entries below are reconstructed from the prior essay-style
`friction.md` material and normalized into this funnel. They preserve the
legacy notion-level signal, but not the full recursive implication tree.

Add new friction entries below these imported observations using the schema
above.

## FN-015: Strict inner carrier, loose outer config

**Trigger:** Tightening the dataflow `project_root` contract after a failed
helper-level repair in the resume/snapshot path cluster.

**Friction:** The repo had already made inner indexed-scan carriers behave as
if `project_root` were strict, while the public `AuditConfig` and a few outer
entry points still allowed omission and local inference. That split contract
made downstream helpers look like the problem even though the ambiguity was
still being injected at ingress.

**Impact:** Refactors gravitate toward local "fixes" in path-format helpers,
then fail ambiguity policy or reintroduce fallback behavior elsewhere. Both
humans and LLMs burn time chasing the wrong seam because the inner and outer
surfaces advertise different truths.

**Hypothesis:** The repo normalized one side of the boundary during earlier
decomposition work but did not finish the ingress cutover, so strict internal
assumptions and loose public config coexisted long enough to feel normal.

**Evidence:**
- `src/gabion/analysis/dataflow/engine/dataflow_analysis_index.py`
- `src/gabion/analysis/dataflow/engine/dataflow_contracts.py`
- `src/gabion/analysis/dataflow/engine/dataflow_pipeline.py`
- `src/gabion/analysis/dataflow/io/dataflow_synthesis.py`

**Workstream/Context:** `PSN` paused during the dedicated `project_root`
ingress-normalization correction unit.

## FN-001: Production modules named `test_*.py` under `src/`

**Trigger:** Navigating the repo by filename while switching between product
code and the real `tests/` tree.

**Friction:** Several production modules under `src/` use `test_*.py` names,
even though the same lexical signal conventionally denotes executable test
files. The path prefix says "production module"; the basename says "test
module."

**Impact:** Both human and LLM agents can misclassify these files during
navigation, evidence gathering, and targeted validation planning. That slows
orientation and increases the chance of reading the wrong surface or trying to
run the wrong file as a test.

**Hypothesis:** The repo models "test" as a semantic evidence domain inside
production analysis code, but the naming convention was never normalized away
from pytest-style expectations once those modules settled under `src/`.

**Evidence:**
- `src/gabion/analysis/surfaces/test_evidence.py`
- `src/gabion/analysis/surfaces/test_behavior.py`
- `src/gabion/tooling/policy_rules/test_subprocess_hygiene_rule.py`

**Workstream/Context:** Reconstructed legacy notion; grounded against current
repo state during friction-funnel migration.

### Higher-order synthesis

- `AA constructs`: The naming choice exposes a real semantic distinction:
  these modules analyze test evidence and test-like behaviors, so "test"
  inside the filename carries domain meaning.
- `AB critiques`: The same token is already a strong operational classifier in
  Python repos. Once a production path and a test-only basename disagree, the
  distinction stops being cheaply behaviorally reachable.
- `Convergence (Mind A)`: The issue is not that "test" is semantically wrong;
  it is that the repo uses one lexical form for two different operational
  classes. That makes the distinction observable only after extra context
  lookup.
- `Mind B wedge product`: This is a constructability failure at the navigation
  layer. A real distinction exists, but the current naming does not make it
  directly reachable from the filename alone.

## FN-002: `gabion.toml` decision-tier coordinates can drift out of reach

**Trigger:** Treating `[decision]` tier coordinates in `gabion.toml` as an
authoritative map of important runtime decision sites.

**Friction:** A substantial portion of the configured tier coordinates now
refer to files that no longer exist. The config still presents precise
location-level distinctions, but many of those distinctions are no longer
constructible from the current tree.

**Impact:** Humans and LLMs using `gabion.toml` as a navigation or audit
surface are sent to dead paths, which makes the metadata look more precise
than it is and weakens trust in the surviving coordinates.

**Hypothesis:** Decision-tier metadata is maintained more loosely than the code
it annotates, and there is no strong guard that revalidates path existence
after refactors or module moves.

**Evidence:**
- Current `gabion.toml` contains coordinates under `[decision].tier1` and
  `[decision].tier3` that reference missing paths such as
  `src/gabion/analysis/dataflow_indexed_file_scan.py` and
  `src/gabion/tooling/governance_audit.py`
- Local validation during migration showed `tier1` had `7/13` missing-path
  entries and `tier3` had `26/38` missing-path entries

**Workstream/Context:** Reconstructed legacy notion; re-grounded against the
current config during friction-funnel migration.

### Higher-order synthesis

- `AA constructs`: Static decision-tier coordinates are valuable because they
  promise a stable bridge between governance intent and concrete code sites.
- `AB critiques`: That promise only holds if the cited coordinates remain
  behaviorally reachable. Dead paths turn a typed distinction into a stale
  historical gesture.
- `Convergence (Mind A)`: The real friction is not mere "config drift"; it is
  a broken observability contract. The file still claims a distinction that
  the current runtime tree cannot instantiate.
- `Mind B wedge product`: Under the constructability axiom, an unreachable
  coordinate is not just inconvenient metadata. It undermines the validity of
  the distinction the coordinate is supposed to carry.

## FN-003: ASPF facade and star-reexport chains obscure ownership

**Trigger:** Following ASPF and timeout-related symbols from public import
surfaces into the implementation that actually owns the behavior.

**Friction:** Several public-facing modules are thin facades over
`analysis.foundation.*_impl` modules, using star re-exports and, in some
cases, direct imports of underscore-prefixed implementation helpers. This
keeps the public surface thin, but it makes symbol ownership and reliable
go-to-definition behavior harder to recover.

**Impact:** Both human and LLM agents pay extra orientation cost when tracing
semantic ownership, especially during refactors or runtime debugging. The
surface looks public and stable, but the real behavior often lives one layer
deeper in an implementation module.

**Hypothesis:** The facade layer was introduced to stabilize neighborhood
surfaces during strictification, but the repo did not complete the later step
of turning the actually imported symbols into explicit public owner-module
interfaces.

**Evidence:**
- `src/gabion/analysis/timeout_context.py` re-exports
  `gabion.analysis.foundation.timeout_context` via `import *`
- `src/gabion/analysis/aspf/aspf.py` re-exports
  `gabion.analysis.foundation.aspf_impl` and also imports
  `_canonicalize_evidence` and `_canonicalize_evidence_value`
- `src/gabion/analysis/aspf/aspf_execution_fibration.py` re-exports
  `gabion.analysis.foundation.aspf_execution_fibration_impl` and also imports
  multiple underscore-prefixed helpers

**Workstream/Context:** Reconstructed legacy notion; still materially aligned
with current `PSN` public-surface normalization work.

### Higher-order synthesis

- `AA constructs`: Facades can be legitimate when they stabilize imports and
  allow the implementation to move without breaking callers.
- `AB critiques`: Once the facade is mostly `import *` plus private-helper
  reach-ins, it no longer clarifies the public surface. It hides ownership
  while still leaking implementation topology.
- `Convergence (Mind A)`: The problem is not "multiple modules exist"; it is
  that the public/private distinction is not made explicit enough for low-cost
  navigation.
- `Mind B wedge product`: This friction links directly to public-surface
  normalization: if a symbol is worth importing across module boundaries, its
  owner should expose it publicly instead of relying on star-reexport facades
  and underscore imports.

## FN-004: Pure-forwarder heuristics overmatch package markers and entrypoints

**Trigger:** Sweeping `PSN` for pure import-forwarder modules before deleting
the first runtime shim layer.

**Friction:** A broad AST-level forwarder detector catches the real shim files,
but it also catches legitimate package barrels, `__main__.py` entrypoints, and
small owner modules that happen to have very little top-level structure. The
raw candidate list is not directly actionable.

**Impact:** Both human and LLM agents have to do manual triage before editing.
Without that extra pass, the detector encourages over-deletion and makes the
closure surface look more uniform than it really is.

**Hypothesis:** The repo has a real policy distinction between forbidden pure
forwarders and allowed thin surfaces, but that distinction is not yet reified
in one canonical detector. Exploratory scans therefore collapse multiple
surface classes into one noisy candidate bucket.

**Evidence:**
- A crude PSN forwarder sweep surfaced actual shim modules alongside
  `src/gabion/__main__.py` and multiple package `__init__.py` files
- Manual follow-up was required to separate delete-now shims from allowed thin
  surfaces

**Workstream/Context:** PSN forwarder-drain execution.

### Higher-order synthesis

- `AA constructs`: A repo-wide heuristic is useful because it makes the hidden
  forwarder surface visible quickly.
- `AB critiques`: Visibility alone is not enough when the detector does not
  encode the policy distinction it is supposed to support.
- `Convergence (Mind A)`: The friction is not "heuristics are bad"; it is that
  the current heuristic is weaker than the architectural rule it is being used
  to approximate.
- `Mind B wedge product`: This is exactly why PSN needs a real no-forwarder
  enforcement surface rather than one-off exploratory scans.

## FN-005: Live registry layers can retain deleted-surface touchsites

**Trigger:** Deleting real forwarder modules during `PSN` and then scanning the
repo for remaining references to confirm the cutover is truthful.

**Friction:** Some planning-substrate and audit registries still carry touchsite
or evidence paths for modules that are now gone. The references are not always
historical-only; some are still part of live workstream surfaces. That makes it
unclear which path references are supposed to be updated as part of the same
correction unit and which are intentionally archival.

**Impact:** Humans and LLM agents have to distinguish "safe historical residue"
from "active-surface drift" by hand. That increases the chance of either
leaving the repo semantically stale or over-editing registries whose role is
only provenance.

**Hypothesis:** The repo has multiple overlapping truth surfaces for code
ownership and migration state, but their lifecycle semantics are not yet made
explicit enough at the point of use. Once a wrapper file is removed, the
remaining path references no longer advertise whether they are declarative
history, current control state, or just stale coupling.

**Evidence:**
- After removing governance wrapper files under `src/gabion_governance/`,
  `src/gabion/tooling/policy_substrate/connectivity_synergy_registry.py` still
  contained live touchsite rows for the deleted paths
- The `PSN` registry required its own touchsites to be updated in the same
  correction unit to keep the workstream truthful

**Workstream/Context:** `PSN` governance/package veneer drain.

### Higher-order synthesis

- `AA constructs`: Multiple registry layers are valuable because they capture
  different dimensions: current work, historical convergence, and audit
  evidence.
- `AB critiques`: Those distinctions stop being cheaply usable when the
  registry entries do not declare their lifecycle strongly enough at the moment
  a path disappears.
- `Convergence (Mind A)`: The friction is not "too many registries"; it is that
  the repo asks the operator to infer whether a surviving path reference is
  still normative, still active, or merely archival.
- `Mind B wedge product`: This is a constructability problem for migration
  truthfulness. Deleting a surface should leave behind machine-visible signals
  for which residual references must move and which may remain as history.

## FN-006: Facade drains are slowed by implicit public export surfaces

**Trigger:** Burning down the ASPF facade cluster under `PSN` after the
runtime-shim and governance-veneer slices had already landed cleanly.

**Friction:** The foundation owner modules do not declare an explicit public
export list, so the facade currently derives its surface from `import *` plus
extra private reach-ins. Removing the facade-level underscore imports first
requires reconstructing which names are intentionally public and which are just
incidental top-level definitions.

**Impact:** Humans and LLM agents have to do extra structural inspection before
making a bounded change. That increases the cost of each public-surface cleanup
unit and makes it easier to miss a name that is still relied on through the
facade.

**Hypothesis:** The facade layer was introduced before the owner-module public
surface was fully reified. As a result, the facade became both the boundary and
the only practical export inventory.

**Evidence:**
- `src/gabion/analysis/aspf/aspf.py`,
  `src/gabion/analysis/aspf/aspf_execution_fibration.py`, and
  `src/gabion/analysis/aspf/aspf_resume_state.py` all derive most of their
  visible surface from `import *`
- their foundation owners do not declare `__all__`, so public-name discovery
  had to be reconstructed by AST/top-level inspection during the PSN unit

**Workstream/Context:** `PSN` ASPF facade drain.

### Higher-order synthesis

- `AA constructs`: A thin facade is easier to keep stable when the owner
  surface is still moving.
- `AB critiques`: Once the facade is the only export inventory, it becomes hard
  to retire because the public/private distinction is no longer declared at the
  owner.
- `Convergence (Mind A)`: The friction is not just `import *`; it is the lack
  of a constructible owner-module public surface once the facade is removed.
- `Mind B wedge product`: Public-surface normalization goes faster when the
  owner module can already answer “what is public here?” without requiring a
  second inference pass.

## FN-007: Ambiguity remediation can require a two-step shape correction

**Trigger:** Running `gabion policy check --ambiguity-contract` after a local
helper cleanup during the ASPF facade drain.

**Friction:** A direct fix for one ambiguity rule shape can immediately trigger
the adjacent rule family. In this case, replacing `match ... case _` fallthrough
branches with `isinstance(...)` conditionals cleared `ACP-005` but introduced
`ACP-003`, so the real accepted form was a third shape: explicit
`singledispatch`-based runtime-shape normalization.

**Impact:** Small semantic-preserving edits can take an extra remediation cycle
 unless the agent already knows the repo’s preferred enforcement idioms. That
 increases correction-unit latency and makes local fixes feel less local than
 they first appear.

**Hypothesis:** The policy rules are individually coherent, but the repo does
 not yet provide a close-at-hand “preferred remediation shapes” index for the
 most common rule interactions, so agents discover the allowed form by failing
 one gate after another.

**Evidence:**
- `src/gabion/analysis/foundation/aspf_impl.py`
- `docs/policy_rules/ambiguity_contract.md#acp-003`
- `docs/policy_rules/ambiguity_contract.md#acp-005`

**Workstream/Context:** `PSN` ASPF facade drain.

### Higher-order synthesis

- `AA constructs`: The rules are doing their job by rejecting both latent
  wildcard ambiguity and ad-hoc deterministic-core type narrowing.
- `AB critiques`: Without a nearby idiom catalog, the shortest local repair is
  often not the repo-approved repair, so the first fix becomes exploratory
  churn.
- `Convergence (Mind A)`: The friction is not that the policy is too strict;
  it is that the accepted repair shape is discoverable mainly by gate failure
  rather than by a direct local cue.
- `Mind B wedge product`: A small remediation index that maps common rule pairs
  to preferred structural rewrites would save both human and LLM compute.

## FN-008: ASPF taint isomorphism governance is too coarse for low-semantic-drift refactors

**Trigger:** Repeatedly hitting the ASPF taint crosswalk acknowledgement gate
while making public-surface normalization changes that do not alter emitted
ASPF identities, witness payloads, or taint semantics.

**Friction:** The current governance model treats a broad set of ASPF-adjacent
file changes as needing the same heavyweight acknowledgement path. Small
surface-shape refactors, import-surface publicization, and other semantic-noop
ownership cleanups still require manual updates to
`docs/aspf_taint_isomorphism_no_change.yaml`, even when the actual taint
crosswalk semantics are unchanged and the justification is nearly formulaic.

**Impact:** Correction units pay recurring governance tax that is weakly
proportional to the real semantic risk. That slows local iteration, adds a
repeated doc-edit burden, and makes policy compliance feel more like ceremony
than signal for this class of change.

**Hypothesis:** The current gate is acting as a coarse changed-path proxy for
ASPF-taint semantic risk because the repo lacks a finer-grained way to
distinguish "touches ASPF-adjacent code" from "changes ASPF taint/crosswalk
meaning." The result is safe but over-broad governance.

**Evidence:**
- repeated `gabion policy check --workflows` failures during PSN slices that
  only publicized existing owner surfaces
- repeated updates to `docs/aspf_taint_isomorphism_no_change.yaml` with
  substantially similar no-change justifications
- recent touched files:
  `src/gabion/analysis/aspf/aspf.py`,
  `src/gabion/analysis/aspf/aspf_execution_fibration.py`,
  `src/gabion/analysis/foundation/aspf_impl.py`,
  `src/gabion/analysis/dataflow/engine/dataflow_evidence_helpers.py`

**Workstream/Context:** `PSN` ASPF facade drain and planning/governance helper
publicization slices.

### Higher-order synthesis

- `AA constructs`: A conservative taint-governance gate is legitimate because
  ASPF witness/crosswalk drift is high-cost and easy to miss.
- `AB critiques`: The present trigger surface is so broad that many
  semantically inert refactors still look like taint-risk events and must pass
  through the full acknowledgement loop.
- `Convergence (Mind A)`: The friction is not the existence of taint
  governance; it is the lack of a finer-grained admissibility test that can
  cheaply separate semantic drift from public-surface hygiene.
- `Mind B wedge product`: A better model likely needs a constructible middle
  tier between "ASPF-adjacent path touched" and "taint/crosswalk meaning
  changed," so governance effort scales with observable semantic risk.

## FN-009: Planning touchsites drift behind the active correction unit

**Trigger:** Re-truthing `PSN-TP-006` after completing the next bounded
dataflow parse-helper publicization slice.

**Friction:** The planning-substrate touchpoint still named the previously
finished ASPF tranche even though the active work had moved into a different
dataflow helper seam. That made the workstream look more converged than the
current implementation reality and forced a second maintenance pass just to
make the registry tell the truth again.

**Impact:** Correction units pay extra bookkeeping cost, and the planning
surface becomes less trustworthy as an execution map. For an LLM, that means
the registry cannot be used as a direct boundary source without re-verifying it
against the live code diff.

**Hypothesis:** Touchsite declarations are currently maintained as a manual
shadow of active ownership boundaries. When the correction unit boundary shifts
between adjacent slices, the registry lags because there is no cheap,
constructible coupling between "files actually changed for this touchpoint" and
"files the workstream says are active."

**Evidence:**
- `src/gabion/tooling/policy_substrate/public_surface_normalization_registry.py`
  still pointed `PSN-TP-006` at the already-finished ASPF surfaces while the
  active worktree only touched dataflow parse-helper consumers
- the dirty worktree for the active slice was concentrated in
  `src/gabion/analysis/dataflow/io/dataflow_parse_helpers.py`,
  `src/gabion/analysis/dataflow/io/dataflow_reporting_helpers.py`, and direct
  engine consumers such as
  `src/gabion/analysis/dataflow/engine/dataflow_post_phase_analyses.py`
- the focused runtime-policy assertion for `PSN-TP-006` had to be updated in
  lockstep once the touchsites were re-truthed

**Workstream/Context:** `PSN` dataflow private-import publicization.

### Higher-order synthesis

- `AA constructs`: Planning touchsites help keep multi-slice refactors legible
  and auditable.
- `AB critiques`: If touchsites are not refreshed at the same rate as the
  correction unit boundary, they stop being a trustworthy execution map and
  become historical residue.
- `Convergence (Mind A)`: The friction is not the existence of touchsites; it
  is the lack of a low-cost coupling between declared active boundaries and the
  files the current correction unit actually moved.
- `Mind B wedge product`: A better planning substrate would make boundary drift
  observable earlier, ideally before the touchpoint metadata can overstate
  convergence.

## FN-010: Duplicate helper ownership hides the real publicization seam

**Trigger:** Burning down the next `PSN-TP-006` slice around
`dataflow_evidence_helpers.py`.

**Friction:** Names such as `module_name` and `is_test_path` were effectively
owned in more than one dataflow helper module. That made the next PSN slice
look smaller than it really was: the repo had to answer not only "which import
should become public?" but also "which module is actually the canonical owner
of this behavior?"

**Impact:** Boundary discovery gets slower and more error-prone. For an LLM,
the cheapest local grep is not enough to identify the right owner; the agent has
to inspect duplicate helper implementations and infer which one active callers
actually depend on.

**Hypothesis:** Earlier decomposition moved behavior into smaller files faster
than it converged owner contracts, so some helper names became operationally
shared across adjacent modules instead of being retired to one canonical owner.

**Evidence:**
- `src/gabion/analysis/dataflow/engine/dataflow_evidence_helpers.py`
- `src/gabion/analysis/dataflow/engine/dataflow_function_index_helpers.py`
- active `PSN-TP-006` consumers were importing private names from the evidence
  helper module while similarly named helper implementations still existed in
  neighboring owner files

**Workstream/Context:** `PSN` dataflow private-import publicization.

### Higher-order synthesis

- `AA constructs`: Helper extraction improves locality and can reduce monolith
  pressure during staged decomposition.
- `AB critiques`: If old and new helper owners both remain viable, public-surface
  normalization turns into ownership archaeology rather than straightforward
  import cleanup.
- `Convergence (Mind A)`: The friction is not decomposition itself; it is the
  absence of a forced canonical-owner collapse once adjacent helper modules start
  sharing behavior.
- `Mind B wedge product`: Public-surface drains go faster when helper extraction
  and owner convergence happen in the same correction loop rather than as
  separate later cleanups.

## FN-011: Public-surface drains spill into shadow metadata and evidence carriers

**Trigger:** Publicizing `merge_counts_by_knobs` as the owner surface for the
next bounded `PSN-TP-006` seam.

**Friction:** Renaming the real owner function was only part of the slice. The
same symbol name also lived in planning touchsites, alias-adapter metadata, and
test evidence annotations. The code seam was small, but the truthful closure
surface was wider than the implementation diff first suggested.

**Impact:** Even a narrow owner publicization pays extra maintenance cost across
several non-obvious surfaces. For an LLM, "the import is fixed" is not enough;
the agent still has to find and normalize shadow metadata that preserves the
old private-name story.

**Hypothesis:** The repo has multiple parallel observability layers for semantic
surfaces, but those layers are not mechanically coupled to owner-name changes.
As a result, public-surface normalization propagates as a manual fan-out across
code, planning metadata, and evidence carriers.

**Evidence:**
- `src/gabion/analysis/dataflow/engine/dataflow_bundle_merge.py`
- `src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan_alias_adapter_decision.py`
- `src/gabion/tooling/policy_substrate/public_surface_normalization_registry.py`
- `tests/gabion/synthesis/test_types.py`
- `tests/gabion/analysis/dataflow_s1/dataflow_kitchen_sink_cases.py`

**Workstream/Context:** `PSN-TP-006` bundle-merge owner publicization.

### Higher-order synthesis

- `AA constructs`: Metadata and evidence carriers are legitimate observability
  surfaces, so they should track public-surface changes.
- `AB critiques`: When those surfaces are loosely coupled, a tiny refactor
  turns into a scavenger hunt across unrelated layers.
- `Convergence (Mind A)`: The friction is not the existence of metadata; it is
  the lack of a constructible propagation path from owner-surface changes to
  the shadow layers that narrate those surfaces.
- `Mind B wedge product`: Better convergence would come from making
  public-surface ownership a shared primitive that planning and evidence layers
  can consume directly instead of copying names as inert strings.

## FN-012: Some PSN seams are really owner-collision repairs, not simple helper promotions

**Trigger:** Burning down the snapshot-path normalization seam under
`PSN-TP-006`.

**Friction:** The touched private import in `dataflow_snapshot_io.py` was not
enough to identify the right public owner. A separate public helper already
existed in `dataflow_resume_paths.py`, but its contract was too loose to serve
as the canonical PSN target, so the seam still had to converge on the stricter
local owner instead of simply following the existing public path.

**Impact:** Seam sizing by direct import sites alone is misleading. For an LLM,
the cheapest “make the imported helper public” move can be the wrong fix when
the repo already contains an overlapping public owner elsewhere.

**Hypothesis:** Earlier decomposition created utility islands faster than it
converged owner contracts, so some stable helpers now exist both as local
implementation detail and as already-public shared utility surfaces, but those
public siblings do not always satisfy the stricter semantic-core contract that
PSN needs.

**Evidence:**
- `src/gabion/analysis/dataflow/io/dataflow_snapshot_io.py`
- `src/gabion/analysis/dataflow/engine/dataflow_resume_paths.py`
- `src/gabion/analysis/dataflow/engine/dataflow_lint_helpers.py`
- `src/gabion/analysis/dataflow/engine/dataflow_fingerprint_helpers.py`

**Workstream/Context:** `PSN-TP-006` snapshot-path normalization seam.

### Higher-order synthesis

- `AA constructs`: It is valid to start from the direct private-import seam
  because that is the observable violation.
- `AB critiques`: If owner selection stops at "already public," PSN can still
  converge on the wrong surface and import policy debt instead of removing it.
- `Convergence (Mind A)`: The real task is owner convergence under the repo's
  semantic contract, not just import hygiene or pre-existing publicity.
- `Mind B wedge product`: A better PSN workflow would distinguish “promote this
  helper” from “collapse onto the already-public sibling owner” before edits
  begin.

## FN-013: Boundary markers can legitimize a seam before they improve its contract

**Trigger:** Testing whether `dataflow_resume_paths.normalize_snapshot_path`
should be repaired by grading it as a semantic carrier boundary.

**Friction:** The marker-only probe was policy-legal, but it still left the
shared owner on the looser `root: object` contract. That made it too easy to
stop at “the gate is green” even though the underlying caller/callee contract
was still broader than it needed to be.

**Impact:** A passing policy gate can conceal an unfinished convergence step.
For an LLM, that creates a strong temptation to treat legality as completion
and ship a boundary marker instead of a stricter owner contract.

**Hypothesis:** The repo correctly distinguishes lawful boundaries from core
logic, but the local optimization pressure of getting back to green can make a
true-but-incomplete boundary grading look like the final fix unless the agent
explicitly prefers contract tightening over marker-only relief.

**Evidence:**
- `src/gabion/analysis/dataflow/engine/dataflow_resume_paths.py`
- the failed `PSN-TP-006` detour through `normalize_snapshot_path`
- the subsequent decision to strictify the shared owner and push fallback to
  caller-local adapters instead

**Workstream/Context:** Dedicated `resume_paths` contract-fix correction unit,
spun out of paused `PSN-TP-006`.

### Higher-order synthesis

- `AA constructs`: If a seam is genuinely a boundary, grading it explicitly is
  correct and useful.
- `AB critiques`: Correct grading can still be an incomplete answer if the
  boundary contract remains broader than the live callers actually need.
- `Convergence (Mind A)`: The durable preference is to make the lawful boundary
  as strict as the observed callers permit, then push any residual fallback
  outward.
- `Mind B wedge product`: “Boundary or core?” is not the only question;
  “maximally strict lawful boundary or convenience boundary?” is the more
  decision-relevant fork once the seam is known to be real.

## FN-014: A small helper seam can hide a larger ingress-contract mismatch

**Trigger:** Trying to fix `dataflow_resume_paths.normalize_snapshot_path` as an
isolated correction unit after the optional-root contract surfaced under
`PSN-TP-006`.

**Friction:** The visible seam was the snapshot-path helper, but the real
instability sits farther out: `AuditConfig.project_root` remains optional while
`_IndexedPassContext.project_root` is already modeled as a strict `Path`. Every
attempt to "just fix the helper" reintroduced fallback logic because the
upstream ingress contract is still unresolved.

**Impact:** A seemingly local helper repair can turn into repeated false starts.
For an LLM, this creates expensive thrash: several plausible boundary shapes
pass a first smell test, but the ambiguity gate keeps revealing that the true
normalization seam is the broader ingress contract rather than the touched
utility.

**Hypothesis:** The repo already contains the intended stricter internal shape,
but the normalization step that should convert optional config state into that
shape was never fully centralized. As a result, downstream helpers inherit an
optional-root residue that looks like their problem until policy checks force
the search back outward.

**Evidence:**
- `src/gabion/analysis/dataflow/engine/dataflow_contracts.py`
- `src/gabion/analysis/dataflow/engine/dataflow_analysis_index.py`
- `src/gabion/analysis/dataflow/engine/dataflow_resume_paths.py`
- `src/gabion/analysis/dataflow/io/dataflow_snapshot_io.py`
- the failed helper-only detour through `normalize_snapshot_path_optional_root`
  and `build_snapshot_path_normalizer(...)`

**Workstream/Context:** Dedicated `resume_paths` contract-fix correction unit,
paused after ambiguity-policy validation showed the true seam is `project_root`
ingress normalization.

### Higher-order synthesis

- `AA constructs`: It is reasonable to start from the observed helper seam,
  because that is where the contract violation first became visible.
- `AB critiques`: If the upstream carrier mismatch remains, local helper
  repairs only relocate the ambiguity instead of discharging it.
- `Convergence (Mind A)`: The durable fix is to normalize `project_root` once
  at the real ingress boundary and let path helpers stay strictly internal.
- `Mind B wedge product`: Better repo guidance would surface "latent strict
  contract already declared upstream" as a first-class diagnostic pattern, so
  agents look for ingress mismatches earlier instead of burning cycles on local
  seam repairs.

## FN-016: Ambiguity-gate resistance is often a correction-unit boundary signal

**Trigger:** Reworking the `project_root` ingress-normalization tranche after
the first strictification pass made `--ambiguity-contract` fail across touched
downstream helper files such as `dataflow_snapshot_io.py`,
`dataflow_resume_paths.py`, and `file_internal_analysis.py`.

**Friction:** A broad "make everything strict now" pass can feel locally
correct, but the ambiguity gate may be telling you that the strictification has
crossed out of the true ingress seam and into helper/publicization work that
belongs in a later correction unit. The failure mode is subtle because the
downstream edits are directionally right, yet still illegal in the current
boundary shape.

**Impact:** Without recognizing that signal, an LLM can burn cycles trying to
force the same policy outcome through increasingly contorted helper rewrites.
That produces exactly the entropy the correction unit was supposed to remove:
extra touched files, wider validation fallout, and muddled commit boundaries.

**Hypothesis:** The repo's ambiguity and grade gates are acting as a
work-partition sensor, not just a correctness checker. When a tranche goes
green only after downstream helper strictification is trimmed back to the true
ingress seam, that is evidence the later helper work is real but belongs to a
different correction unit.

**Evidence:**
- the failed intermediate `--ambiguity-contract` run during strict
  `project_root` ingress normalization
- the subsequent rollback of downstream helper strictifications in
  `dataflow_snapshot_io.py`, `dataflow_resume_paths.py`,
  `dataflow_function_index_helpers.py`, and
  `file_internal_analysis.py`
- the restored green ambiguity gate after narrowing the tranche back to ingress
  contracts and callsite signatures

**Workstream/Context:** Dedicated dataflow `project_root` ingress-normalization
correction unit, recut after `FN-014` exposed that the helper seam was not the
right first landing zone.

### Higher-order synthesis

- `AA constructs`: If strict inputs are the goal, propagating that strictness
  outward through every reachable helper seems like disciplined follow-through.
- `AB critiques`: The ambiguity gate is not only checking directionality; it is
  also checking whether the current correction unit has crossed into a
  different seam with its own ownership and validation burden.
- `Convergence (Mind A)`: When the gate resists a downstream helper
  strictification but accepts the narrowed ingress-only slice, treat that as a
  trustworthy cue to split the work rather than to push harder.
- `Mind B wedge product`: The highest-value synthesis is to read policy fallout
  as boundary discovery. The gate is not merely vetoing code; it is helping
  locate the lawful correction-unit frontier.

## FN-017: A semantic edge in intent may still be a non-boundary in repo policy

**Trigger:** Trying to strictify `project_root` on
`dataflow_snapshot_io.render_structure_snapshot(...)`,
`dataflow_snapshot_io.render_decision_snapshot(...)`, and
`dataflow_function_index_helpers._module_name(...)` immediately after the
ingress-normalization tranche landed.

**Friction:** These surfaces look edge-like to a human reader because they are
rendering helpers and path-normalization helpers. But the ambiguity gate still
treated them as ordinary internal edges, not as lawful final-boundary sites for
this correction unit. That means "furthest from core in intent" is not the same
thing as "currently lawful boundary in repo policy."

**Impact:** An LLM can overfit to architectural intuition and keep pushing
strictification outward through surfaces that feel like edges, even when the
repo's policy model says those surfaces still inherit core/helper obligations.
That wastes momentum and risks turning one good correction unit into a second
failed slice.

**Hypothesis:** The repo's real boundary model depends on more than human
semantic naming. Output helpers and path shapers may still be policy-visible as
ordinary internal functions until they are reified under an explicit boundary or
ownership model. Without that reification, a signature-only strictification can
be directionally right but still non-landable.

**Evidence:**
- the failed ambiguity run after strictifying
  `dataflow_snapshot_io.py` and `dataflow_function_index_helpers.py`
- the immediate rollback of that slice while leaving the earlier
  `project_root` ingress tranche intact
- the contrast between green targeted pytest/workflows and red
  `--ambiguity-contract`

**Workstream/Context:** Follow-on `project_root` strictification after
`GH-214 Strictify dataflow project_root ingress`.

### Higher-order synthesis

- `AA constructs`: Rendering and path-shaping helpers are natural candidates for
  "push strictness to the outer edge."
- `AB critiques`: Repo policy is not judging by naming or intuitive role alone;
  it is judging by the currently declared structural/boundary semantics.
- `Convergence (Mind A)`: The correct move is to treat this as a boundary-model
  discovery failure, roll the slice back, and choose the next seam whose
  lawfulness is clearer.
- `Mind B wedge product`: There are two different "edges" in play:
  architectural edges and policy-admitted edges. Progress depends on not
  conflating them.

## FN-018: Tightened owner contracts still leak if the request carrier stays loose

**Trigger:** Strictifying `project_root` at refactor ingress while
`RefactorRequest.target_path` and `LoopGeneratorRequest.target_path` still
entered the core as `str`.

**Friction:** It is easy to tighten a core owner and still leave the incoming
carrier loose enough that the next layer has to classify shape again. Here the
core remained stuck deciding whether `target_path` was relative or absolute,
even after `project_root` itself had been made strict.

**Impact:** An LLM under pressure can mistake that situation for a local helper
problem and keep adding `Path(...)`, `is_absolute()`, or runtime type checks in
core methods. That preserves the same ambiguity while merely moving the branch
site, and the ambiguity gate correctly pushes back.

**Hypothesis:** When a strictification does not commute, inspect the request
carrier one level upstream before touching more helpers. If the carrier still
publishes a looser type than the core actually wants, the real fix is to
reify and normalize there, not to defend the core repeatedly.

**Evidence:**
- `RefactorRequest.target_path` and `LoopGeneratorRequest.target_path` still
  being string carriers while refactor core methods wanted `Path`
- the failed ambiguity run flagging `isinstance(...)` and `path.is_absolute()`
  as downstream reclassification
- the cleaner follow-on shape: boundary normalization in `server.py`, strict
  `Path` carriers in `refactor/model.py`, and direct `Path` consumption in
  `engine.py` and `loop_generator.py`

**Workstream/Context:** Follow-on refactor-ingress strictification after
`GH-214 Strictify dataflow project_root ingress`, while continuing the broader
"push ambiguity to origin" policy burn-down.

### Higher-order synthesis

- `AA constructs`: Once `project_root` is strict, it is tempting to polish the
  remaining path handling locally in whichever core method still has the branch.
- `AB critiques`: If the request carrier is still loose, that local polish is
  fake progress because the branch is being regenerated from upstream shape.
- `Convergence (Mind A)`: The lawful move is to strictify the request carrier
  and absolutize it at the outer boundary, then let the core consume one shape.
- `Mind B wedge product`: The deeper pattern is that "strict owner plus loose
  request DTO" is a stable ambiguity generator. Eliminating it gives the repo a
  reusable precedent for future ingress work.

## FN-019: Once ingress ambiguity is removed, the next gate may be a boundary-disclosure failure

**Trigger:** Returning to `project_root` strictification in
`dataflow_snapshot_io.py` after the dataflow and refactor ingress carriers were
made strict.

**Friction:** The first failed snapshot/render attempt looked like another
helper-level `project_root` issue, but once the upstream ambiguity was gone the
ambiguity gate was actually pointing at two different distinctions:
- path containment inside `normalize_snapshot_path(...)`
- undeclared output/materialization boundaries in
  `render_structure_snapshot(...)` and `render_decision_snapshot(...)`

That is a different class of problem than the original optional-root seam, but
it emerges only after the origin ambiguity is removed.

**Impact:** Humans and LLMs can misread the next red gate as "the previous
strictification was wrong" and start reintroducing fallback or optionality.
That loses momentum and obscures the useful signal: the repo is now ready for a
more explicit boundary/decision declaration at the downstream owner.

**Hypothesis:** Some repo surfaces are not truly discoverable as boundaries by
name or directory alone. Once a loose ingress is fixed, the remaining latent
decision or materialization semantics become newly observable, and the gate
expects those to be declared rather than inferred from role intuition.

**Evidence:**
- the first red `--ambiguity-contract` run after strictifying
  `dataflow_snapshot_io.py`
- the green rerun after making path containment explicit and declaring the
  snapshot renderers as semantic-carrier adapters
- the contrast between already-green focused tests/workflow policy and the
  initially red ambiguity gate

**Workstream/Context:** `PSN-TP-006` snapshot/render owner slice after
`GH-214 Strictify refactor target-path ingress and preserve project_root call
shapes`.

## FN-020: Performative optionality hides the real boundary

**Trigger:** Chasing the next `project_root` origin after the refactor and
snapshot ingress slices were already strict.

**Friction:** A surface can still advertise `config=None` or `config.project_root
or root` long after every real caller has converged on explicit strict input.
That optionality is no longer carrying legitimate behavior; it is only masking
the fact that the function is already acting as a true ingress or
normalization boundary.

**Impact:** Humans and LLMs waste time treating the API as more permissive than
the repo actually is. That encourages local fallback repairs and delays the
moment when the boundary gets named and validated as such.

**Hypothesis:** Repo evolution often tightens callers first, leaving the public
surface lagging behind as a compatibility-shaped shell. Once that shell stops
matching real use, it becomes ambiguity debt rather than compatibility value.

**Evidence:**
- `src/gabion/analysis/dataflow/engine/dataflow_pipeline.py` still advertising
  `config=None` despite real callers already supplying `AuditConfig`
- `src/gabion/analysis/surfaces/test_evidence_suggestions.py` still carrying
  `config.project_root or root` after `root` and `AuditConfig.project_root`
  were both already strict in practice
- the ambiguity gate only settled once those surfaces were treated as real
  boundaries/adapters instead of pseudo-flexible helpers

**Workstream/Context:** `PSN-TP-006` follow-on root-origin burn-down after the
snapshot-render strictification tranche.

## FN-021: Structural caches turn into hidden ingress policy if they retain origin-derived state

**Trigger:** Continuing the next `PSN-TP-006` report/lint slice after
`project_root` had already been made strict at the main dataflow and refactor
ingresses.

**Friction:** `BundleProjection` looked like a purely structural intermediate,
but it still carried a cached `root` derived from `file_paths`. That let the
reporting and lint surfaces keep consuming an inferred origin without
admitting that they were depending on origin policy at all.

**Impact:** The seam is easy to misread as a leaf-helper problem inside
`_normalize_snapshot_path(...)` or `render_component_callsite_evidence(...)`.
An LLM under pressure can try to "fix the helper" while leaving the actual
ambiguity alive in the projection builder. That recreates the same fallback
behavior under a more indirect name.

**Hypothesis:** When an intermediate dataclass or projection artifact stores a
value that was inferred from inputs rather than computed from its own semantic
domain, it is often acting as a disguised ingress boundary. The right move is
to strip the cached origin-derived field and force real callers to publish the
strict value explicitly.

**Evidence:**
- `src/gabion/analysis/dataflow/io/dataflow_graph_rendering.py`
- `src/gabion/analysis/dataflow/io/dataflow_reporting.py`
- `src/gabion/analysis/dataflow/engine/dataflow_lint_helpers.py`
- the green focused rerun only after `BundleProjection.root` was removed and
  reporting/lint callers started passing explicit `project_root`

**Workstream/Context:** `PSN-TP-006` report/lint projection slice immediately
after the snapshot-render strictification tranche.

## FN-022: Duplicate helper owners preserve dead ambiguity after the real boundary is already strict

**Trigger:** Tracing the remaining snapshot-path `root` ambiguity after the
report/lint, indexed-scan, and deadline `project_root` slices were already
green.

**Friction:** The remaining red shape was no longer coming from a live ingress.
The strict public owner already existed in
`dataflow_snapshot_io.normalize_snapshot_path(path, root: Path)`, but a second
copy in `dataflow_resume_paths` still advertised `root: object`, and local
wrapper functions in `dataflow_post_phase_analyses`,
`dataflow_projection_materialization`, and `dataflow_graph_rendering` kept that
dead shape circulating.

**Impact:** Humans and LLMs can waste time chasing a non-existent upstream
alternation because the codebase still contains multiple near-identical owners
with different contracts. That slows PSN work and makes it look like ambiguity
is still entering the system when the real issue is owner duplication drift.

**Hypothesis:** Once a strict owner lands, stale helper copies can keep the old
signature alive longer than the real behavior. If they are not collapsed
promptly, they become fossilized pseudo-boundaries that manufacture confusion
without adding semantic value.

**Evidence:**
- `src/gabion/analysis/dataflow/io/dataflow_snapshot_io.py`
- `src/gabion/analysis/dataflow/engine/dataflow_resume_paths.py`
- `src/gabion/analysis/dataflow/engine/dataflow_post_phase_analyses.py`
- `src/gabion/analysis/dataflow/engine/dataflow_projection_materialization.py`
- `src/gabion/analysis/dataflow/io/dataflow_graph_rendering.py`

**Workstream/Context:** `PSN-TP-006` snapshot-path owner collapse after the
indexed-scan and deadline project-root publication slices.

## FN-023: Single-caller owners can still preserve fake optionality long after their origin has converged

**Trigger:** Resuming PSN after the snapshot-path owner collapse and scanning for
the next remaining `project_root` ambiguity.

**Friction:** `schema_audit.find_anonymous_schema_surfaces(...)` still advertised
`project_root: object = None` even though its only live product-code caller in
`dataflow_reporting.py` already published a strict `Path`. The owner looked
general-purpose, but the repo had already converged on one real ingress shape.

**Impact:** Humans and LLMs can mistake leftover owner optionality for evidence
of a broader compatibility requirement and spend time preserving it. That keeps
fake alternation alive in the owner even after the real upstream source has
already been narrowed.

**Hypothesis:** Once a function is only called from one strict carrier path,
optional parameters in the owner are usually historical residue rather than live
contract. Treating that residue as a compatibility obligation recreates the same
ambiguity debt that earlier origin-level strictification already paid down.

**Evidence:**
- `src/gabion/analysis/semantics/schema_audit.py`
- `src/gabion/analysis/dataflow/io/dataflow_reporting.py`
- `tests/gabion/analysis/misc_s1/test_schema_audit.py`

**Workstream/Context:** `PSN` follow-on root-origin strictification after
`GH-214 Collapse duplicate snapshot-path owner surface`.

## FN-024: After strictification, policy may require boundary disclosure before it accepts the slice as lawful

**Trigger:** Tightening `schema_audit` so `project_root` became strict at the
owner after confirming that its real reporting caller already published `Path`.

**Friction:** The first ambiguity-gate failure was no longer about loose root
shape. It was about the owner still looking like an ordinary core helper while
performing a real scan/materialization role. Once the data shape was made
truthful, the next policy demand was to disclose that boundary explicitly.

**Impact:** Without reading the gate carefully, an LLM can misdiagnose the red
result as evidence that the strictification was wrong and start reintroducing
optionality. That backtracks the real progress instead of satisfying the newly
visible requirement.

**Hypothesis:** In this repo, some owners only become legible as boundaries
after fake compatibility has been removed. Strictifying the contract first can
surface a second, legitimate obligation: declare the owner as the real carrier
or materialization seam.

**Evidence:**
- `src/gabion/analysis/semantics/schema_audit.py`
- the first post-strictification `gabion policy check --ambiguity-contract`
  failure for `schema_audit`

**Workstream/Context:** `PSN` schema-audit root-origin slice immediately after
the snapshot-path owner-collapse tranche.

## FN-025: Test scaffolding can be the last place a dead optional contract survives

**Trigger:** Tracing the next remaining `project_root` ambiguity in
`test_evidence_suggestions.py` after the product-code caller chain was already
strict.

**Friction:** The live code path had already converged on a strict root, but an
edge-case test still called `_build_test_index(..., None)`. That makes the owner
look like it still needs optional-root support even though the only remaining
source of that shape is the test harness itself.

**Impact:** Humans and LLMs can overestimate how much compatibility still exists
in product code and keep dead optionality alive to satisfy tests that are
actually asserting obsolete behavior.

**Hypothesis:** Once the real caller chain converges, low-level tests often
become the last refuge of a retired contract shape. If they are not recut
promptly, they distort the apparent API and slow downstream strictification.

**Evidence:**
- `src/gabion/analysis/surfaces/test_evidence_suggestions.py`
- `tests/gabion/analysis/evidence/evidence_suggestions_edges_cases.py`

**Workstream/Context:** `PSN` follow-on strictification after the
`schema_audit` root-owner slice.

## FN-026: Entry-point defaults can fossilize as fake owner optionality

**Trigger:** Tracing the next remaining root fallback in
`impact_index.build_impact_index(...)` after the product-code caller chain was
already passing a strict `Path`.

**Friction:** The semantic owner still advertised `repo_root`, `root`, and
`Path.cwd()` fallback even though the only live product-code caller already
published `root: Path`. The apparent dual-shape contract survived only because
the module `__main__` entrypoint had left its cwd default parked inside the
owner.

**Impact:** Humans and LLMs can misread owner-local defaulting as evidence of a
real multi-origin contract and spend time preserving compatibility that no
longer exists in the owned code path.

**Hypothesis:** When the last live fallback source is an executable entrypoint,
the correct repair is usually to move that default all the way out to the
entrypoint boundary and keep the semantic owner strict. Leaving the default in
the owner hides the true origin and keeps dead optionality circulating.

**Evidence:**
- `src/gabion/analysis/semantics/impact_index.py`
- `src/gabion_governance/governance_audit_impl.py`
- `tests/gabion/tooling/impact/test_impact_index.py`

**Workstream/Context:** `PSN` follow-on root-origin strictification after the
`test_evidence_suggestions` slice.

## FN-027: Artifact-carried metadata can linger as a fake semantic input contract

**Trigger:** Tracing the next root seam in `compute_structure_reuse(...)` after
the structure-reuse callers had already converged on an explicit
`project_root: Path`.

**Friction:** The structure-reuse core was still reaching back into the raw
snapshot JSON for `"root"` even though the live caller chain could publish
`project_root` directly. The artifact field had outlived its role as emitted
metadata and was still being treated as if it were the authoritative internal
carrier.

**Impact:** Humans and LLMs can mistake artifact payload metadata for a live
semantic-core dependency and keep parsing it deep in the core. That preserves
fake optionality and makes boundary cleanup look bigger than it really is.

**Hypothesis:** Once an emitted artifact field stops being the first place a
decision is introduced, it should stop being consumed as an internal carrier.
Otherwise the artifact shape fossilizes into a pseudo-ingress even after the
real caller chain has already converged.

**Evidence:**
- `src/gabion/analysis/dataflow/io/dataflow_structure_reuse.py`
- `src/gabion/server.py`
- `tests/gabion/analysis/dataflow_s1/dataflow_structure_reuse_edges_cases.py`
- `tests/gabion/cli/cli_server_parity_cases.py`

**Workstream/Context:** `PSN-TP-006` follow-on strictification after the
`impact_index` root-boundary slice.

## FN-028: Strict root repairs keep flushing out preview-path omissions

**Trigger:** Burning down the optional-root problem in
`analysis.foundation.timeout_context` by replacing `project_root=None` with an
explicit scoped-vs-unscoped carrier.

**Friction:** Once the main timeout APIs were made strict, the next failures did
not come from the obvious deadline checks. They came from side paths like
incremental report previews and timeout-context tests that were still relying on
omission to mean "unscoped". Those paths had already converged semantically,
but they had not been forced to publish that intent.

**Impact:** Humans and LLMs can think a strictification slice is complete after
the primary execution path is clean, then lose time when preview/report/test
surfaces reintroduce the retired shape. This makes the repo look more
multi-shape than it really is and encourages helper-local fallbacks.

**Hypothesis:** In this repo, once a root carrier becomes strict, the remaining
ambiguity often hides in secondary surfaces that were never forced to declare
scope explicitly. They are not separate domains; they are stale side channels
for the same retired omission contract.

**Evidence:**
- `src/gabion/analysis/foundation/timeout_context.py`
- `src/gabion/server.py`
- `src/gabion/server_core/command_orchestrator.py`
- `tests/gabion/analysis/timeout_deadline/test_timeout_context.py`

**Workstream/Context:** `PSN-TP-006` follow-on strictification while burning
down the `timeout_context.py` optional-root seam.
