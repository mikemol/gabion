---
doc_revision: 5
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
