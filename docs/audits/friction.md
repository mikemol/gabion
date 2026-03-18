---
doc_revision: 2
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
