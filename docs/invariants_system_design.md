---
doc_revision: 2
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: invariants_system_design
doc_role: design
doc_scope:
  - repo
  - semantics
  - governance
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - AGENTS.md#agent_obligations
  - CONTRIBUTING.md#contributing_contract
  - docs/normative_clause_index.md#normative_clause_index
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 52
  glossary.md#contract: 43
  AGENTS.md#agent_obligations: 30
  CONTRIBUTING.md#contributing_contract: 113
  docs/normative_clause_index.md#normative_clause_index: 10
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev52 (forward-remediation order, ci_watch failure-bundle durability, and enforced execution-coverage policy wording)."
  glossary.md#contract: "Reviewed glossary.md rev43 (glossary contract + semantic typing discipline)."
  AGENTS.md#agent_obligations: "Reviewed AGENTS.md rev30 (required validation stack, forward-remediation preference, and ci_watch failure-bundle triage guidance)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev113 (two-stage dual-sensor cadence, correction-unit validation stack, and strict-coverage trigger guidance)."
  docs/normative_clause_index.md#normative_clause_index: "Reviewed normative_clause_index rev10 (extended existing dual-sensor/shift-ambiguity/deadline clauses without introducing new clause IDs)."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="invariants_system_design"></a>
# Invariants System Design

This document describes the current invariants subsystem centered on:

- `src/gabion/invariants.py`
- `src/gabion/analysis/foundation/marker_protocol.py`
- `src/gabion/runtime/policy_runtime.py`

The model is **factory-first**: `invariant_factory()` is the execution core, while `never()`, `todo()`, and `deprecated()` are thin marker entrypoints.

## 1. System overview

### 1.1 Core contracts

- **Structured reasoning contract**: `summary`, `control`, `blocking_dependencies`.
- **Marker payload contract**: normalized marker kind + lifecycle metadata + structured reasoning.
- **Runtime behavior contract** (profile-driven):
  - `throws`
  - `emits_warning`
  - `warning_limit`

### 1.2 Execution path

1. Caller invokes `never()/todo()/deprecated()`.
2. Marker helper delegates to `invariant_factory(marker_kind, reasoning, **env)`.
3. Factory normalizes reasoning and payload.
4. Factory resolves behavior via active profile.
5. Factory applies warning behavior and throw behavior.

### 1.3 Governance layering

- **Marker kind mapping layer** (`marker_protocol`): maps semantic kind by governance profile.
- **Runtime behavior layer** (`invariants` + `policy_runtime`): decides throw/warn/no-op behavior.
- **Policy integration layer** (`RuntimePolicyConfig`): injects profile behavior at runtime scope boundaries.

---

## 2. Notions and orthogonal implication lattice

The requested structure is represented as 3 notions, each with:

- 3 first-order implications (mutually orthogonal)
- each first-order implication has 3 second-order implications
- each second-order implication has 3 third-order homotopies (orthogonal siblings)

### Notation

- `N*` = notion
- `F*` = first-order implication
- `S*` = second-order implication
- `H*` = third-order homotopy implication

---

## N1 - Runtime Semantics as Data (not hard-coded behavior)

### N1.F1 - Throw policy is profile data
- N1.F1.S1 - Throw decision is looked up by marker kind
  - N1.F1.S1.H1 - profile can make `never` non-throwing (diagnostic mode)
  - N1.F1.S1.H2 - profile can make `todo` throwing (strict debt gate)
  - N1.F1.S1.H3 - profile can make `deprecated` throwing (sunset enforcement)
- N1.F1.S2 - Throw decision is independent of marker-kind remapping
  - N1.F1.S2.H1 - kind remaps for analysis without changing runtime throw
  - N1.F1.S2.H2 - runtime throw changes without altering marker identity scheme
  - N1.F1.S2.H3 - throw contract can evolve while payload schema remains stable
- N1.F1.S3 - Throw decision is testable as profile fixture
  - N1.F1.S3.H1 - table-driven tests for marker x profile x throws
  - N1.F1.S3.H2 - scope-based tests for temporary behavior overrides
  - N1.F1.S3.H3 - regression tests for default behavior invariants

### N1.F2 - Emission policy is profile data
- N1.F2.S1 - emission channel is explicit (warning/no warning)
  - N1.F2.S1.H1 - `deprecated` warning path enabled by profile
  - N1.F2.S1.H2 - emission disabled for silent collection mode
  - N1.F2.S1.H3 - future extension to alternate channel (telemetry/event bus)
- N1.F2.S2 - emission is independent of throw decision
  - N1.F2.S2.H1 - warn + throw profile
  - N1.F2.S2.H2 - warn + no-throw profile
  - N1.F2.S2.H3 - no-warn + throw profile
- N1.F2.S3 - emission category stability is explicit
  - N1.F2.S3.H1 - warning class remains API surface
  - N1.F2.S3.H2 - warning content uses structured fields
  - N1.F2.S3.H3 - warning generation is deterministic under same reasoning

### N1.F3 - Rate policy is profile data
- N1.F3.S1 - warning cardinality bound is configurable
  - N1.F3.S1.H1 - low limit for CI noise control
  - N1.F3.S1.H2 - high limit for forensic runs
  - N1.F3.S1.H3 - zero limit to disable warning emission
- N1.F3.S2 - rate key derives from structured reasoning
  - N1.F3.S2.H1 - summary contributes to identity
  - N1.F3.S2.H2 - control contributes to identity
  - N1.F3.S2.H3 - blocking dependencies contribute to identity
- N1.F3.S3 - rate behavior is monotone under cache growth
  - N1.F3.S3.H1 - first-seen keys emit until cap
  - N1.F3.S3.H2 - repeated keys suppress duplicates
  - N1.F3.S3.H3 - cap saturation suppresses new keys

---

## N2 - Structured Reasoning as Semantic Carrier

### N2.F1 - Reasoning is fielded, not parsed text
- N2.F1.S1 - `summary` expresses the local semantic claim
  - N2.F1.S1.H1 - human diagnosis anchor
  - N2.F1.S1.H2 - report rendering anchor
  - N2.F1.S1.H3 - marker identity ingredient
- N2.F1.S2 - `control` identifies policy/control surface
  - N2.F1.S2.H1 - local guard or feature gate id
  - N2.F1.S2.H2 - protocol/decision surface id
  - N2.F1.S2.H3 - ownership seam identifier
- N2.F1.S3 - `blocking_dependencies` captures resolution blockers
  - N2.F1.S3.H1 - dependency ids are normalized list items
  - N2.F1.S3.H2 - ordering noise removed by normalization
  - N2.F1.S3.H3 - missing dependency set is explicit empty tuple

### N2.F2 - Reasoning survives all marker paths
- N2.F2.S1 - payload carries reasoning regardless of marker helper
  - N2.F2.S1.H1 - `never` payload includes reasoning
  - N2.F2.S1.H2 - `todo` payload includes reasoning
  - N2.F2.S1.H3 - `deprecated` payload includes reasoning
- N2.F2.S2 - reasoning survives governance kind remap
  - N2.F2.S2.H1 - remapped kind keeps summary
  - N2.F2.S2.H2 - remapped kind keeps control
  - N2.F2.S2.H3 - remapped kind keeps dependency set
- N2.F2.S3 - reasoning contributes to stable marker identity
  - N2.F2.S3.H1 - identity changes when summary changes
  - N2.F2.S3.H2 - identity changes when control changes
  - N2.F2.S3.H3 - identity changes when dependency set changes

### N2.F3 - Reasoning is normalization-first
- N2.F3.S1 - input shape coercion is boundary-level
  - N2.F3.S1.H1 - dataclass input path
  - N2.F3.S1.H2 - mapping input path
  - N2.F3.S1.H3 - scalar fallback path
- N2.F3.S2 - normalization is deterministic
  - N2.F3.S2.H1 - trimming applied to text fields
  - N2.F3.S2.H2 - dependency set deduplicated
  - N2.F3.S2.H3 - dependency order canonicalized
- N2.F3.S3 - empty summary has explicit fallback
  - N2.F3.S3.H1 - fallback avoids empty semantic payloads
  - N2.F3.S3.H2 - fallback remains profile-independent
  - N2.F3.S3.H3 - fallback remains analyzable by reporting paths

---

## N3 - Governance/Profile Integration Across Runtime and Analysis

### N3.F1 - Profile scoping is explicit runtime context
- N3.F1.S1 - `RuntimePolicyConfig` carries marker profile
  - N3.F1.S1.H1 - default profile is explicit
  - N3.F1.S1.H2 - scope override is composable with other runtime knobs
  - N3.F1.S1.H3 - apply-runtime and scope-runtime stay aligned
- N3.F1.S2 - governance config is contextvar-based
  - N3.F1.S2.H1 - thread/task-safe dynamic scoping
  - N3.F1.S2.H2 - nested scope restoration via token reset
  - N3.F1.S2.H3 - deterministic lookup in factory hot path
- N3.F1.S3 - profile behavior and profile mapping are separated
  - N3.F1.S3.H1 - runtime behavior profile config in `invariants`
  - N3.F1.S3.H2 - marker kind mapping profile config in `marker_protocol`
  - N3.F1.S3.H3 - both are composed, not conflated

### N3.F2 - Analysis sees profile-adjusted marker semantics
- N3.F2.S1 - marker alias extraction remains syntax-level
  - N3.F2.S1.H1 - alias list is canonicalized
  - N3.F2.S1.H2 - default aliases include fully-qualified helpers
  - N3.F2.S1.H3 - custom aliases can be layered
- N3.F2.S2 - extracted kind is profile-resolved before emission
  - N3.F2.S2.H1 - profile can collapse todo/deprecated into never
  - N3.F2.S2.H2 - profile can preserve native kind distinctions
  - N3.F2.S2.H3 - profile changes report semantics without AST shape changes
- N3.F2.S3 - invariants evidence remains structurally stable
  - N3.F2.S3.H1 - site identifiers remain path/function/span based
  - N3.F2.S3.H2 - marker metadata retains owner/expiry/links
  - N3.F2.S3.H3 - forest sink topology remains compatible

### N3.F3 - Factory-centric architecture preserves extension seams
- N3.F3.S1 - helper functions are API aliases over one core engine
  - N3.F3.S1.H1 - no duplicated throw/warn logic across helpers
  - N3.F3.S1.H2 - behavior changes localize to one factory path
  - N3.F3.S1.H3 - test matrix centralizes around factory semantics
- N3.F3.S2 - new marker kinds can reuse existing behavior contracts
  - N3.F3.S2.H1 - add mapping entry for kind semantics
  - N3.F3.S2.H2 - add profile behavior row for throws/emits/rate
  - N3.F3.S2.H3 - keep payload schema stable while adding behavior modes
- N3.F3.S3 - policy evolution remains forward-remediation friendly
  - N3.F3.S3.H1 - tighten profile without API break in callsites
  - N3.F3.S3.H2 - relax profile for migration windows
  - N3.F3.S3.H3 - stage rollout by runtime scope boundaries

---

## 3. Merge-conflict handling guidance

Because marker behavior now has a shared factory and profile data, merge conflict
resolution should preserve the following invariants:

1. `never/todo/deprecated` must delegate to `invariant_factory`.
2. Structured reasoning fields must remain present in payload/identity surfaces.
3. Runtime behavior knobs (`throws`, `emits_warning`, `warning_limit`) must remain profile-configurable through runtime policy wiring.
4. Profile-driven marker-kind mapping in analysis should remain independent from runtime throw/warn behavior.

## 4. Non-goals (current design)

- Do not introduce string-matching classifiers over free-form reason text.
- Do not duplicate behavior logic in individual marker helpers.
- Do not hard-code warning rates per helper when profile data is available.
