---
doc_revision: 2
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: deprecated_substrate_strictification_ledger
doc_role: audit
doc_scope:
  - repo
  - analysis
  - policy
  - tests
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - AGENTS.md#agent_obligations
  - glossary.md#contract
  - docs/normative_clause_index.md#normative_clause_index
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 55
  AGENTS.md#agent_obligations: 33
  glossary.md#contract: 44
  docs/normative_clause_index.md#normative_clause_index: 14
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev55; fix-forward correction units and impossible-by-construction post-ingress rules anchor this ledger."
  AGENTS.md#agent_obligations: "Reviewed AGENTS.md rev33; required validation stack and never()-after-ingress obligations remain aligned with this debt surface."
  glossary.md#contract: "Reviewed glossary.md rev44; bundle/protocol/decision-surface language matches the strictification backlog tracked here."
  docs/normative_clause_index.md#normative_clause_index: "Reviewed clause index rev14; runtime-narrowing and fiber-trace-boundary clauses define the allowed boundary move for this refactor."
doc_sections:
  deprecated_substrate_strictification_ledger: 1
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="deprecated_substrate_strictification_ledger"></a>
# Deprecated Substrate Strictification Ledger

This ledger tracks fix-forward debt in
`src/gabion/analysis/core/deprecated_substrate.py`. It exists because a local
strictification attempt proved that replacing runtime exceptions with
`never(...)` is not a local edit: the refactor signal propagates across the same
fiber into payload ingress, constructor protocols, perf-sample parsing, and the
deprecated non-erasability policy wrapper.

This is a backlog surface, not a rollback trigger.

## Status Key

- `open`: not started.
- `in_progress`: active correction unit.
- `mitigated`: partial boundary move landed, but semantic core still carries reachable narrowing.
- `closed`: ingress normalization is upstream of semantic core, `never(...)` only appears on dead post-invariant paths, and the ambiguity/policy gates pass.

## Boundary Rule

The governing move is not "swap `ValueError` for `never(...)`". The governing
move is:

1. normalize raw payload alternation at a named ingress boundary,
2. reify the allowed constructor states as an explicit protocol,
3. make semantic-core carriers impossible-by-construction, and only then
4. discharge dead fallback paths with `never(...)`.

The relevant clauses are:

- `NCI-RUNTIME-NARROWING-BOUNDARY`: runtime narrowing belongs at ingress, not in semantic core.
- `NCI-FIBER-TRACE-BOUNDARY`: a valid strictification must move the boundary upstream on the same fiber; lateral relocation is concealment.

## Breadcrumb Contract

The temporary proof surface for this debt should not be string-only comments or
freehand TODO text. When a work-in-progress site must remain marked in code, use
structured invariant markers:

- `never(...)` only for dead post-invariant paths,
- `deprecated(...)` for blocked legacy semantic surfaces, and
- `todo(...)` for explicit staged construction that has not landed yet.

For this workstream, those markers should carry structured reasoning with:

- `reasoning.summary`: the local blocked obligation,
- `reasoning.control`: the governing control/boundary move,
- `reasoning.blocking_dependencies`: one or more `DSD-*` row IDs from this ledger,
- `links`: at minimum `{"kind": "doc_id", "value": "deprecated_substrate_strictification_ledger"}` and, where relevant, governing `policy_id` links such as `NCI-RUNTIME-NARROWING-BOUNDARY` or `NCI-FIBER-TRACE-BOUNDARY`.

Those markers are breadcrumbs, not substitutes for the boundary move. Their job
is to preserve structural traceability through the code-flow labyrinth while the
upstream normalization/protocol refactor is still open.

## Propagation Scope

| surface | current role | why the strictification propagates |
| --- | --- | --- |
| `src/gabion/analysis/core/deprecated_substrate.py` | semantic-core carrier construction and extraction helpers | `DeprecatedFiber.from_payload`, `deprecated(...)`, and `ingest_perf_samples(...)` still normalize raw shapes inside the core |
| `scripts/policy/deprecated_nonerasability_policy_check.py` | child policy wrapper over deprecated-fiber payloads | wrapper loads raw JSON and delegates shape alternation into `DeprecatedFiber.from_payload(...)` |
| `tests/gabion/analysis/misc_s2/test_deprecated_substrate.py` | contract coverage for permissive payload edges | current tests prove reachable fallback/default behavior that must move to ingress before `never(...)` becomes lawful |
| `tests/gabion/tooling/runtime_policy/test_deprecated_nonerasability_policy_check.py` | wrapper/policy payload coverage | current tests reflect raw-payload ingestion semantics rather than canonical carrier-only inputs |

## Active Debt Rows

| debt_id | surface | signal_source | blocking? | target_cu | status | impact | evidence_links | owner | expiry | fix_forward_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DSD-001` | `DeprecatedFiber.from_payload(...)` performs payload alternation and normalization inside semantic core | local strictification attempt: `never(...)` on resolved-without-metadata path caused grade-monotonicity fallout instead of a lawful dead-path discharge | yes | `DS-CU-01` | open | semantic core still accepts raw mapping shape decisions that belong at ingress | `src/gabion/analysis/core/deprecated_substrate.py`; `tests/gabion/analysis/misc_s2/test_deprecated_substrate.py` | codex | 2026-03-18 | introduce a named ingress DTO/decision protocol for deprecated-fiber payloads, keep `DeprecatedFiber` construction on canonical inputs only, and use structured marker breadcrumbs with `blocking_dependencies=("DSD-001", ...)` until the boundary move lands |
| `DSD-002` | `deprecated(...)` still encodes resolved-vs-blocked state via runtime checks over `object`-typed inputs | local strictification attempt around `resolution_metadata` showed the constructor protocol is underspecified rather than merely under-guarded | yes | `DS-CU-01` | open | `never(...)` cannot lawfully replace runtime exceptions until resolved and unresolved fiber states are separate explicit constructor cases | `src/gabion/analysis/core/deprecated_substrate.py`; `docs/normative_clause_index.md#clause-fiber-trace-boundary` | codex | 2026-03-18 | split deprecated-fiber construction into an explicit decision protocol or dedicated constructor variants for active/blocked/resolved cases, and use structured `deprecated(...)`/`todo(...)` breadcrumbs to identify the still-blocked constructor cases |
| `DSD-003` | `ingest_perf_samples(...)` still accepts malformed sample payloads and repairs them in semantic core | local strictification attempt on weight/default branch showed the wildcard/default path is reachable under current ingress semantics | yes | `DS-CU-02` | open | perf-sample normalization remains a core concern, so `never(...)` would be lying about deadness | `src/gabion/analysis/core/deprecated_substrate.py`; `tests/gabion/analysis/misc_s2/test_deprecated_substrate.py` | codex | 2026-03-19 | add a perf-sample ingress parser/DTO that classifies invalid rows before semantic-core construction, leave only dead post-invariant fallthroughs, and annotate any transitional blocked sites with structured marker reasoning tied to `DSD-003` |
| `DSD-004` | deprecated non-erasability child policy check still ingests raw JSON into `DeprecatedFiber.from_payload(...)` | propagation map from the failed strictification: policy wrapper remains on the same fiber and would reintroduce raw-shape alternation even after partial core cleanup | yes | `DS-CU-03` | open | impossible-by-construction cannot hold end-to-end while the wrapper owns raw payload alternation | `scripts/policy/deprecated_nonerasability_policy_check.py`; `tests/gabion/tooling/runtime_policy/test_deprecated_nonerasability_policy_check.py`; `src/gabion/analysis/core/deprecated_substrate.py` | codex | 2026-03-20 | move deprecated-fiber payload normalization into a named boundary loader, make the policy wrapper consume canonical carriers only, and keep any temporary wrapper markers structurally linked back to `DSD-004` |

## Closure Conditions

Do not close this ledger until all of the following are true:

1. raw deprecated-fiber payload alternation is normalized at a named ingress boundary,
2. `DeprecatedFiber` construction no longer accepts semantically underdetermined `object` inputs,
3. `ingest_perf_samples(...)` receives canonical sample inputs rather than repairing malformed payloads in semantic core,
4. `scripts/policy/deprecated_nonerasability_policy_check.py` consumes canonical deprecated-fiber carriers rather than raw JSON row shapes, and
5. any remaining `never(...)` sites in this workstream are dead post-invariant paths that pass ambiguity, policy, and targeted-test gates, and
6. any temporary `never(...)`, `deprecated(...)`, or `todo(...)` breadcrumbs in this workstream use structured reasoning and `blocking_dependencies` tied back to the relevant `DSD-*` rows rather than string-only reasons.
