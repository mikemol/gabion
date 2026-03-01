---
doc_revision: 5
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: progress_transition_contract
doc_role: contract
doc_scope:
  - repo
  - progress
  - telemetry
doc_authority: normative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/normative_clause_index.md#normative_clause_index
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  docs/normative_clause_index.md#normative_clause_index: 2
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Progress transition contract remains impossible-by-construction and never()-enforced at invariant boundaries."
  glossary.md#contract: "Parent/child semantics now recurse as a self-similar tree contract."
  docs/normative_clause_index.md#normative_clause_index: "No new clause IDs introduced; this contract reifies existing ambiguity-left and semantic-core obligations."
---

<a id="progress_transition_contract"></a>
# Progress Transition Contract (Normative)

This contract defines recursive progress transition semantics for progress
notifications.

`progress_transition_v2` is authoritative when present. `progress_transition_v1`
is a compatibility projection from `v2` at transport/display boundaries.

## Recursive model

- Progress state is a tree of `ProgressNode`.
- Every node has the same shape: `identity`, `unit`, `done`, `total`,
  `marker_text`, `marker_family`, `marker_step`, `children`.
- `active_path` identifies the active node from root to frontier.
- Parent/child semantics are recursive: any node may be treated as a local
  parent with local children.

## Parent stage identity and step index semantics

- Root parent identity is represented by:
  - envelope identity: `(phase, analysis_state)`
  - root node identity: `root.identity`
- Parent step index is root progress tuple:
  - `(root.unit, root.done, root.total)`
- For post phase, `root.unit` is expected to be `post_tasks`.

## Child stage identity and step semantics

- Active child identity is the node at `active_path`.
- Child step boundary uses active marker decomposition:
  - `marker_family` and `marker_step` from marker text
  - example `fingerprint:normalize` -> family=`fingerprint`, step=`normalize`
  - boundary marker uses `*:done` (for example `fingerprint:done`)
- `complete` is terminal marker text at the active frontier.

## Transition classes

- `hold`: parent index unchanged.
- `advance`: parent index changes and active marker lands on `*:done`.
- `terminal`: enters post terminal boundary (`complete` with `done == total > 0`).
- `terminal_keepalive`: unchanged terminal state re-emitted only as heartbeat.

## Forbidden transitions

1. Invalid active path.
2. Duplicate sibling identities under a single parent.
3. Impossible node status transition (regressed `done` or `total`).
4. Parent identity drift without parent index change.
5. Parent index change without active child transition.
6. Parent index change without `*:done` child boundary.
7. `complete` before parent completion boundary.
8. Terminal replay mutation (unchanged terminal replay that mutates tree/path).
9. Terminal keepalive without terminal state.

## Terminal replay and heartbeat rules

- First terminal entry may normalize `event_kind=progress` to `terminal`.
- Unchanged terminal replay as active `progress`/`terminal` is suppressed.
- Unchanged terminal replay as `heartbeat` is valid `terminal_keepalive`.
- Terminal keepalive must preserve tree state and active path exactly.

## Concrete examples

### Valid

- `analysis_post_in_progress | post_tasks=5/6 | fingerprint:normalize`
  ->
  `analysis_post_in_progress | post_tasks=5/6 | fingerprint:warnings`
  (`hold`)
- `analysis_post_in_progress | post_tasks=5/6 | fingerprint:rewrite_plans`
  ->
  `analysis_post_in_progress | post_tasks=6/6 | fingerprint:done`
  (`advance`)
- `analysis_post_in_progress | post_tasks=6/6 | complete`
  ->
  unchanged tree with `event_kind=heartbeat`
  (`terminal_keepalive`)

### Invalid

- `analysis_post_in_progress | post_tasks=5/6 | fingerprint:normalize`
  ->
  `analysis_post_in_progress | post_tasks=6/6 | fingerprint:normalize`
  (parent advanced without `*:done`)
- `analysis_post_in_progress | post_tasks=5/6 | complete`
  (complete before parent completion boundary)
- `analysis_post_in_progress | post_tasks=6/6 | complete`
  ->
  unchanged state with `event_kind=progress`
  (suppressed terminal replay)

## Timeline row structure

Timeline row shaping is formal and transition-derived, not marker-heuristic.

- `progress_marker`: active frontier marker text.
- `primary`: root parent progress summary (`root.done/root.total root.unit`).
- `progress_path`: `active_path` joined by ` > `.
- `active_primary`: active frontier node progress summary.
- `active_depth`: `len(active_path) - 1`.
- `transition_reason`: validator/emitter reason code (for example
  `parent_held`, `parent_advanced`, `terminal_transition`,
  `terminal_keepalive`).
- `root_identity`: root node identity for parent-stage correlation.
- `active_identity`: active frontier node identity.
- `marker_family` and `marker_step`: active marker decomposition.
- `active_children`: immediate child count at active frontier.

Row columns are emitted in this order:

`ts_utc`, `event_seq`, `event_kind`, `phase`, `analysis_state`,
`classification`, `progress_marker`, `primary`, `files`, `stale_for_s`,
`dimensions`, `progress_path`, `active_primary`, `active_depth`,
`transition_reason`, `root_identity`, `active_identity`, `marker_family`,
`marker_step`, `active_children`.
