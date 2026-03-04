---
doc_revision: 4
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: identity_debt_rebase_map
doc_role: architecture
doc_scope:
  - repo
  - analysis
  - identity
doc_authority: informative
doc_requires:
  - docs/identity_substrate_bridge_matrix.md
  - src/gabion/server_core/command_orchestrator.py
  - src/gabion/analysis/foundation/identity_shadow_runtime.py
  - src/gabion/analysis/dataflow/engine/dataflow_event_algebra_adapter.py
  - src/gabion/analysis/aspf/aspf_event_algebra_adapter.py
doc_reviewed_as_of:
doc_review_notes:
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

# Identity Debt Rebase Map

## Phase 2A/2B objective

Rebase identity substrate debt by making canonical progress envelopes primary on
`progress-v2` while retaining `progress-v1` as a temporary compatibility
adapter, and by mirroring fingerprint-lane prime assignments into the shared
identity-space allocation ledger.

## Reuse-now surfaces (keep)

| Surface | Status | Debt tag | Why reused |
| --- | --- | --- | --- |
| `PrimeRegistry` + `PrimeIdentityAdapter` | keep | `reuse` | already canonical unique-structure allocator with seed replay |
| `GlobalIdentitySpace` | keep | `reuse` | already provides interned atom/path + projection witness |
| `CanonicalRunContext` + event algebra envelope | keep | `reuse` | provides typed sequence/run boundary contract |
| Dataflow event algebra adapters | keep (extended) | `reuse` | already normalize collection/phase progress into canonical envelopes |
| ASPF event adapter stack | keep | `reuse` | already converges on typed replay envelopes and sink/visitor model |

## New bridge seams (Phase 2A)

| Seam | Status | Debt tag | Lifecycle |
| --- | --- | --- | --- |
| `IdentityShadowRuntime` | added | `keep` | active shadow lane for dataflow progress sidecars |
| `IntegerCarrierProtocol` + `BitPrimeIntegerCarrier` | active | `keep` | bit-lowered integer anchor tokenization for high-cardinality anchors |
| `FastIntegerCarrier` | retained | `sunset` | explicit compatibility implementation pending carrier simplification gate |
| `IdentityRegistryMirror` | added | `keep` | active Phase 2B write-through mirror (`PrimeRegistry` -> `GlobalIdentitySpace`) for `type_base`/`type_ctor`/`synth` |
| Canonical primary progress (`gabion.dataflowAudit/progress-v2`) | added, default-on | `keep` | authoritative progress carrier (`valid`/`rejected` adaptation kinds) |
| Legacy compatibility progress (`gabion.dataflowAudit/progress-v1`) | retained | `boundary-only` | temporary boundary adapter during dual-publish migration |
| Progress sidecars (`canonical_event_v1`, `identity_allocation_delta_v1`, `canonical_event_error_v1`) | retained | `boundary-only` | additive compatibility lane while v1 exists; `identity_allocation_delta_v1` now includes mirrored fingerprint namespaces |
| Command response sidecar (`identity_seed_v1`) | added | `keep` | replay/debug seed output |

## Adapter inventory (retirement tags)

| Adapter | Lane | Debt tag | Decision | Action |
| --- | --- | --- | --- | --- |
| `progress-v2` canonical primary emitter/parser | dataflow progress | `keep` | canonical transport boundary | retain as authoritative carrier |
| `progress-v1` dual-publish compatibility adapter | dataflow progress | `boundary-only` | temporary transport boundary adapter | remove at repo-consumer-zero gate |
| transcript fixture event adapter | transcript fixture-contract lane | `boundary-only` | fixture-only bridge; not production lane | retain fixture-only unless transcript runtime lane is explicitly accepted |
| `FastIntegerCarrier` | dataflow identity tokenization | `sunset` | temporary compatibility carrier | remove after carrier simplification gate |
| legacy lane-local seed export helpers | cross-lane response shaping | `sunset` | duplicated seed handoff paths | collapse into one `identity_seed_v1` path |
| `PrimeRegistry`/`GlobalIdentitySpace` allocator core | identity substrate | `reuse` | canonical allocator and projection substrate | keep and extend via boundary adapters only |

## Rebase candidates (next phases)

| Debt area | Current fragmentation | Rebase action |
| --- | --- | --- |
| integer anchors | ad-hoc text tokens for high-cardinality ints | replace `FastIntegerCarrier` with lowered bit-prime carrier |
| lane-specific envelope emission | separate emitters in dataflow/ASPF/transcript experiments | route through shared envelope + sink adapters on `progress-v2` first |
| seed handoff points | registry seed emitted inconsistently by lane | standardize via `identity_seed_v1` and lane finalizers |
| interning registries | separate name/path/prime intern domains at runtime boundaries | consolidate namespace governance and import/export adapters |

## Temporary compatibility adapter lifecycle metadata

| field | value |
| --- | --- |
| actor | `gabion-maintainers` |
| rationale | Preserve existing in-repo progress consumers while v2 canonical primary is adopted. |
| scope | `$/progress` token `gabion.dataflowAudit/progress-v1` in dataflow lane only |
| start | `2026-03-04` |
| expiry | `repo-consumer-zero gate reached` |
| rollback_condition | v2 canonical emission causes operational regressions in progress consumers; restore v1-only emission until corrected |
| evidence_links | `tests/gabion/server/server_execute_command_edges_cases.py`, `tests/gabion/server_core/command_orchestrator_coverage_cases.py`, `tests/gabion/commands/test_progress_contract_edges.py` |

### Transcript fixture boundary metadata

| field | value |
| --- | --- |
| actor | `gabion-maintainers` |
| rationale | Preserve transcript substrate comparability without introducing a production transcript runtime lane. |
| scope | transcript fixture adapters only (`in/transcript.md` contract references and fixture tests) |
| start | `2026-03-04` |
| expiry | `explicit transcript runtime lane acceptance` |
| rollback_condition | fixture adapter assumptions diverge from canonical envelope contract and fail bridge parity checks |
| evidence_links | `docs/identity_substrate_bridge_matrix.md`, `in/transcript.md` |

## Deferred deletion list (post-cutover)

| Candidate | Delete condition |
| --- | --- |
| ad-hoc progress-only identity token branches (`sunset`) | once all progress producers use shared integer-carrier + canonical envelope adapter |
| bespoke lane-local allocation delta bookkeeping (`sunset`) | once all lanes emit `identity_allocation_delta_v1` from shared shadow runtime |
| duplicated seed export helpers (`sunset`) | once all command responses use one `identity_seed_v1` path |

## Guardrails

1. Keep semantic outputs unchanged; sidecars remain additive.
2. Reject semantic-core compatibility wrappers; boundary adapters only with explicit lifecycle metadata.
3. Preserve deterministic ordering/canonicalization before transport emission.
