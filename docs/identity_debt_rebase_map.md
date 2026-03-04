---
doc_revision: 8
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
  - docs/event_algebra_bridge_matrix.md
  - src/gabion/server_core/command_orchestrator.py
  - src/gabion/server_core/command_orchestrator_primitives.py
  - src/gabion/server.py
  - src/gabion/commands/progress_contract.py
  - src/gabion/analysis/foundation/identity_shadow_runtime.py
  - src/gabion/analysis/foundation/identity_registry_mirror.py
  - src/gabion/analysis/dataflow/engine/dataflow_event_algebra_adapter.py
  - src/gabion/analysis/aspf/aspf_event_algebra_adapter.py
  - tests/gabion/analysis/foundation/test_identity_registry_mirror.py
  - tests/gabion/analysis/foundation/transcript_event_fixtures.py
  - tests/gabion/analysis/foundation/test_transcript_event_adapter.py
  - tests/gabion/server/server_execute_command_edges_cases.py
  - tests/gabion/server_core/command_orchestrator_coverage_cases.py
  - tests/gabion/commands/test_progress_contract_edges.py
  - tests/gabion/cli/cli_helpers_cases.py
  - tests/gabion/tooling/delta/test_delta_script_telemetry.py
  - tests/gabion/tooling/delta/test_delta_emit_runtime.py
  - tests/gabion/lsp_client/lsp_client_direct_cases.py
  - tests/gabion/commands/test_command_boundary_order.py
doc_reviewed_as_of:
doc_review_notes:
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

# Identity Debt Rebase Map

## Phase 2A/2B objective

Rebase identity substrate debt by making canonical progress envelopes primary on
`progress-v2`, retiring `progress-v1` transport from `src/gabion`, and
mirroring fingerprint-lane prime assignments into the shared identity-space
allocation ledger.

## As-Of Snapshot (2026-03-04)

| Surface | Status | Evidence | Drain note |
| --- | --- | --- | --- |
| `IdentityShadowRuntime` | `done` | `src/gabion/analysis/foundation/identity_shadow_runtime.py`; `src/gabion/server_core/command_orchestrator.py` | Active shadow runtime emits canonical adaptation sidecars and seed payload. |
| `BitPrimeIntegerCarrier` default | `done` | `src/gabion/analysis/foundation/identity_shadow_runtime.py` | Integer-anchor encoding defaults to bit-lowered tokenization. |
| `IdentityRegistryMirror` write-through | `done` | `src/gabion/analysis/foundation/identity_registry_mirror.py`; `tests/gabion/analysis/foundation/test_identity_registry_mirror.py` | Fingerprint namespaces hydrate and mirror into identity-space allocation ledger. |
| `progress-v2` canonical emission | `done` | `src/gabion/server_core/command_orchestrator.py`; `src/gabion/commands/progress_contract.py` | Canonical progress payload emitted on `gabion.dataflowAudit/progress-v2`. |
| `identity_seed_v1` unified response sidecar | `done` | `src/gabion/server_core/command_orchestrator.py`; `tests/gabion/server/server_execute_command_edges_cases.py` | Command success and timeout paths return one seed sidecar surface. |
| `progress-v1` compatibility adapter | `done` | `src/gabion/server_core/command_orchestrator.py`; `src/gabion/server_core/command_orchestrator_primitives.py`; `src/gabion/commands/progress_contract.py`; `tests/gabion/commands/test_progress_contract_edges.py`; `tests/gabion/server_core/command_orchestrator_coverage_cases.py`; `tests/gabion/server/server_execute_command_edges_cases.py` | `IDR-001` closed after hard-cut removal from `src/gabion`; gate `G-004` now pass. |
| `FastIntegerCarrier` sunset | `done` | `src/gabion/analysis/foundation/identity_shadow_runtime.py`; `tests/gabion/analysis/foundation/test_identity_shadow_runtime.py` | `IDR-002` closed after runtime/test removal; gate `G-003` now pass (doc scan excludes this map file). |
| transcript fixture boundary governance | `done` | `tests/gabion/analysis/foundation/transcript_event_fixtures.py`; `tests/gabion/analysis/foundation/test_transcript_event_adapter.py`; `docs/event_algebra_bridge_matrix.md`; `in/transcript.md` | `IDR-003` closed with fixture-only governance enforcement; `G-002` and `G-005` pass. |
| interning namespace consolidation | `open` | `src/gabion/analysis/foundation/identity_registry_mirror.py`; `src/gabion/analysis/foundation/identity_shadow_runtime.py`; `docs/identity_substrate_bridge_matrix.md` | Tracked by `IDR-004`; gated by `G-001`. |

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
| `FastIntegerCarrier` | removed | `sunset` | removed in `IDR-002`; gate `G-003` satisfied with tracked-surface scan |
| `IdentityRegistryMirror` | added | `keep` | active Phase 2B write-through mirror (`PrimeRegistry` -> `GlobalIdentitySpace`) for `type_base`/`type_ctor`/`synth` |
| Canonical primary progress (`gabion.dataflowAudit/progress-v2`) | added, default-on | `keep` | authoritative progress carrier (`valid`/`rejected` adaptation kinds) |
| Legacy compatibility progress (`gabion.dataflowAudit/progress-v1`) | removed | `boundary-only` | retired in `IDR-001` stage 2 hard cut; no `progress-v1` transport remains in `src/gabion` |
| Progress sidecars (`canonical_event_v1`, `identity_allocation_delta_v1`, `canonical_event_error_v1`) | retained | `boundary-only` | additive compatibility lane while v1 exists; `identity_allocation_delta_v1` now includes mirrored fingerprint namespaces |
| Command response sidecar (`identity_seed_v1`) | added | `keep` | replay/debug seed output |

## Adapter inventory (retirement tags)

| Adapter | Lane | Debt tag | Decision | Action |
| --- | --- | --- | --- | --- |
| `progress-v2` canonical primary emitter/parser | dataflow progress | `keep` | canonical transport boundary | retain as authoritative carrier |
| `progress-v1` dual-publish compatibility adapter | dataflow progress | `boundary-only` | temporary transport boundary adapter | removed via `IDR-001` stage 2 hard cut; keep only historical debt record |
| transcript fixture event adapter | transcript fixture-contract lane | `boundary-only` | fixture-only bridge; not production lane | retain fixture-only unless transcript runtime lane is explicitly accepted |
| `FastIntegerCarrier` | dataflow identity tokenization | `sunset` | temporary compatibility carrier | removed via `IDR-002`; keep only historical debt record |
| legacy lane-local seed export helpers | cross-lane response shaping | `sunset` | duplicated seed handoff paths | collapse into one `identity_seed_v1` path |
| `PrimeRegistry`/`GlobalIdentitySpace` allocator core | identity substrate | `reuse` | canonical allocator and projection substrate | keep and extend via boundary adapters only |

## Remaining Work Drain Queue

| Debt ID | Status | Debt area | Current blocker | Required change | Exit gate | Evidence links |
| --- | --- | --- | --- | --- | --- | --- |
| `IDR-001` | `done` | progress-v1 retirement | none | hard-cut completed: removed v1 token constants/aliases, removed v1 emission path, removed v1 parser ingress branch, and migrated affected fixtures/tests to v2 canonical transport | `G-004A` (pass), `G-004` (pass) | `src/gabion/commands/progress_contract.py`; `src/gabion/server_core/command_orchestrator_primitives.py`; `src/gabion/server_core/command_orchestrator.py`; `src/gabion/server.py`; `tests/gabion/commands/test_progress_contract_edges.py`; `tests/gabion/server_core/command_orchestrator_coverage_cases.py`; `tests/gabion/server/server_execute_command_edges_cases.py`; `tests/gabion/cli/cli_helpers_cases.py`; `tests/gabion/tooling/delta/test_delta_script_telemetry.py`; `tests/gabion/tooling/delta/test_delta_emit_runtime.py`; `tests/gabion/lsp_client/lsp_client_direct_cases.py`; `tests/gabion/commands/test_command_boundary_order.py` |
| `IDR-002` | `done` | `FastIntegerCarrier` deletion | none | removed `FastIntegerCarrier` implementation/export and direct test coverage; bit-prime carrier is sole runtime path | `G-003` (pass) | `src/gabion/analysis/foundation/identity_shadow_runtime.py`; `tests/gabion/analysis/foundation/test_identity_shadow_runtime.py` |
| `IDR-003` | `done` | transcript fixture lane governance tightening | none | governance tightened with static fixture-lane boundary enforcement and parity regression coverage; transcript runtime remains out of production surfaces | `G-002` (pass), `G-005` (pass) | `tests/gabion/analysis/foundation/transcript_event_fixtures.py`; `tests/gabion/analysis/foundation/test_transcript_event_adapter.py`; `tests/gabion/analysis/foundation/test_event_algebra.py`; `docs/event_algebra_bridge_matrix.md`; `in/transcript.md` |
| `IDR-004` | `open` | namespace governance/import-export consolidation | path/prime intern seams remain split across mirror, shadow runtime, and lane adapters | consolidate namespace governance and import/export boundary adapters while preserving deterministic ordering and sidecar additivity | `G-001` | `src/gabion/analysis/foundation/identity_registry_mirror.py`; `src/gabion/analysis/foundation/identity_shadow_runtime.py`; `src/gabion/server_core/command_orchestrator.py`; `docs/identity_substrate_bridge_matrix.md` |

## Gate Definitions (Measurable)

| Gate | Purpose | Command | Current expectation |
| --- | --- | --- | --- |
| `G-001` | dual-publish parity and sidecar behavior remains healthy | `mise exec -- python -m pytest -q tests/gabion/server/server_execute_command_edges_cases.py tests/gabion/server_core/command_orchestrator_coverage_cases.py tests/gabion/commands/test_progress_contract_edges.py` | pass |
| `G-002` | transcript fixture adapter parity remains healthy | `mise exec -- python -m pytest -q tests/gabion/analysis/foundation/test_transcript_event_adapter.py tests/gabion/analysis/foundation/test_event_algebra.py` | pass |
| `G-005` | transcript fixture lane remains test-only and does not leak into production source | `rg -n "transcript_event_fixtures|adapt_transcript_fixture_event|NodeDiscovered|EdgeFormed|ComponentSealed|StreamTerminated|NameInterned|transcript\\.scout" src/gabion` | pass (no matches) |
| `G-003` | `FastIntegerCarrier` fully removed from tracked code/tests/docs surfaces | `rg -n "FastIntegerCarrier" src tests docs --glob '!docs/identity_debt_rebase_map.md'` | pass |
| `G-004A` | stage-1 alias cutover complete (`LSP_PROGRESS_TOKEN` / `_LSP_PROGRESS_TOKEN` no longer default to v1) | `rg -n "LSP_PROGRESS_TOKEN\\s*=\\s*LSP_PROGRESS_TOKEN_V1|_LSP_PROGRESS_TOKEN\\s*=\\s*_?LSP_PROGRESS_TOKEN_V1" src/gabion` | pass (no matches) |
| `G-004` | progress-v1 carrier fully retired from `src` | `rg -n "LSP_PROGRESS_TOKEN\\s*=\\s*LSP_PROGRESS_TOKEN_V1|_LSP_PROGRESS_TOKEN\\s*=\\s*_?LSP_PROGRESS_TOKEN_V1|gabion\\.dataflowAudit/progress-v1" src/gabion` | pass (no matches) |

## Temporary compatibility adapter lifecycle metadata

| field | value |
| --- | --- |
| actor | `gabion-maintainers` |
| rationale | Historical boundary adapter that preserved in-repo progress consumers during v2 cutover. |
| scope | historical: `$/progress` token `gabion.dataflowAudit/progress-v1` in dataflow lane only (now retired from `src/gabion`) |
| start | `2026-03-04` |
| expiry | `2026-03-04` |
| rollback_condition | none (stage-2 hard cut completed and tracked by strict gate closure) |
| evidence_links | `src/gabion/commands/progress_contract.py`, `src/gabion/server_core/command_orchestrator_primitives.py`, `src/gabion/server_core/command_orchestrator.py`, `src/gabion/server.py`, `tests/gabion/commands/test_progress_contract_edges.py`, `tests/gabion/server/server_execute_command_edges_cases.py`, `tests/gabion/server_core/command_orchestrator_coverage_cases.py`, `tests/gabion/cli/cli_helpers_cases.py`, `tests/gabion/tooling/delta/test_delta_script_telemetry.py`, `tests/gabion/tooling/delta/test_delta_emit_runtime.py`, `tests/gabion/lsp_client/lsp_client_direct_cases.py`, `tests/gabion/commands/test_command_boundary_order.py`, `gate:G-004` |

### Transcript fixture boundary metadata

| field | value |
| --- | --- |
| actor | `gabion-maintainers` |
| rationale | Preserve transcript substrate comparability without introducing a production transcript runtime lane. |
| scope | transcript fixture adapters only (`in/transcript.md` is informative context; tracked fixtures/tests are executable evidence). |
| start | `2026-03-04` |
| expiry | `explicit transcript runtime lane acceptance` |
| rollback_condition | fixture adapter assumptions diverge from canonical envelope contract or fixture-only boundary checks fail (`G-002`/`G-005`). |
| evidence_links | `tests/gabion/analysis/foundation/transcript_event_fixtures.py`, `tests/gabion/analysis/foundation/test_transcript_event_adapter.py`, `tests/gabion/analysis/foundation/test_event_algebra.py`, `docs/event_algebra_bridge_matrix.md`, `docs/identity_substrate_bridge_matrix.md`, `in/transcript.md`, `gate:G-002`, `gate:G-005` |

## Deferred deletion list (post-cutover)

| Candidate | Gate | Delete condition (concrete) |
| --- | --- | --- |
| ad-hoc progress-only identity token branches (`sunset`) | `G-003` | delete when `rg -n "FastIntegerCarrier" src tests docs --glob '!docs/identity_debt_rebase_map.md'` returns no matches. |
| bespoke lane-local allocation delta bookkeeping (`sunset`) | `G-001` | delete when `G-001` passes and `tests/gabion/server/server_execute_command_edges_cases.py` keeps asserting `identity_allocation_delta_v1` on canonical progress notifications. |
| duplicated seed export helpers (`sunset`) | `G-001` | delete when `G-001` passes and `tests/gabion/server/server_execute_command_edges_cases.py` keeps asserting `identity_seed_v1` on success and timeout response paths. |

## Guardrails

1. Keep semantic outputs unchanged; sidecars remain additive.
2. Reject semantic-core compatibility wrappers; boundary adapters only with explicit lifecycle metadata.
3. Preserve deterministic ordering/canonicalization before transport emission.
