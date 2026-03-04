---
doc_revision: 4
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: identity_substrate_bridge_matrix
doc_role: architecture
doc_scope:
  - repo
  - analysis
  - identity
doc_authority: informative
doc_requires:
  - in/transcript.md
  - src/gabion/analysis/foundation/event_algebra.py
  - src/gabion/analysis/dataflow/engine/dataflow_event_algebra_adapter.py
  - src/gabion/analysis/aspf/aspf_event_algebra_adapter.py
  - src/gabion/analysis/core/type_fingerprints.py
doc_reviewed_as_of:
doc_review_notes:
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

# Identity Substrate Bridge Matrix

## Canonical envelope (phase-agnostic)

| Canonical field | Meaning | Existing carrier sources |
| --- | --- | --- |
| `sequence` | monotonic per run context | `CanonicalRunContext.sequencer` |
| `kind` | typed event variant | transcript events, ASPF replay variants, dataflow progress `event_kind` |
| `phase` | pipeline stage or boundary stage | dataflow phases, ASPF boundary phases, transcript stream epochs |
| `source` | producer namespace | `dataflow.*`, `aspf.trace_replay`, transcript scout/planner lanes |
| `identity_projection` | ordered basis path + prime-product witness | `GlobalIdentitySpace.project(...)` |
| `payload` | lane-native payload | progress payloads, ASPF payload blocks, transcript event payloads |
| `causal_refs` | predecessor references | transition parent identities, run-boundary refs, transcript causal edges |
| `event_id` | canonical run-scoped id | `run_id:sequence` |

## Field-level substrate bridge matrix

| Source lane | Source carrier | Canonical envelope compatibility | Identity-space compatibility | Replay/finalization semantics | Source -> canonical -> sink adapter |
| --- | --- | --- | --- | --- | --- |
| Transcript fixture lane | `StreamEvent` union (`NodeDiscovered`, `EdgeFormed`, `ComponentSealed`, `StreamTerminated`, optional `NameInterned`) | direct map to `kind`, `phase`, `source`, and `causal_refs` | `name -> int` and tuple-path tokens are isomorphic to atom/path projection basis | terminal barrier via `StreamTerminated`; planner folds partial state until barrier | transcript fixture event -> `CanonicalEventEnvelope` -> fixture consumer/replay adapter |
| Indexed/Dataflow lane | collection/phase/index progress callbacks (`collection`, `forest`, `edge`, `post`) | direct map to envelope with adaptation `valid`/`rejected` status at transport boundary | deterministic identity tokens with integer-anchor carrier (`event_seq` lowered tokens) | explicit phase transitions plus terminal event boundary | progress payload -> dataflow event algebra adapter -> `progress-v2` canonical primary sink |
| ASPF lane | replay event union (`AspfOneCellEvent`, `AspfTwoCellEvent`, `AspfCofibrationEvent`, `AspfRunBoundaryEvent`) | direct map to envelope axes; payload remains witness/domain specific | ASPF node interning + event indexes project into canonical identity refs without semantic collapse | append/fold/replay with sink finalization and trace index materialization | ASPF event stream -> replay adapter -> visitor/sink stack |
| Fingerprint lane | prime assignment and seed carriers (`type_base`, `type_ctor`, `synth`) | envelope-adjacent through identity sidecars and seed evidence | exact atom allocation isomorphic to transcript intern allocation; tuple-path witness aligns with prime-product projection | deterministic assignment order; seed/replay and mirror hydration/write-through semantics | registry assignment event -> `IdentityRegistryMirror` -> `GlobalIdentitySpace` allocation ledger |

## Identity allocator compatibility matrix

| Allocator concern | Transcript shape | Gabion shape | Bridge seam |
| --- | --- | --- | --- |
| atom allocation | `name -> int` | `PrimeRegistry.get_or_assign(raw_key(namespace, token))` | `GlobalIdentitySpace.intern_atom` |
| tuple-path identity | tuple of interned atoms | `IdentityPath(namespace, atoms)` | `GlobalIdentitySpace.intern_path` |
| commutative alias | multiplied primes | `IdentityProjection.prime_product` | `GlobalIdentitySpace.project` witness |
| stable projection id | deterministic event/node id | digest alias + `event_id` | `canonical_event_id`, digest alias carrier |
| replay/debug seed | serialized intern table | registry seed payload | `identity_seed_v1` |
| fingerprint write-through | post-allocation mirror from prime registry | `PrimeAssignmentEvent -> register_atom(...)` | `identity_allocation_delta_v1` with fingerprint namespaces |

## Bridgeability tiers

| Tier | Substrate seam | Notes |
| --- | --- | --- |
| high | event envelope + adapter/sink composition | carrier polymorphism is already normalized around typed payload unions |
| high | checkpoint/resume + delta replay | ASPF and progress lanes both expose append/fold/replay-compatible ledgers |
| high | determinism and canonical ordering | WL/canonical encoders/sort contracts already provide shared determinism substrate |
| medium-high | interning unification | transcript name/path, prime registry, and identity space now bridge through write-through mirroring |
| medium | SCC/WL cross-lane reuse | algorithmic mechanics commute; graph carriers and payload semantics remain lane-specific |

## Near-compatibility constraints

1. Envelope fields are purpose-agnostic; payload schemas are intentionally lane-specific and should not be collapsed.
2. Transcript tuple-path identities and fingerprint prime allocations are substrate-isomorphic, but provenance witness payloads remain domain-scoped.
3. `progress-v1` remains a temporary boundary adapter; canonical authority is `progress-v2`.
4. Transcript runtime remains fixture-only; bridge matrix assumes adapter reuse, not production transcript lane cutover.

## Progress bridge (Phase 2A Dataflow lane)

| Sidecar field | Producer | Consumer intent |
| --- | --- | --- |
| `canonical_progress_event_v1` (`$/progress`, token `gabion.dataflowAudit/progress-v2`) | orchestrator canonical primary emission | v2 authoritative progress carrier (valid/rejected adaptation outcomes) |
| `fallback_payload_v1` (v2 rejected-only) | canonical v2 rejected adaptation payload | preserve observability during rejection without dropping v1 payload semantics |
| `canonical_event_v1` | identity shadow runtime valid adaptation | cross-lane canonical replay and adapter parity |
| `identity_allocation_delta_v1` | identity shadow runtime delta cursor + fingerprint write-through mirror | deterministic incremental allocation telemetry across path and fingerprint namespaces (`type_base`, `type_ctor`, `synth`) |
| `canonical_event_error_v1` | identity shadow runtime rejected adaptation | non-fatal shadow fault visibility |
| `gabion.dataflowAudit/progress-v1` (legacy token) | dual-publish compatibility adapter | temporary migration lane until repo-consumer-zero gate |
| `identity_seed_v1` | final command response | replay/debug seed handoff |
