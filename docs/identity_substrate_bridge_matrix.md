---
doc_revision: 3
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

## Source -> envelope adaptation matrix

| Source lane | Source type(s) | Adapter surface | Identity token basis | Envelope sink |
| --- | --- | --- | --- | --- |
| Transcript lane | `NodeDiscovered`, `EdgeFormed`, `ComponentSealed`, stream boundary | transcript fixture adapter pattern (foundation tests) | node/path/edge anchors (name->int, tuple-path) | `CanonicalEventEnvelope` |
| Dataflow progress lane | collection + phase progress payloads | `adapt_dataflow_collection_progress_event`, `adapt_dataflow_phase_progress_event` | deterministic tokens (`phase`, marker, root identity, `event_seq`, index digests) | `canonical_event_v1` sidecar in `$/progress` |
| ASPF replay lane | `AspfOneCellEvent`, `AspfTwoCellEvent`, `AspfCofibrationEvent`, `AspfRunBoundaryEvent` | `adapt_aspf_replay_event` | index + payload digest + boundary ids | `CanonicalEventEnvelope` / replay sinks |
| Fingerprint lane | base/ctor/provenance/synth assignments | `PrimeRegistry` + `PrimeIdentityAdapter` | namespace-key tokenization (`raw_key`) | seed payload + decoded witness surfaces |

## Identity allocator compatibility matrix

| Allocator concern | Transcript shape | Gabion shape | Bridge seam |
| --- | --- | --- | --- |
| atom allocation | `name -> int` | `PrimeRegistry.get_or_assign(raw_key(namespace, token))` | `GlobalIdentitySpace.intern_atom` |
| tuple-path identity | tuple of interned atoms | `IdentityPath(namespace, atoms)` | `GlobalIdentitySpace.intern_path` |
| commutative alias | multiplied primes | `IdentityProjection.prime_product` | `GlobalIdentitySpace.project` witness |
| stable projection id | deterministic event/node id | digest alias + `event_id` | `canonical_event_id`, digest alias carrier |
| replay/debug seed | serialized intern table | registry seed payload | `identity_seed_v1` |
| fingerprint write-through | post-allocation mirror from prime registry | `PrimeAssignmentEvent -> register_atom(...)` | `identity_allocation_delta_v1` with fingerprint namespaces |

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
