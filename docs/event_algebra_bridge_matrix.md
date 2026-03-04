---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: event_algebra_bridge_matrix
doc_role: architecture
doc_scope:
  - repo
  - analysis
  - event-algebra
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - AGENTS.md#agent_obligations
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  AGENTS.md#agent_obligations: 2
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed execution constraints; this matrix is adapter-only and does not alter execution trust boundaries."
  glossary.md#contract: "Reviewed bundle/protocol/decision-surface terms; matrix documents structural mapping only."
  AGENTS.md#agent_obligations: "Reviewed correction-unit and policy-check requirements for analysis-core additions."
doc_sections:
  bridge_matrix: 1
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="bridge_matrix"></a>
# Event Algebra Bridge Matrix

Phase 2 introduces a canonical envelope contract in
`analysis/foundation/event_algebra.py` and converges source systems via adapters
only.

## Canonical Envelope Fields

- `schema_version`
- `sequence`
- `run_id`
- `source`
- `phase`
- `kind`
- `identity_projection`
- `payload`
- `causal_refs`
- `event_id`

## Source -> Canonical Mapping

| Source event type | Adapter function | Canonical `source` | Canonical `phase` | Canonical `kind` | Identity derivation |
| --- | --- | --- | --- | --- | --- |
| `AspfOneCellEvent` | `adapt_aspf_replay_event` | `aspf.trace_replay` | `trace_replay` | `one_cell` | `(source, phase, kind, index, payload_digest)` |
| `AspfTwoCellEvent` | `adapt_aspf_replay_event` | `aspf.trace_replay` | `trace_replay` | `two_cell` | `(source, phase, kind, index, payload_digest)` |
| `AspfCofibrationEvent` | `adapt_aspf_replay_event` | `aspf.trace_replay` | `trace_replay` | `cofibration` | `(source, phase, kind, index, payload_digest)` |
| `AspfSurfaceUpdateEvent` | `adapt_aspf_replay_event` | `aspf.trace_replay` | `trace_replay` | `surface_update` | `(source, phase, kind, surface, representative)` |
| `AspfRunBoundaryEvent` | `adapt_aspf_replay_event` | `aspf.trace_replay` | `run_boundary` | `run_boundary` | `(source, phase, kind, boundary, payload_digest)` |
| dataflow phase progress payload | `adapt_dataflow_phase_progress_event` | `dataflow.phase_progress` | payload `phase` | payload/transition `event_kind` | `(source, phase, kind, marker/root/event_seq/... deterministic anchors)` |
| dataflow collection/index progress payload | `adapt_dataflow_collection_progress_event` | `dataflow.collection_progress` | `collection` | `collection_progress` or `analysis_index_progress` | `(source, phase, kind, event_seq/cache identities/digests)` |
| transcript fixture `NodeDiscovered` | `adapt_transcript_fixture_event` | `transcript.scout` | `scout` | `node_discovered` | `(source, kind, node_id, module_path)` |
| transcript fixture `EdgeFormed` | `adapt_transcript_fixture_event` | `transcript.scout` | `scout` | `edge_formed` | `(source, kind, src, dst, relation)` |
| transcript fixture `ComponentSealed` | `adapt_transcript_fixture_event` | `transcript.scout` | `scout` | `component_sealed` | `(source, kind, component_id, members_hash)` |
| transcript fixture `StreamTerminated` | `adapt_transcript_fixture_event` | `transcript.scout` | `scout` | `stream_terminated` | `(source, kind, reason, total_events)` |
| transcript fixture `NameInterned` | `adapt_transcript_fixture_event` | `transcript.scout` | `scout` | `name_interned` | `(source, kind, namespace, payload_digest)` |

## Hard-Fail Rule

Each adapter has:

- `adapt_*` returning `CanonicalAdaptationDecision` (`valid`/`rejected`).
- `adapt_*_or_raise` that raises `CanonicalEventAdaptationError` on `rejected`.

No canonical envelope is emitted without `identity_projection`.
