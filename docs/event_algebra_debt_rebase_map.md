---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: event_algebra_debt_rebase_map
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
  POLICY_SEED.md#policy_seed: "Reviewed correction-unit obligations and policy checks; map marks staged adapter lifecycle only."
  glossary.md#contract: "Reviewed decision protocol and transport semantics for event-adapter staging."
  AGENTS.md#agent_obligations: "Reviewed shift-left ambiguity and no-compat-layer constraints; map marks boundary adapters as temporary seams."
doc_sections:
  debt_rebase: 1
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="debt_rebase"></a>
# Event Algebra Debt Rebase Map

This map tracks which surfaces are reusable immediately versus staged for later
deletion once producers can emit canonical envelopes directly.

## Reuse Now (`now`)

- `analysis/foundation/event_algebra.py` canonical in-memory contract.
- `analysis/foundation/event_algebra_codec.py` JSON/proto envelope codecs.
- `analysis/aspf/aspf_event_algebra_adapter.py` replay-event adapter.
- `analysis/dataflow/engine/dataflow_event_algebra_adapter.py` phase and collection/index progress adapters.
- `tests/gabion/analysis/foundation/transcript_event_fixtures.py` transcript compatibility fixtures for contract testing.

## Deferred Deletion (`phase3`)

- Legacy ad-hoc stream-shape adapters that duplicate event field normalization
  downstream of source-specific progress/replay carriers.
- Any caller-local event wrappers that only exist to bridge to a canonical
  downstream sink after adapter convergence.
- Redundant source-local event identity helper code once identity derivation is
  centralized on the canonical adapter seam.

## Deferred Deletion (`phase4`)

- Producer-specific parallel envelope codecs once producers can emit canonical
  envelope natively.
- Duplicate progress/replay routing paths in orchestration layers that become
  no-ops after producer-native canonical envelope emission.

## Lifecycle Notes

- Phase 2 is adapter convergence only.
- Existing producer payload contracts remain unchanged.
- Deletions are blocked until equivalent canonical coverage has parity tests for
  each source surface.
