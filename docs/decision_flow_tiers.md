---
doc_revision: 1
reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.
doc_id: decision_flow_tiers
doc_role: reference
doc_scope:
  - repo
  - analysis
  - docs
doc_authority: informative
doc_requires:
  - glossary.md#decision_table
  - glossary.md#decision_bundle
  - glossary.md#decision_protocol
  - docs/sppf_checklist.md#sppf_checklist
doc_reviewed_as_of:
  glossary.md#decision_table: 1
  glossary.md#decision_bundle: 1
  glossary.md#decision_protocol: 1
  docs/sppf_checklist.md#sppf_checklist: 7
doc_review_notes:
  glossary.md#decision_table: Reviewed glossary decision table contract.
  glossary.md#decision_bundle: Reviewed glossary decision bundle contract.
  glossary.md#decision_protocol: Reviewed glossary decision protocol contract.
  docs/sppf_checklist.md#sppf_checklist: Reviewed checklist status + GH mapping for decision tiers.
doc_sections:
  decision_flow_tiers: 1
doc_section_requires:
  decision_flow_tiers:
    - glossary.md#decision_table
    - glossary.md#decision_bundle
    - glossary.md#decision_protocol
    - docs/sppf_checklist.md#sppf_checklist
doc_section_reviews:
  decision_flow_tiers:
    glossary.md#decision_table:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Decision table tier contract applied.
    glossary.md#decision_bundle:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Decision bundle tier contract applied.
    glossary.md#decision_protocol:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Decision protocol tier contract applied.
    docs/sppf_checklist.md#sppf_checklist:
      dep_version: 7
      self_version_at_review: 1
      outcome: no_change
      note: Checklist node references aligned to GH-47/48/49.
doc_change_protocol: POLICY_SEED.md#change_protocol
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="decision_flow_tiers"></a>

# Decision-flow tier artifacts

This document is the decision-flow companion for GH-47/48/49 and describes the concrete artifacts emitted by snapshot/report projections.

## <a id="decision-flow-tier3"></a>Tier-3 decision tables (GH-47)

| Artifact field | Meaning | Analysis evidence keys | Checklist node |
| --- | --- | --- | --- |
| `decision_id` | Deterministic ID for one decision surface/value-encoded surface. | Derived from `E:decision_surface/<mode>::<path>::<qual>::<param>` entries in `analysis_evidence_keys`. | `docs/sppf_checklist.md#decision-flow-tier3` |
| `mode` / `path` / `qual` / `params` | Normalized decision site payload. | `E:decision_surface/direct::...` and `E:decision_surface/value::...` | `docs/sppf_checklist.md#decision-flow-tier3` |
| `analysis_evidence_keys` | Stable key list proving source observations. | One key per normalized parameter. | `docs/sppf_checklist.md#decision-flow-tier3` |

## <a id="decision-flow-tier2"></a>Tier-2 repeated-guard bundles (GH-48)

| Artifact field | Meaning | Analysis evidence keys | Checklist node |
| --- | --- | --- | --- |
| `bundle_id` | Deterministic ID for a repeated decision parameter bundle. | Computed from member decision IDs emitted by Tier-3 tables. | `docs/sppf_checklist.md#decision-flow-tier2` |
| `params` / `occurrences` | Canonical bundle members + repeat count. | Back-references Tier-3 `analysis_evidence_keys` through `member_decision_ids`. | `docs/sppf_checklist.md#decision-flow-tier2` |
| `member_decision_ids` | Sorted IDs of linked Tier-3 tables. | IDs map to `decision_tables[*].decision_id`. | `docs/sppf_checklist.md#decision-flow-tier2` |

## <a id="decision-flow-tier1"></a>Tier-1 protocol enforcement hooks (GH-49)

| Violation code | Contract drift detected | Triggered by | Checklist node |
| --- | --- | --- | --- |
| `DECISION_PROTOCOL_EMPTY_MEMBERS` | Tier-2 bundle does not point to any decision table. | Empty `member_decision_ids`. | `docs/sppf_checklist.md#decision-flow-tier1` |
| `DECISION_PROTOCOL_MISSING_TABLE` | Tier-2 bundle references a non-existent Tier-3 table. | Missing `decision_id` in table map. | `docs/sppf_checklist.md#decision-flow-tier1` |
| `DECISION_PROTOCOL_MISSING_EVIDENCE` | Critical decision path lacks evidence keys. | Empty `analysis_evidence_keys` on linked Tier-3 table. | `docs/sppf_checklist.md#decision-flow-tier1` |
| `DECISION_PROTOCOL_MISSING_CHECKLIST_LINK` | Critical decision path lacks checklist linkage. | Empty `checklist_nodes` on linked Tier-3 table. | `docs/sppf_checklist.md#decision-flow-tier1` |

## Determinism contract

- IDs use deterministic schema hashing over normalized payloads.
- Artifact arrays are sorted canonically by path/qual/mode/params or bundle rank.
- Snapshot summaries include explicit counts for Tier-3 tables, Tier-2 bundles, and Tier-1 violations.
