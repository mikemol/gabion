---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: universal_curve_lab_tc_v2_decision_crosswalk
doc_role: research_mapping
doc_scope:
  - in
  - research
  - universal_curve_lab
  - tc
doc_authority: informative
doc_requires:
  - in/universal-curve-lab-bundle/docs/tc-design-bridge.md
  - src/gabion/server.py
  - src/gabion/server_core/command_orchestrator.py
  - docs/architecture_zones.md
  - docs/normative_clause_index.md#clause-shift-ambiguity-left
doc_relations:
  refines:
    - in/universal-curve-lab-bundle/docs/tc-design-bridge.md
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

# TC v2 decision crosswalk (Agda -> runtime contracts)

This crosswalk links proposed Agda decision surfaces to current runtime contract loci.
Status labels are runtime-readiness tags:

- **already enforced**: explicit runtime guard/protocol is present.
- **partially enforced**: related guards exist, but no single explicit contract exactly matches the Agda surface yet.
- **research-only**: only architectural intent or research framing exists.

## Crosswalk table

| Agda decision surface | Runtime analogue status | Runtime contract loci | Why this status |
| --- | --- | --- | --- |
| `CoverDichotomy` | **partially enforced** | `src/gabion/server_core/command_orchestrator.py` timeout/success classification shaping (`classification`, `analysis_state`) and terminal response shaping; `src/gabion/server.py` response normalization through `_normalize_dataflow_response`; architecture boundary contract in `docs/architecture_zones.md` | Runtime has a practical dichotomy (`succeeded` / timeout / failed classes), but it is encoded as string-valued classifications and progress-state transforms rather than a first-class, single Decision Protocol named as a dichotomy contract. |
| `BoundaryMeansMaxOnly` | **already enforced** | Boundary normalization module anchors in `src/gabion/server.py` (`# gabion:boundary_normalization_module`, `_require_payload`, `_ordered_command_response`, `_normalize_dataflow_response`) plus boundary-normalization marker in `src/gabion/server_core/command_orchestrator.py` | Current runtime enforces ingress/egress normalization at server boundaries and routes normalized carriers into server core; this directly matches a boundary-first, ambiguity-left discipline. |
| `CollapseOnlyIfBoth` | **partially enforced** | Conjunctive collapse logic in timeout classification upgrade (`resume_supported` **and** prior no-progress classification **and** substantive progress gate) in `src/gabion/server_core/command_orchestrator.py`; aux-operation gating requiring domain/action + baseline-path conjunction; boundary/core policy language in `docs/architecture_zones.md` | Runtime contains multiple "collapse/upgrade only when conjunction holds" checks, but they are local decision bundles rather than one canonical Decision Protocol surface dedicated to this invariant. |

## Trace anchors for semantic ownership

Use these exact search anchors when tracing implementation ownership:

- `# gabion:boundary_normalization_module`
- `# gabion:decision_protocol_module`
- `@decision_protocol`
- `Decision Protocol`
- `NCI-SHIFT-AMBIGUITY-LEFT`

## Implementation notes for follow-up promotion

If these Agda surfaces are promoted from research into runtime contracts, prefer:

1. a dedicated Protocol/dataclass bundle in boundary modules,
2. one explicit validation surface per decision family,
3. structural outcomes instead of sentinel/stringly control flow where practical,

while preserving the boundary-first rule from `NCI-SHIFT-AMBIGUITY-LEFT`.
