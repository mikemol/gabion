---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: universal_curve_lab_tc_design_bridge
doc_role: research_mapping
doc_scope:
  - in
  - research
  - universal_curve_lab
  - tc
doc_authority: informative
doc_requires:
  - in/universal-curve-lab-bundle/README.md
  - glossary.md#contract
doc_relations:
  refines:
    - in/universal-curve-lab-bundle/README.md
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---
# TC design bridge (research scope)

This bridge maps the **TC module concepts** in `agda/UniversalCurve/TC/` to existing
Gabion runtime surfaces so experiments can stay aligned with practical outputs.

> Status: **research/inspiration only**. These mappings are descriptive and do not
> claim production enforcement in Gabion runtime paths.

## Concept-to-surface map

| TC concept (Agda) | Gabion runtime surface | Notes |
| --- | --- | --- |
| `TracePoint` (`SIG`) | ASPF traces emitted by audit/check command pipelines | Conceptual alignment with span/phase-style trace slices used for evidence exports. |
| `PayloadKey` (`SIG`) | DTO payload keys in serialized audit/check results | Models required vs optional payload fields at a research-contract level. |
| `CommandSurface` (`SIG`) | `gabion` command outputs (e.g., check/audit style commands) | Treats command names + maturity epoch as a typed surface descriptor. |
| Constructor helpers (`CONSTR`) | Runtime payload assembly pathways | Mirrors how handlers build output records, but currently only as lab constructors. |
| `BridgePlan` (`GLUE`) | Cross-surface traceability docs and prototype adapters | Encodes mapping bundles and an explicit non-production-enforced flag. |

## Intended usage

- Keep TC signatures stable while iterating on experimental adapters.
- Use this map to evaluate whether a TC concept should later be promoted into
  first-class runtime contracts.
- Preserve the current contract boundary: no direct claim that Agda TC modules
  are wired into production command enforcement yet.
