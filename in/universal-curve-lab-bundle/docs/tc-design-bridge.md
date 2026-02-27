---
doc_revision: 2
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

## SIGv2 decomposition to runtime surfaces

`SIGv2` introduces sub-records so bridge coverage can be tracked field-by-field while
`SIG`, `CONSTR`, and `GLUE` remain compatibility modules.

| `SIGv2` concept | Closest runtime surface today | Bridge coverage status |
| --- | --- | --- |
| `KanSandwich` | Ingress validation + evidence egress boundaries in command handlers | Mapped conceptually; **not runtime-enforced.** |
| `StratifiedSite` | Layered ambiguity/correction semantics in policy and docs | Documented in governance language; **not runtime-enforced.** |
| `QuantaleMetric` | Evidence-drift / projection-distance ideas in audit outputs | Descriptive naming only; **not runtime-enforced.** |
| `Cotower` | Snapshot/baseline continuity across check-delta style runs | Operationally adjacent only; **not runtime-enforced.** |
| `TowerOps` | Composition/normalization steps around payload shaping | No first-class runtime carrier; **not runtime-enforced.** |
| `Stabilization` | Fixed-point style convergence expectations in policy loop docs | Conceptual traceability only; **not runtime-enforced.** |

The v2 modules (`CONSTRv2.agda`, `GLUEv2.agda`) intentionally remain dual-tracked
with the existing modules until each `SIGv2` field has explicit bridge coverage
captured in this document and any future runtime contract docs.

## Intended usage

- Keep TC signatures stable while iterating on experimental adapters.
- Use this map to evaluate whether a TC concept should later be promoted into
  first-class runtime contracts.
- Preserve the current contract boundary: no direct claim that Agda TC modules
  are wired into production command enforcement yet.
