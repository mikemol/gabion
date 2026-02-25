---
doc_revision: 4
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: universal_curve_lab_experiments
doc_role: research_protocol
doc_scope:
  - in
  - research
  - universal_curve_lab
doc_authority: informative
doc_requires:
  - in/universal-curve-lab-bundle/README.md
doc_relations:
  refines:
    - in/universal-curve-lab-bundle/README.md
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---
# Experiments

See `python/`.

- Exact rational pipelines (Fractions) for forcing gaps Δ₂/Δ₃.
- Prime-by-prime forcing atlases.
- Interning curves: collision-rate vs scale for GF(p)/mod p/cap.
- SPPF memoization and JSON exports (audit artifacts).

## Gabion fingerprint bridge harness

The integration harness at `python/gabion_fingerprint_bridge.py` maps minimal lab
adapter payloads (`python/adapter_schema.json`) into Gabion-compatible
fingerprint dimensions.

Adapter schema (`gabion-lab-adapter@1`) case fields:

- `base_atoms`: maps to base type atoms (`type_base` namespace).
- `ctor_atoms`: maps to constructor atoms (`type_ctor` namespace).
- `provenance_atoms` (optional): maps to provenance/evidence atoms
  (`evidence_kind` namespace).
- `synth_atoms` (optional): maps to direct synth atoms (`synth` namespace)
  before repeated-tail synth assignment.

Harness outputs are written to
`artifacts/gabion_fingerprint_bridge.json`, including:

- dimensional fingerprints (`base` / `ctor` / `provenance` / `synth`),
- `keys_with_remainder` diagnostics per dimension,
- pairwise carrier-soundness checks via `fingerprint_carrier_soundness`,
- synth-registry payload snapshots (`synth_registry_payload`) and registry seed.

### Boundary-focused case: end-window asymmetry

The `end-window-asymmetry` case encodes the lab claim that boundary behavior is
structural rather than noise (`boundary:asymmetric`,
`claim:boundary-is-structure`, `window:end`).

Gabion interpretation in this bridge:

- **Provenance/coherence:** boundary asymmetry appears in the provenance
  dimension as explicit evidence atoms, separating structural location semantics
  from base and constructor carriers.
- **Coherence check:** `keys_with_remainder.remainder == 1` across dimensions
  indicates the prime basis fully explains the encoded boundary state.
- **Carrier interpretation:** pairwise carrier soundness remains true, so the
  boundary-structure evidence commutes with base/ctor carriers without hidden
  overlap in mask-gcd diagnostics.
