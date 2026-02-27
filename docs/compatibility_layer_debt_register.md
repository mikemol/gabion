---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: compatibility_layer_debt_register
doc_role: register
doc_scope:
  - repo
  - governance
  - refactor
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - CONTRIBUTING.md#contributing_contract
  - AGENTS.md#agent_obligations
  - docs/normative_clause_index.md#normative_clause_index
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  CONTRIBUTING.md#contributing_contract: 2
  AGENTS.md#agent_obligations: 2
  docs/normative_clause_index.md#normative_clause_index: 2
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev2 and applied compatibility-layer hardening under NCI-SHIFT-AMBIGUITY-LEFT with boundary-only temporary exceptions."
  CONTRIBUTING.md#contributing_contract: "Reviewed contributing contract rev2; this register mirrors correction-unit and lifecycle metadata requirements."
  AGENTS.md#agent_obligations: "Reviewed agent obligations rev2; register aligns with refusal/forward-remediation expectations."
  docs/normative_clause_index.md#normative_clause_index: "Reviewed clause index rev2; register tracks debt against NCI-SHIFT-AMBIGUITY-LEFT."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="compatibility_layer_debt_register"></a>
# Compatibility Layer Debt Register

Canonical clause: [`NCI-SHIFT-AMBIGUITY-LEFT`](./normative_clause_index.md#clause-shift-ambiguity-left).

## Policy status
- Existing compatibility layers are legacy remediation debt.
- Net-new semantic-core compatibility layers are disallowed.
- Sunset deadline for listed debt: `2026-03-31`.

## Required fields
Every debt row must provide:
- owner
- rationale
- removal correction unit
- expiry
- exit criteria

## Debt ledger

| Surface | Owner | Rationale | Removal correction unit | Expiry | Exit criteria | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `src/gabion/schema.py` (`compatibility_shim` union surface) | Maintainer (unassigned) | Legacy request shape still allows compatibility-shim alternation. | `CU-compat-schema-surface` | `2026-03-31` | Replace union compatibility entry with deterministic refactor contract type and remove compatibility shim DTO path. | Open |
| `src/gabion/server.py` (compatibility normalization path) | Maintainer (unassigned) | Server still normalizes compatibility-shim alternation for refactor requests. | `CU-compat-server-normalization` | `2026-03-31` | Remove compatibility normalization branch and require a single validated ingress contract. | Open |
| `src/gabion/cli.py` (`--compat-shim*` flags/payload) | Maintainer (unassigned) | CLI exposes compatibility toggles that preserve dual-path behavior. | `CU-compat-cli-flags` | `2026-03-31` | Remove `--compat-shim*` options and emit only deterministic refactor payloads. | Open |
| `src/gabion/refactor/model.py` (shim normalization helper) | Maintainer (unassigned) | Model layer still lifts generic shim input into compatibility config. | `CU-compat-refactor-model` | `2026-03-31` | Remove `normalize_compatibility_shim` and compatibility config transport from the model contract. | Open |
| `src/gabion/refactor/engine.py` (shim emission path) | Maintainer (unassigned) | Engine still emits compatibility wrappers and optional overload/deprecation surfaces. | `CU-compat-refactor-engine` | `2026-03-31` | Remove compatibility-wrapper emission and keep only deterministic refactor output path. | Open |
