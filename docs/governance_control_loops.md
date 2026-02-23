---
doc_revision: 1
doc_id: governance_control_loops
doc_role: governance
doc_scope:
  - repo
  - policy
  - tooling
doc_authority: normative
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

# Governance control loops

This document defines correction modes and transition criteria for baseline and delta governance loops.

## Correction modes

- `advisory`: emit diagnostics and summary only; do not block execution.
- `ratchet`: allow bounded movement only toward tighter policy; block regressions.
- `hard-fail`: block any blocking-threshold regression immediately.

## Transition criteria

Each loop transitions using `warning_threshold` and `blocking_threshold` values from `docs/governance_rules.yaml`:

1. `advisory -> ratchet` when warning-threshold events recur and no override token is recorded.
2. `ratchet -> hard-fail` when blocking-threshold events recur under ratchet mode.
3. `hard-fail -> ratchet|advisory` only with an explicit policy override token and an annotated rationale in the change artifact.

## Bounded-step correction rules

Baseline updates are bounded by these invariants:

1. Baseline writes require explicit `--write-*` flags.
2. Strictness cannot be reduced unless both are present:
   - an explicit policy override token (`GABION_POLICY_OVERRIDE_TOKEN`), and
   - an annotated rationale (`GABION_POLICY_OVERRIDE_RATIONALE`).
3. Delta gates use shared severity mapping from `docs/governance_rules.yaml`; ad-hoc threshold branching is prohibited.

## Loop table

| Loop | Correction mode | Warning threshold | Blocking threshold |
| --- | --- | --- | --- |
| obsolescence opaque | hard-fail | 0 | 1 |
| obsolescence unmapped | ratchet | 0 | 1 |
| annotation orphaned | ratchet | 0 | 1 |
| ambiguity total | hard-fail | 0 | 1 |
| docflow contradictions | advisory | 0 | 1 |

The source of truth for loop thresholds, mode defaults, and transitions is `docs/governance_rules.yaml`.
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: governance_control_loops
doc_role: policy
doc_scope:
  - repo
  - governance
  - ci
  - tooling
doc_authority: normative
doc_requires:
  - README.md#repo_contract
  - AGENTS.md#agent_obligations
  - POLICY_SEED.md#policy_seed
  - CONTRIBUTING.md#contributing_contract
  - glossary.md#contract
doc_reviewed_as_of:
  README.md#repo_contract: 1
  AGENTS.md#agent_obligations: 1
  POLICY_SEED.md#policy_seed: 1
  CONTRIBUTING.md#contributing_contract: 1
doc_review_notes:
  README.md#repo_contract: "Repo contract reviewed; loop registry aligns with governance scope and entry points."
  AGENTS.md#agent_obligations: "Agent obligations reviewed; loop enforcement remains mechanized and refusal-safe."
  POLICY_SEED.md#policy_seed: "Policy seed reviewed; governance loops reify mechanized control surfaces."
  CONTRIBUTING.md#contributing_contract: "Contributor workflow reviewed; loop verification commands align with required local checks."
doc_sections:
  governance_control_loops: 1
doc_section_requires:
  governance_control_loops:
    - README.md#repo_contract
    - AGENTS.md#agent_obligations
    - POLICY_SEED.md#policy_seed
    - CONTRIBUTING.md#contributing_contract
    - glossary.md#contract
doc_section_reviews:
  governance_control_loops:
    README.md#repo_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Repo contract reviewed; governed domains map to declared loop entries."
    AGENTS.md#agent_obligations:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Agent obligations reviewed; loop declarations stay machine-checkable."
    POLICY_SEED.md#policy_seed:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Policy constraints map cleanly onto loop schema fields."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Contributor commands remain valid loop verification operators."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Glossary contract reviewed; loop predicates remain semantically aligned."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
loop_domains:
  - security/workflows
  - docs/docflow
  - LSP architecture
  - dataflow grammar
  - baseline ratchets
doc_invariants:
  - mechanized_governance_invariant
  - lsp_first_invariant
  - tier2_reification
  - tier3_documentation
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="governance_control_loops"></a>
# Governance control loops

This file defines normalized control-loop entries for every governed normative domain.
Docflow treats `loop_domains` as the declared registry used by `scripts/audit_tools.py`
for loop-entry completeness checks. Canonical governance anchors: `README.md#repo_contract`, `AGENTS.md#agent_obligations`, `CONTRIBUTING.md#contributing_contract`, and `glossary.md#contract`.

## Normalized schema

Each loop entry MUST define:

- `sensor`
- `state artifact`
- `target predicate`
- `error signal`
- `actuator`
- `max correction step`
- `verification command`
- `escalation threshold`

## Loop registry

### 1) security/workflows

- **sensor:** workflow YAML + policy posture probes from `scripts/policy_check.py --workflows` and `scripts/policy_check.py --posture`.
- **state artifact:** `.github/workflows/*.yml`, plus policy snapshots under `artifacts/out/policy_posture*.json` when emitted.
- **target predicate:** all workflow actions pinned to allow-listed full SHAs; self-hosted trust boundaries satisfy `POLICY_SEED.md#policy_seed` Prime Invariant.
- **error signal:** `policy_check` non-zero exit or posture violation rows.
- **actuator:** patch workflow files; refresh allow-list expectations; re-run policy checks.
- **max correction step:** one workflow/guardrail patchset per loop iteration.
- **verification command:** `mise exec -- python scripts/policy_check.py --workflows && mise exec -- python scripts/policy_check.py --posture`.
- **escalation threshold:** any Prime Invariant contradiction or unresolved posture failure after one correction step.

### 2) docs/docflow

- **sensor:** docflow compliance emitters in `scripts/audit_tools.py` and `gabion docflow`.
- **state artifact:** `artifacts/out/docflow_compliance.json`, `artifacts/out/docflow_compliance_delta.json`, and `artifacts/audit_reports/docflow_compliance.md`.
- **target predicate:** normative docs satisfy frontmatter/review invariants; contradictions delta remains zero; every normative domain has a loop registry entry.
- **error signal:** docflow violations or positive contradiction delta.
- **actuator:** update governance docs, references, review pins, and loop declarations.
- **max correction step:** one doc revision cycle affecting the minimum coherent set of governance files.
- **verification command:** `mise exec -- python -m gabion docflow --fail-on-violations`.
- **escalation threshold:** repeated docflow violation after one coherent doc revision cycle.

### 3) LSP architecture

- **sensor:** code-path checks over server/CLI split plus governance contract references.
- **state artifact:** `src/gabion/server.py`, `src/gabion/cli.py`, and generated check artifacts under `artifacts/audit_reports/`.
- **target predicate:** LSP-first invariant holds: server remains semantic core and CLI is a thin LSP client.
- **error signal:** architecture drift findings in checks/tests or violations against policy text.
- **actuator:** move semantics from CLI to server boundary; restore thin-client behavior.
- **max correction step:** one refactor slice that removes one architectural drift source at a time.
- **verification command:** `scripts/checks.sh --no-docflow`.
- **escalation threshold:** semantic logic remains duplicated or CLI-owned after one refactor slice.

### 4) dataflow grammar

- **sensor:** `gabion check` dataflow audit and synthesis outputs.
- **state artifact:** `artifacts/audit_reports/dataflow_report.md`, `artifacts/out/dataflow_raw.json`, and related synthesis outputs.
- **target predicate:** recurring cross-boundary bundles are reified (Protocol/dataclass) or explicitly documented with `# dataflow-bundle:` markers.
- **error signal:** bundle-tier violations, decision-surface lint findings, or failed check contract.
- **actuator:** reify bundles, tighten contracts, and remove sentinel control outcomes.
- **max correction step:** one bundle family reification (or equivalent Tier-3 documentation set) per loop.
- **verification command:** `mise exec -- python -m gabion check`.
- **escalation threshold:** unresolved Tier-2 bundle violations after one reification step.

### 5) baseline ratchets

- **sensor:** baseline delta guards and refresh tooling.
- **state artifact:** `baselines/docflow_compliance_baseline.json`, `artifacts/out/docflow_compliance.json`, `artifacts/out/docflow_compliance_delta.json`.
- **target predicate:** baseline refresh never masks new contradictions; ratchet movement is monotone toward stricter compliance.
- **error signal:** baseline refresh refusal, positive contradiction delta, or baseline drift inconsistent with current audit output.
- **actuator:** run guarded refresh via `scripts/refresh_baselines.py`; inspect and reduce deltas before update.
- **max correction step:** one baseline refresh transaction per loop with guard checks passing.
- **verification command:** `mise exec -- python scripts/refresh_baselines.py --docflow --timeout 600`.
- **escalation threshold:** guard rejects refresh twice consecutively for the same unresolved contradiction source.
