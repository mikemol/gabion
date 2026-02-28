---
doc_revision: 4
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
  README.md#repo_contract: 2
  AGENTS.md#agent_obligations: 2
  POLICY_SEED.md#policy_seed: 2
  CONTRIBUTING.md#contributing_contract: 2
  glossary.md#contract: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md rev2 (removed stale ASPF action-plan CLI/examples; continuation docs now state/delta only)."
  AGENTS.md#agent_obligations: "Reviewed AGENTS.md rev2 (required validation stack, forward-remediation preference, and ci_watch failure-bundle triage guidance)."
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev2 (forward-remediation order, ci_watch failure-bundle durability, and enforced execution-coverage policy wording)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev2 (two-stage dual-sensor cadence, correction-unit validation stack, and strict-coverage trigger guidance)."
  glossary.md#contract: "Glossary contract reviewed; loop predicates remain semantically coherent with enforcement terms."
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
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Repo contract rev2 reviewed; command and artifact guidance remains aligned."
    AGENTS.md#agent_obligations:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Agent obligations rev2 reviewed; clause and cadence links remain aligned."
    POLICY_SEED.md#policy_seed:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed rev2 reviewed; governance obligations remain aligned."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Contributor contract rev2 reviewed; dual-sensor cadence and correction gates remain aligned."
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
  - execution coverage
  - GitHub status API process
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

This document defines the normalized control-loop registry for governed normative domains.
It is coupled to `README.md#repo_contract`, `AGENTS.md#agent_obligations`,
`POLICY_SEED.md#policy_seed`, `CONTRIBUTING.md#contributing_contract`, and
`glossary.md#contract`.

## Correction modes

- `advisory`: emit diagnostics and summary only; do not block execution.
- `ratchet`: allow bounded movement only toward tighter policy; block regressions.
- `hard-fail`: block any blocking-threshold regression immediately.

## Transition criteria

Loop transitions are governed by `warning_threshold` and `blocking_threshold` values in
`docs/governance_rules.yaml`:

1. `advisory -> ratchet` when warning-threshold events recur and no override token is recorded.
2. `ratchet -> hard-fail` when blocking-threshold events recur under ratchet mode.
3. `hard-fail -> ratchet|advisory` only with an explicit override token and annotated rationale.

## Bounded-step correction rules

1. Baseline writes require explicit `--write-*` flags.
2. Strictness reduction requires both:
- `GABION_POLICY_OVERRIDE_TOKEN`
- `GABION_POLICY_OVERRIDE_RATIONALE`
3. Delta gates use shared severity mapping from `docs/governance_rules.yaml`; ad-hoc threshold branching is prohibited.

## Normalized loop schema

Each loop entry must define:

- `sensor`
- `state artifact`
- `target predicate`
- `error signal`
- `actuator`
- `max correction step`
- `verification command`
- `escalation threshold`

## First-order loop registry

### 1) security/workflows

- **sensor:** workflow YAML and posture probes from `python -m scripts.policy_check --workflows` and `python -m scripts.policy_check --posture`.
- **state artifact:** `.github/workflows/*.yml` and policy posture snapshots under `artifacts/out/policy_posture*.json`.
- **target predicate:** all workflow actions are pinned to allow-listed full SHAs and self-hosted trust boundaries satisfy the prime invariant.
- **error signal:** policy-check non-zero exit or posture violations.
- **actuator:** patch workflow files, refresh allow-list expectations, and rerun policy checks.
- **max correction step:** one workflow guardrail patchset per iteration.
- **verification command:** `mise exec -- python -m scripts.policy_check --workflows && mise exec -- python -m scripts.policy_check --posture`.
- **escalation threshold:** any prime-invariant contradiction or unresolved posture failure after one correction step.

### 2) docs/docflow

- **sensor:** docflow compliance emitters in `src/gabion/tooling/governance_audit.py` and `gabion docflow`.
- **state artifact:** `artifacts/out/docflow_compliance.json`, `artifacts/out/docflow_compliance_delta.json`, and `artifacts/audit_reports/docflow_compliance.md`.
- **target predicate:** normative docs satisfy frontmatter/review invariants, contradiction delta remains zero, and every required loop domain is declared.
- **error signal:** docflow violations or positive contradiction delta.
- **actuator:** update governance docs, references, review pins, and loop declarations.
- **max correction step:** one coherent doc revision cycle.
- **verification command:** `mise exec -- python -m gabion docflow --fail-on-violations`.
- **escalation threshold:** repeated docflow violation after one coherent cycle.

### 3) LSP architecture

Clause links: [`NCI-LSP-FIRST`](docs/normative_clause_index.md#clause-lsp-first), [`NCI-COMMAND-MATURITY-PARITY`](docs/normative_clause_index.md#clause-command-maturity-parity).

- **sensor:** server/CLI split checks and governance contract conformance checks.
- **state artifact:** `src/gabion/server.py`, `src/gabion/cli.py`, and audit outputs under `artifacts/audit_reports/`.
- **target predicate:** LSP-first invariant holds; semantic logic remains server-owned and CLI stays thin.
- **error signal:** architecture drift findings or policy-contradiction findings.
- **actuator:** migrate semantics from CLI to server boundaries and restore thin-client behavior.
- **max correction step:** one refactor slice per drift source.
- **verification command:** `scripts/checks.sh --no-docflow`.
- **escalation threshold:** semantic duplication remains after one refactor slice.

### 4) dataflow grammar

- **sensor:** `gabion check` dataflow audit and synthesis outputs.
- **state artifact:** `artifacts/audit_reports/dataflow_report.md`, `artifacts/out/dataflow_raw.json`, and synthesis outputs.
- **target predicate:** recurring cross-boundary bundles are reified or documented via `# dataflow-bundle:` markers.
- **error signal:** tier violations, decision-surface lint findings, or failed check contracts.
- **actuator:** reify bundles, tighten contracts, and remove sentinel control outcomes.
- **max correction step:** one bundle-family reification or equivalent Tier-3 documentation set.
- **verification command:** `mise exec -- python -m gabion check`.
- **escalation threshold:** unresolved Tier-2 violations after one reification step.

### 5) baseline ratchets

- **sensor:** baseline delta guards and refresh tooling.
- **state artifact:** `baselines/docflow_compliance_baseline.json`, `artifacts/out/docflow_compliance.json`, `artifacts/out/docflow_compliance_delta.json`.
- **target predicate:** baseline refresh does not mask new contradictions and ratchet movement is monotone toward stricter compliance.
- **error signal:** refresh refusal, positive contradiction delta, or drift inconsistent with current audit output.
- **actuator:** run guarded refresh via `python -m scripts.refresh_baselines`, inspect deltas, then update.
- **max correction step:** one guarded baseline refresh transaction.
- **verification command:** `mise exec -- python -m scripts.refresh_baselines --docflow --timeout 600`.
- **escalation threshold:** guard rejects refresh twice for the same unresolved source.

### 6) execution coverage

Clause links: [`NCI-DUAL-SENSOR-CORRECTION-LOOP`](docs/normative_clause_index.md#clause-dual-sensor-correction-loop), [`NCI-SHIFT-AMBIGUITY-LEFT`](docs/normative_clause_index.md#clause-shift-ambiguity-left).

- **sensor:** strict coverage gate (`pytest --cov=src/gabion --cov-branch --cov-fail-under=100`) in local repro and CI.
- **state artifact:** terminal coverage report plus `artifacts/test_runs/coverage.xml`.
- **target predicate:** total and branch coverage remain at 100% with no uncovered semantic-core regressions.
- **error signal:** any coverage drop below gate.
- **actuator:** open a dedicated fix-forward correction unit to cover the uncovered branches/lines; prefer simplification/reification where feasible before adding broad compatibility behavior.
- **max correction step:** one coverage-blocking surface per push.
- **verification command:** `mise exec -- python -m pytest --cov=src/gabion --cov-branch --cov-report=term-missing --cov-fail-under=100`.
- **escalation threshold:** repeated coverage failure for the same uncovered surface after one correction step.

### 7) GitHub status API process

Clause links: [`NCI-DUAL-SENSOR-CORRECTION-LOOP`](docs/normative_clause_index.md#clause-dual-sensor-correction-loop), [`NCI-CONTROLLER-ADAPTATION-LAW`](docs/normative_clause_index.md#clause-controller-adaptation-law).

- **sensor:** API invocation outcomes for `gh api` / `gh run view --json` monitoring calls.
- **state artifact:** CI-watch summaries and local monitoring transcripts.
- **target predicate:** status monitoring remains low-noise, high-density, and error-free under workstream cadence bounds.
- **error signal:** any API transport/auth/rate-limit error during monitoring or forensics.
- **actuator:** remediate the API-access process itself (query consolidation, cadence control, fallback wiring, and parse robustness) in a dedicated correction unit; backoff-only responses are insufficient.
- **max correction step:** one monitoring-process patchset per blocking API error class.
- **verification command:** one-call-per-interval high-density status query plus local parse checks over returned JSON.
- **escalation threshold:** repeated API error for the same access path after one process-remediation step.

## Second-order controller loop (cybernetic meta-loop)

Clause links: [`NCI-CONTROLLER-ADAPTATION-LAW`](docs/normative_clause_index.md#clause-controller-adaptation-law), [`NCI-OVERRIDE-LIFECYCLE`](docs/normative_clause_index.md#clause-override-lifecycle), [`NCI-CONTROLLER-DRIFT-LIFECYCLE`](docs/normative_clause_index.md#clause-controller-drift-lifecycle).

Second-order governance closes drift between normative anchors and enforcement scripts.
This loop governs first-order loop integrity and prevents controller drift.

- **sensor:** `scripts/governance_controller_audit.py` reading controller anchors and command references.
- **state artifact:** `artifacts/out/controller_drift.json`.
- **target predicate:** every normative controller anchor has a live enforcing check, and every enforcing check has a normative anchor.
- **error signal:** policy clauses without checks, checks without anchors, contradictory anchors, or stale command references.
- **actuator:** update normative anchors and workflow/script wiring together in one patchset.
- **max correction step:** one coherent governance patchset across policy + workflow + tooling.
- **verification command:** `mise exec -- python scripts/governance_controller_audit.py --out artifacts/out/controller_drift.json`.
- **escalation threshold:** repeated high-severity controller drift after one correction step.
