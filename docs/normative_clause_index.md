---
doc_revision: 9
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: normative_clause_index
doc_role: normative_index
doc_scope:
  - repo
  - ci
  - agents
  - governance
doc_authority: normative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - AGENTS.md#agent_obligations
  - glossary.md#contract
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  README.md#repo_contract: 2
  CONTRIBUTING.md#contributing_contract: 2
  AGENTS.md#agent_obligations: 2
  glossary.md#contract: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev2 (forward-remediation order, ci_watch failure-bundle durability, and enforced execution-coverage policy wording)."
  README.md#repo_contract: "Reviewed README.md rev2 (removed stale ASPF action-plan CLI/examples; continuation docs now state/delta only)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev2 (two-stage dual-sensor cadence, correction-unit validation stack, and strict-coverage trigger guidance)."
  AGENTS.md#agent_obligations: "Reviewed AGENTS.md rev2 (required validation stack, forward-remediation preference, and ci_watch failure-bundle triage guidance)."
  glossary.md#contract: "Dataflow tier references remain governed by glossary contract."
doc_sections:
  normative_clause_index: 2
doc_section_requires:
  normative_clause_index:
    - POLICY_SEED.md#policy_seed
    - README.md#repo_contract
    - CONTRIBUTING.md#contributing_contract
    - AGENTS.md#agent_obligations
    - glossary.md#contract
doc_section_reviews:
  normative_clause_index:
    POLICY_SEED.md#policy_seed:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Policy seed rev2 reviewed; governance obligations remain aligned."
    README.md#repo_contract:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Repo contract rev2 reviewed; command and artifact guidance remains aligned."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Contributor contract rev2 reviewed; dual-sensor cadence and correction gates remain aligned."
    AGENTS.md#agent_obligations:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Agent obligations rev2 reviewed; clause and cadence links remain aligned."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 2
      outcome: no_change
      note: "Dataflow tier clauses stay glossary-aligned."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="normative_clause_index"></a>
# Normative Clause Index

This document provides stable clause IDs for high-signal obligations that are
repeated across governance-facing documents. Other docs should summarize and
link to clause IDs instead of duplicating long-form normative prose.

## Canonical clauses

<a id="clause-lsp-first"></a>
### `NCI-LSP-FIRST` — LSP-first semantic core
- The language server is the semantic core.
- The CLI must remain a thin LSP client and must not duplicate core analysis.
- Canonical sources: `POLICY_SEED.md#policy_seed`, `CONTRIBUTING.md#contributing_contract`.

<a id="clause-actions-pinned"></a>
### `NCI-ACTIONS-PINNED` — Workflow action pinning
- Workflow actions must be pinned to full commit SHAs.
- Canonical sources: `POLICY_SEED.md#policy_seed`, `README.md#repo_contract`.

<a id="clause-actions-allowlist"></a>
### `NCI-ACTIONS-ALLOWLIST` — Workflow action allow-list
- Workflow actions must be allow-listed.
- Canonical source of allow-listed entries: `docs/allowed_actions.txt`.
- Canonical policy source: `POLICY_SEED.md#policy_seed`.

<a id="clause-dataflow-bundle-tiers"></a>
### `NCI-DATAFLOW-BUNDLE-TIERS` — Dataflow bundle tier obligations
- Recurring parameter bundles crossing function boundaries are type-level obligations.
- Tier-2 bundles must be reified before merge.
- Tier-3 bundles must be reified or documented with `# dataflow-bundle:`.
- Canonical sources: `[glossary.md#contract](../glossary.md#contract)`, `AGENTS.md#agent_obligations`, `CONTRIBUTING.md#contributing_contract`.


<a id="clause-shift-ambiguity-left"></a>
### `NCI-SHIFT-AMBIGUITY-LEFT` — Boundary-first ambiguity discharge
- Ambiguity must be classified at ingress and discharged before semantic-core execution.
- Reify ambiguity as explicit Protocol/Decision Protocol surfaces at boundaries.
- Semantic core modules must not add ad-hoc branch/sentinel/type-alternation shortcuts as first response.
- Semantic core modules must not introduce or preserve compatibility-layer wrappers, dual-shape bridges, or legacy fallback paths as steady-state behavior.
- Temporary compatibility is permitted only at boundary ingress with an explicit Decision Protocol plus lifecycle metadata (`actor`, `rationale`, `scope`, `start`, `expiry`, `rollback_condition`, `evidence_links`).
- Existing compatibility layers are remediation debt and must carry dated removal commitments.
- ACP/branchless/defensive-fallback violations discovered during simplification are transition signals for forward boundary reification.
- Rollback-first is disallowed by default; rollback is permitted only when forward remediation cannot preserve behavior or cannot converge.
- `# pragma: no cover` is allowed only when the branch is protected by a `never()` invariant.
- Enum exhaustiveness fallbacks should pair `never(...)` with `# pragma: no cover` so drift is explicit and correction is local.
- Canonical sources: `POLICY_SEED.md#policy_seed` (§4.8), `CONTRIBUTING.md#contributing_contract`, `AGENTS.md#agent_obligations`.

<a id="clause-baseline-ratchet"></a>
### `NCI-BASELINE-RATCHET` — Baseline ratchet integrity
- Baselines are ratchet checkpoints, not bypass levers.
- Do not refresh baselines to bypass positive deltas while gates are enabled.
- Canonical source: `CONTRIBUTING.md#contributing_contract`.


<a id="clause-deadline-timeout-propagation"></a>
### `NCI-DEADLINE-TIMEOUT-PROPAGATION` — Deadline carrier propagation
- Timeout/deadline tokens must propagate across CLI dispatch, LSP transport, and CI wrappers.
- Timeout recovery state must be emitted as deterministic machine-readable artifacts.
- Helper-level script functions that perform iterative parsing or subprocess orchestration are deadline-carrier surfaces and must check/propagate deadlines.
- Canonical sources: `POLICY_SEED.md#policy_seed`, `CONTRIBUTING.md#contributing_contract`.

<a id="clause-controller-adaptation-law"></a>
### `NCI-CONTROLLER-ADAPTATION-LAW` — Second-order controller adaptation law
- Adaptation is trigger-driven (parity instability, timeout resume churn, gate noise).
- Allowed moves are bounded to declared knobs and require telemetry/evidence links.
- Forbidden compensations include baseline-refresh bypasses and silent strictness downgrades.
- Canonical source: `POLICY_SEED.md#policy_seed`.

<a id="clause-override-lifecycle"></a>
### `NCI-OVERRIDE-LIFECYCLE` — Override lifecycle governance
- Override channels require machine-readable records with actor, rationale, scope, start, expiry, rollback condition, and evidence links.
- Expired or metadata-incomplete overrides fail governance gates.
- Post-override convergence requires consecutive clean runs before stabilization is declared.
- Canonical sources: `POLICY_SEED.md#policy_seed`, `docs/governance_rules.yaml`.

<a id="clause-controller-drift-lifecycle"></a>
### `NCI-CONTROLLER-DRIFT-LIFECYCLE` — Controller drift enforcement and override lifecycle
- Controller drift gates must enforce severity thresholds from governance rules and fail closed when blocking findings exceed policy and no valid override record exists.
- Active controller-drift overrides must use non-expired machine-readable lifecycle records and emit normalized diagnostics/telemetry fields.
- Canonical sources: `docs/governance_rules.yaml`, `scripts/ci_controller_drift_gate.py`, `POLICY_SEED.md#policy_seed`.

<a id="clause-command-maturity-parity"></a>
### `NCI-COMMAND-MATURITY-PARITY` — Command maturity, carrier, and parity governance
- `beta`/`production` command maturity requires LSP-carrier validation (`require_lsp_carrier`) and cannot treat direct execution as normative without valid override evidence + lifecycle record.
- Parity-governed commands (`parity_required`) must retain probe validation payload coverage and declared parity-ignore semantics.
- Canonical sources: `docs/governance_rules.yaml`, `src/gabion/commands/transport_policy.py`, `src/gabion/cli.py`, `POLICY_SEED.md#policy_seed`.

<a id="clause-dual-sensor-correction-loop"></a>
### `NCI-DUAL-SENSOR-CORRECTION-LOOP` — Temporal dual-sensor correction loop
- Agents must run local repro tooling and GitHub status-check monitoring concurrently when both are available.
- Agents must act on the first actionable failure signal and avoid serialized waiting when one sensor already produced actionable information.
- First actionable remote failure may preempt an in-progress local lane; agents must not wait for local completion once the remote signal is actionable.
- Bounded dependency-cluster publication is allowed before actionable signals exist; once actionable signals exist, agents must use one blocking-surface correction unit per push.
- Correction-unit validation stack must include workflow policy check, ambiguity-contract check, targeted pytest, and evidence-carrier drift refresh/check when tests or semantic surfaces changed.
- A correction unit is one failing signal (or a tightly coupled set) targeting one blocking surface; after local validation, stage/commit/push immediately.
- Multiple CI runs in flight are expected; fallout is handled by subsequent detect/correct/push iterations.
- Watcher-based failure forensics should collect deterministic bundles under `artifacts/out/ci_watch/run_<run_id>/`.
- If one sensor is unavailable, proceed with the available sensor and restore dual-sensor operation when possible.
- Applicability: mandatory for agents; recommended interoperability posture for contributors.
- Canonical sources: `AGENTS.md#agent_obligations`, `CONTRIBUTING.md#contributing_contract`, `docs/user_workflows.md#user_workflows`.


## Enforcement completeness ledger

Machine-readable clause-to-enforcement traceability is maintained in `docs/normative_enforcement_map.yaml`.
The map is exhaustive: every canonical clause listed above must appear as a top-level clause key,
including entries that are currently `partial` or `document-only`.
Policy checks validate canonical-clause completeness plus CI/workflow anchor integrity via
`scripts/policy_check.py --normative-map`.

## Usage rule

When referencing one of these obligations in `README.md`, `CONTRIBUTING.md`,
or `AGENTS.md`, use a short summary with direct clause links, for example:

- `NCI-LSP-FIRST` (`docs/normative_clause_index.md#clause-lsp-first`)
- `NCI-DATAFLOW-BUNDLE-TIERS` (`docs/normative_clause_index.md#clause-dataflow-bundle-tiers`)
- `NCI-SHIFT-AMBIGUITY-LEFT` (`docs/normative_clause_index.md#clause-shift-ambiguity-left`)
- `NCI-DUAL-SENSOR-CORRECTION-LOOP` (`docs/normative_clause_index.md#clause-dual-sensor-correction-loop`)
