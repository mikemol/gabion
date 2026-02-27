---
doc_revision: 81
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: readme
doc_role: readme
doc_scope:
  - repo
  - overview
  - tooling
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - AGENTS.md#agent_obligations
  - CONTRIBUTING.md#contributing_contract
  - docs/normative_clause_index.md#normative_clause_index
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  AGENTS.md#agent_obligations: 2
  CONTRIBUTING.md#contributing_contract: 2
  docs/normative_clause_index.md#normative_clause_index: 2
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev2 (forward-remediation order, ci_watch failure-bundle durability, and enforced execution-coverage policy wording)."
  glossary.md#contract: "Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline)."
  AGENTS.md#agent_obligations: "Reviewed AGENTS.md rev2 (required validation stack, forward-remediation preference, and ci_watch failure-bundle triage guidance)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev2 (two-stage dual-sensor cadence, correction-unit validation stack, and strict-coverage trigger guidance)."
  docs/normative_clause_index.md#normative_clause_index: "Reviewed normative_clause_index rev2 (extended existing dual-sensor/shift-ambiguity/deadline clauses without introducing new clause IDs)."
doc_sections:
  repo_contract: 2
doc_section_requires:
  repo_contract:
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
    - AGENTS.md#agent_obligations
    - CONTRIBUTING.md#contributing_contract
    - docs/normative_clause_index.md#normative_clause_index
doc_section_reviews:
  repo_contract:
    POLICY_SEED.md#policy_seed:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Policy seed rev2 reviewed; governance obligations remain aligned."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 2
      outcome: no_change
      note: "Glossary contract reviewed; repo contract semantics unchanged."
    AGENTS.md#agent_obligations:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Agent obligations rev2 reviewed; clause and cadence links remain aligned."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Contributor contract rev2 reviewed; dual-sensor cadence and correction gates remain aligned."
    docs/normative_clause_index.md#normative_clause_index:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Clause index rev2 reviewed; canonical clause references remain aligned."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="repo_contract"></a>
# Gabion

[![CI](https://github.com/mikemol/gabion/actions/workflows/ci.yml/badge.svg)](https://github.com/mikemol/gabion/actions/workflows/ci.yml)

Gabion is an architectural linter that stabilizes loose parameters into
structural bundles. It discovers recurring argument groups in a Python codebase
and guides their reification into dataclass-based Protocols.

This repo contains scaffolded infrastructure plus a prototype dataflow audit.
Synthesis and refactoring exist as evolving prototypes and are intentionally
conservative.

## Why Gabion
- **Find implicit structure:** detect “dataflow grammar” bundles that repeatedly
  travel together across function boundaries ([`NCI-DATAFLOW-BUNDLE-TIERS`](docs/normative_clause_index.md#clause-dataflow-bundle-tiers)).
- **Refactor safely:** promote bundles into explicit dataclass Protocols.
- **Govern meaning:** enforce semantics via a normative glossary.

## Status
- CLI uses the LSP server as its semantic core.
- Transport maturity policy: `experimental`/`debug` commands may use direct diagnostics, while `beta`/`production` commands require validated LSP-carrier execution.
- A feature is not `beta`/`production` unless it has passed LSP-carrier validation.
- Dataflow grammar audit is implemented (prototype).
- Type-flow, constant-flow, and unused-argument smells are implemented (prototype).
- Refactor engine can rewrite signatures/call sites for targeted functions (prototype).
- Governance layer is active.

## Versioning (pre-1.0)
Gabion is pre-1.0. Until a 1.0 release, minor version bumps (0.x) may include
breaking changes; patch releases target fixes. Breaking changes will be called
out in release notes.

## Branching model
- `stage` is the integration branch for routine pushes; CI runs on `stage` pushes.
- `main` is protected and receives changes via PRs from `stage`.
- Merges to `main` are regular merge commits (no squash or rebase).
- `stage` accumulates changes and may include merge commits from `main` as it stays in sync.
- `next` mirrors `main` (no unique commits) and is updated after `main` merges.
- `release` mirrors `next` (no unique commits) and is updated only after `test-v*` succeeds.
- Tags are cut via the `release-tag` workflow on `next` (test) and `release` (prod).
- `next` and `release` are automation-only branches; `mirror-next` and
  `promote-release` keep them in sync.

## Convergence checklist
Bottom-up convergence targets live in `docs/sppf_checklist.md`.

## Governance addenda (optional)
See `docs/doer_judge_witness.md` for optional role framing.

## Agent Control Surface
The canonical agent instruction graph is emitted by docflow at:
- `artifacts/out/agent_instruction_drift.json` (machine-readable)
- `artifacts/audit_reports/agent_instruction_drift.md` (human-readable)

Use this graph as the single source of truth for mandatory directive deduping,
scoped-delta validation, and precedence/conflict checks.

## Non-goals (for now)
- Docflow is a repo-local convenience feature, not a Gabion product feature.
- Public-API compatibility shims for refactors are not yet implemented.
- Multi-language support is out of scope (Python-first).

## Quick start
Need practical remediation loops? See `docs/user_workflows.md#user_workflows`.

Install toolchain with `mise` (once):
```
mise install
mise trust --yes
```

`mise trust --yes` marks this repo's `mise.toml` as trusted so local `mise exec`
matches CI behavior. CI sets `MISE_TRUSTED_CONFIG_PATHS=${{ github.workspace }}`
in workflows, so the workspace is already trusted there.

Install from source (editable):
```
mise exec -- python -m pip install -e .
```
Dependencies are locked in `requirements.lock` (generated via `uv`).
CI installs from the lockfile inside an explicit venv to prevent drift.

Install git hooks (optional):
```
scripts/install_hooks.sh
```

Commands below assume the package is installed (editable) or `PYTHONPATH=src`.

Run the dataflow grammar audit (strict defaults):
```
mise exec -- python -m gabion check run
```
`gabion check run` writes a Markdown report to
`artifacts/audit_reports/dataflow_report.md` by default.
Violation enforcement remains independent of report generation.
Use `--baseline path/to/baseline.txt --baseline-mode enforce` to ratchet existing
violations and `--baseline path/to/baseline.txt --baseline-mode write` to
generate/update the baseline file
([`NCI-BASELINE-RATCHET`](docs/normative_clause_index.md#clause-baseline-ratchet)).

For iterative local cleanup, use ASPF snapshot + delta as the continuation
substrate:
```bash
mise exec -- python -m gabion check run \
  --aspf-state-json artifacts/out/aspf_state/session-a/0001_check-run.snapshot.json \
  --aspf-delta-jsonl artifacts/out/aspf_state/session-a/0001_check-run.delta.jsonl
```

### ASPF Cross-Execution Equivalence + Cross-Script Handoff (phase 1)

Phase-1 supports both:
- trace/equivalence/opportunity artifacts (`--aspf-trace-json`, `--aspf-import-trace`)
- serialized ASPF state objects for cross-script reuse (`--aspf-state-json`, `--aspf-import-state`)
- append-only mutation ledgers (`--aspf-delta-jsonl`)

Capture a baseline lane as a first-class state object:
```bash
mise exec -- python -m gabion check run \
  --aspf-state-json artifacts/out/aspf_state/session-a/0001_check-run.snapshot.json \
  --aspf-delta-jsonl artifacts/out/aspf_state/session-a/0001_check-run.delta.jsonl \
  --aspf-semantic-surface groups_by_path \
  --aspf-semantic-surface decision_surfaces \
  --aspf-semantic-surface rewrite_plans \
  --aspf-semantic-surface violation_summary
```

Run another lane and import prior state for glued equivalence reasoning:
```bash
mise exec -- python -m gabion check run \
  --aspf-state-json artifacts/out/aspf_state/session-a/0002_check-run.snapshot.json \
  --aspf-delta-jsonl artifacts/out/aspf_state/session-a/0002_check-run.delta.jsonl \
  --aspf-import-state artifacts/out/aspf_state/session-a/0001_check-run.snapshot.json \
  --aspf-opportunities-json artifacts/out/aspf_opportunities.json
```

For script orchestration, use `scripts/aspf_handoff.py` to reserve state paths
and cumulative imports through `artifacts/out/aspf_handoff_manifest.json`.
The repo scripts `scripts/checks.sh`, `scripts/ci_local_repro.sh`,
`scripts/refresh_baselines.py`, and `scripts/audit_snapshot.sh` now enable this
handoff loop by default (disable with `--no-aspf-handoff`).

Phase-1 ASPF outputs:
- `artifacts/out/aspf_trace.json`
- `artifacts/out/aspf_equivalence.json`
- `artifacts/out/aspf_opportunities.json`
- `artifacts/out/aspf_state/<session>/<seq>_<step>.snapshot.json`
- `artifacts/out/aspf_state/<session>/<seq>_<step>.delta.jsonl`
- `artifacts/out/aspf_handoff_manifest.json`

See `docs/aspf_execution_fibration.md` for surface/witness/handoff details.

Legacy timeout/resume checkpoint flags were removed from `gabion check run`.
Use ASPF state import (`--aspf-import-state`) plus per-run snapshot/delta
artifacts for continuation and progress tracking.

Run the dataflow grammar audit in raw profile mode (prototype):
```
mise exec -- python -m gabion check raw -- path/to/project
```
Modality commands replace the legacy check-flag matrix:
```
mise exec -- python -m gabion check obsolescence delta --baseline baselines/test_obsolescence_baseline.json
mise exec -- python -m gabion check annotation-drift baseline-write --baseline baselines/test_annotation_drift_baseline.json
mise exec -- python -m gabion check ambiguity state
```
Delta-analysis fast path (single analysis pass + gate-only follow-up):
```
mise exec -- python -m gabion check delta-bundle
mise exec -- python -m gabion check delta-gates
```
Removed top-level wrappers:
- `gabion delta-state-emit` -> `gabion check delta-bundle`
- `gabion delta-triplets` -> `gabion check delta-gates`

Global runtime controls are now:
- `--timeout <duration>` (for example `750ms`, `2s`, `1m30s`)
- `--carrier {lsp|direct}`
- `--carrier-override-record <path>`

Repo defaults are driven by `gabion.toml` (see `[dataflow]`).
By default, `in/` (inspiration) is excluded from enforcement there.
Use `--synthesis-plan` to emit a JSON plan and `--synthesis-report` to append a
summary section to the Markdown report. Use `--synthesis-protocols` to emit
dataclass stubs (prototype) for review, or add
`--synthesis-protocols-kind protocol` for typing.Protocol stubs.
Use `--refactor-plan` to append a per-bundle refactoring schedule and
`--refactor-plan-json` to emit the JSON plan.

Generate protocol refactor edits (prototype):
```
mise exec -- python -m gabion refactor-protocol \
  --protocol-name BundleProtocol \
  --bundle a --bundle b \
  --target-path path/to/module.py \
  --target-function foo
```

Run audit + synthesis in one step (timestamped output under `artifacts/synthesis`):
```
mise exec -- python -m gabion synth path/to/project
```

Run the docflow audit (governance docs; `in/` is included for dependency resolution):
```
mise exec -- python -m gabion docflow
```
Run governance graph/status checks through the same CLI entrypoint:
```
mise exec -- python -m gabion sppf-graph
mise exec -- python -m gabion status-consistency --fail-on-violations
```

Compute a normative-docs versus code/tooling symmetric-difference report:
```
mise exec -- python -m gabion normative-symdiff --root .
```
Default artifacts:
- `artifacts/out/normative_symdiff.json`
- `artifacts/audit_reports/normative_symdiff.md`

Note: docflow is a repo-local convenience feature. It is not a core Gabion
capability and is not intended to generalize beyond this repository.

Generate a synthesis plan from a JSON payload (prototype scaffolding):
```
mise exec -- python -m gabion synthesis-plan --input path/to/payload.json --output plan.json
```
Example payload:
```json
{
  "bundles": [
    { "bundle": ["ctx", "config"], "tier": 2 }
  ],
  "field_types": {
    "ctx": "Context",
    "config": "Config"
  }
}
```
Payload schema: `docs/synthesis_payload.md`.

Capture an audit snapshot (reports + DOT graph under `artifacts/`):
```
scripts/audit_snapshot.sh
```
Snapshots now include a synthesis plan JSON and protocol stub file.
Show the latest snapshot paths:
```
scripts/latest_snapshot.sh
```

## Editor integration
The VS Code extension stub lives in `extensions/vscode` and launches the
Gabion LSP server over stdio. It is a thin wrapper only.
For end-to-end editor + CLI iteration guidance, see
`docs/user_workflows.md#user_workflows`.

## Quick commands (make)
```
make bootstrap
make check
make check-ci
make test
make test-logs
make clean-artifacts
make docflow
make dataflow
make lsp-smoke
make audit-snapshot
make audit-latest
```

## CI
GitHub-hosted CI runs `gabion check run`, docflow audit, and pytest using `mise`
as defined in `.github/workflows/ci.yml`.
If `POLICY_GITHUB_TOKEN` is set, the posture check also runs on pushes.

The `dataflow-grammar` job now performs ASPF handoff by default across staged
invocations, emitting per-step snapshot/delta/action-plan artifacts under
`artifacts/out/aspf_state/` and recording cumulative imports in
`artifacts/out/aspf_handoff_manifest.json`.

Cross-step effectiveness can be audited in CI logs and summaries from the
handoff manifest (`sequence`, `import_state_paths`, `status`) plus generated
action-plan artifacts. Unified all-phase progress telemetry is also persisted to:
- `artifacts/audit_reports/dataflow_phase_timeline.md`
- `artifacts/audit_reports/dataflow_phase_timeline.jsonl`

Allow-listed actions are defined in `docs/allowed_actions.txt` and governed by
[`NCI-ACTIONS-ALLOWLIST`](docs/normative_clause_index.md#clause-actions-allowlist).

Pull requests also get a dataflow-grammar report artifact (and a comment on
same-repo PRs) via `.github/workflows/pr-dataflow-grammar.yml`.

## GitHub Action (redistributable)
A composite action wrapper lives at `.github/actions/gabion`.
It installs Gabion via pip and runs `gabion check run` (or another subcommand).
See `.github/actions/gabion/README.md#repo_contract` for usage and pinning guidance.
Example workflow (with pinned SHA placeholders):
`docs/workflows/gabion_action_example.yml`.
Pinning guide: `docs/pinning_actions.md`.

## Architecture (planned shape)
- **LSP-first:** [`NCI-LSP-FIRST`](docs/normative_clause_index.md#clause-lsp-first).
  Editor integrations remain thin wrappers over the same server.
- **Analysis:** import resolution, alias-aware identity tracking, fixed-point
  bundle propagation, and tiering. Type-flow, constant-flow, and unused-argument
  audits are part of the prototype coverage.
- **Reports:** Mermaid/DOT graph outputs and a violations list from the audit.
- **Synthesis:** Protocol generation, bundle-merge heuristics, and refactoring
  assistance (callee-first/topological schedule).

See `in/` for design notes and the prototype audit script.

## Governance
This repository is governed by two co-equal contracts:
- `POLICY_SEED.md#policy_seed` (execution and CI safety)
- `[glossary.md#contract](glossary.md#contract)` (semantic meanings and commutation obligations)

LLM/agent behavior is governed by `AGENTS.md#agent_obligations`.

## Cross-references
- `CONTRIBUTING.md#contributing_contract` defines workflow guardrails and dataflow grammar rules.
- `AGENTS.md#agent_obligations` defines LLM/agent obligations.
- `POLICY_SEED.md#policy_seed` defines execution and CI safety constraints.
- `docs/normative_clause_index.md#normative_clause_index` defines stable clause IDs for repeated obligations.
- `[glossary.md#contract](glossary.md#contract)` defines semantic meanings, axes, and commutation obligations.
- `docs/enforceable_rules_cheat_sheet.md#enforceable_rules_cheat_sheet` provides an authoritative-by-proxy operator reference with clause-level source links.
- `docs/governance_loop_matrix.md#governance_loop_matrix` maps governance gate entrypoints, artifacts, thresholds, and override controls in one matrix.

## License
Apache-2.0. See `LICENSE`.
