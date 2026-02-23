---
doc_revision: 97
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: contributing
doc_role: guide
doc_scope:
  - repo
  - contributors
  - workflows
  - tooling
doc_authority: normative
doc_requires:
  - README.md#repo_contract
  - AGENTS.md#agent_obligations
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/coverage_semantics.md#coverage_semantics
doc_reviewed_as_of:
  README.md#repo_contract: 1
  CONTRIBUTING.md#contributing_contract: 1
  AGENTS.md#agent_obligations: 1
  POLICY_SEED.md#policy_seed: 1
  glossary.md#contract: 1
  docs/coverage_semantics.md#coverage_semantics: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md rev1 (docflow audit now scans in/ by default); no conflicts with contributor scope."
  CONTRIBUTING.md#contributing_contract: "Self-review via Grothendieck analysis (cofibration/dedup/contrast); docflow now fails on missing GH references for SPPF-relevant changes; baseline guardrail + ci_cycle helper affirmed."
  AGENTS.md#agent_obligations: "Agent review discipline aligns with contributor workflow."
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev1 (mechanized governance default; branch/tag CAS + check-before-use constraints); no conflicts with this document's scope."
  glossary.md#contract: "Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline)."
  docs/coverage_semantics.md#coverage_semantics: "Reviewed docs/coverage_semantics.md#coverage_semantics v1 (glossary-lifted projection + explicit core anchors); contributor guidance unchanged."
doc_sections:
  contributing_contract: 1
doc_section_requires:
  contributing_contract:
    - README.md#repo_contract
    - AGENTS.md#agent_obligations
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
    - docs/coverage_semantics.md#coverage_semantics
doc_section_reviews:
  contributing_contract:
    README.md#repo_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Repo contract reviewed; contributor contract unchanged."
    AGENTS.md#agent_obligations:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Agent obligations reviewed; contributor contract unchanged."
    POLICY_SEED.md#policy_seed:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed reviewed; contributor contract unchanged."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Glossary contract reviewed; contributor contract unchanged."
    docs/coverage_semantics.md#coverage_semantics:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Coverage semantics reviewed; contributor contract unchanged."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_invariants:
  - policy_glossary_handshake
  - dataflow_grammar_invariant
  - tier3_documentation
  - lsp_first_invariant
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="contributing_contract"></a>
# Contributing

Thanks for contributing. This repo enforces a strict execution policy to protect
self-hosted runners. Please read `POLICY_SEED.md#policy_seed` before making changes.

## Contract handshake (normative)
Execution safety is governed by `POLICY_SEED.md#policy_seed`. Semantic correctness is
governed by `[glossary.md#contract](glossary.md#contract)`. Both contracts must be satisfied for any change to be
valid.

## Documentation review discipline (normative)
- `doc_reviewed_as_of` updates must reflect a real content review.
- Each update must include a non-empty `doc_review_notes` entry describing the
  dependency interaction.
- Mechanical version stamping is prohibited and treated as a governance breach.

## Architectural invariants (normative)
- **LSP-first invariant:** the language server is the semantic core; the CLI is
  a thin LSP client and must not import or reimplement analysis logic.
- **Single source of truth:** diagnostics and code actions must be derived from
  the server, not duplicated in client code.

## Optional governance framing
See `docs/doer_judge_witness.md` for a lightweight Doer/Judge/Witness workflow
that can be adopted when helpful.

## Cross-references (normative pointers)
- `README.md#repo_contract` defines project scope, status, and entry points.
- `AGENTS.md#agent_obligations` defines LLM/agent obligations and refusal rules.
- `POLICY_SEED.md#policy_seed` defines execution and CI safety constraints.
- `[glossary.md#contract](glossary.md#contract)` defines semantic meanings, axes, and commutation obligations.
- `docs/enforceable_rules_cheat_sheet.md#enforceable_rules_cheat_sheet` provides a day-to-day implementation checklist backed by canonical clauses.

## Dataflow grammar invariant
Recurring parameter bundles are treated as type-level obligations. Any bundle
that crosses function boundaries must be promoted to a Protocol (dataclass
config/local bundle), or explicitly documented in-place with:

```
# dataflow-bundle: a1, a2, a3
```

Tier-2 bundles must be reified before merge (see `[glossary.md#contract](glossary.md#contract)`).
Tier-3 bundles must be documented with `# dataflow-bundle:` or reified.

## Refactor Under Ambiguity Pressure (normative)
When ambiguity appears during refactors, contributors must apply the following
sequence in order:

1. **Step A — classify ambiguity at boundary vs core.** Determine whether the
   uncertainty belongs in an adapter/interface boundary or in the semantic core.
2. **Step B — create/extend a Protocol or Decision Protocol.** Reify the
   expected shape/decision surface as an explicit contract.
3. **Step C — normalize incoming values once (adapter layer).** Perform
   conversion/defaulting/disambiguation at ingress.
4. **Step D — remove downstream `isinstance`/`Optional`/sentinel checks.**
   Core flows must consume deterministic contract types, not repeated ambiguity
   guards.
5. **Step E — verify no new ambiguity signatures were introduced.** Confirm the
   change did not add new ambiguous unions, sentinel branches, or fallback-only
   control paths.

## Construction-first callback and decode seams (normative)
- Test seams must be DI-based; avoid runtime patch mutation and callable-probe fallback logic.
- Normalize optional boundary inputs once; internal decode/analysis paths should consume validated shapes.
- Do not use sentinel parse outcomes for control decisions in core flows.
- If an internal state is impossible after ingress validation, enforce it with `never()`.

## Sortedness Disclosure Ratchet (normative)
When sortedness is enforced, it must be treated as part of semantic behavior.

1. **Ingress:** normalize ordered carriers before entering functional code.
2. **Core:** preserve ordering through functional paths; do not actively sort a
   carrier that was already normalized.
3. **Egress:** enforce ordering at protocol/artifact boundaries before
   externalization; do not use serializer-level fallback sorting for already
   canonical carriers.
4. **Single-sort lifetime rule:** each carrier may consume active sorting at
   most once; subsequent active sorting is a contract violation.
5. **Disclosure:** every enforced sort must document:
   * sort key/function (or comparator tuple shape),
   * lexical vs non-lexical semantics, and
   * rationale for that ordering.
6. **Shared helper usage:** if a shared helper enforces ordering, document the
   sort contract once at the helper and reuse it; do not introduce conflicting
   callsite-specific semantics.

## Pull request checklist (normative)
- [ ] Describe where ambiguity was discharged and what deterministic contract
      replaced it (Protocol, Decision Protocol, or equivalent typed boundary).
- [ ] For each new/modified ordered surface, describe the sort key/function (or
      comparator), whether it is lexical/non-lexical, and why that ordering is
      semantically required.
- [ ] Confirm each ordered carrier consumes active sorting at most once, and
      that egress paths enforce order without serializer `sort_keys=True`
      fallback for canonical carriers.
- [ ] For governance/tooling changes, include PR template fields `controller impact`
      and `loop updated?`, and describe any controller-drift sensor/anchor changes.

## Branching model (normative)
- Routine work goes to `stage`; CI runs on every `stage` push and must be green.
- CI does not run on `main` pushes; PRs to `main` are for review and status checks.
- `main` is protected and receives changes via PRs from `stage`.
- Merges to `main` are regular merge commits (no squash or rebase).
- `stage` accumulates changes and may include merge commits from `main`.
- `next` mirrors `main` (no unique commits) and is updated after `main` merges.
- `release` mirrors `next` (no unique commits) and is updated only after `test-v*` succeeds.
- Test release tags are created via the `release-tag` workflow on `next`.
- Release tags are created via the `release-tag` workflow on `release` (no manual tags).
- `next` and `release` are automation-only branches. Human pushes are forbidden.
  The `mirror-next` and `promote-release` workflows update them.

## Workflow authoring (normative)
Workflow logic lives in `scripts/`. YAML files should only orchestrate steps
and invoke scripts rather than embed long inline logic.

## Current analysis coverage (non-binding)
These describe current coverage so contributors keep changes aligned:
- import resolution / symbol table
- alias-aware identity tracking
- fixed-point bundle propagation
- type-flow tightening audit
- constant-flow (dead knob) audit
- unused-argument pass detection

## Planned analysis expansions (non-binding)
- Protocol/dataclass synthesis (prototype scaffolding in `gabion.synthesis`)
- bundle-merge heuristics (fragmentation control, prototype scaffolding)

## SPPF tracking (non-binding)
Checklist nodes in `docs/sppf_checklist.md` map to GitHub issues (`GH-####`).
To keep issue state synced without CI write permissions, use commit trailers
like `SPPF: GH-17` or `Closes #17`.

Non-mutating lifecycle validation can run locally and in CI:

```
mise exec -- python scripts/sppf_sync.py --validate --only-when-relevant --range origin/stage..HEAD --require-state open --require-label done-on-stage --require-label status/pending-release
```

Mutating operations remain local-only. Use them explicitly when needed:

```
mise exec -- python scripts/sppf_sync.py --comment --range origin/stage..HEAD --label done-on-stage --label status/pending-release
```

Use `--close` when you want to close the issue on `stage`, or keep it open
until a merge to `main` with `Closes #17` (GitHub auto-closes on merge).

To automate this locally on `stage`, set `GABION_SPPF_SYNC=1` and re-run
`scripts/install_hooks.sh` to enable a pre-push sync (validation always; optional
comments + lifecycle labels when `GABION_SPPF_SYNC` is set).

### SPPF happy path (canonical)

```
# 1) include GH references in commits touching src/, in/, or docs/sppf_checklist.md
git commit -m "Implement X" -m "SPPF: GH-123"

# 2) run non-mutating validation
mise exec -- python scripts/sppf_sync.py --validate --only-when-relevant --range origin/stage..HEAD --require-state open --require-label done-on-stage --require-label status/pending-release

# 3) apply lifecycle labels locally (mutating)
mise exec -- python scripts/sppf_sync.py --comment --range origin/stage..HEAD --label done-on-stage --label status/pending-release

# 4) push stage
git push origin stage
```

## Issue lifecycle / kanban (normative)
Issues are not closed until a release containing the fix is published.
When work lands on `stage`, apply `done-on-stage` + `status/pending-release`.
On release, swap to `status/released` and close the issue.
Recommended status labels:
- `status/backlog`
- `status/in-progress`
- `status/pending-release`
- `status/released`

## Development setup
This project ships prototype analysis + refactor features. Treat outputs as
advisory outside this repo until the convergence checklist says otherwise.

Local environment (via `mise`):
```
mise install
mise exec -- python -m pip install -e .
```

Bootstrap everything (toolchain + deps + smoke test):
```
scripts/bootstrap.sh
```

Commands below assume the package is installed (editable) or `PYTHONPATH=src`.

Run the dataflow grammar audit (strict defaults):
```
mise exec -- python -m gabion check
```
`gabion check` writes a Markdown report to
`artifacts/audit_reports/dataflow_report.md` by default, and fails on type
ambiguities for this repo.
Violation enforcement remains independent of report generation.
Use `--baseline path/to/baseline.txt` to allowlist existing violations and
`--baseline-write` to generate/update the baseline (ratchet mode). Baseline
writes are a local, explicit action and should not run in CI.

For local edit→check loops, use a persistent resume checkpoint to keep caches warm:
```
mise exec -- python -m gabion check \
  --resume-checkpoint artifacts/audit_reports/dataflow_resume_checkpoint_local.json \
  --resume-on-timeout 1
```
Recommended practice:
- Keep the checkpoint path stable across local runs.
- Keep semantic identity knobs stable while iterating (strictness,
  allow-external/external filter mode, fingerprint seed/forest spec inputs).
- If outputs appear stale or semantics changed intentionally, delete the local
  checkpoint file and re-run cold once.

Run the dataflow grammar audit (prototype):
```
mise exec -- python -m gabion check --profile raw path/to/project
```
Defaults live in `gabion.toml` (see `[dataflow]`).
`in/` (inspiration) is excluded from enforcement there by default.
Use `--synthesis-plan` to emit a JSON plan and `--synthesis-report` to append a
summary section to the Markdown report. Use `--synthesis-protocols` to emit
dataclass stubs (prototype) for review.
Use `--refactor-plan` to append a per-bundle refactoring schedule and
`--refactor-plan-json` to emit the JSON plan.

Run audit + synthesis in one step (timestamped output under `artifacts/synthesis`):
```
mise exec -- python -m gabion synth path/to/project
```

Run the docflow audit (governance docs; `in/` is included for dependency resolution):
```
mise exec -- python -m gabion docflow
```
Control-loop declarations are normative: `docs/governance_control_loops.md#governance_control_loops` must declare every governed domain loop entry.

Run governance graph/status checks through the same CLI entrypoint:
```
mise exec -- python -m gabion sppf-graph
mise exec -- python -m gabion status-consistency --fail-on-violations
```

Docflow now fails when commits touching SPPF-relevant paths (`src/`, `in/`, or
`docs/sppf_checklist.md`) lack GH references in commit messages. Use `GH-####`
trailers or run `scripts/sppf_sync.py --comment` after adding references.

Note: docflow is a repo-local convenience feature. It is not a core Gabion
capability and is not intended to generalize beyond this repository.

Generate a synthesis plan from a JSON payload (prototype scaffolding):
```
mise exec -- python -m gabion synthesis-plan --input path/to/payload.json --output plan.json
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

Install git hooks (optional):
```
scripts/install_hooks.sh
```
To bypass hooks for a one-off command:
```
GABION_SKIP_HOOKS=1 git commit
```

## Locked dependencies
Dependencies are locked in `requirements.lock` (generated via `uv`).
Install the locked set and the editable package:
```
mise exec -- uv pip sync requirements.lock
mise exec -- uv pip install -e .
```
CI creates an explicit venv and installs the lock into it.
Regenerate the lockfile (after updating dependencies):
```
uv pip compile pyproject.toml --extra dev -o requirements.lock
```

## Editor integration (optional)
The VS Code extension stub lives in `extensions/vscode` and launches the
Gabion LSP server over stdio. It is a thin wrapper only.

Run the LSP smoke test (optional):
```
mise exec -- python scripts/lsp_smoke_test.py --root .
```

## Testing
LSP smoke test (pytest) requires `pygls` to be installed. It will be skipped
automatically if the dependency is missing.

Run tests:
```
mise exec -- pytest
```

Run coverage (advisory):
```
mise exec -- python -m pytest --cov=src/gabion --cov-report=term-missing
```
Coverage meaning is defined in `docs/coverage_semantics.md#coverage_semantics`.

Run tests with durable logs:
```
scripts/run_tests.sh
```

Logs are written under `artifacts/test_runs/` and are ignored by git.

Clean artifacts (keep `.gitkeep`):
```
scripts/clean_artifacts.sh
```

Run all checks in one sweep:
```
scripts/checks.sh
```

Reproduce the `ci.yml` workflow locally (checks + dataflow jobs):
```
scripts/ci_local_repro.sh
```

Run only one CI job locally when iterating:
```
scripts/ci_local_repro.sh --checks-only
scripts/ci_local_repro.sh --dataflow-only
```

Reproduce the PR dataflow status-check path locally:
```
scripts/ci_local_repro.sh --pr-dataflow-only --pr-base-sha <base-sha> --pr-head-sha <head-sha>
```
`--pr-base-sha`/`--pr-head-sha` are optional; when omitted, the script falls
back to environment values or local branch ancestry.
PR mode now also runs the governance template check and controller-drift audit.
For stricter parity with `.github/workflows/pr-dataflow-grammar.yml`, use:
```
scripts/ci_local_repro.sh --pr-dataflow-only --verify-pr-stage-ci --pr-stage-ci-timeout-minutes 70
```
When governance/template checks need PR body context, provide one with:
```
scripts/ci_local_repro.sh --pr-dataflow-only --pr-body-file <path-to-pr-body.md>
```

SPPF lifecycle validation in that script defaults to auto (run when GH auth is
available); use `--skip-sppf-sync` to bypass or `--run-sppf-sync` to require it.
GitHub-interactive steps prefer authenticated `gh` API calls (`gh auth status`).
If `gh` auth is unavailable, set `GH_TOKEN` or `GITHUB_TOKEN` explicitly for
non-interactive fallback paths.
`gh` does not mint a new ephemeral PAT here; it uses your existing local auth.
For long-running dataflow reproductions, set
`GABION_DATAFLOW_DEBUG_DUMP_INTERVAL_SECONDS=<seconds>` to emit periodic
state dumps; you can also send `SIGUSR1` to `gabion run-dataflow-stage`
to force an immediate dump. CI uses a 60-second interval by default. Unified
phase telemetry is written to:
- `artifacts/audit_reports/dataflow_phase_timeline.md` (human-readable table)
- `artifacts/audit_reports/dataflow_phase_timeline.jsonl` (machine-readable mirror)

Run CI checks (docflow omitted):
```
scripts/checks.sh --no-docflow
```

Run targeted checks:
```
scripts/checks.sh --docflow-only
scripts/checks.sh --dataflow-only
scripts/checks.sh --tests-only
```

Preview what will run:
```
scripts/checks.sh --list
```

Baseline refresh helpers:

```
mise exec -- python scripts/refresh_baselines.py --obsolescence
mise exec -- python scripts/refresh_baselines.py --annotation-drift
mise exec -- python scripts/refresh_baselines.py --ambiguity
mise exec -- python scripts/refresh_baselines.py --all
```

Baseline refresh guardrail (normative):
- **Never** refresh a baseline to bypass a ratchet. `refresh_baselines.py` will
  refuse to refresh when the corresponding gate is enabled and the delta is
  positive. Clear the delta via real fixes first, then refresh at a checkpoint.
- Use `--timeout <seconds>` if a baseline refresh risks hanging.

No-op CI cycle helper:

```
mise exec -- python scripts/ci_cycle.py --push --watch
```

CI watch helper:

```
mise exec -- python scripts/ci_watch.py --branch stage
```

By default this prefers active runs (in-progress/queued). If you want the most
recent run regardless of status, pass:

```
mise exec -- python scripts/ci_watch.py --branch stage --no-prefer-active
```

## Make targets (optional)
If you prefer `make`, the following targets wrap the scripts:
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
The GitHub-hosted workflow in `.github/workflows/ci.yml` runs:
- `gabion check`
- docflow audit
- pytest

It uses `mise` (via `gabion.toml`) to install the toolchain.

For push-driven `dataflow-grammar`, prefer warm caches:
- CI restores the previous same-branch `dataflow-report` artifact's resume
  checkpoint (`dataflow_resume_checkpoint_ci.json`) on a best-effort basis.
- `gabion run-dataflow-stage` emits resume metrics in logs/step-summary
  (`completed_paths`, `hydrated_paths`, `paths_parsed_after_resume`) so cache
  impact can be verified explicitly.
- Keep resume identity stable (forest spec / fingerprint seed / strictness
  knobs) when you expect high cache hit rates.

Pull requests also run `.github/workflows/pr-dataflow-grammar.yml`, which
uploads a dataflow report artifact and comments on same-repo PRs.

If `POLICY_GITHUB_TOKEN` is set, the CI workflow also runs the posture check
(`scripts/policy_check.py --posture`) on pushes.

## Policy guardrails
- Workflow changes must preserve the Prime Invariant in `POLICY_SEED.md#policy_seed`.
- Actions must be pinned to full commit SHAs and allow-listed.
- Self-hosted jobs must use the required labels and actor guard.
Allow-listed actions are defined in `docs/allowed_actions.txt` and enforced by
`scripts/policy_check.py`.

Workflow policy checks live in `scripts/policy_check.py` (requires `pyyaml`).
Run:
```
mise exec -- python -m pip install pyyaml
mise exec -- python scripts/policy_check.py --workflows
```
Posture checks require `POLICY_GITHUB_TOKEN` with admin read access:
```
mise exec -- python scripts/policy_check.py --posture
```

## Doc front-matter
Markdown docs include a YAML front-matter block with:
- `doc_revision` (integer)
- `reader_reintern` (reader-only guidance)

Bump `doc_revision` for conceptual changes.
