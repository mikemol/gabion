---
doc_revision: 71
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
  - README.md
  - AGENTS.md
  - POLICY_SEED.md
  - glossary.md
  - docs/coverage_semantics.md
doc_reviewed_as_of:
  README.md: 58
  AGENTS.md: 12
  POLICY_SEED.md: 28
  glossary.md: 17
  docs/coverage_semantics.md: 6
doc_change_protocol: "POLICY_SEED.md §6"
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

# Contributing

Thanks for contributing. This repo enforces a strict execution policy to protect
self-hosted runners. Please read `POLICY_SEED.md` before making changes.

## Contract handshake (normative)
Execution safety is governed by `POLICY_SEED.md`. Semantic correctness is
governed by `glossary.md`. Both contracts must be satisfied for any change to be
valid.

## Architectural invariants (normative)
- **LSP-first invariant:** the language server is the semantic core; the CLI is
  a thin LSP client and must not import or reimplement analysis logic.
- **Single source of truth:** diagnostics and code actions must be derived from
  the server, not duplicated in client code.

## Optional governance framing
See `docs/doer_judge_witness.md` for a lightweight Doer/Judge/Witness workflow
that can be adopted when helpful.

## Cross-references (normative pointers)
- `README.md` defines project scope, status, and entry points.
- `AGENTS.md` defines LLM/agent obligations and refusal rules.
- `POLICY_SEED.md` defines execution and CI safety constraints.
- `glossary.md` defines semantic meanings, axes, and commutation obligations.

## Dataflow grammar invariant
Recurring parameter bundles are treated as type-level obligations. Any bundle
that crosses function boundaries must be promoted to a Protocol (dataclass
config/local bundle), or explicitly documented in-place with:

```
# dataflow-bundle: a1, a2, a3
```

Tier-2 bundles must be reified before merge (see `glossary.md`).
Tier-3 bundles must be documented with `# dataflow-bundle:` or reified.

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
like `SPPF: GH-17` or `Closes #17`, then run:

```
scripts/sppf_sync.py --comment
```

Use `--close` when you want to close the issue on `stage`, or keep it open
until a merge to `main` with `Closes #17` (GitHub auto-closes on merge).

To automate this locally on `stage`, set `GABION_SPPF_SYNC=1` and re-run
`scripts/install_hooks.sh` to enable a pre-push sync (comments + `done-on-stage`
label) before pushing.

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
`gabion check` enforces violations even without `--report` output, and fails on
type ambiguities for this repo.
Use `--baseline path/to/baseline.txt` to allowlist existing violations and
`--baseline-write` to generate/update the baseline (ratchet mode). Baseline
writes are a local, explicit action and should not run in CI.

Run the dataflow grammar audit (prototype):
```
mise exec -- python -m gabion dataflow-audit path/to/project
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

Run the docflow audit (governance docs only):
```
mise exec -- python -m gabion docflow-audit
```

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
Coverage meaning is defined in `docs/coverage_semantics.md`.

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

Pull requests also run `.github/workflows/pr-dataflow-grammar.yml`, which
uploads a dataflow report artifact and comments on same-repo PRs.

If `POLICY_GITHUB_TOKEN` is set, the CI workflow also runs the posture check
(`scripts/policy_check.py --posture`) on pushes.

## Policy guardrails
- Workflow changes must preserve the Prime Invariant in `POLICY_SEED.md`.
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
