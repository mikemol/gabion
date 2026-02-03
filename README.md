---
doc_revision: 58
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: readme
doc_role: readme
doc_scope:
  - repo
  - overview
  - tooling
doc_authority: informative
doc_requires:
  - POLICY_SEED.md
  - glossary.md
  - AGENTS.md
  - CONTRIBUTING.md
doc_reviewed_as_of:
  POLICY_SEED.md: 28
  glossary.md: 14
  AGENTS.md: 12
  CONTRIBUTING.md: 70
doc_change_protocol: "POLICY_SEED.md §6"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

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
  travel together across function boundaries.
- **Refactor safely:** promote bundles into explicit dataclass Protocols.
- **Govern meaning:** enforce semantics via a normative glossary.

## Status
- CLI uses the LSP server as its semantic core.
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

## Non-goals (for now)
- Docflow is a repo-local convenience feature, not a Gabion product feature.
- Public-API compatibility shims for refactors are not yet implemented.
- Multi-language support is out of scope (Python-first).

## Quick start
Install toolchain with `mise` (once):
```
mise install
```

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
mise exec -- python -m gabion check
```
`gabion check` enforces violations even without `--report` output.
Use `--baseline path/to/baseline.txt` to ratchet existing violations and
`--baseline-write` to generate/update the baseline file.

Run the dataflow grammar audit (prototype):
```
mise exec -- python -m gabion dataflow-audit path/to/project
```
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
GitHub-hosted CI runs `gabion check`, docflow audit, and pytest using `mise`
as defined in `.github/workflows/ci.yml`.
If `POLICY_GITHUB_TOKEN` is set, the posture check also runs on pushes.

Allow-listed actions are defined in `docs/allowed_actions.txt`.

Pull requests also get a dataflow-grammar report artifact (and a comment on
same-repo PRs) via `.github/workflows/pr-dataflow-grammar.yml`.

## GitHub Action (redistributable)
A composite action wrapper lives at `.github/actions/gabion`.
It installs Gabion via pip and runs `gabion check` (or another subcommand).
See `.github/actions/gabion/README.md` for usage and pinning guidance.
Example workflow (with pinned SHA placeholders):
`docs/workflows/gabion_action_example.yml`.
Pinning guide: `docs/pinning_actions.md`.

## Architecture (planned shape)
- **LSP-first:** the language server is the semantic core; the CLI is a thin
  LSP client. Editor integrations remain thin wrappers over the same server.
  The server is the single source of truth for diagnostics and code actions.
- **Analysis:** import resolution, alias-aware identity tracking, fixed-point
  bundle propagation, and tiering. Type-flow, constant-flow, and unused-argument
  audits are part of the prototype coverage.
- **Reports:** Mermaid/DOT graph outputs and a violations list from the audit.
- **Synthesis:** Protocol generation, bundle-merge heuristics, and refactoring
  assistance (callee-first/topological schedule).

See `in/` for design notes and the prototype audit script.

## Governance
This repository is governed by two co-equal contracts:
- `POLICY_SEED.md` (execution and CI safety)
- `glossary.md` (semantic meanings and commutation obligations)

LLM/agent behavior is governed by `AGENTS.md`.

## Cross-references
- `CONTRIBUTING.md` defines workflow guardrails and dataflow grammar rules.
- `AGENTS.md` defines LLM/agent obligations.
- `POLICY_SEED.md` defines execution and CI safety constraints.
- `glossary.md` defines semantic meanings, axes, and commutation obligations.

## License
Apache-2.0. See `LICENSE`.
