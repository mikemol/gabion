---
doc_revision: 31
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
Synthesis and automated refactoring are intentionally staged for later
integration.

## Why Gabion
- **Find implicit structure:** detect “dataflow grammar” bundles that repeatedly
  travel together across function boundaries.
- **Refactor safely:** promote bundles into explicit dataclass Protocols.
- **Govern meaning:** enforce semantics via a normative glossary.

## Status
- CLI uses the LSP server as its semantic core.
- Dataflow grammar audit is implemented (prototype).
- Type-flow, constant-flow, and unused-argument smells are implemented (prototype).
- Governance layer is active.

## Branching model
- `stage` is the integration branch for routine pushes; CI runs on every push.
- `main` is protected and receives changes via PRs from `stage`.
- Merge commits are allowed; merges to `main` should be regular merges (no squash).
- `stage` accumulates changes and may include merge commits from `main` as it stays in sync.

## Convergence checklist
Bottom-up convergence targets live in `docs/sppf_checklist.md`.

## Governance addenda (optional)
See `docs/doer_judge_witness.md` for optional role framing.

## Non-goals (for now)
- Docflow is a repo-local convenience feature, not a Gabion product feature.
- Automated refactoring is not yet in scope; analysis is advisory only.
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

Run the dataflow grammar audit (prototype):
```
mise exec -- python -m gabion dataflow-audit path/to/project
```
Repo defaults are driven by `gabion.toml` (see `[dataflow]`).
By default, `in/` (inspiration) is excluded from enforcement there.

Run the docflow audit (governance docs only):
```
mise exec -- python -m gabion docflow-audit
```

Note: docflow is a repo-local convenience feature. It is not a core Gabion
capability and is not intended to generalize beyond this repository.

Capture an audit snapshot (reports + DOT graph under `artifacts/`):
```
scripts/audit_snapshot.sh
```
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

Pull requests also get a dataflow-grammar report artifact (and a comment on
same-repo PRs) via `.github/workflows/pr-dataflow-grammar.yml`.

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
TBD.
