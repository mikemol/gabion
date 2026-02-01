---
doc_revision: 2
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
---

# Contributing

Thanks for contributing. This repo enforces a strict execution policy to protect
self-hosted runners. Please read `POLICY_SEED.md` before making changes.

## Policy requirements (summary)
- Self-hosted workflows must trigger only on `push` to trusted branches.
- Self-hosted jobs must include `self-hosted`, `gpu`, and `local` labels.
- Self-hosted jobs must be guarded with
  `if: github.actor == github.repository_owner`.
- Workflow actions must be pinned to full commit SHAs and allow-listed.
- Workflows must declare `permissions: contents: read`.

## Contract handshake (normative)
Execution safety is governed by `POLICY_SEED.md`. Semantic correctness is governed
by `in/glossary.md`. Both contracts must be satisfied for any change to be valid.

## Guardrails
Install the advisory hooks:
```
scripts/install_policy_hooks.sh
```

Install the policy-check dependency (once):
```
mise exec -- python -m pip install pyyaml
```

Run the policy checks manually when editing workflows:
```
mise exec -- python scripts/policy_check.py --workflows
```

CI also runs `scripts/policy_check.py --workflows --posture`, which checks the
GitHub Actions settings for this repository.

## Doc front-matter (reader-only re-internment signal)
Markdown docs include a YAML front-matter block with:
- `doc_revision` (integer)
- `reader_reintern` (reader-only guidance)

When you make a conceptual change, bump `doc_revision`. This is a reader-only
signal to re-intern; it is not enforced by tooling or repo state.

## GPU tests and sandboxed environments
Some tests rely on CUDA/JAX GPU backends. If you are running in a sandboxed
environment, GPU access may require explicit sandbox escalation/privileged
execution. Without GPU access, CUDA backend init can fail. Do not mask these
failures; rerun with GPU access enabled, or explicitly select a CPU-only path
when that is the intent of the test run.

## Test runner helper (durable logs)
Use the helper to run pytest with durable logs in `artifacts/test_runs/`:
```
scripts/run_tests.sh [pytest args...]
```
If running under Codex or another sandboxed runner, increase the sandbox
command timeout before invoking the helper.

## Agda proofs
Agda checks run in a pinned container image. See `agda/README.md` for details.
Local run:
```
scripts/check_agda_container.sh
```

## Dataflow grammar invariant
Recurring parameter bundles are treated as type-level smells. Any bundle that
crosses function boundaries must be promoted to a dataclass (config or local
bundle), or explicitly documented.

The dataflow grammar audit enforces:
- Tier 1/2 bundles (crossing config or recurring across functions) must be
  promoted to a dataclass bundle.
- Tier 3 bundles (single-site) must either be promoted or documented in-place.

To document an intentional unbundled tuple, add a marker comment:
```
# dataflow-bundle: a1, a2, a3
```
and explain why it must remain unbundled.
