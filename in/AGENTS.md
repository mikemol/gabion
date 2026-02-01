---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
---

# AGENTS.md

This repository is governed by `POLICY_SEED.md`. Treat it as authoritative.

## Required behavior
- Read `POLICY_SEED.md` before proposing or applying changes.
- Do not weaken or bypass self-hosted runner protections.
- Keep workflow actions pinned to full commit SHAs and allow-listed.
- Run `python scripts/policy_check.py --workflows` when changing workflows.

## Local guardrails
- Install advisory hooks: `scripts/install_policy_hooks.sh`.
- Hooks are advisory; CI policy checks are authoritative.
- Use `mise exec -- python` for policy tooling so dependencies resolve as expected.

If any request conflicts with `POLICY_SEED.md`, stop and ask for guidance.
