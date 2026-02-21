---
doc_revision: 1
reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.
doc_change_protocol: POLICY_SEED.md#change_protocol
doc_requires:
  - POLICY_SEED.md#policy_seed
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 38
doc_review_notes:
  POLICY_SEED.md#policy_seed: Confirms the agent obligations align with the current execution policy contract.
doc_id: in_agents
doc_role: agent
doc_scope:
  - repo
  - governance
  - agents
doc_authority: normative
doc_sections:
  in_agents: 1
doc_section_requires:
  in_agents:
    - POLICY_SEED.md#policy_seed
doc_section_reviews:
  in_agents:
    POLICY_SEED.md#policy_seed:
      dep_version: 38
      self_version_at_review: 1
      outcome: no_change
      note: Agent guardrails remain consistent with POLICY_SEED execution policy requirements.
---

<a id="in_agents"></a>

# AGENTS.md#agent_obligations

This repository is governed by `POLICY_SEED.md#policy_seed`. Treat it as authoritative.

## Required behavior
- Read `POLICY_SEED.md#policy_seed` before proposing or applying changes.
- Do not weaken or bypass self-hosted runner protections.
- Keep workflow actions pinned to full commit SHAs and allow-listed.
- Run `python scripts/policy_check.py --workflows` when changing workflows.

## Local guardrails
- Install advisory hooks: `scripts/install_policy_hooks.sh`.
- Hooks are advisory; CI policy checks are authoritative.
- Use `mise exec -- python` for policy tooling so dependencies resolve as expected.

If any request conflicts with `POLICY_SEED.md#policy_seed`, stop and ask for guidance.
