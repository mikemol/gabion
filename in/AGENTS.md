---
doc_revision: 1
reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.
doc_change_protocol: POLICY_SEED.md#change_protocol
doc_requires:
  - POLICY_SEED.md#policy_seed
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 42
doc_review_notes:
  POLICY_SEED.md#policy_seed: Reviewed policy seed execution constraints (mechanized governance, explicit review-discipline invariant, and mise-based tooling expectations); scoped in/ delta remains consistent.
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
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Re-reviewed policy seed anchor; in/ agent deltas still match execution-safety and review-discipline requirements.
---

<a id="in_agents"></a>

# AGENTS.md#agent_obligations

This repository is governed by `POLICY_SEED.md#policy_seed`. Treat it as authoritative.

## Required behavior
- [delta] Read root `AGENTS.md#agent_obligations` and apply canonical directives before any in-scope changes.
- [delta] Run `mise exec -- python scripts/policy_check.py --workflows` for workflow edits touching `in/` support tooling.
- [delta] When adding scoped obligations, mark them with `[delta]` and avoid repeating canonical directives verbatim.

## Local guardrails
- Install advisory hooks: `scripts/install_policy_hooks.sh`.
- Hooks are advisory; CI policy checks are authoritative.
- [delta] Use `mise exec -- python` for policy tooling so dependencies resolve as expected.

If any request conflicts with `POLICY_SEED.md#policy_seed`, stop and ask for guidance.
