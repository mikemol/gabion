---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: doer_judge_witness
doc_role: protocol
doc_scope:
  - repo
  - governance
  - workflow
doc_authority: informative
doc_requires:
  - POLICY_SEED.md
  - glossary.md
  - AGENTS.md
  - CONTRIBUTING.md
  - README.md
doc_change_protocol: "POLICY_SEED.md §6"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

# Doer / Judge / Witness (Optional Framing)

This framing is an optional workflow lens. It separates responsibility for
changes into three roles so the governance layer remains explicit and testable.

## Roles

- **Doer:** produces candidate changes (code, docs, scripts).
- **Judge:** evaluates changes against `POLICY_SEED.md`, `glossary.md`, and CI.
- **Witness:** records outcomes and ensures the governance record stays coherent.

## Why this exists

The governance layer is a self-stabilizing system. Separating these roles helps
avoid conflating implementation with validation, and helps keep policy checks
repeatable under automation.

## How to use it (lightweight)

1. **Doer** proposes a change on `stage`.
2. **Judge** runs `scripts/checks.sh --no-docflow` and reviews any policy or
   dataflow violations.
3. **Witness** updates governance docs or the checklist if the change shifts
   the repo's obligations.

If you do not need this framing, ignore it.
