---
doc_revision: 3
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: influence_index
doc_role: index
doc_scope:
  - repo
  - governance
  - planning
  - documentation
doc_authority: informative
doc_requires:
  - POLICY_SEED.md
  - glossary.md
  - CONTRIBUTING.md
  - README.md
doc_reviewed_as_of:
  POLICY_SEED.md: 21
  glossary.md: 9
  CONTRIBUTING.md: 68
  README.md: 58
doc_change_protocol: "POLICY_SEED.md §6"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

# Influence Index (`in/` → `out/`)

This index records which inbound documents (`in/`) have been reviewed, and how
(or whether) they have been reflected in `out/`, `docs/`, or the checklist. It
is a lightweight bridge between the inbox and the rest of the repo.

Normative anchors: `POLICY_SEED.md`, `glossary.md`, `CONTRIBUTING.md`, `README.md`.

Status legend:
- **untriaged**: not yet reviewed.
- **queued**: reviewed; awaiting adoption.
- **partial**: partially adopted; remaining items tracked in checklist.
- **adopted**: reflected in `out/`/`docs/`/code.
- **rejected**: explicitly out of scope.

## Inbox entries

- in/in-1.md — **adopted** (core dataflow audit + type/constant audits + tiered reporting implemented; see `docs/sppf_checklist.md`.)
- in/in-2.md — **partial** (import/alias/class hierarchy resolved; dynamic dispatch + decorator transparency remain limited.)
- in/in-3.md — **adopted** (alias‑aware forwarding / identity tracking implemented.)
- in/in-4.md — **adopted** (identity vs. symbol and chain‑of‑custody logic implemented.)
- in/in-5.md — **partial** (wildcard forwarding, import resolution, synthesis, schedule done; partial‑application merging still open.)
- in/in-6.md — **adopted** (deterministic import resolution implemented.)
- in/in-7.md — **adopted** (symbol table + aliasing + propagation implemented in dataflow audit.)
- in/in-8.md — **adopted** (protocol/dataclass synthesis + type aggregation implemented.)
- in/in-9.md — **adopted** (topological refactor scheduling + SCC detection implemented.)
- in/in-10.md — **adopted** (root anchoring, excludes, ignore params, external filter, strictness, naming stubs implemented.)
- in/in-11.md — **adopted** (duplicate of in-10; same adoption status.)
- in/in-12.md — **partial** (LSP integration + LibCST refactor engine present; analysis still AST‑based.)
- in/in-13.md — **adopted** (LSP‑first architecture + CLI as thin client implemented.)
- in/in-14.md — **adopted** (repo scaffold + CLI/LSP split aligned with current structure.)
