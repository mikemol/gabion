---
doc_revision: 1
reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.
doc_id: docflow_cycles
doc_role: report
doc_scope:
  - repo
  - docflow
  - report
doc_authority: informative
doc_change_protocol: POLICY_SEED.md#change_protocol
doc_requires: []
doc_reviewed_as_of: {}
doc_review_notes: {}
doc_sections:
  docflow_cycles: 1
doc_section_requires:
  docflow_cycles: []
doc_section_reviews:
  docflow_cycles: {}
---

<a id="docflow_cycles"></a>

Docflow dependency cycles
Summary (raw graph):
- mixed: 1

Summary (dependency projection):
- core: 1

Cycles (raw graph):
- (mixed) AGENTS.md, CONTRIBUTING.md, POLICY_SEED.md, README.md, docs/coverage_semantics.md

Cycles (dependency projection):
- (core) AGENTS.md, CONTRIBUTING.md, POLICY_SEED.md, README.md

Guidance:
- core: expected governance cycle; break only with policy change.
- mixed: consider lifting shared semantics to glossary and removing non-core back-edges.
- non_core: lift shared semantics to glossary to break the cycle.
