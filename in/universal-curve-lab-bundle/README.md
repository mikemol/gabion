---
doc_revision: 3
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: universal_curve_lab_readme
doc_role: research_overview
doc_scope:
  - in
  - research
  - universal_curve_lab
doc_authority: informative
doc_requires:
  - in/README.md#repo_contract
doc_relations:
  informs:
    - in/universal-curve-lab-bundle/docs/overview.md
    - in/universal-curve-lab-bundle/docs/proofs.md
    - in/universal-curve-lab-bundle/docs/experiments.md
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---
# Universal Curve Lab

A research/engineering lab for **probabilistic canonical labeling** of finite window graphs via:
- symbolic **WL/Kleene\*** refinement (1-WL color refinement),
- graded **outer algebra** feature lifts (valuation moments / exterior-style grades),
- **Faithful-on-support** evaluation bridges to concrete codomains (finite fields / hashes),
- and a **complexity/probability wrapper** (PIT / Schwartz–Zippel hook).

Structure:
- `agda/` constructive core and executable specs
- `python/` empirical harnesses + artifact generation
- `notes/` lab notes + proof sketches
- `docs/` overview + proof schemas + experiment protocol
- `artifacts/` exported runs / forests / tables

License: MIT (`LICENSE`).
