---
doc_revision: 3
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: universal_curve_lab_roadmap
doc_role: planning_note
doc_scope:
  - in
  - research
  - universal_curve_lab
doc_authority: informative
doc_requires:
  - in/universal-curve-lab-bundle/README.md
doc_relations:
  informs:
    - in/universal-curve-lab-bundle/docs/experiments.md
  supersedes: []
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---
# Roadmap

1) Agda executable core:
- Sym ordering + normalize
- WindowGraph topologies
- WL engine (gas + optional early-exit)
- Faithful-on-support + checkFaithful

2) Probabilistic wrapper:
- depth/size metrics
- PIT/Schwartz–Zippel hook
- safetyMargin computation

3) Empirical suites:
- basis shift invariance (triples/quads)
- mod p / ring projections
- hyperoperator window families
- defect atlases (boundary included)
