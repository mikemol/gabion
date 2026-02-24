---
doc_revision: 3
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: universal_curve_lab_overview
doc_role: research_overview
doc_scope:
  - in
  - research
  - universal_curve_lab
doc_authority: informative
doc_requires:
  - in/universal-curve-lab-bundle/README.md
doc_relations:
  refines:
    - in/universal-curve-lab-bundle/README.md
  informs:
    - in/universal-curve-lab-bundle/docs/proofs.md
    - in/universal-curve-lab-bundle/docs/experiments.md
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---
# Overview

This repo defines a **Universal Curve** as the isomorphism class of a **context-closed quotient**
of a finite windowed overlap graph.

Core:
1. Window graph `G=(I,E)` from a window generator.
2. Outer algebra lift (valuation moments / exterior-style grades) gives base labeling `ℓ₀`.
3. WL/Kleene* refinement yields stabilized symbolic labeling `ℓ̂ = WL*(ℓ₀)` (boundary included).
4. Faithful-on-support evaluation maps symbolic labels into concrete codomains without collisions on observed support.
5. Complexity/probability wrapper exports metrics (depth/size) + PIT hook.
