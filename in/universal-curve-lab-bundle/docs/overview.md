# Overview

This repo defines a **Universal Curve** as the isomorphism class of a **context-closed quotient**
of a finite windowed overlap graph.

Core:
1. Window graph `G=(I,E)` from a window generator.
2. Outer algebra lift (valuation moments / exterior-style grades) gives base labeling `ℓ₀`.
3. WL/Kleene* refinement yields stabilized symbolic labeling `ℓ̂ = WL*(ℓ₀)` (boundary included).
4. Faithful-on-support evaluation maps symbolic labels into concrete codomains without collisions on observed support.
5. Complexity/probability wrapper exports metrics (depth/size) + PIT hook.
