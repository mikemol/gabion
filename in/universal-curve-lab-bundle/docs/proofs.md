# Proof Schemas

## Faithful ⇒ Global Isomorphism (boundary included)
Let `ℓ̂_sym = WL*(ℓ₀)` be the stabilized symbolic labeling.

A concrete evaluation `Interp` is **Faithful on support** if it is injective on the finite set `{ℓ̂_sym(i)}`.
Then `Interp` preserves all equalities/inequalities among stabilized labels, so the induced partition
(and quotient graph) is identical to the symbolic one. Boundary nodes are preserved automatically.

## Basis shift invariance
Empirically, for consecutive prime bases (triples/quads), the quotient graphs
`G / ~_{Star(K2_P)}` are globally isomorphic. A symbolic proof strategy lifts to term algebras,
then uses Faithful(P) to conclude concrete stability.
