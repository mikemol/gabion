{-# OPTIONS --safe #-}

module UniversalCurve.Engine where

open import Agda.Builtin.Nat
open import Agda.Builtin.List
open import Data.List using (List; map)
open import Data.Fin using (Fin)

open import UniversalCurve.Structure
open import UniversalCurve.WindowGraph

step : ∀ {V : Set} {G : Graph} →
       (norm : List (Sym V) → Sym V) →
       (Graph.Node G → Sym V) →
       (Graph.Node G → Sym V)
step {G = G} norm ℓ i =
  let nbrs  = Graph.neighbors G i
      blobs = map ℓ nbrs
  in pair (ℓ i) (norm blobs)

runWL : ∀ {V : Set} {G : Graph} →
        (norm : List (Sym V) → Sym V) →
        Nat →
        (Graph.Node G → Sym V) →
        (Graph.Node G → Sym V)
runWL norm zero    ℓ = ℓ
runWL {G = G} norm (suc n) ℓ =
  let next = step {G = G} norm ℓ
  in runWL norm n next

canonicalize : ∀ {len : Nat} {V : Set} →
               (cmpV : V → V → Order) →
               (init : Fin len → Sym V) →
               (Fin len → Sym V)
canonicalize {len} {V} cmpV init =
  let open SymOrd cmpV
      G = windowStructure len
  in runWL {G = G} normalize len init
