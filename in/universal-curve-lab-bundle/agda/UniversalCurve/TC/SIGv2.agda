{-# OPTIONS --safe #-}

module UniversalCurve.TC.SIGv2 where

open import Agda.Builtin.Equality using (_≡_)

record SliceBridge : Set₁ where
  constructor mkSliceBridge
  field
    Hist₀Slice : Set
    Hist⋆Slice : Set
    embed : Hist₀Slice → Hist⋆Slice
    project : Hist⋆Slice → Hist₀Slice
    project-embed : (s₀ : Hist₀Slice) → project (embed s₀) ≡ s₀

record NonMaxCoverWitness : Set where
  constructor mkNonMaxCoverWitness

record Boundary : Set where
  constructor mkBoundary

record Unknown : Set where
  constructor mkUnknown
