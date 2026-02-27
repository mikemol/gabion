{-# OPTIONS --safe #-}

module UniversalCurve.TC.GLUEv2 where

open import Agda.Builtin.Equality using (_≡_)
open import Data.Sum using (_⊎_)

open import UniversalCurve.TC.SIGv2

StabilizationProvenance : Set
StabilizationProvenance = NonMaxCoverWitness ⊎ Boundary ⊎ Unknown

record GLUEv2 : Set₁ where
  constructor mkGLUEv2
  field
    bridge : SliceBridge
    admittedImage : SliceBridge.Hist⋆Slice bridge → Set
    refine : SliceBridge.Hist⋆Slice bridge → SliceBridge.Hist⋆Slice bridge
    gateProvenance :
      (s₀ : SliceBridge.Hist₀Slice bridge) →
      (p : StabilizationProvenance) →
      admittedImage (refine (SliceBridge.embed bridge s₀)) →
      StabilizationProvenance
    gateProvenance-preserved :
      (s₀ : SliceBridge.Hist₀Slice bridge) →
      (p : StabilizationProvenance) →
      (admitted : admittedImage (refine (SliceBridge.embed bridge s₀))) →
      gateProvenance s₀ p admitted ≡ p
