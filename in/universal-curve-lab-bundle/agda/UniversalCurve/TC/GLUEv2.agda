{-# OPTIONS --safe #-}

module UniversalCurve.TC.GLUEv2 where

open import Agda.Builtin.List using (List)
open import Agda.Builtin.String using (String)
open import Agda.Builtin.Bool using (Bool; false)
open import Agda.Builtin.Equality using (_≡_)
open import Data.Sum using (_⊎_)

open import UniversalCurve.TC.SIGv2
open import UniversalCurve.TC.CONSTRv2

record RuntimeMapping : Set where
  constructor mkRuntimeMapping
  field
    tcConcept : String
    runtimeSurface : String
    bridgeCoverage : String

record BridgePlanV2 : Set where
  constructor mkBridgePlanV2
  field
    contract : TraceContractV2
    mappings : List RuntimeMapping
    productionEnforced : Bool

mkMapping : String → String → String → RuntimeMapping
mkMapping concept surface coverage = mkRuntimeMapping concept surface coverage

bridgePlanV2 : List RuntimeMapping → BridgePlanV2
bridgePlanV2 maps = mkBridgePlanV2 sampleContractV2 maps false

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
