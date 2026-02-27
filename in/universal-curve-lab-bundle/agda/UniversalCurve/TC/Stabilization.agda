{-# OPTIONS --safe #-}

module UniversalCurve.TC.Stabilization where

open import Agda.Builtin.Equality using (_≡_)

open import UniversalCurve.TC.SIGv2
open import UniversalCurve.TC.GLUEv2

horizon :
  (bridge : SliceBridge) →
  SliceBridge.Hist₀Slice bridge →
  SliceBridge.Hist⋆Slice bridge
horizon bridge s₀ = SliceBridge.embed bridge s₀

horizon-provenance :
  (glue : GLUEv2) →
  (s₀ : SliceBridge.Hist₀Slice (GLUEv2.bridge glue)) →
  (p : StabilizationProvenance) →
  (admitted : GLUEv2.admittedImage glue
    (GLUEv2.refine glue (horizon (GLUEv2.bridge glue) s₀))) →
  GLUEv2.gateProvenance glue s₀ p admitted ≡ p
horizon-provenance glue s₀ p admitted =
  GLUEv2.gateProvenance-preserved glue s₀ p admitted
