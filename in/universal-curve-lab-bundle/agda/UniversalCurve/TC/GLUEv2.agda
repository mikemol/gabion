{-# OPTIONS --safe #-}

module UniversalCurve.TC.GLUEv2 where

open import Agda.Builtin.List using (List)
open import Agda.Builtin.String using (String)
open import Agda.Builtin.Bool using (Bool; false)

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
