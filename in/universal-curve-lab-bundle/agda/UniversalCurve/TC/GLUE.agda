{-# OPTIONS --safe #-}

module UniversalCurve.TC.GLUE where

open import Agda.Builtin.List using (List)
open import Agda.Builtin.String using (String)
open import Agda.Builtin.Bool using (Bool; false)

open import UniversalCurve.TC.SIG
open import UniversalCurve.TC.CONSTR

record RuntimeMapping : Set where
  constructor mkRuntimeMapping
  field
    tcConcept : String
    runtimeSurface : String

record BridgePlan : Set where
  constructor mkBridgePlan
  field
    contract : TraceContract
    mappings : List RuntimeMapping
    productionEnforced : Bool

mkMapping : String → String → RuntimeMapping
mkMapping concept surface = mkRuntimeMapping concept surface

bridgePlan : List RuntimeMapping → BridgePlan
bridgePlan maps = mkBridgePlan sampleContract maps false
