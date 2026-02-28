{-# OPTIONS --safe #-}

module UniversalCurve.TC.CONSTR where

open import Agda.Builtin.List using (List; []; _∷_)
open import Agda.Builtin.String using (String)
open import Agda.Builtin.Nat using (Nat)
open import Agda.Builtin.Bool using (Bool; true; false)

open import UniversalCurve.TC.SIG

mkPoint : String → String → TracePoint
mkPoint span phase = mkTracePoint span phase

requiredKey : String → PayloadKey
requiredKey key = mkPayloadKey key true

optionalKey : String → PayloadKey
optionalKey key = mkPayloadKey key false

surface : String → Nat → CommandSurface
surface name since = mkCommandSurface name since

buildTraceContract :
  List TracePoint →
  List PayloadKey →
  List CommandSurface →
  TraceContract
buildTraceContract points keys surfaces =
  mkTraceContract points keys surfaces

sampleContract : TraceContract
sampleContract =
  buildTraceContract
    (mkPoint "aspf.emit" "analysis" ∷ mkPoint "dto.encode" "projection" ∷ [])
    (requiredKey "entries" ∷ optionalKey "metadata" ∷ [])
    (surface "gabion check-delta" 1 ∷ surface "gabion dataflow-audit" 1 ∷ [])
