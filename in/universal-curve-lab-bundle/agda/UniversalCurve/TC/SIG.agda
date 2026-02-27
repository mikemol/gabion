{-# OPTIONS --safe #-}

module UniversalCurve.TC.SIG where

open import Agda.Builtin.List using (List)
open import Agda.Builtin.String using (String)
open import Agda.Builtin.Nat using (Nat)
open import Agda.Builtin.Bool using (Bool)

-- TC (Trace Contract) signatures are intentionally minimal for research staging.
-- They model shape and labels only; runtime enforcement lives in Gabion handlers.

record TracePoint : Set where
  constructor mkTracePoint
  field
    spanLabel : String
    phaseLabel : String

record PayloadKey : Set where
  constructor mkPayloadKey
  field
    keyName : String
    required : Bool

record CommandSurface : Set where
  constructor mkCommandSurface
  field
    commandName : String
    stableSince : Nat

record TraceContract : Set where
  constructor mkTraceContract
  field
    points : List TracePoint
    payloadKeys : List PayloadKey
    commandSurfaces : List CommandSurface
