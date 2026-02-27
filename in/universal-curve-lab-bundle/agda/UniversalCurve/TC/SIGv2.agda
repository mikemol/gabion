{-# OPTIONS --safe #-}

module UniversalCurve.TC.SIGv2 where

open import Agda.Builtin.List using (List)
open import Agda.Builtin.String using (String)
open import Agda.Builtin.Nat using (Nat)
open import Agda.Builtin.Bool using (Bool)

-- TC v2 introduces an explicit decomposition of contract concerns into
-- sub-records that can be mapped independently to runtime surfaces.

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

record KanSandwich : Set where
  constructor mkKanSandwich
  field
    ingressBoundary : String
    egressBoundary : String

record StratifiedSite : Set where
  constructor mkStratifiedSite
  field
    siteName : String
    stratumCount : Nat

record QuantaleMetric : Set where
  constructor mkQuantaleMetric
  field
    metricCarrier : String
    metricLaw : String

record Cotower : Set where
  constructor mkCotower
  field
    cotowerHeight : Nat
    cotowerCarrier : String

record TowerOps : Set where
  constructor mkTowerOps
  field
    composeOp : String
    contractOp : String

record Stabilization : Set where
  constructor mkStabilization
  field
    witnessName : String
    stabilizedAtEpoch : Nat

record TraceContractV2 : Set where
  constructor mkTraceContractV2
  field
    points : List TracePoint
    payloadKeys : List PayloadKey
    commandSurfaces : List CommandSurface
    kanSandwich : KanSandwich
    stratifiedSite : StratifiedSite
    quantaleMetric : QuantaleMetric
    cotower : Cotower
    towerOps : TowerOps
    stabilization : Stabilization

-- Compatibility record preserved for existing imports that still target SIG.
record TraceContract : Set where
  constructor mkTraceContract
  field
    points : List TracePoint
    payloadKeys : List PayloadKey
    commandSurfaces : List CommandSurface

forgetV2 : TraceContractV2 → TraceContract
forgetV2 contract =
  mkTraceContract
    (TraceContractV2.points contract)
    (TraceContractV2.payloadKeys contract)
    (TraceContractV2.commandSurfaces contract)
