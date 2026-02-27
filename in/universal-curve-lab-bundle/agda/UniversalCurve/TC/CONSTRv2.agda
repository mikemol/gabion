{-# OPTIONS --safe #-}

module UniversalCurve.TC.CONSTRv2 where

open import Agda.Builtin.List using (List; []; _∷_)
open import Agda.Builtin.String using (String)
open import Agda.Builtin.Nat using (Nat)
open import Agda.Builtin.Bool using (Bool; true; false)

open import UniversalCurve.TC.SIGv2

mkPoint : String → String → TracePoint
mkPoint span phase = mkTracePoint span phase

requiredKey : String → PayloadKey
requiredKey key = mkPayloadKey key true

optionalKey : String → PayloadKey
optionalKey key = mkPayloadKey key false

surface : String → Nat → CommandSurface
surface name since = mkCommandSurface name since

kanSandwich : String → String → KanSandwich
kanSandwich ingress egress = mkKanSandwich ingress egress

stratifiedSite : String → Nat → StratifiedSite
stratifiedSite name strata = mkStratifiedSite name strata

quantaleMetric : String → String → QuantaleMetric
quantaleMetric carrier law = mkQuantaleMetric carrier law

cotower : Nat → String → Cotower
cotower height carrier = mkCotower height carrier

towerOps : String → String → TowerOps
towerOps compose contract = mkTowerOps compose contract

stabilization : String → Nat → Stabilization
stabilization witness epoch = mkStabilization witness epoch

buildTraceContractV2 :
  List TracePoint →
  List PayloadKey →
  List CommandSurface →
  KanSandwich →
  StratifiedSite →
  QuantaleMetric →
  Cotower →
  TowerOps →
  Stabilization →
  TraceContractV2
buildTraceContractV2 points keys surfaces ks site metric tower ops stable =
  mkTraceContractV2 points keys surfaces ks site metric tower ops stable

sampleContractV2 : TraceContractV2
sampleContractV2 =
  buildTraceContractV2
    (mkPoint "aspf.emit" "analysis" ∷ mkPoint "dto.encode" "projection" ∷ [])
    (requiredKey "entries" ∷ optionalKey "metadata" ∷ [])
    (surface "gabion check-delta" 1 ∷ surface "gabion dataflow-audit" 1 ∷ [])
    (kanSandwich "runtime ingress" "evidence egress")
    (stratifiedSite "ambiguity lattice" 3)
    (quantaleMetric "evidence-distance" "monotone-join")
    (cotower 2 "continuation snapshots")
    (towerOps "compose-carriers" "contract-carriers")
    (stabilization "fixed-point witness" 1)
