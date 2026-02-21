{-# OPTIONS --safe #-}

module UniversalCurve.WindowGraph where

open import Agda.Builtin.Nat
open import Agda.Builtin.List
open import Data.Fin using (Fin; zero; suc; toNat; fromNat)
open import Data.List using (List; _∷_; []; map)
open import Data.Nat using (_==_)

record Graph : Set₁ where
  field
    Node : Set
    neighbors : Node → List Node

windowStructure : Nat → Graph
windowStructure len = record
  { Node = Fin len
  ; neighbors = adj
  }
  where
    adj : Fin len → List (Fin len)
    adj i with toNat i
    ... | zero = if len == 0 then [] else if len == 1 then [] else (fromNat 1 ∷ [])
    ... | suc k =
      let prev = fromNat k in
      if (suc (suc k)) == len
      then (prev ∷ [])
      else (prev ∷ fromNat (suc (suc k)) ∷ [])
