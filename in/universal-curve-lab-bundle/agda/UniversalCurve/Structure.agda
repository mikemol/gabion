{-# OPTIONS --safe #-}

module UniversalCurve.Structure where

open import Agda.Builtin.Nat
open import Agda.Builtin.List
open import Relation.Nullary using (Dec)

data Order : Set where
  LT EQ GT : Order

data Sym (V : Set) : Set where
  var  : V → Sym V
  pair : Sym V → Sym V → Sym V
  ms   : List (Sym V) → Sym V  -- assumed sorted/canonical

module SymOrd {V : Set} (cmpV : V → V → Order) where
  {-# TERMINATING #-}
  compareSym : Sym V → Sym V → Order
  compareSymList : List (Sym V) → List (Sym V) → Order

  compareSym (var x) (var y) = cmpV x y
  compareSym (var _) _       = LT
  compareSym _ (var _)       = GT

  compareSym (pair a b) (pair c d) with compareSym a c
  ... | LT = LT
  ... | GT = GT
  ... | EQ = compareSym b d
  compareSym (pair _ _) _ = LT
  compareSym _ (pair _ _) = GT

  compareSym (ms xs) (ms ys) = compareSymList xs ys

  compareSymList [] [] = EQ
  compareSymList [] (_ ∷ _) = LT
  compareSymList (_ ∷ _) [] = GT
  compareSymList (x ∷ xs) (y ∷ ys) with compareSym x y
  ... | LT = LT
  ... | GT = GT
  ... | EQ = compareSymList xs ys

  insert : Sym V → List (Sym V) → List (Sym V)
  insert x [] = x ∷ []
  insert x (y ∷ ys) with compareSym x y
  ... | LT = x ∷ y ∷ ys
  ... | EQ = x ∷ y ∷ ys
  ... | GT = y ∷ insert x ys

  sort : List (Sym V) → List (Sym V)
  sort = foldr insert []

  normalize : List (Sym V) → Sym V
  normalize xs = ms (sort xs)
