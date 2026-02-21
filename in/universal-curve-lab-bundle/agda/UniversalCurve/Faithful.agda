{-# OPTIONS --safe #-}

module UniversalCurve.Faithful where

open import Agda.Builtin.Nat
open import Agda.Builtin.List
open import Agda.Builtin.Equality
open import Data.List using (List; map; _∷_; [])
open import Data.Fin using (Fin; zero; suc)
open import Data.Product using (_×_; _,_)
open import Relation.Binary.PropositionalEquality using (cong)

open import UniversalCurve.Structure
open import UniversalCurve.Engine

record EvalAlg (C : Set) : Set₁ where
  field
    encPair : C → C → C
    encMS   : List C → C

Interp : ∀ {V C : Set} → EvalAlg C → (V → C) → Sym V → C
Interp A ρ (var x)    = ρ x
Interp A ρ (pair a b) = EvalAlg.encPair A (Interp A ρ a) (Interp A ρ b)
Interp A ρ (ms xs)    = EvalAlg.encMS   A (map (Interp A ρ) xs)

allFin : (n : Nat) → List (Fin n)
allFin zero    = []
allFin (suc n) = zero ∷ map suc (allFin n)

data _∈_ {A : Set} (x : A) : List A → Set where
  here  : ∀ {xs} → x ∈ (x ∷ xs)
  there : ∀ {y xs} → x ∈ xs → x ∈ (y ∷ xs)

map-∈ : ∀ {A B : Set} {f : A → B} {x : A} {xs : List A} →
        x ∈ xs → f x ∈ map f xs
map-∈ here      = here
map-∈ (there p) = there (map-∈ p)

allFin-complete : ∀ {n} (k : Fin n) → k ∈ allFin n
allFin-complete {suc n} zero    = here
allFin-complete {suc n} (suc k) = there (map-∈ (allFin-complete k))

stabilizedLabels : ∀ {len V : Set} →
                   (canon : (List (Sym V) → Sym V) → Fin len → Sym V) →
                   (norm : List (Sym V) → Sym V) →
                   (init : Fin len → Sym V) →
                   List (Sym V)
stabilizedLabels canon norm init =
  map (canon norm init) (allFin _)

InjectiveOn : ∀ {A B : Set} → (f : A → B) → List A → Set
InjectiveOn f xs =
  ∀ {x y} → x ∈ xs → y ∈ xs → f x ≡ f y → x ≡ y

Faithful : ∀ {len V C : Set} →
           (A : EvalAlg C) →
           (ρ : V → C) →
           (canon : (List (Sym V) → Sym V) → Fin len → Sym V) →
           (norm : List (Sym V) → Sym V) →
           (init : Fin len → Sym V) →
           Set
Faithful {len} A ρ canon norm init =
  InjectiveOn (Interp A ρ) (stabilizedLabels {len} canon norm init)

label-in-set :
  ∀ {len V : Set}
  (canon : (List (Sym V) → Sym V) → Fin len → Sym V)
  (norm : List (Sym V) → Sym V)
  (init : Fin len → Sym V)
  (i : Fin len) →
  canon norm init i ∈ stabilizedLabels canon norm init
label-in-set canon norm init i =
  map-∈ (allFin-complete i)

faithful-soundness :
  ∀ {len V C : Set}
  (A : EvalAlg C) (ρ : V → C)
  (canon : (List (Sym V) → Sym V) → Fin len → Sym V)
  (norm : List (Sym V) → Sym V)
  (init : Fin len → Sym V) →
  Faithful A ρ canon norm init →
  ∀ (i j : Fin len) →
  Interp A ρ (canon norm init i) ≡ Interp A ρ (canon norm init j) →
  canon norm init i ≡ canon norm init j
faithful-soundness A ρ canon norm init isFaithful i j eq =
  isFaithful (label-in-set canon norm init i)
             (label-in-set canon norm init j)
             eq

record _↔_ (P Q : Set) : Set where
  constructor iff
  field to : P → Q
        from : Q → P

global-isomorphism :
  ∀ {len V C1 C2 : Set}
  (A1 : EvalAlg C1) (ρ1 : V → C1)
  (A2 : EvalAlg C2) (ρ2 : V → C2)
  (canon : (List (Sym V) → Sym V) → Fin len → Sym V)
  (norm : List (Sym V) → Sym V)
  (init : Fin len → Sym V) →
  Faithful A1 ρ1 canon norm init →
  Faithful A2 ρ2 canon norm init →
  ∀ (i j : Fin len) →
  (Interp A1 ρ1 (canon norm init i) ≡ Interp A1 ρ1 (canon norm init j))
  ↔
  (Interp A2 ρ2 (canon norm init i) ≡ Interp A2 ρ2 (canon norm init j))
global-isomorphism A1 ρ1 A2 ρ2 canon norm init f1 f2 i j =
  iff dir1 dir2
  where
    symLabels = canon norm init

    dir1 : Interp A1 ρ1 (symLabels i) ≡ Interp A1 ρ1 (symLabels j) →
           Interp A2 ρ2 (symLabels i) ≡ Interp A2 ρ2 (symLabels j)
    dir1 eq1 = cong (Interp A2 ρ2) (faithful-soundness A1 ρ1 canon norm init f1 i j eq1)

    dir2 : Interp A2 ρ2 (symLabels i) ≡ Interp A2 ρ2 (symLabels j) →
           Interp A1 ρ1 (symLabels i) ≡ Interp A1 ρ1 (symLabels j)
    dir2 eq2 = cong (Interp A1 ρ1) (faithful-soundness A2 ρ2 canon norm init f2 i j eq2)
