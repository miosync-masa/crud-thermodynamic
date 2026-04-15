/-
  CRUDThermo/Basic/MutualInformation.lean

  Mutual information I(X;Y) defined via KL divergence.
  Used for the Read operation's thermodynamic bound (Eq. 4).

  Reference: "From Erasure to CRUD", Eq. (4), Sagawa-Ueda bound
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import CRUDThermo.Basic.KLDivergence

namespace CRUDThermo

variable {X Y : Type*} [Fintype X] [Fintype Y]

/-- Marginal distribution over X from a joint distribution on X × Y. -/
noncomputable def marginalX (p : FinDist (X × Y)) : X → ℝ :=
  fun x => ∑ y : Y, p.prob (x, y)

/-- Marginal distribution over Y from a joint distribution on X × Y. -/
noncomputable def marginalY (p : FinDist (X × Y)) : Y → ℝ :=
  fun y => ∑ x : X, p.prob (x, y)

lemma marginalX_nonneg (p : FinDist (X × Y)) (x : X) :
    0 ≤ marginalX p x :=
  Finset.sum_nonneg fun y _ => p.nonneg (x, y)

lemma marginalY_nonneg (p : FinDist (X × Y)) (y : Y) :
    0 ≤ marginalY p y :=
  Finset.sum_nonneg fun x _ => p.nonneg (x, y)

lemma marginalX_sum (p : FinDist (X × Y)) :
    ∑ x : X, marginalX p x = 1 := by
  simp only [marginalX, ← Fintype.sum_prod_type, p.sum_one]

/-- Product distribution p_X ⊗ p_Y. -/
noncomputable def productDist (p : FinDist (X × Y))
    (hmX : ∀ x, 0 ≤ marginalX p x)
    (hmY : ∀ y, 0 ≤ marginalY p y)
    (hsum : ∑ xy : X × Y, marginalX p xy.1 * marginalY p xy.2 = 1) :
    FinDist (X × Y) where
  prob := fun ⟨x, y⟩ => marginalX p x * marginalY p y
  nonneg := fun ⟨x, y⟩ => mul_nonneg (hmX x) (hmY y)
  sum_one := hsum

/-- Mutual information I(X;Y) = D_KL(p_{XY} ‖ p_X ⊗ p_Y).
    Measures the correlation between system and meter in Read. -/
noncomputable def mutualInfo (p : FinDist (X × Y))
    (hmX : ∀ x, 0 ≤ marginalX p x)
    (hmY : ∀ y, 0 ≤ marginalY p y)
    (hsum : ∑ xy : X × Y, marginalX p xy.1 * marginalY p xy.2 = 1) :
    ℝ :=
  klDiv p (productDist p hmX hmY hsum)

/-- **Mutual information is nonneg**: I(X;Y) ≥ 0.
    Immediate from Gibbs' inequality (klDiv_nonneg).
    This is the foundation for the Read measurement bound. -/
theorem mutualInfo_nonneg (p : FinDist (X × Y))
    (hmX : ∀ x, 0 < marginalX p x)
    (hmY : ∀ y, 0 < marginalY p y)
    (hsum : ∑ xy : X × Y, marginalX p xy.1 * marginalY p xy.2 = 1) :
    0 ≤ mutualInfo p
      (fun x => le_of_lt (hmX x))
      (fun y => le_of_lt (hmY y))
      hsum := by
  unfold mutualInfo
  exact klDiv_nonneg p _ (fun ⟨x, y⟩ => by
    simp only [productDist]
    exact mul_pos (hmX x) (hmY y))

end CRUDThermo-- Mutual Information
