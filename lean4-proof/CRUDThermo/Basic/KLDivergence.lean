/-
  CRUDThermo/Basic/KLDivergence.lean

  KL divergence for finite discrete distributions.
  Foundation for the CRUD information-thermodynamic framework.

  Reference: "From Erasure to CRUD", Eq. (1)-(3)
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Topology.Algebra.InfiniteSum.Real

namespace CRUDThermo

/-- A probability distribution over a finite type. -/
structure FinDist (Ω : Type*) [Fintype Ω] where
  prob : Ω → ℝ
  nonneg : ∀ x, 0 ≤ prob x
  sum_one : ∑ x : Ω, prob x = 1

variable {Ω : Type*} [Fintype Ω]

/-- KL divergence for finite distributions.
    D_KL(p ‖ q) = ∑_x p(x) * log(p(x) / q(x)) -/
noncomputable def klDiv (p q : FinDist Ω) : ℝ :=
  ∑ x : Ω, p.prob x * Real.log (p.prob x / q.prob x)

/-- KL divergence of a distribution with itself is zero. -/
theorem klDiv_self (p : FinDist Ω) : klDiv p p = 0 := by
  unfold klDiv
  have : ∀ x, p.prob x * Real.log (p.prob x / p.prob x) = 0 := by
    intro x
    by_cases hx : p.prob x = 0
    · simp [hx]
    · rw [div_self hx, Real.log_one, mul_zero]
  exact Finset.sum_eq_zero fun x _ => this x

/-- log x ≤ x - 1 for x > 0. Central to Gibbs' inequality.
    Proof: from exp(log t) = t and 1 + y ≤ exp(y). -/
private lemma log_le_sub_one {t : ℝ} (ht : 0 < t) : Real.log t ≤ t - 1 := by
  linarith [Real.add_one_le_exp (Real.log t), Real.exp_log ht]

/-- Gibbs' inequality: D_KL(p ‖ q) ≥ 0 when q has full support.
    This is the information-theoretic foundation for all CRUD bounds.

    Proof strategy:
    1. Pointwise: p(x) - q(x) ≤ p(x) · log(p(x)/q(x))
       via log(q/p) ≤ q/p - 1
    2. Sum: Σ(p - q) = 1 - 1 = 0 ≤ D_KL -/
theorem klDiv_nonneg (p q : FinDist Ω)
    (hq : ∀ x, 0 < q.prob x) :
    0 ≤ klDiv p q := by
  unfold klDiv
  -- Step 1: Pointwise bound
  suffices pointwise : ∀ x, p.prob x - q.prob x ≤
      p.prob x * Real.log (p.prob x / q.prob x) by
    -- Step 2: Sum and use Σp = Σq = 1
    have h_sum : (∑ x : Ω, (p.prob x - q.prob x)) = 0 := by
      simp only [Finset.sum_sub_distrib, p.sum_one, q.sum_one, sub_self]
    linarith [Finset.sum_le_sum (fun x (_ : x ∈ Finset.univ) => pointwise x)]
  -- Prove pointwise bound
  intro x
  by_cases hpx : p.prob x = 0
  · -- Case p(x) = 0: need 0 - q(x) ≤ 0, true since q(x) > 0
    simp [hpx]; linarith [hq x]
  · -- Case p(x) > 0
    have hpx_pos : 0 < p.prob x := lt_of_le_of_ne (p.nonneg x) (Ne.symm hpx)
    have hqx_pos := hq x
    -- Key: log(p/q) = -log(q/p)
    have h_log_flip : Real.log (p.prob x / q.prob x) =
        -Real.log (q.prob x / p.prob x) := by
      rw [Real.log_div (ne_of_gt hpx_pos) (ne_of_gt hqx_pos),
          Real.log_div (ne_of_gt hqx_pos) (ne_of_gt hpx_pos)]
      ring
    -- Chain: p - q = -(p·(q/p - 1)) ≤ -(p·log(q/p)) = p·log(p/q)
    calc p.prob x - q.prob x
        = -(p.prob x * (q.prob x / p.prob x - 1)) := by
          field_simp; ring
      _ ≤ -(p.prob x * Real.log (q.prob x / p.prob x)) := by
          apply neg_le_neg
          exact mul_le_mul_of_nonneg_left
            (log_le_sub_one (div_pos hqx_pos hpx_pos))
            (le_of_lt hpx_pos)
      _ = p.prob x * Real.log (p.prob x / q.prob x) := by
          rw [h_log_flip]; ring

end CRUDThermo
