/-
  CRUDThermo/Basic/ChainRule.lean

  Chain rule of KL divergence: decomposition into
  macro (inter-basin) and intra (within-basin) contributions.

  Reference: "From Erasure to CRUD", Eq. (3)
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Ring.Finset
import CRUDThermo.Basic.KLDivergence

namespace CRUDThermo

variable {M Ω : Type*} [Fintype M] [Fintype Ω]

/-- A joint distribution with a partition (macro label m, micro state x).
    Models the double-well memory: m ∈ {L, R} labels the basin,
    x is the full state within that basin. -/
structure JointDist (M Ω : Type*) [Fintype M] [Fintype Ω] where
  joint : M → Ω → ℝ
  joint_nonneg : ∀ m x, 0 ≤ joint m x
  sum_one : ∑ m : M, ∑ x : Ω, joint m x = 1

namespace JointDist

variable (d : JointDist M Ω)

/-- Marginal over micro states: p(m) = Σ_x joint(m,x) -/
noncomputable def marginal (m : M) : ℝ := ∑ x : Ω, d.joint m x

lemma marginal_nonneg (m : M) : 0 ≤ d.marginal m :=
  Finset.sum_nonneg fun x _ => d.joint_nonneg m x

lemma marginal_sum_one : ∑ m : M, d.marginal m = 1 := by
  unfold marginal
  exact d.sum_one

/-- Conditional distribution: p(x|m) = joint(m,x) / p(m)
    Returns 0 when p(m) = 0. -/
noncomputable def conditional (m : M) (x : Ω) : ℝ :=
  if d.marginal m = 0 then 0 else d.joint m x / d.marginal m

lemma conditional_nonneg (m : M) (x : Ω) : 0 ≤ d.conditional m x := by
  unfold conditional
  split
  · exact le_refl 0
  · exact div_nonneg (d.joint_nonneg m x) (d.marginal_nonneg m)

lemma conditional_sum_one (m : M) (hm : 0 < d.marginal m) :
    ∑ x : Ω, d.conditional m x = 1 := by
  unfold conditional
  simp only [ne_of_gt hm, ↓reduceIte]
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt hm)

/-- Reconstruction: joint(m,x) = p(m) · p(x|m) -/
lemma joint_eq_marginal_mul_conditional (m : M) (x : Ω)
    (hm : d.marginal m ≠ 0) :
    d.joint m x = d.marginal m * d.conditional m x := by
  unfold conditional
  simp only [hm, ↓reduceIte]
  exact (mul_div_cancel₀ (d.joint m x) hm).symm

end JointDist

/-- KL divergence of marginals. -/
noncomputable def klDiv_marginal (p q : JointDist M Ω) : ℝ :=
  ∑ m : M, p.marginal m * Real.log (p.marginal m / q.marginal m)

/-- Conditional KL divergence: Σ_m p(m) · D_KL(p(·|m) ‖ q(·|m)) -/
noncomputable def klDiv_conditional (p q : JointDist M Ω) : ℝ :=
  ∑ m : M, p.marginal m *
    (∑ x : Ω, p.conditional m x *
      Real.log (p.conditional m x / q.conditional m x))

/-- Full joint KL divergence: D_KL(p(m,x) ‖ q(m,x)) -/
noncomputable def klDiv_joint (p q : JointDist M Ω) : ℝ :=
  ∑ m : M, ∑ x : Ω,
    p.joint m x * Real.log (p.joint m x / q.joint m x)

/-- **Chain rule of KL divergence** (Eq. 3 of the paper):
    D_KL(p ‖ q) = D_KL^macro(p_m ‖ q_m) + Σ_m p_m · D_KL(p(·|m) ‖ q(·|m))

    This is the structural heart of the CRUD classification:
    Delete has large macro term ≈ kT ln 2,
    Create has macro term ≈ 0. -/
theorem klDiv_chain_rule [Nonempty Ω] (p q : JointDist M Ω)
    (hp_marg : ∀ m, 0 < p.marginal m)
    (hq_joint : ∀ m x, 0 < q.joint m x) :
    klDiv_joint p q = klDiv_marginal p q + klDiv_conditional p q := by
  -- Derive q marginal positivity
  have hq_marg : ∀ m, 0 < q.marginal m := by
    intro m; unfold JointDist.marginal
    exact Finset.sum_pos (fun x _ => hq_joint m x) Finset.univ_nonempty
  unfold klDiv_joint klDiv_marginal klDiv_conditional
  rw [← Finset.sum_add_distrib]
  congr 1; ext m
  have hpm_pos := hp_marg m
  have hpm_ne := ne_of_gt hpm_pos
  have hqm_pos := hq_marg m
  have hqm_ne := ne_of_gt hqm_pos
  -- q.conditional m x > 0 from q.joint > 0
  have hcq_pos : ∀ x, 0 < q.conditional m x := by
    intro x; unfold JointDist.conditional
    simp only [hqm_ne, ↓reduceIte]
    exact div_pos (hq_joint m x) hqm_pos
  have hcq_ne : ∀ x, q.conditional m x ≠ 0 := fun x => ne_of_gt (hcq_pos x)
  have h_cond_sum := p.conditional_sum_one m hpm_pos
  -- Expand p(m)*log(p(m)/q(m)) = Σ_x p(m)*cond_p(x)*log(p(m)/q(m))
  rw [show p.marginal m * Real.log (p.marginal m / q.marginal m) =
    ∑ x : Ω, p.marginal m * p.conditional m x *
      Real.log (p.marginal m / q.marginal m) from by
    simp_rw [← Finset.sum_mul, ← Finset.mul_sum, h_cond_sum, mul_one]]
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  congr 1; ext x
  rw [p.joint_eq_marginal_mul_conditional m x hpm_ne,
      q.joint_eq_marginal_mul_conditional m x hqm_ne]
  by_cases hcpx : p.conditional m x = 0
  · simp [hcpx]
  · rw [mul_div_mul_comm,
        Real.log_mul (div_ne_zero hpm_ne hqm_ne) (div_ne_zero hcpx (hcq_ne x))]
    ring

end CRUDThermo
