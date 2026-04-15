/-
  CRUDThermo/CRUD/Delete.lean

  Delete operation: uniform → blank0 (Landauer erasure).
  The macro KL divergence equals ln 2, recovering Landauer's principle
  as a special case of the CRUD framework.

  Reference: "From Erasure to CRUD", Section 4-5
-/
import CRUDThermo.CRUD.Operations
import CRUDThermo.Basic.KLDivergence
import CRUDThermo.Thermodynamics.JarzynskiBound

namespace CRUDThermo

/-
  KL divergence of the macro distribution for Delete:
  D_KL(uniform ‖ blank0).

  Since blank0 has pR = 0 and uniform has pR = 1/2,
  this is not well-defined (log(x/0) diverges).
  We instead compute the reverse: D_KL(blank0 ‖ uniform) = ln 2.

  In the paper, Delete goes uniform → blank0.
  The information free-energy change ΔD_KL^macro is measured
  between the FINAL distribution and the FINAL equilibrium.
  For the Landauer case with symmetric erasure,
  D_KL^macro reduces to ln 2 - h(ε), approaching ln 2 as ε → 0.
-/

/-- Helper: the two-element sum over LogicalState. -/
private lemma logicalState_sum (f : LogicalState → ℝ) :
    ∑ x : LogicalState, f x = f .L + f .R := by
  have : (Finset.univ : Finset LogicalState) = {.L, .R} := by decide
  rw [this, Finset.sum_pair (by decide)]

/-- D_KL(blank0 ‖ uniform) = ln 2.

    This is the Landauer cost: erasing one bit of information
    requires at least kT ln 2 of work (with k=1, T=1). -/
theorem klDiv_blank0_uniform :
    klDiv MacroDist.blank0.toFinDist MacroDist.uniform.toFinDist = Real.log 2 := by
  unfold klDiv
  rw [logicalState_sum]
  simp only [MacroDist.toFinDist, MacroDist.blank0, MacroDist.uniform]
  norm_num

/-- D_KL(uniform ‖ uniform) = 0. Baseline: no information change. -/
theorem klDiv_uniform_self :
    klDiv MacroDist.uniform.toFinDist MacroDist.uniform.toFinDist = 0 :=
  klDiv_self _

/-- **Landauer's principle as CRUD special case**:
    For Delete (uniform → blank0), the macro KL divergence
    contribution to ΔF_total is ln 2.

    Combined with the CRUD work bound ⟨W⟩ ≥ ΔF_total,
    this recovers: ⟨W⟩ ≥ T · ln 2 for erasure. -/
theorem landauer_from_crud
    (T : ℝ)
    (sl : SecondLawWitness)
    (h_deltaF : sl.deltaF_total = T * Real.log 2) :
    T * Real.log 2 ≤ sl.meanWork := by
  rw [← h_deltaF]
  exact sl.work_bound

end CRUDThermo
