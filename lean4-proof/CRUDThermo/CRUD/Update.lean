/-
  CRUDThermo/CRUD/Update.lean

  Update operation: uniform → logical1 (overwrite).
  Contains an erase-like component because the old unknown value
  is not retained. Macro KL ≈ kT ln 2, similar to Delete.

  Reference: "From Erasure to CRUD", Section 4-5
-/
import CRUDThermo.CRUD.Operations
import CRUDThermo.CRUD.Delete
import CRUDThermo.Basic.KLDivergence
import CRUDThermo.CRUD.Create

namespace CRUDThermo

private lemma logicalState_sum'' (f : LogicalState → ℝ) :
    ∑ x : LogicalState, f x = f .L + f .R := by
  have : (Finset.univ : Finset LogicalState) = {.L, .R} := by decide
  rw [this, Finset.sum_pair (by decide)]

/-- D_KL(logical1 ‖ uniform) = ln 2.

    Update overwrites unknown → logical1. The final state is
    concentrated in the right well, while the equilibrium reference
    is uniform. This gives the same ln 2 as Delete. -/
theorem klDiv_logical1_uniform :
    klDiv MacroDist.logical1.toFinDist MacroDist.uniform.toFinDist = Real.log 2 := by
  unfold klDiv
  rw [logicalState_sum'']
  simp only [MacroDist.toFinDist, MacroDist.logical1, MacroDist.uniform]
  norm_num

/-- **Update contains Landauer-scale cost**:
    D_KL(logical1 ‖ uniform) = D_KL(blank0 ‖ uniform) = ln 2.

    Both Delete and Update compress an initially uncertain logical
    state into a specified target. The macro KL cost is the same
    regardless of which basin is targeted. -/
theorem update_delete_same_macro_cost :
    klDiv MacroDist.logical1.toFinDist MacroDist.uniform.toFinDist =
    klDiv MacroDist.blank0.toFinDist MacroDist.uniform.toFinDist := by
  rw [klDiv_logical1_uniform, klDiv_blank0_uniform]

/-- **Update work bound**: ⟨W⟩ ≥ T · ln 2 for overwrite.
    Same Landauer-scale bound as Delete. -/
theorem update_work_bound
    (T : ℝ)
    (sl : SecondLawWitness)
    (h_deltaF : sl.deltaF_total = T * Real.log 2) :
    T * Real.log 2 ≤ sl.meanWork := by
  rw [← h_deltaF]
  exact sl.work_bound

/-- **CRUD classification theorem** (macro KL level):
    Delete and Update both incur ln 2 macro cost.
    Create incurs zero. This is the formal content of Fig. 1(b). -/
theorem crud_macro_classification :
    klDiv MacroDist.blank0.toFinDist MacroDist.uniform.toFinDist = Real.log 2 ∧
    klDiv MacroDist.logical1.toFinDist MacroDist.uniform.toFinDist = Real.log 2 ∧
    klDiv MacroDist.blank0.toFinDist MacroDist.logical1.toFinDist = 0 := by
  exact ⟨klDiv_blank0_uniform, klDiv_logical1_uniform, klDiv_blank0_logical1⟩

end CRUDThermo
