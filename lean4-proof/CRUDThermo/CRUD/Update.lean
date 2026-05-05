/-
  CRUDThermo/CRUD/Update.lean

  Update operation: uniform → logical1 (overwrite).
  Contains an erase-like component because the old unknown value
  is not retained. Macro KL ≈ kT ln 2, similar to Delete.

  This file also contains the **physically meaningful** macro-KL
  classification of state-transforming CRUD operations, expressed
  against the full-support uniform prior. These statements are
  free of the `Real.log 0 = 0` / `_ / 0 = 0` conventions and are
  the intended citation targets in the paper.

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

/-- **Both definite macrostates are equally far from the uniform prior**.

    A geometric fact in the macro probability simplex: `blank0` and
    `logical1` sit at equal KL distance from the uniform prior π_unif.
    This is the physically meaningful equality behind Create's
    classification: `blank0` and `logical1` are *not identified* with
    each other; rather, each is `ln 2` away from π_unif. -/
theorem definite_macrostates_same_macro_KL_to_uniform :
    klDiv MacroDist.logical1.toFinDist MacroDist.uniform.toFinDist =
    klDiv MacroDist.blank0.toFinDist MacroDist.uniform.toFinDist := by
  exact update_delete_same_macro_cost

/-- **ΔD_KL^macro for Create vanishes** (physically meaningful statement).

    For Create, interpreted as the state-transforming operation
    `blank0 → logical1`, the macro-level information free-energy change
    is measured against the uniform macro equilibrium prior:

      ΔD_KL^macro(Create)
        := D_KL(logical1 ‖ uniform) - D_KL(blank0 ‖ uniform)
         = ln 2 - ln 2 = 0.

    Physical interpretation: both endpoints of Create are already
    macroscopically specified delta states, and both are equidistant
    from the uniform prior. Create therefore carries no macro Landauer
    compression term.

    This theorem is **distinct** from any expression of the form
    `D_KL(blank0 ‖ logical1)`. The latter involves support mismatch
    and may depend on formal conventions such as `Real.log 0 = 0`;
    it is not used in the physical CRUD classification. -/
theorem create_macro_DKL_change_zero :
    klDiv MacroDist.logical1.toFinDist MacroDist.uniform.toFinDist -
    klDiv MacroDist.blank0.toFinDist MacroDist.uniform.toFinDist = 0 := by
  exact sub_eq_zero.mpr definite_macrostates_same_macro_KL_to_uniform

/-- **CRUD classification theorem** (macro KL level, original form):
    Delete and Update both incur ln 2 macro cost.
    Create incurs zero. This is the formal content of Fig. 1(b).

    NOTE: The third conjunct uses the convention-dependent identity
    `klDiv_blank0_logical1`. For the convention-independent version,
    see `crud_macro_classification_physical` below. -/
theorem crud_macro_classification :
    klDiv MacroDist.blank0.toFinDist MacroDist.uniform.toFinDist = Real.log 2 ∧
    klDiv MacroDist.logical1.toFinDist MacroDist.uniform.toFinDist = Real.log 2 ∧
    klDiv MacroDist.blank0.toFinDist MacroDist.logical1.toFinDist = 0 := by
  exact ⟨klDiv_blank0_uniform, klDiv_logical1_uniform, klDiv_blank0_logical1⟩

/-- **Physical macro-KL classification of state-transforming CRUD operations**.

    All KL values here are taken against the full-support uniform
    macro prior, so they are genuine Gibbs quantities — independent
    of the `Real.log 0 = 0` convention.

    The theorem states:

      • Delete (uniform → blank0):    ΔD_KL^macro = + ln 2
      • Update (uniform → logical1):  ΔD_KL^macro = + ln 2
      • Create (blank0   → logical1): ΔD_KL^macro = 0

    Read is intentionally not included here, because Read is not a
    logical compression operation in this framework. It is handled
    separately through system–meter correlation and the Sagawa–Ueda
    measurement bound (see `CRUD/Read.lean`).

    This is the **convention-independent counterpart** of
    `crud_macro_classification` and is the recommended citation
    target in the paper. -/
theorem crud_macro_classification_physical :
    -- Delete: uniform → blank0
    (klDiv MacroDist.blank0.toFinDist MacroDist.uniform.toFinDist -
     klDiv MacroDist.uniform.toFinDist MacroDist.uniform.toFinDist
       = Real.log 2) ∧
    -- Update-as-overwrite: uniform → logical1
    (klDiv MacroDist.logical1.toFinDist MacroDist.uniform.toFinDist -
     klDiv MacroDist.uniform.toFinDist MacroDist.uniform.toFinDist
       = Real.log 2) ∧
    -- Create: blank0 → logical1
    (klDiv MacroDist.logical1.toFinDist MacroDist.uniform.toFinDist -
     klDiv MacroDist.blank0.toFinDist MacroDist.uniform.toFinDist
       = 0) := by
  refine ⟨?_, ?_, ?_⟩
  · rw [klDiv_blank0_uniform, klDiv_uniform_self]
    ring
  · rw [klDiv_logical1_uniform, klDiv_uniform_self]
    ring
  · exact create_macro_DKL_change_zero

end CRUDThermo
