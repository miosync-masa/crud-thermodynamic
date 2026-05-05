/-
  CRUDThermo/CRUD/Create.lean

  Create operation: blank0 → logical1 (known → known).

  Both endpoints are macroscopically specified delta distributions,
  equidistant from the uniform prior. Hence the macro contribution
  to ΔF_total vanishes — Create is NOT kT ln 2 limited.

  The physically meaningful statement of this fact lives in
  `Update.lean` as `create_macro_DKL_change_zero`. This file
  contains the convention-dependent computational identities
  (suffixed `_formal`) that arise from `Real.log 0 = 0` in
  Mathlib, and the work-bound corollary `create_not_landauer_limited`.

  Reference: "From Erasure to CRUD", Section 4-5.3
-/
import CRUDThermo.CRUD.Operations
import CRUDThermo.CRUD.Delete
import CRUDThermo.Basic.KLDivergence

namespace CRUDThermo

private lemma logicalState_sum' (f : LogicalState → ℝ) :
    ∑ x : LogicalState, f x = f .L + f .R := by
  have : (Finset.univ : Finset LogicalState) = {.L, .R} := by decide
  rw [this, Finset.sum_pair (by decide)]

/-- **Convention-dependent formal identity** (NOT a genuine KL value).

    The mathematical KL divergence `D_KL(δ_L ‖ δ_R)` between two delta
    distributions on disjoint support is `+∞` (or undefined under the
    strict Gibbs hypothesis `q(x) > 0` everywhere).

    Under Mathlib's conventions
      `(x : ℝ) / 0 = 0`  and  `Real.log 0 = 0`,
    the syntactic expression
      `klDiv blank0.toFinDist logical1.toFinDist`
    evaluates to `0`. This is a **formal artifact** of those
    conventions, not a physical claim, and it does NOT satisfy the
    Gibbs-style nonnegativity hypothesis of `klDiv_nonneg`.

    The physically meaningful statement for Create — that the macro
    contribution to ΔF_total vanishes — is captured by
    `create_macro_DKL_change_zero` (in `Update.lean`), which expresses
    the equality of `D_KL(blank0 ‖ uniform)` and `D_KL(logical1 ‖ uniform)`
    against the full-support uniform prior.

    This lemma is retained only as a computational stepping stone for
    the alternate identity `create_delete_contrast_formal` and any
    downstream simp use. It should NOT be cited in the paper as a
    Gibbs KL value. -/
theorem klDiv_blank0_logical1_formal :
    klDiv MacroDist.blank0.toFinDist MacroDist.logical1.toFinDist = 0 := by
  unfold klDiv
  rw [logicalState_sum']
  simp only [MacroDist.toFinDist, MacroDist.blank0, MacroDist.logical1]
  norm_num

/-- D_KL(logical1 ‖ logical1) = 0. Trivially (genuine Gibbs value). -/
theorem klDiv_logical1_self :
    klDiv MacroDist.logical1.toFinDist MacroDist.logical1.toFinDist = 0 :=
  klDiv_self _

/-- **Create is not Landauer-limited** (paper result: ΔF_total ≈ −0.0373).

    Hypothesis: `ΔF_total = 0` for the Create process. This hypothesis
    is justified at the macro level by `create_macro_DKL_change_zero`
    (in `Update.lean`), which establishes that the macro contribution
    `ΔD_KL^macro(Create) = D_KL(logical1 ‖ uniform) − D_KL(blank0 ‖ uniform)`
    vanishes; combined with `ΔF_eq = 0` for an isothermal known→known
    transition, the total free-energy change collapses to zero.

    Conclusion: the work bound `⟨W⟩ ≥ ΔF_total` reduces to
    `⟨W⟩ ≥ 0` — there is no irreducible kT ln 2 floor.
    The thermodynamic cost is dominated by protocol dissipation. -/
theorem create_not_landauer_limited
    (sl : SecondLawWitness)
    (h_deltaF : sl.deltaF_total = 0) :
    0 ≤ sl.meanWork := by
  rw [← h_deltaF]
  exact sl.work_bound

/-- **Convention-dependent contrast identity** (NOT a physical claim).

    Under Mathlib's `Real.log 0 = 0` convention, the syntactic difference
      klDiv(blank0 ‖ uniform) − klDiv(blank0 ‖ logical1) = ln 2 − 0 = ln 2.

    The first term is a genuine Gibbs value; the second is the formal
    artifact `klDiv_blank0_logical1_formal`. This identity is therefore
    convention-dependent and is retained only as a computational lemma.

    For the **physical** Delete-vs-Create contrast at the macro level,
    see `crud_macro_classification_physical` in `Update.lean`, which
    uses only genuine Gibbs values against the uniform prior. -/
theorem create_delete_contrast_formal :
    klDiv MacroDist.blank0.toFinDist MacroDist.uniform.toFinDist -
    klDiv MacroDist.blank0.toFinDist MacroDist.logical1.toFinDist
    = Real.log 2 := by
  rw [klDiv_blank0_uniform, klDiv_blank0_logical1_formal]
  ring

end CRUDThermo
-- Create: Write-like Operation
