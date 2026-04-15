/-
  CRUDThermo/Thermodynamics/JarzynskiBound.lean

  The unified CRUD work bound:
    ⟨W⟩ ≥ ΔF_eq + T · ΔD_KL^full ≡ ΔF_total

  This is the generalized Landauer inequality (Eq. 2).
  Jarzynski equality is taken as a physical axiom (hypothesis),
  and the bound is derived via Jensen's inequality for exp.

  Reference: "From Erasure to CRUD", Eq. (2)
-/
import CRUDThermo.Thermodynamics.FreeEnergy

namespace CRUDThermo

variable {Ω : Type*} [Fintype Ω]

/-- Expectation of a real observable over a finite distribution. -/
noncomputable def expectation (p : FinDist Ω) (f : Ω → ℝ) : ℝ :=
  ∑ x : Ω, p.prob x * f x

/-- Affine observables commute with finite expectation. -/
theorem expectation_affine (p : FinDist Ω) (f : Ω → ℝ) (a b : ℝ) :
    expectation p (fun x => a * f x + b) = a * expectation p f + b := by
  unfold expectation
  calc
    ∑ x : Ω, p.prob x * (a * f x + b)
      = ∑ x : Ω, (a * (p.prob x * f x) + p.prob x * b) := by
          apply Finset.sum_congr rfl
          intro x hx
          ring
    _ = (∑ x : Ω, a * (p.prob x * f x)) + (∑ x : Ω, p.prob x * b) := by
          rw [Finset.sum_add_distrib]
    _ = a * (∑ x : Ω, p.prob x * f x) + (∑ x : Ω, p.prob x) * b := by
          rw [Finset.mul_sum, Finset.sum_mul]
    _ = a * (∑ x : Ω, p.prob x * f x) + b := by
          rw [p.sum_one]
          ring

/-- Pointwise tangent bound `x ≤ exp x - 1`. This is the `exp`
    Jensen ingredient used to derive the second law from Jarzynski. -/
private lemma le_exp_sub_one (x : ℝ) : x ≤ Real.exp x - 1 := by
  linarith [Real.add_one_le_exp x]

/-- Special-case Jensen for `exp` on a finite distribution:
    `E[X] ≤ E[exp X] - 1`. -/
theorem expectation_le_exp_sub_one (p : FinDist Ω) (f : Ω → ℝ) :
    expectation p f ≤ expectation p (fun x => Real.exp (f x)) - 1 := by
  unfold expectation
  have hpoint :
      ∀ x, p.prob x * f x ≤ p.prob x * (Real.exp (f x) - 1) := by
    intro x
    exact mul_le_mul_of_nonneg_left (le_exp_sub_one (f x)) (p.nonneg x)
  calc
    ∑ x : Ω, p.prob x * f x
      ≤ ∑ x : Ω, p.prob x * (Real.exp (f x) - 1) := by
          exact Finset.sum_le_sum (fun x hx => hpoint x)
    _ = ∑ x : Ω, (p.prob x * Real.exp (f x) - p.prob x * (1 : ℝ)) := by
          apply Finset.sum_congr rfl
          intro x hx
          ring
    _ = (∑ x : Ω, p.prob x * Real.exp (f x)) - (∑ x : Ω, p.prob x * (1 : ℝ)) := by
          rw [Finset.sum_sub_distrib]
    _ = (∑ x : Ω, p.prob x * Real.exp (f x)) - 1 := by
          simpa using congrArg (fun t : ℝ => (∑ x : Ω, p.prob x * Real.exp (f x)) - t)
            (by simpa using p.sum_one)

/-- Rewriting the centered Jarzynski random variable through expectation. -/
theorem expectation_neg_div_centered
    (p : FinDist Ω) (f : Ω → ℝ) (μ T : ℝ) (hT : T ≠ 0) :
    expectation p (fun x => -((f x - μ) / T)) =
      -((expectation p f - μ) / T) := by
  have hfun :
      (fun x => -((f x - μ) / T)) = fun x => (-(1 / T)) * f x + μ / T := by
    funext x
    field_simp [hT]
    ring
  rw [hfun, expectation_affine]
  field_simp [hT]
  ring

/-- A common second-law layer: work is decomposed into
    `meanWork = deltaF_total + dissipativeWork`, and the dissipative
    contribution is nonnegative. This packages the physical content
    once so downstream CRUD theorems do not each assume it separately. -/
structure SecondLawWitness where
  /-- Total nonequilibrium free-energy change -/
  deltaF_total : ℝ
  /-- Average work performed on the system -/
  meanWork : ℝ
  /-- Dissipative work -/
  dissipativeWork : ℝ
  /-- Defining relation: W_diss = ⟨W⟩ - ΔF_total -/
  diss_def : dissipativeWork = meanWork - deltaF_total
  /-- Second-law content: dissipative work is nonnegative -/
  dissipative_nonneg : 0 ≤ dissipativeWork

namespace SecondLawWitness

variable (w : SecondLawWitness)

/-- The second-law work bound `ΔF_total ≤ ⟨W⟩`. -/
theorem work_bound : w.deltaF_total ≤ w.meanWork := by
  linarith [w.diss_def, w.dissipative_nonneg]

/-- Work decomposes into free-energy change plus dissipation. -/
theorem work_decomposition :
    w.meanWork = w.deltaF_total + w.dissipativeWork := by
  linarith [w.diss_def]

end SecondLawWitness

/-- A thermodynamic protocol applied to a CRUD process.
    Records the average work and the process it acts on. -/
structure ThermodynamicProtocol (Ω : Type*) [Fintype Ω] where
  process : CRUDProcess Ω
  /-- Average work performed on the system -/
  meanWork : ℝ
  /-- Dissipative work: W_diss = ⟨W⟩ - ΔF_total -/
  dissipativeWork : ℝ
  /-- Defining relation: W_diss = ⟨W⟩ - ΔF_total -/
  diss_def : dissipativeWork = meanWork - process.deltaF_total

/-- A thermodynamic protocol together with the second-law statement
    that its dissipative work is nonnegative. -/
structure SecondLawProtocol (Ω : Type*) [Fintype Ω] where
  toProtocol : ThermodynamicProtocol Ω
  dissipative_nonneg : 0 ≤ toProtocol.dissipativeWork

/-- A Jarzynski certificate for a thermodynamic protocol.
    It supplies a finite trajectory ensemble whose work statistics
    reproduce `meanWork`, together with the Jarzynski equality
    `E[exp(-(W-ΔF_total)/T)] = 1`. -/
structure JarzynskiCertificate
    (proto : ThermodynamicProtocol Ω) (Γ : Type*) [Fintype Γ] where
  trajectories : FinDist Γ
  work : Γ → ℝ
  meanWork_def : proto.meanWork = expectation trajectories work
  jarzynski_eq :
    expectation trajectories
      (fun γ =>
        Real.exp (-((work γ - proto.process.deltaF_total) / proto.process.initial.T)))
    = 1

namespace JarzynskiCertificate

/-- Jarzynski plus the `exp` Jensen inequality imply that
    dissipative work is nonnegative. -/
theorem dissipative_nonneg
    {Γ : Type*} [Fintype Γ]
    {proto : ThermodynamicProtocol Ω}
    (cert : JarzynskiCertificate proto Γ) :
    0 ≤ proto.dissipativeWork := by
  let X : Γ → ℝ :=
    fun γ => -((cert.work γ - proto.process.deltaF_total) / proto.process.initial.T)
  have h_jensen : expectation cert.trajectories X ≤ 0 := by
    have h := expectation_le_exp_sub_one cert.trajectories X
    dsimp [X] at h
    rw [cert.jarzynski_eq] at h
    linarith
  have h_expect :
      expectation cert.trajectories X =
        -((proto.meanWork - proto.process.deltaF_total) / proto.process.initial.T) := by
    dsimp [X]
    rw [expectation_neg_div_centered cert.trajectories cert.work
      proto.process.deltaF_total proto.process.initial.T
      (ne_of_gt proto.process.initial.T_pos)]
    rw [← cert.meanWork_def]
  have h_gap_nonneg : 0 ≤ proto.meanWork - proto.process.deltaF_total := by
    rw [h_expect] at h_jensen
    have h_scaled :
        (-((proto.meanWork - proto.process.deltaF_total) / proto.process.initial.T)) *
          proto.process.initial.T ≤ 0 := by
      simpa using
        mul_le_mul_of_nonneg_right h_jensen (le_of_lt proto.process.initial.T_pos)
    have h_cancel :
        (-((proto.meanWork - proto.process.deltaF_total) / proto.process.initial.T)) *
          proto.process.initial.T =
        -(proto.meanWork - proto.process.deltaF_total) := by
      field_simp [ne_of_gt proto.process.initial.T_pos]
    rw [h_cancel] at h_scaled
    linarith
  linarith [proto.diss_def, h_gap_nonneg]

/-- A Jarzynski certificate canonically upgrades a protocol to the
    shared second-law layer. -/
def toSecondLawProtocol
    {Γ : Type*} [Fintype Γ]
    {proto : ThermodynamicProtocol Ω}
    (cert : JarzynskiCertificate proto Γ) :
    SecondLawProtocol Ω where
  toProtocol := proto
  dissipative_nonneg := dissipative_nonneg cert

end JarzynskiCertificate

namespace SecondLawProtocol

variable (sl : SecondLawProtocol Ω)

/-- Forgetting the state-space details yields the common second-law witness. -/
noncomputable def witness : SecondLawWitness where
  deltaF_total := sl.toProtocol.process.deltaF_total
  meanWork := sl.toProtocol.meanWork
  dissipativeWork := sl.toProtocol.dissipativeWork
  diss_def := sl.toProtocol.diss_def
  dissipative_nonneg := sl.dissipative_nonneg

/-- `ΔF_total ≤ ⟨W⟩` for any protocol satisfying the second law. -/
theorem crud_work_bound :
    sl.toProtocol.process.deltaF_total ≤ sl.toProtocol.meanWork :=
  sl.witness.work_bound

/-- Expanded form `ΔF_eq + T·ΔD_KL^full ≤ ⟨W⟩`. -/
theorem crud_work_bound_expanded :
    sl.toProtocol.process.deltaF_eq +
      sl.toProtocol.process.initial.T * sl.toProtocol.process.deltaDKL
    ≤ sl.toProtocol.meanWork := by
  have h := sl.crud_work_bound
  rwa [CRUDProcess.deltaF_total] at h

/-- Work decomposition in protocol form. -/
theorem work_decomposition :
    sl.toProtocol.meanWork =
      sl.toProtocol.process.deltaF_total + sl.toProtocol.dissipativeWork :=
  sl.witness.work_decomposition

/-- Hamiltonian-cycle specialization of the CRUD work bound. -/
theorem crud_bound_hamiltonian_cycle
    (h_cycle : sl.toProtocol.process.deltaF_eq = 0) :
    sl.toProtocol.process.initial.T * sl.toProtocol.process.deltaDKL
    ≤ sl.toProtocol.meanWork := by
  have h := sl.crud_work_bound_expanded
  linarith

end SecondLawProtocol

namespace ThermodynamicProtocol

variable (proto : ThermodynamicProtocol Ω)

/-- **The CRUD work bound** (Eq. 2 of the paper):
    ⟨W⟩ ≥ ΔF_total

    This is the unified generalization of Landauer's principle.
    We take nonnegativity of dissipative work as a physical hypothesis
    (derived from Jarzynski equality + Jensen's inequality).

    The hypothesis `nonneg_diss` encodes the second law:
    no process can extract work beyond the free-energy difference. -/
theorem crud_work_bound
    (nonneg_diss : 0 ≤ proto.dissipativeWork) :
    proto.process.deltaF_total ≤ proto.meanWork := by
  let sl : SecondLawProtocol Ω := {
    toProtocol := proto
    dissipative_nonneg := nonneg_diss
  }
  exact sl.crud_work_bound

/-- **Decomposed form**: ⟨W⟩ ≥ ΔF_eq + T · ΔD_KL^full -/
theorem crud_work_bound_expanded
    (nonneg_diss : 0 ≤ proto.dissipativeWork) :
    proto.process.deltaF_eq +
      proto.process.initial.T * proto.process.deltaDKL
    ≤ proto.meanWork := by
  let sl : SecondLawProtocol Ω := {
    toProtocol := proto
    dissipative_nonneg := nonneg_diss
  }
  exact sl.crud_work_bound_expanded

/-- **Work decomposition**: ⟨W⟩ = ΔF_total + W_diss -/
theorem work_decomposition :
    proto.meanWork = proto.process.deltaF_total + proto.dissipativeWork := by
  linarith [proto.diss_def]

/-- When the process is a Hamiltonian cycle (ΔF_eq = 0, same endpoints),
    the bound simplifies to: ⟨W⟩ ≥ T · ΔD_KL^full.
    This is the relevant form for Read (Eq. 4 context). -/
theorem crud_bound_hamiltonian_cycle
    (nonneg_diss : 0 ≤ proto.dissipativeWork)
    (h_cycle : proto.process.deltaF_eq = 0) :
    proto.process.initial.T * proto.process.deltaDKL
    ≤ proto.meanWork := by
  let sl : SecondLawProtocol Ω := {
    toProtocol := proto
    dissipative_nonneg := nonneg_diss
  }
  exact sl.crud_bound_hamiltonian_cycle h_cycle

end ThermodynamicProtocol

end CRUDThermo
