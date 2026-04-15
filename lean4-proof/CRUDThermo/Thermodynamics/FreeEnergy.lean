/-
  CRUDThermo/Thermodynamics/FreeEnergy.lean

  Nonequilibrium free energy and its decomposition.
  F[p;λ] = F_eq(λ) + kT · D_KL(p ‖ π_λ)

  Reference: "From Erasure to CRUD", Eq. (1), (2), (8)
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import CRUDThermo.Basic.KLDivergence

namespace CRUDThermo

variable {Ω : Type*} [Fintype Ω]

/-- Physical memory state at a given protocol parameter λ.
    Bundles the nonequilibrium distribution, equilibrium distribution,
    equilibrium free energy, and temperature. -/
structure MemoryState (Ω : Type*) [Fintype Ω] where
  /-- Nonequilibrium state distribution p(x) -/
  p : FinDist Ω
  /-- Equilibrium (Boltzmann) distribution π_λ(x) -/
  π : FinDist Ω
  /-- Equilibrium free energy F_eq(λ) -/
  F_eq : ℝ
  /-- Temperature (in units where k=1) -/
  T : ℝ
  T_pos : 0 < T

namespace MemoryState

variable (s : MemoryState Ω)

/-- Nonequilibrium free energy: F[p;λ] = F_eq(λ) + T · D_KL(p ‖ π_λ)
    Eq. (1) of the paper (with k=1). -/
noncomputable def freeEnergy : ℝ :=
  s.F_eq + s.T * klDiv s.p s.π

/-- F[p;λ] ≥ F_eq(λ) when π has full support.
    The nonequilibrium free energy is always at least the equilibrium value. -/
theorem freeEnergy_ge_eq (hπ : ∀ x, 0 < s.π.prob x) :
    s.F_eq ≤ s.freeEnergy := by
  unfold freeEnergy
  linarith [mul_nonneg (le_of_lt s.T_pos) (klDiv_nonneg s.p s.π hπ)]

/-- Equality holds iff D_KL = 0, i.e., p = π (equilibrium). -/
theorem freeEnergy_eq_iff_eq (hπ : ∀ x, 0 < s.π.prob x) :
    s.freeEnergy = s.F_eq ↔ klDiv s.p s.π = 0 := by
  unfold freeEnergy
  constructor
  · intro h
    have hkl := klDiv_nonneg s.p s.π hπ
    nlinarith [s.T_pos]
  · intro h; rw [h, mul_zero, add_zero]

end MemoryState

/-- A CRUD process: initial and final memory states. -/
structure CRUDProcess (Ω : Type*) [Fintype Ω] where
  initial : MemoryState Ω
  final : MemoryState Ω
  T_eq : initial.T = final.T  -- isothermal

namespace CRUDProcess

variable (proc : CRUDProcess Ω)

/-- ΔF_eq = F_eq(final) - F_eq(initial) -/
noncomputable def deltaF_eq : ℝ :=
  proc.final.F_eq - proc.initial.F_eq

/-- ΔD_KL^full = D_KL(p_f ‖ π_f) - D_KL(p_i ‖ π_i) -/
noncomputable def deltaDKL : ℝ :=
  klDiv proc.final.p proc.final.π - klDiv proc.initial.p proc.initial.π

/-- ΔF_total = ΔF_eq + T · ΔD_KL^full
    The unified CRUD state-function bound. Eq. (8) of the paper. -/
noncomputable def deltaF_total : ℝ :=
  proc.deltaF_eq + proc.initial.T * proc.deltaDKL

/-- ΔF_total equals the difference of nonequilibrium free energies.
    This is the key identity connecting Eq. (1) and Eq. (8). -/
theorem deltaF_total_eq_freeEnergy_diff :
    proc.deltaF_total =
      proc.final.freeEnergy - proc.initial.freeEnergy := by
  unfold deltaF_total deltaF_eq deltaDKL
  unfold MemoryState.freeEnergy
  rw [proc.T_eq]
  ring

end CRUDProcess

end CRUDThermo
