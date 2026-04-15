/-
  CRUDThermo/CRUD/Operations.lean

  Formal definitions of CRUD operations as thermodynamic processes.
  Each operation is characterized by its initial/final state conditions,
  which determine the information free-energy contribution ΔF_total.

  Reference: "From Erasure to CRUD", Section 2-4
-/
import CRUDThermo.Thermodynamics.FreeEnergy
import CRUDThermo.Basic.KLDivergence

namespace CRUDThermo

/-- Logical state in a double-well memory: Left (0) or Right (1). -/
inductive LogicalState
  | L  -- left well = logical 0 = "blank"
  | R  -- right well = logical 1
  deriving Fintype, DecidableEq, Repr

/-- Basin-level (macro) probability distribution over {L, R}. -/
structure MacroDist where
  pL : ℝ
  pR : ℝ
  pL_nonneg : 0 ≤ pL
  pR_nonneg : 0 ≤ pR
  sum_one : pL + pR = 1

namespace MacroDist

/-- Convert MacroDist to a FinDist over LogicalState. -/
noncomputable def toFinDist (d : MacroDist) : FinDist LogicalState where
  prob := fun s => match s with
    | .L => d.pL
    | .R => d.pR
  nonneg := fun s => match s with
    | .L => d.pL_nonneg
    | .R => d.pR_nonneg
  sum_one := by
    show ∑ x : LogicalState, _ = 1
    have : (Finset.univ : Finset LogicalState) = {.L, .R} := by decide
    rw [this, Finset.sum_pair (by decide)]
    exact d.sum_one

/-- Uniform (maximally uncertain) distribution: p = (1/2, 1/2). -/
noncomputable def uniform : MacroDist where
  pL := 1 / 2
  pR := 1 / 2
  pL_nonneg := by norm_num
  pR_nonneg := by norm_num
  sum_one := by norm_num

/-- Concentrated in left basin: p = (1, 0). "blank0" state. -/
def blank0 : MacroDist where
  pL := 1
  pR := 0
  pL_nonneg := by norm_num
  pR_nonneg := le_refl 0
  sum_one := by norm_num

/-- Concentrated in right basin: p = (0, 1). "logical 1" state. -/
def logical1 : MacroDist where
  pL := 0
  pR := 1
  pL_nonneg := le_refl 0
  pR_nonneg := by norm_num
  sum_one := by norm_num

end MacroDist

/-- Classification of CRUD operation types by their initial/final
    macro distributions. This is the core taxonomy of the paper. -/
inductive CRUDType
  | Create  -- blank0 → logical1 (known → known)
  | Read    -- measured via system-meter coupling (separate treatment)
  | Update  -- uniform → logical1 (unknown → known, overwrite)
  | Delete  -- uniform → blank0 (unknown → known, Landauer case)
  deriving Repr

/-- A typed CRUD operation: bundles the operation type with
    its initial and final macro distributions. -/
structure CRUDOperation where
  opType : CRUDType
  initial : MacroDist
  final : MacroDist

namespace CRUDOperation

/-- Create: blank0 → logical1 -/
def mkCreate : CRUDOperation where
  opType := .Create
  initial := MacroDist.blank0
  final := MacroDist.logical1

/-- Update: uniform → logical1 (overwrite) -/
noncomputable def mkUpdate : CRUDOperation where
  opType := .Update
  initial := MacroDist.uniform
  final := MacroDist.logical1

/-- Delete: uniform → blank0 (Landauer erasure) -/
noncomputable def mkDelete : CRUDOperation where
  opType := .Delete
  initial := MacroDist.uniform
  final := MacroDist.blank0

end CRUDOperation

end CRUDThermo
