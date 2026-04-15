# CRUDThermo

Lean 4 formalization of the information-thermodynamic CRUD framework, companion to the paper *"From Erasure to CRUD"* by Masamichi Iizumi (Miosync, Inc.).

## Overview

This project provides machine-checked proofs that database-style CRUD (Create, Read, Update, Delete) operations have distinct thermodynamic signatures when viewed through the lens of nonequilibrium statistical mechanics. The central result is a unified work bound

$$\langle W \rangle \geq \Delta F_{\mathrm{eq}} + T \cdot \Delta D_{\mathrm{KL}}$$

from which Landauer's principle ($\langle W \rangle \geq kT \ln 2$ for erasure) emerges as the Delete special case.

## Formalized Results

| Theorem | File | Statement |
|---------|------|-----------|
| Gibbs' inequality | `Basic/KLDivergence.lean` | $D_{\mathrm{KL}}(p \| q) \geq 0$ for full-support $q$ |
| KL chain rule | `Basic/ChainRule.lean` | $D_{\mathrm{KL}}^{\mathrm{joint}} = D_{\mathrm{KL}}^{\mathrm{macro}} + \sum_m p_m \, D_{\mathrm{KL}}^{\mathrm{cond}}$ |
| Mutual info non-negativity | `Basic/MutualInformation.lean` | $I(X;Y) \geq 0$ |
| Free energy bound | `Thermodynamics/FreeEnergy.lean` | $F[p;\lambda] \geq F_{\mathrm{eq}}(\lambda)$ |
| CRUD work bound (Eq. 2) | `Thermodynamics/JarzynskiBound.lean` | $\langle W \rangle \geq \Delta F_{\mathrm{total}}$ |
| Sagawa-Ueda bound (Eq. 4) | `Thermodynamics/SagawaUeda.lean` | $W_{\mathrm{diss}}^{\mathrm{read}} \geq -T \cdot I(X:Y)$ |
| Landauer's principle | `CRUD/Delete.lean` | $D_{\mathrm{KL}}(\mathrm{blank0} \| \mathrm{uniform}) = \ln 2$ |
| Create escapes Landauer | `CRUD/Create.lean` | $D_{\mathrm{KL}}(\mathrm{blank0} \| \mathrm{logical1}) = 0$ |
| Update/Delete same macro cost | `CRUD/Update.lean` | Both have macro $D_{\mathrm{KL}} = \ln 2$ |
| CRUD macro classification | `CRUD/Update.lean` | Delete, Update: $\ln 2$; Create: $0$ |

All proofs are complete with **zero `sorry`** statements.

## Architecture

```
CRUDThermo/
  Basic/
    KLDivergence.lean      -- FinDist, klDiv, Gibbs' inequality
    ChainRule.lean          -- JointDist, marginal/conditional, chain rule
    MutualInformation.lean  -- I(X;Y) via KL divergence
  Thermodynamics/
    FreeEnergy.lean         -- MemoryState, CRUDProcess, F[p;lambda]
    JarzynskiBound.lean     -- SecondLawWitness, unified work bound
    SagawaUeda.lean         -- MeasurementProtocol, Read bound
  CRUD/
    Operations.lean         -- LogicalState, MacroDist, CRUDType
    Delete.lean             -- Landauer erasure (uniform -> blank0)
    Create.lean             -- bit flip (blank0 -> logical1)
    Update.lean             -- overwrite (uniform -> logical1)
    Read.lean               -- measurement (system-meter coupling)
```

### Dependency graph

```
KLDivergence <-- ChainRule
      ^              ^
      |              |
MutualInformation    |
      ^              |
      |              |
FreeEnergy --> JarzynskiBound
      |              |
      v              v
Operations --> Delete/Create/Update/Read
      ^
      |
SagawaUeda
```

## Key Structures

| Structure | Description |
|-----------|-------------|
| `FinDist Omega` | Probability distribution over a finite type |
| `JointDist M Omega` | Joint distribution with macro/micro partition |
| `MacroDist` | Basin-level distribution over `{L, R}` |
| `MemoryState Omega` | Physical memory state $(p, \pi, F_{\mathrm{eq}}, T)$ |
| `CRUDProcess Omega` | Initial/final memory states in a CRUD process |
| `SecondLawWitness` | Second-law constraint $0 \leq W_{\mathrm{diss}}$ |
| `MeasurementProtocol X Y` | System-meter coupling for Read |
| `PassiveMeasurementProtocol X Y` | Measurement with $W_{\mathrm{diss}} \geq 0$ |

## Building

Requires Lean 4 v4.29.0 and Mathlib v4.29.0.

```bash
lake build
```

## Physical Assumptions

The second law ($W_{\mathrm{diss}} \geq 0$) is either:
- Encoded as a field in `SecondLawWitness` / `PassiveMeasurementProtocol`, or
- Derived from a `JarzynskiCertificate` (finite trajectory ensemble satisfying Jarzynski equality).

No `axiom` or `sorry` is used anywhere in the project.

## Reference

> Masamichi Iizumi. *"From Erasure to CRUD: A Unified Information-Thermodynamic Framework."*

## License

See [LICENSE](LICENSE) for details.
