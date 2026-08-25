```@meta
CurrentModule = ClassicalDFT
```

# Self-Consistent Field Theory (SCFT)

SCFT takes a different theoretical route to the same block-copolymer melts the classical
DFT family reaches via [`LamellarStack*`/`HexLattice*`/`BCC3DCart`/`Gyroid3DCart`](@ref
"Block-Copolymer Microphase Morphologies") structures (see the
[Copolymer Microphase Morphologies tutorial](../tutorials/copolymer_morphology.md) for
that alternative): rather than a particle-based free-energy functional evaluated on a
density profile, SCFT solves for a set of self-consistent mean fields `w_α(r)`, one per
*species* (monomer type, e.g. `"A"`/`"B"` in a diblock — not one per bead position along
the chain), such that the density a single chain produces in that field matches the
density that generated it. This makes it substantially cheaper for large, flexible chain
architectures, at the cost of the coarser Flory-Huggins/Gaussian-chain approximations it
relies on in place of a full pairwise free-energy functional.

- [`SCFTSystem`](@ref ClassicalDFT.SCFTSystem) (see [System](../api/system.md)) — composes an
  `SCFTLatticeFluid` bulk model with a structure and chain architecture, mirroring
  `DFTSystem`.
- [`DiscreteGaussianChainPropagator`](@ref ClassicalDFT.DiscreteGaussianChainPropagator) (see
  [Propagators](../api/propagators.md)) — the chain propagator every `SCFTSystem` uses.

See the [Self-Consistent Field Theory tutorial](../tutorials/scft.md) for a worked
example.

## Contents

```@contents
Pages = ["scft.md"]
Depth = 1
```

## Lattice Fluid Model

`SCFTLatticeFluid` supplies the bulk interaction model: local Flory-Huggins
`χ`-interactions between species plus a Helfand compressibility penalty `κ` that
softly enforces incompressibility (`Σ_α ρ_α ≈ ρ₀`) rather than solving a hard
equation-of-state constraint. Chain architecture (which species occupy which positions
along each molecule type) is supplied separately, via `SCFTSystem`'s `mol_structure`
keyword — the same `custom_structure`/connectivity mechanism `HeterogcPCPSAFT`/
`SAFTgammaMie` use.

```@docs
ClassicalDFT.SCFTLatticeFluid
```

## Utilities

`compute_bulk_densities` returns the already-correct, per-species bulk density implied by
`structure.ρbulk`/`ensemble`/`n_molecules` — computed once when the `SCFTSystem` is built
(not recomputed on every call), and used internally by `initialize_profiles`/`converge!`,
but also handy on its own for inspecting a system's intended bulk composition.

```@docs
ClassicalDFT.compute_bulk_densities
```
