```@meta
CurrentModule = ClassicalDFT
```

## Contents

```@contents
Pages = ["methods.md"]
Depth = 1
```

## Solvers

```@docs
ClassicalDFT.cDFTProblem
ClassicalDFT.DFTProblem
ClassicalDFT.SCFTProblem
ClassicalDFT.converge!
```

### Anderson solver

```@docs
ClassicalDFT.aasol
ClassicalDFT.AASol
```

## Properties

```@docs
ClassicalDFT.surface_tension
ClassicalDFT.interfacial_tension
ClassicalDFT.adsorption
```

## Orientation (WLC only)

A [`WLCPropagator`](@ref ClassicalDFT.WLCPropagator) system tracks each species' full 3D
bond orientation alongside position, so density isn't the only quantity that can be
resolved in space — see the [Self-Consistent Field Theory](../tutorials/scft.md)
tutorial. Every function below (and in the two sections after it) takes the converged
`ρ` directly; `q_in`/`q_out` are computed from it internally
([`compute_propagator_fields`](@ref)) unless passed as the optional `q_in`/`q_out`
keywords, for callers who already have them (or want to reuse them across several calls
without recomputing).

```@docs
ClassicalDFT.compute_densities!
ClassicalDFT.orientation_order_parameter
ClassicalDFT.compute_orientation_tensor
ClassicalDFT.compute_orientation_moments
ClassicalDFT.mean_orientation_field
```

## Chain Conformation Sampling (WLC only)

Every function above is an ENSEMBLE-AVERAGED marginal (a density, an order parameter, a
polar vector field) — none of them is a single chain's actual shape. These reconstruct
one concrete bead-by-bead conformation consistent with the converged mean field,
conditioned on one bead sitting at a chosen (or most-likely/mean) position and
orientation:

```@docs
ClassicalDFT.compute_propagator_fields
ClassicalDFT.bead_marginal_density
ClassicalDFT.terminal_orientation_anchor
ClassicalDFT.sample_chain
ClassicalDFT.periodic_interp
```

## Visualization (WLC only, requires Makie)

`plot_orientation_field`/`plot_chain_conformations` are stubs in the main package;
loading a Makie backend (e.g. `using CairoMakie`) provides the actual methods, following
the same pattern `Makie.plot(system, ρ)` (see [System](../api/system.md)) already uses
for density profiles. Like every function above, both take `ρ` directly and compute
`q_in`/`q_out` internally unless given as keywords.

```@docs
ClassicalDFT.plot_orientation_field
ClassicalDFT.plot_chain_conformations
```
