```@meta
CurrentModule = ClassicalDFT
```

## Contents

```@contents
Pages = ["propagators.md"]
Depth = 1
```

## Propagators

A propagator carries the connectivity between bonded beads of a chain molecule (or, for
[`IdealPropagator`](@ref ClassicalDFT.IdealPropagator), signals that a model has no chains at
all). It's how [`converge!`](@ref ClassicalDFT.converge!)'s fixed-point map turns each species'
own field into a chain-connectivity contribution to the functional derivative — for
SCFT (see [SCFT](../models/scft.md)), the same [`DiscreteGaussianChainPropagator`](@ref
ClassicalDFT.DiscreteGaussianChainPropagator) instead builds the forward/backward propagators used
to assemble density profiles directly. [`WLCPropagator`](@ref ClassicalDFT.WLCPropagator) is the
same idea for a bead-rod (worm-like) chain instead of a Gaussian one — see the [Worm-Like
Chains & Orientational Order](../tutorials/wlc.md) tutorial.

```@docs
ClassicalDFT.IdealPropagator
ClassicalDFT.TangentHSPropagator
ClassicalDFT.DiscreteGaussianChainPropagator
ClassicalDFT.WLCPropagator
```

## Functions

```@docs
ClassicalDFT.propagate!
ClassicalDFT.preallocate_propagator
```

## Worm-Like Chain Statistics

The discrete freely-rotating-chain bending statistics behind [`WLCPropagator`](@ref
ClassicalDFT.WLCPropagator) — a bond's own persistence-length-to-bond-length ratio `κ = lp/b`
maps to a per-spherical-harmonic-degree correlation via `bending_eigenvalues`, and
[`bend_kernel_value`](@ref ClassicalDFT.bend_kernel_value) evaluates the corresponding
real-space bond-angle density directly (used by [`sample_chain`](@ref
ClassicalDFT.sample_chain), see [Methods](../api/methods.md)):

```@docs
ClassicalDFT.bending_eigenvalues
ClassicalDFT.bend_kernel_value
ClassicalDFT.legendre_all
```
