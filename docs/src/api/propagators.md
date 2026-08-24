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
to assemble density profiles directly.

```@docs
ClassicalDFT.IdealPropagator
ClassicalDFT.TangentHSPropagator
ClassicalDFT.DiscreteGaussianChainPropagator
```

## Functions

```@docs
ClassicalDFT.propagate!
ClassicalDFT.preallocate_propagator
```
