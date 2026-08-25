```@meta
CurrentModule = ClassicalDFT
```

# Other Functionals

## Contents

```@contents
Pages = ["other.md"]
Depth = 1
```

## COFFEE and PeTS

Two further Weighted Density Functionals developed by Langenbach (2017), neither requiring
a chain propagator. Bulk equations of state are again obtained from Clapeyron.

```@docs
ClassicalDFT.COFFEE
ClassicalDFT.PeTS
```

## Density Gradient Theory (DGT)

A cheaper alternative route to interfacial properties: rather than a full weighted-density
functional, [`DGTSystem`](@ref ClassicalDFT.DGTSystem) augments a bulk `model` with a square-gradient
correction, parameterised by an influence parameter supplied by a `GradientModel` such as
`ConstGradient`.

```@docs
ClassicalDFT.ConstGradient
```
