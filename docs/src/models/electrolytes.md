```@meta
CurrentModule = ClassicalDFT
```

# Electrolytes

As with previous DFT models, ClassicalDFT does not define its own bulk electrolyte
equations of state — those come directly from Clapeyron (e.g. `ePCSAFT`, built from a
neutral model, an ion model and a set of charges via Clapeyron's `ElectrolyteModel`).
ClassicalDFT's contribution is the machinery that turns a Clapeyron `ElectrolyteModel` into an
inhomogeneous DFT calculation, which is split across two other reference pages:

- [`ElectrolyteDFTSystem`](@ref ClassicalDFT.ElectrolyteDFTSystem) (see [System](../api/system.md)) —
  composes the neutral-model functional with the ion-model functional below, and
  automatically attaches the mean-field electrostatic external field.
- [`ElectrostaticPotential`](@ref ClassicalDFT.ElectrostaticPotential) (see
  [External Fields](@ref)) — the mean-field Coulomb term, added automatically; you should
  not normally need to construct it yourself.

See the [Electrolytes tutorial](../tutorials/electrolytes.md) for a worked example.

## Contents

```@contents
Pages = ["electrolytes.md"]
Depth = 1
```

## Ion Models

`DH` is the Debye-Hückel ion-ion correction used as the `ionmodel` half of a Clapeyron
`ElectrolyteModel`.

```@docs
ClassicalDFT.DH
```
