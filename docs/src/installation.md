# Installation

## Installing Julia

The latest version of Julia can be downloaded [here](https://julialang.org/downloads/),
with OS-specific instructions [here](https://julialang.org/downloads/platform). ClassicalDFT.jl
should function on all platforms; GPU acceleration additionally requires a CUDA-capable
GPU and driver (see [GPU Acceleration](@ref)).

If you're new to Julia, the [official tutorials](https://julialang.org/learning/tutorials/)
are a good starting point; basic use of ClassicalDFT does not require deep Julia knowledge, but
writing your own functional or extending an existing one benefits from familiarity with
multiple dispatch and broadcasting.

## Installing ClassicalDFT.jl

ClassicalDFT.jl is a registered package. Since every calculation needs a bulk equation of state,
you'll almost always want [Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl)
installed alongside it:

```julia
julia> using Pkg

julia> Pkg.add(["ClassicalDFT", "Clapeyron"])
```

Then, in any script or REPL session:

```julia
julia> using Clapeyron, ClassicalDFT
```

## Optional extensions

ClassicalDFT.jl uses Julia's package extension mechanism for a number of optional features — each
only activates once the corresponding package is also loaded, so you only pay for what you
use:

| To use...                                                       | Also install and `using`                              |
|:-----------------------------------------------------------------|:-------------------------------------------------------|
| Automatic group-contribution connectivity from a SMILES string or chemical name (see [Group-Contribution & Heterosegmented Chains](@ref)) | `GCIdentifier`, `ChemicalIdentifiers` |
| Plotting density profiles                                        | `Makie` (+ a backend such as `CairoMakie` or `GLMakie`), `Plots`, or `PlotlyJS` |
| Dynamic DFT time evolution (see [Dynamic DFT](@ref))              | `SciMLBase` and an ODE solver, e.g. `OrdinaryDiffEq`   |
| GPU acceleration (see [GPU Acceleration](@ref))                   | `CUDA`                                                  |
| Pinning CPU threads to specific cores                             | `ThreadPinning`                                         |

```julia
julia> Pkg.add(["GCIdentifier", "ChemicalIdentifiers", "CairoMakie"])

julia> using GCIdentifier, ChemicalIdentifiers, CairoMakie
```
