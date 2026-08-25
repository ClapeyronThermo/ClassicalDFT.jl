# FAQ

## Why are spherical and cylindrical geometries CPU-only?

[`Uniform1DSphr`](@ref ClassicalDFT.Uniform1DSphr) and [`Uniform1DCyl`](@ref ClassicalDFT.Uniform1DCyl) use
a quasi-discrete Hankel transform ([Hankel.jl](https://github.com/jwscook/Hankel.jl)) rather
than an FFT to perform radial convolutions, and that implementation does not currently run
on the GPU. If you need a curved geometry on the GPU, embed it in a Cartesian box instead
(e.g. [`TwoPhase3DSphrCart`](@ref ClassicalDFT.TwoPhase3DSphrCart) for a spherical droplet) — see
[Choosing a Geometry & Adsorption](@ref).

## Why did my density profile turn into `NaN`?

The most common cause is a density that underflows to an exact `0.0` right at a hard wall
or excluded-volume boundary: `log`/association terms in the free-energy functional produce
`NaN` for exactly-zero density, and this then spreads to the entire profile after one
convolution pass. Wall-type external fields (e.g. [`Steele`](@ref ClassicalDFT.Steele)) clamp the
minimum wall distance to `0.5*minimum(σ)` for exactly this reason — if you're writing a
custom external field, make sure it does the same near any hard boundary.

## Why is my first calculation so slow?

Model free-energy kernels are automatically differentiated with
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) and compiled the first time they're
called for a given model/structure/device combination — this one-time compilation can take
anywhere from several seconds to a few minutes for the more complex functionals (e.g.
electrolyte models), independent of the actual convergence time. Subsequent calls with the
same types are fast. If you're iterating on a script, keep a persistent Julia session
(e.g. via [Revise.jl](https://github.com/timholy/Revise.jl)) rather than restarting Julia
for every run.

## Why do I get an error asking me to load `GCIdentifier`/`ChemicalIdentifiers`?

Resolving group-contribution connectivity automatically from a SMILES string or chemical
name (rather than a hand-written [`custom_structure`](@ref ClassicalDFT.custom_structure)) is
implemented as a package extension. Add and load both packages:

```julia
julia> using Pkg; Pkg.add(["GCIdentifier", "ChemicalIdentifiers"])

julia> using GCIdentifier, ChemicalIdentifiers, ClassicalDFT
```

See [Group-Contribution & Heterosegmented Chains](@ref).

## How do I run a calculation on the GPU?

Load `CUDA` *before* constructing your [`DFTOptions`](@ref ClassicalDFT.DFTOptions), then pass a
`CUDABackend()`:

```julia
julia> using CUDA

julia> options = ClassicalDFT.DFTOptions(CUDABackend())
```

Remember that spherical/cylindrical structures cannot run on the GPU (see above). See
[GPU Acceleration](@ref) for when GPU acceleration is actually worth the transfer overhead.
