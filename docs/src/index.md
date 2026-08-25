````@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: ClassicalDFT.jl
  text: Classical Density Functional Theory in Julia
  image:
    src: logo.png
    alt: ClassicalDFT.jl
  tagline: A comprehensive, extensible library of classical DFT and Self-Consistent Field Theory models, built directly on top of Clapeyron.jl
  actions:
    - theme: brand
      text: Getting started
      link: /tutorials/getting_started
    - theme: alt
      text: Installation
      link: /installation
    - theme: alt
      text: View on GitHub
      link: https://github.com/ClapeyronThermo/ClassicalDFT.jl

features:
  - icon: ⚛️
    title: Many free-energy functionals
    details: PC-SAFT, SAFT-VR Mie, SAFT-γ Mie, COFFEE, electrolytes, Self-Consistent Field Theory and more, all sharing Clapeyron.jl's equations of state as the underlying bulk model
    link: /models/saft

  - icon: 🧱
    title: Flexible geometries
    details: Planar, cylindrical, spherical and fully 3D structures, plus block-copolymer microphase morphologies (BCC, gyroid, hex, lamellar)
    link: /structures

  - icon: ⚡
    title: GPU acceleration
    details: The same model code runs unchanged on CPU or GPU via KernelAbstractions.jl and Enzyme.jl
    link: /tutorials/gpu_acceleration

  - icon: ⏱️
    title: Dynamic DFT
    details: Evolve density profiles in time as well as space, to watch phase separation and microphase ordering actually happen
    link: /tutorials/dynamic_dft
---
````

```@meta
CurrentModule = ClassicalDFT
```

## What is this?

ClassicalDFT intends to provide a comprehensive library of classical Density Functional Theory
(ClassicalDFT) and Self-Consistent Field Theory models, as well as a simple framework to develop
your own! ClassicalDFT is built directly on top of [Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl)
and reuses its equations of state as the bulk free-energy model underlying every
inhomogeneous calculation — the two packages are meant to be used together.

With ClassicalDFT you can compute density profiles, surface/interfacial tensions and adsorption
isotherms for fluids next to walls, in pores, around solutes, at vapour-liquid and
liquid-liquid interfaces, in microphase-separated copolymer melts, and for electrolytes
near surfaces — in 1D, 2D or 3D, on the CPU or GPU, and (via Dynamic DFT) as a
function of time as well as space.

## Quick start guide

### Minimal Installation

```julia
using Pkg
Pkg.add(["ClassicalDFT", "Clapeyron"])
```

See [Installation](@ref) for optional extensions (plotting, GPU, group-contribution
connectivity, Dynamic DFT).

### A first calculation

```julia
using Clapeyron, ClassicalDFT

model = PCSAFT(["methane"])
T, p = 150.0, 1e7
v = Clapeyron.volume(model, p, T, [1.0]; phase=:liquid)
ρbulk = [1/v]
L = ClassicalDFT.length_scale(model)

width = 5L
surface = Steele(["graphite"], width)
structure = Uniform1DCart((p, T), ρbulk, [0.5L, width-0.5L], 201)

system = DFTSystem(model, structure, surface)
ρ = initialize_profiles(system)
converge!(system, ρ)
```

`ρ` is now the converged density profile of liquid methane next to a graphite wall. See
[Getting Started](@ref) for the full walkthrough, including plotting the result.

## Citing ClassicalDFT.jl

ClassicalDFT.jl does not yet have a dedicated publication — for now, please cite the
[GitHub repository](https://github.com/ClapeyronThermo/ClassicalDFT.jl) directly. Since every ClassicalDFT
calculation is built on top of a Clapeyron.jl bulk equation of state, please also cite
[Clapeyron.jl](https://pubs.acs.org/doi/10.1021/acs.iecr.2c00326) itself, along with the
specific equation of state used (obtainable via `Clapeyron.cite(model)`) and, where
relevant, the original ClassicalDFT functional reference (e.g. Sauer & Gross, 2017 for the
weighted-density PC-SAFT functional; see each entry under [Available Models](@ref "SAFT-based Models")
for its reference).

## Related packages

````@raw html
<div class="related-pkg-grid">
  <a class="related-pkg-card" href="https://clapeyronthermo.github.io/Clapeyron.jl/" target="_blank" rel="noreferrer">
    <div class="related-pkg-logo-wrap">
      <img class="related-pkg-logo" src="/assets/related_clapeyron_logo.svg" alt="Clapeyron.jl" />
    </div>
    <h3 class="related-pkg-title">Clapeyron.jl</h3>
    <p class="related-pkg-details">Provides every bulk equation of state ClassicalDFT builds its inhomogeneous functionals on top of, and is required alongside ClassicalDFT for essentially all use.</p>
  </a>
  <a class="related-pkg-card" href="https://clapeyronthermo.github.io/GCIdentifier.jl/" target="_blank" rel="noreferrer">
    <div class="related-pkg-logo-wrap">
      <img class="related-pkg-logo" src="/assets/related_gcidentifier_logo.png" alt="GCIdentifier.jl" />
    </div>
    <h3 class="related-pkg-title">GCIdentifier.jl</h3>
    <p class="related-pkg-details">Group contribution identification from SMILES, used for building heterosegmented and group-contribution ClassicalDFT models.</p>
  </a>
  <a class="related-pkg-card" href="https://clapeyronthermo.github.io/Langmuir.jl/" target="_blank" rel="noreferrer">
    <div class="related-pkg-logo-wrap">
      <img class="related-pkg-logo" src="/assets/related_langmuir_logo.png" alt="Langmuir.jl" />
    </div>
    <h3 class="related-pkg-title">Langmuir.jl</h3>
    <p class="related-pkg-details">Single- and multi-component adsorption equilibrium models, complementary to ClassicalDFT's own adsorption isotherm calculations.</p>
  </a>
</div>

<style>
.related-pkg-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
  margin: 16px 0 32px;
}

.related-pkg-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 24px;
  border: 1px solid var(--vp-c-bg-soft);
  border-radius: 12px;
  background-color: var(--vp-c-bg-soft);
  text-decoration: none !important;
  transition: border-color 0.25s, background-color 0.25s;
}

.related-pkg-card:hover {
  border-color: var(--vp-c-brand-1);
}

.related-pkg-logo-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 96px;
  margin-bottom: 8px;
}

.related-pkg-logo {
  max-height: 100%;
  max-width: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
}

.related-pkg-title {
  margin: 0;
  line-height: 24px;
  font-size: 16px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  border-top: none;
  padding-top: 0;
}

.related-pkg-details {
  flex-grow: 1;
  margin: 8px 0 0;
  line-height: 22px;
  font-size: 14px;
  font-weight: 500;
  color: var(--vp-c-text-2);
}
</style>
````

## Authors

- [Pierre J. Walker](mailto:pjwalker@caltech.edu), California Institute of Technology
- [Andrés Riedemann](mailto:andres.riedemann@gmail.com), University of Concepción

## License

ClassicalDFT.jl is licensed under the [MIT license](https://github.com/ClapeyronThermo/ClassicalDFT.jl/blob/main/LICENSE.md).
