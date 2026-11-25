# Self-Consistent Field Theory

[Self-Consistent Field Theory (SCFT)](../models/scft.md) reaches block-copolymer
microphase morphologies via mean fields `w_α(r)` per species rather than a particle-based
free-energy functional. This tutorial builds an `SCFTSystem` for a symmetric AB diblock
melt and converges it to the classic lamellar phase — the SCFT counterpart to the
[Copolymer Microphase Morphologies](@ref "Copolymer Microphase Morphologies") tutorial's
classical-DFT route to the same physics — then swaps components of the model.

## Building the model

An [`SCFTModel`](@ref ClassicalDFT.SCFTModel) takes a list of `(name, groups)`
components — here one component `"diblock"` made of 10 `A`-beads and 10 `B`-beads — a
reference total segment density `rho0`, a chain representation, and a tuple of
interactions:

```julia
julia> using ClassicalDFT

julia> N_seg = 20

julia> model = SCFTModel([("diblock", ["A"=>N_seg÷2, "B"=>N_seg÷2])];
                         rho0 = 1.0,
                         chain = GaussianChain(b=1.0),
                         interactions = (FloryHugginsChi(Dict(("A","B") => 1.5)),
                                         HelfandCompressibility(25.0)))
```

[`FloryHugginsChi`](@ref ClassicalDFT.FloryHugginsChi) is the A–B contact interaction
(`χN = 30`, well above the order–disorder transition) and
[`HelfandCompressibility`](@ref ClassicalDFT.HelfandCompressibility) a soft penalty
keeping `ρ_A + ρ_B ≈ rho0`.

As with the group-contribution DFT models, the chain's bonding topology is supplied
separately via `mol_structure`, using the same [`custom_structure`](@ref
ClassicalDFT.custom_structure) mechanism as [Group-Contribution & Heterosegmented Chains](@ref):

```julia
julia> mol_structure = Dict("diblock" => custom_structure("A"^(N_seg÷2) * "B"^(N_seg÷2)))
```

## Seeding a lamellar unit cell

[`LamellarStack1DCart`](@ref ClassicalDFT.LamellarStack1DCart) (see [Block-Copolymer Microphase
Morphologies](@ref)) seeds a periodic layered profile directly — a much more reliable
starting point for a symmetric diblock than random noise, which can decay back to a
uniform melt before the Anderson solver has a chance to grow the instability. `core_groups`
picks which named group (from `custom_structure`'s letter prefixes) forms the
minority/core layer:

```julia
julia> L = 7.0

julia> ngrid = 65

julia> structure = LamellarStack1DCart((0.0, 0.0), [1.0], [0.0, L], ngrid; core_groups=["A"])
```

`SCFTSystem` composes the model, structure and chain architecture, mirroring
[`DFTSystem`](@ref ClassicalDFT.DFTSystem). Because this is a melt of a single molecule type
filling the box, use the canonical ensemble with the chain count implied by the box
length and chain length:

```julia
julia> system = SCFTSystem(model, structure, DFTOptions();
                           mol_structure = mol_structure,
                           ensemble = [:canonical],
                           n_molecules = [L / N_seg])
```

## Converging

`initialize_profiles` and `converge!` follow the same pattern as every other tutorial.
For SCFT, `converge!` also returns the converged fields and partition functions:

```julia
julia> ρ = initialize_profiles(system)

julia> result = converge!(system, ρ; verbose=true)
[ Info: SCFT converged after 5027 iterations: err = 3.84e-6 | F = 1.61701

julia> result.converged
true
```

```julia
julia> plot(system, ρ)
```

![Converged SCFT lamellar diblock morphology](../assets/scft_lamellar.png)

The two species separate into alternating layers, each reaching a volume fraction `φ`
near `1.0` at its layer center and near `0.0` in the opposing layer — a genuine
microphase-separated lamellar melt.

## Swapping components

Every part of the model can be exchanged without touching the rest. Replacing the soft
Helfand penalty with strict incompressibility, [`Incompressible`](@ref
ClassicalDFT.Incompressible), enforces `ρ_A + ρ_B = rho0` exactly and, for this melt,
converges an order of magnitude faster:

```julia
julia> model = SCFTModel([("diblock", ["A"=>N_seg÷2, "B"=>N_seg÷2])];
                         rho0 = 1.0,
                         chain = GaussianChain(b=1.0),
                         interactions = (FloryHugginsChi(Dict(("A","B") => 1.5)),
                                         Incompressible(lambda=5.0)))

julia> system = SCFTSystem(model, structure, DFTOptions(); mol_structure,
                           ensemble = [:canonical], n_molecules = [L / N_seg])

julia> ρ = initialize_profiles(system); result = converge!(system, ρ; verbose=true)
[ Info: SCFT converged after 311 iterations: err = 8.64e-7 | F = 1.62108

julia> maximum(abs.(sum(ρ, dims=2) .- 1))   # 0.018 with HelfandCompressibility(25.0)
2.7e-6
```

The chain representation is chosen per species. Making the A block a rigid rod gives a
rod-coil block copolymer — the propagator switches to the orientation-resolved
[`WLCPropagator`](@ref ClassicalDFT.WLCPropagator) automatically, with Gaussian bonds for
the coil:

```julia
julia> rodcoil = SCFTModel([("rodcoil", ["A"=>8, "B"=>12])];
                           rho0 = 1.0, L_max = 6,
                           chain = Dict("A" => RigidRod(b=1.0), "B" => GaussianChain(b=1.0)),
                           interactions = (FloryHugginsChi(Dict(("A","B") => 2.0)),
                                           Incompressible(lambda=5.0)))
```

Other interactions — finite-range χ, Edwards excluded volume for solutions, Maier-Saupe
alignment — and sequence-defined chains are described in [SCFT](../models/scft.md).

## Next steps

- The [Worm-Like Chains & Orientational Order](../tutorials/wlc.md) tutorial covers
  semiflexible and rigid species: orientation order parameters, mean orientation fields
  and sampled chain conformations.
- `LamellarStack1DCart`/`HexLattice2DCart`/`HexLattice3DCart`/`BCC3DCart`/`Gyroid3DCart`
  all work as `SCFTSystem` seeds the same way — see [Block-Copolymer Microphase
  Morphologies](@ref) for the full structure list, and [Copolymer Microphase
  Morphologies](@ref "Copolymer Microphase Morphologies") for the equivalent classical-DFT
  workflow using `HeterogcPCPSAFT`/`DFTSystem` instead.
