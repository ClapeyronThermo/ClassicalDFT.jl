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
architectures, at the cost of the mean-field interaction and coarse-grained chain models
it relies on in place of a full pairwise free-energy functional.

Every SCFT calculation uses one model type, [`SCFTModel`](@ref), assembled from
swappable components in the style of Clapeyron's cubic equations of state:

- a **chain representation per species** — [`GaussianChain`](@ref),
  [`WormLikeChain`](@ref) or [`RigidRod`](@ref) — so one molecule can mix them (e.g. a
  rod-coil block copolymer);
- a **tuple of interactions** — [`FloryHugginsChi`](@ref), [`SmearedFloryHugginsChi`](@ref),
  [`HelfandCompressibility`](@ref), [`Incompressible`](@ref),
  [`EdwardsExcludedVolume`](@ref), [`MaierSaupe`](@ref) — whose fields and free energies add;
- **sequence and topology** given per molecule type through `mol_structure` (linear or
  branched, via [`custom_structure`](@ref ClassicalDFT.custom_structure)) or by a
  sequence descriptor ([Sequence-defined models](@ref)).

The same components define the inhomogeneous problem ([`SCFTSystem`](@ref
ClassicalDFT.SCFTSystem), solved by [`converge!`](@ref ClassicalDFT.converge!)) and the
bulk thermodynamics, including nematic phases ([Bulk thermodynamics](@ref)). See the
[Self-Consistent Field Theory](../tutorials/scft.md) and [Worm-Like Chains & Orientational
Order](../tutorials/wlc.md) tutorials for worked examples.

## Contents

```@contents
Pages = ["scft.md"]
Depth = 2
```

## Building a model

```julia
using ClassicalDFT

# Symmetric AB diblock: Gaussian chains, χ between A and B, soft compressibility.
model = SCFTModel([("diblock", ["A"=>10, "B"=>10])];
    rho0 = 1.0,
    chain = GaussianChain(b=1.0),
    interactions = (FloryHugginsChi(Dict(("A","B") => 1.5)), HelfandCompressibility(25.0)))

# Rod-coil diblock: rigid A block, flexible B block, strict incompressibility and
# Maier-Saupe alignment of the rods.
rodcoil = SCFTModel([("rodcoil", ["A"=>8, "B"=>12])];
    rho0 = 1.0, L_max = 8,
    chain = Dict("A" => RigidRod(b=1.0), "B" => GaussianChain(b=1.0)),
    interactions = (FloryHugginsChi(Dict(("A","B") => 2.5)),
                    Incompressible(lambda=5.0),
                    MaierSaupe(Dict(("A","A") => 1.0))))
```

Parameters indexed by species accept a matrix in species order
(`model.groups.flattenedgroups`) or a `Dict` keyed by species names. Combinations are
checked when the model is built: e.g. [`MaierSaupe`](@ref) on a Gaussian species, or
[`Incompressible`](@ref) together with [`HelfandCompressibility`](@ref), is rejected.

```@docs
ClassicalDFT.SCFTModel
```

## Chain representations

| Chain | Statistics | Propagator |
|---|---|---|
| [`GaussianChain`](@ref) | discrete Gaussian (bead-spring) bonds, segment length `b` | [`DiscreteGaussianChainPropagator`](@ref) |
| [`WormLikeChain`](@ref) | bead-rod bonds of fixed length `b`, bending stiffness `κ = lp/b` | [`WLCPropagator`](@ref) |
| [`RigidRod`](@ref) | worm-like with `lp = Inf` | [`WLCPropagator`](@ref) |

A system whose species are all Gaussian uses the cheaper orientation-free propagator;
otherwise [`WLCPropagator`](@ref) is used, which also handles Gaussian beads in the same
chain:

- a bond is a **rod bond** (bending + fixed-length translation) only when both of its
  beads are orientation-resolved; a bond touching a Gaussian bead is a **Gaussian bond**
  with `b = √((b_α² + b_β²)/2)`;
- a rod bond between two species uses the RMS of their `κ`, except that a junction with an
  exactly rigid species inherits the flexible side's `κ`;
- branched chains are swept on their bond tree rooted at a chain end, and **branch points
  are freely jointed** (no bending correlation between the bonds that meet there).

`L_max` (a model keyword) sets the spherical-harmonic resolution of bond orientations for
all orientation-resolved species. Stiffer chains and strongly aligned (nematic) states
need more: for a rigid-rod melt at Maier-Saupe coupling `Nν = 36` (`S ≈ 0.97`),
`L_max = 8` underestimates `S` by 0.07 and `L_max = 14` by 0.01.

```@docs
ClassicalDFT.SCFTChainModel
ClassicalDFT.GaussianChain
ClassicalDFT.WormLikeChain
ClassicalDFT.RigidRod
```

## Interactions

| Interaction | Mean field added to `w_α` | Notes |
|---|---|---|
| [`FloryHugginsChi`](@ref)`(χ)` | `Σ_β (χ_αβ/ρ₀) ρ_β` | local contact interaction |
| [`SmearedFloryHugginsChi`](@ref)`(χ, a)` | `Σ_β (χ_αβ/ρ₀) (h*h*ρ_β)` | Gaussian-smeared, range `a`; same bulk as local χ |
| [`HelfandCompressibility`](@ref)`(κ)` | `(κ/ρ₀)(ρ₊/ρ₀ − 1)` | soft incompressibility |
| [`Incompressible`](@ref)`(lambda)` | pressure field `ξ(r)` | enforces `Σρ = ρ₀` exactly |
| [`EdwardsExcludedVolume`](@ref)`(v)` | `Σ_β v_αβ ρ_β` | good-solvent excluded volume |
| [`MaierSaupe`](@ref)`(ν)` | orientation-dependent, from `R = ρ⟨uu⟩` | liquid-crystalline alignment |

The model's mean field, bulk free energy and inhomogeneous free energy are sums over the
interaction tuple, so a new interaction only has to implement the
[`SCFTInteractionModel`](@ref) interface — no changes to the solver or the propagators.
For a strongly segregated lamella, the iteration with [`HelfandCompressibility`](@ref)
stopped converging above `κ ≈ 30` in our tests; use [`Incompressible`](@ref) for the
strict limit (it also converged faster there).

```@docs
ClassicalDFT.SCFTInteractionModel
ClassicalDFT.FloryHugginsChi
ClassicalDFT.SmearedFloryHugginsChi
ClassicalDFT.HelfandCompressibility
ClassicalDFT.Incompressible
ClassicalDFT.EdwardsExcludedVolume
ClassicalDFT.MaierSaupe
```

## Solving

[`converge!`](@ref ClassicalDFT.converge!) Anderson-accelerates the scalar field `w` and,
with an active [`MaierSaupe`](@ref), the orientation moments `R = ρ⟨uu⟩` jointly, so its
convergence test covers both. It returns `(; converged, err, iterations, w, Q, R_tensor)`.
Because the isotropic state is an exact fixed point of the orientational problem, a
nematic solution only develops from an aligned start: pass `director` (and optionally
`nematic_order`) to start orientation-resolved species aligned along it:

```julia
result = converge!(system, ρ; director=(0, 0, 1), nematic_order=0.5)
result.converged
```

See [`SCFTProblem`](@ref ClassicalDFT.SCFTProblem) for all solver keywords.

## Sequence-defined models

Following Xie & Olsen (*Macromolecules* 2022, 55, 6516–6524), a chain can also be
described by a descriptor function `f(s)` along its contour, evaluated on `Nc` nodes, each
of which becomes its own species. [`sequence_model`](@ref) builds such a model with any
chain representation — one for all nodes, a vector, or a function of the contour position
`s` — and any interactions; [`sequence_pair_matrix`](@ref) turns a kernel of two
descriptor values into a node-pair parameter. A step descriptor reproduces the
corresponding block model exactly.

```julia
d = taper_descriptor(0.0, 1.0, 40)                 # linear taper from f = 0 to f = 1
model = sequence_model(d; rho0 = 1.0, L_max = 6,
    chain = s -> WormLikeChain(b=1.0, lp=1 + 9s),  # stiffness grows along the chain
    gamma = (f1, f2) -> 300 * sum(abs2, f1 .- f2)) # sequence χ (Xie & Olsen eq. 1)
system = SCFTSequenceSystem(model, structure; ensemble=:canonical, n_molecules=n)
```

```@docs
ClassicalDFT.sequence_model
ClassicalDFT.sequence_pair_matrix
ClassicalDFT.SequenceDescriptor
ClassicalDFT.step_descriptor
ClassicalDFT.taper_descriptor
ClassicalDFT.sequence_chi_b
ClassicalDFT.SCFTSequenceFluid
ClassicalDFT.SCFTSequenceSystem
```

## Bulk thermodynamics

For a uniform isotropic phase, `a_res` (and so every Clapeyron bulk property) and the
bulk fields come from the interactions' bulk free energies
([`bulk_free_energy_density`](@ref)); like the SCFT free energy, they exclude chain
translational entropy, which SCFT counts through the single-chain partition function.

Nematic phases need each molecule's orientational partition function, which depends on
its sequence, topology and stiffness. [`NematicBulk`](@ref) precomputes that from a system
(or a model and `mol_structure`), and gives the self-consistent order parameter, free
energy, chemical potentials, pressure and isotropic–nematic coexistence. Its free energy
is exactly the uniform limit of the SCFT free energy, and for rigid rods it reproduces
classical Maier-Saupe theory.

```julia
nb = NematicBulk(system)
bulk_order_parameter(nb, [ρ_molecules])                  # (; S, Δf, converged)
coex = isotropic_nematic_coexistence(nb, [ρ_iso_guess], [ρ_nem_guess])
coex.ρ_iso, coex.ρ_nem, coex.S_nem
```

```@docs
ClassicalDFT.bulk_free_energy_density
ClassicalDFT.compute_bulk_fields
ClassicalDFT.NematicBulk
ClassicalDFT.bulk_order_parameter
ClassicalDFT.nematic_free_energy
ClassicalDFT.orientational_logq
ClassicalDFT.bulk_free_energy
ClassicalDFT.bulk_chemical_potential
ClassicalDFT.bulk_pressure
ClassicalDFT.isotropic_nematic_coexistence
```

## Legacy constructors

These earlier constructors remain and now build an [`SCFTModel`](@ref).

```@docs
ClassicalDFT.SCFTLatticeFluid
ClassicalDFT.SCFTWormLikeChainFluid
```

## Utilities

`compute_bulk_densities` returns the per-species bulk density implied by
`structure.ρbulk`/`ensemble`/`n_molecules` — computed once when the `SCFTSystem` is built,
and used internally by `initialize_profiles`/`converge!`.

```@docs
ClassicalDFT.compute_bulk_densities
```
