import Clapeyron: a_res, lb_volume, T_scale
using Clapeyron: EoSModel, LatticeFluidModel, IdealModel, BasicIdeal, GroupParam, init_model

"""
    SCFTLatticeFluidModel

Abstract supertype of every SCFT bulk/inhomogeneous model. The only concrete subtype is
[`SCFTModel`](@ref); the legacy constructors ([`SCFTLatticeFluid`](@ref),
[`SCFTWormLikeChainFluid`](@ref), [`SCFTSequenceFluid`](@ref)) all build an `SCFTModel`.
"""
abstract type SCFTLatticeFluidModel <: LatticeFluidModel end

include("chains.jl")
include("interactions/interactions.jl")

"""
    SCFTModel(grouplist; rho0, chain=GaussianChain(), interactions=(), L_max=8,
              idealmodel=BasicIdeal, references=String[])
    SCFTModel(groups::GroupParam; kwargs...)

Modular SCFT model, assembled from swappable components in the style of Clapeyron's
cubic equations of state (`PR(components; alpha=…, mixing=…)`):

- **species and molecules** — `grouplist` is Clapeyron's group-contribution format, e.g.
  `[("diblock", ["A"=>10, "B"=>10]), ("solvent", ["S"=>1])]`. `components` are molecule
  types; species are `groups.flattenedgroups` (`"A"`, `"B"`, `"S"`). Sequence and topology
  (linear, branched) are given per molecule type by `mol_structure` when building the
  [`SCFTSystem`](@ref), or directly by a pre-built `GroupParam` (see [`SCFTSequenceFluid`](@ref)).
- **chain representation, per species** — `chain` is one [`SCFTChainModel`](@ref) applied
  to every species, a `Vector` in species order, or a `Dict` from species name, e.g.
  `Dict("A" => RigidRod(b=1.0), "B" => GaussianChain(b=1.0))`.
- **interactions** — a tuple of [`SCFTInteractionModel`](@ref)s, e.g.
  `(FloryHugginsChi(chi), HelfandCompressibility(20.0), MaierSaupe(nu))`. Every mean field,
  bulk free energy and inhomogeneous free energy is the sum of their contributions.
- `rho0` — reference (total segment) density; `L_max` — spherical-harmonic truncation for
  orientation-resolved species (one shared orientation grid).

`a_res` (see [`bulk_free_energy_density`](@ref)) deliberately omits chain translational
and mixing entropy: SCFT counts that once, through the single-chain partition function `Q`.

# Example
```julia
model = SCFTModel([("rodcoil", ["A"=>8, "B"=>8])];
    rho0 = 1.0,
    chain = Dict("A" => WormLikeChain(b=1.0, lp=10.0), "B" => GaussianChain(b=1.0)),
    interactions = (FloryHugginsChi(Dict(("A","B") => 1.2)), HelfandCompressibility(20.0)))
```
"""
struct SCFTModel{I<:IdealModel,U<:Tuple} <: SCFTLatticeFluidModel
    components::Vector{String}
    groups::GroupParam
    rho0::Float64
    chains::Vector{SCFTChainModel}
    interactions::U
    L_max::Int
    idealmodel::I
    references::Vector{String}
end

SCFTModel(grouplist; kwargs...) = SCFTModel(GroupParam(grouplist); kwargs...)

function SCFTModel(groups::GroupParam; rho0::Real, chain = GaussianChain(), interactions = (),
                   L_max::Int = 8, idealmodel = BasicIdeal, references = String[])
    species = groups.flattenedgroups
    chains = _resolve_chains(chain, species)
    ints = interactions isa SCFTInteractionModel ? (interactions,) : Tuple(interactions)
    all(i -> i isa SCFTInteractionModel, ints) || throw(ArgumentError(
        "interactions must be SCFTInteractionModels, e.g. (FloryHugginsChi(chi), HelfandCompressibility(20.0))."))
    ints = map(i -> bind_interaction(i, species), ints)
    ideal = init_model(idealmodel, groups.components, String[], false)
    model = SCFTModel(groups.components, groups, Float64(rho0), chains, ints, L_max, ideal, references)
    foreach(i -> validate_interaction(i, model), ints)
    return model
end

export SCFTModel

#= ---------------------------------------------------------------------------------
   Accessors (the only way the propagators, solver and sampling code read the model)
--------------------------------------------------------------------------------- =#

species_names(model::SCFTModel) = model.groups.flattenedgroups
nspecies(model::SCFTModel) = length(model.groups.flattenedgroups)

"""Per-species segment/bond length `b`, in species order."""
segment_lengths(model::SCFTModel) = Float64[segment_length(c) for c in model.chains]

"""Per-species persistence length `lp` (`Inf` for rods, `NaN` for Gaussian species, which have none)."""
persistence_lengths(model::SCFTModel) = Float64[is_oriented(c) ? persistence_length(c) : NaN for c in model.chains]

"""Whether any species is orientation-resolved (needs the WLC orientation grid)."""
has_orientation(model::SCFTModel) = any(is_oriented, model.chains)

"""First interaction of type `T` in `model.interactions`, or `nothing`."""
function get_interaction(model::SCFTModel, ::Type{T}) where T<:SCFTInteractionModel
    for int in model.interactions
        int isa T && return int
    end
    return nothing
end

"""Maier-Saupe coupling matrix `ν` (all zeros when the model has no `MaierSaupe`)."""
function maier_saupe_nu(model::SCFTModel)
    ms = get_interaction(model, MaierSaupe)
    n = nspecies(model)
    return ms === nothing ? zeros(n, n) : ms.nu
end

has_maier_saupe(model::SCFTModel) = any(!iszero, maier_saupe_nu(model))

"""
    length_scale(model::SCFTModel)

Largest segment/bond length across species — the SCFT analogue of
`length_scale(model::SAFTModel) = maximum(sigma)`.
"""
length_scale(model::SCFTLatticeFluidModel) = maximum(segment_lengths(model))

"""
    SCFTSpecies(sequence, nbeads, levels, i_groups, n_intergroups, ensemble, n_molecules, molecule_bulk_density, bulk_density)

Everything SCFT-calculation-specific that isn't part of the bulk EoS model: `sequence`
(species indices per molecule type's flattened group/node list, for the chain
propagator — `length(sequence[i])` is the total node count `Ni`; for a branched molecule
this is *node* order, i.e. `model.groups.i_groups[i]`'s order, not a linear chain
position), `nbeads` (the *number of distinct species* used by molecule type `i`,
`length.(model.groups.i_groups)` — **not** the node count; matches the field
`PCSAFTSpecies`/`gcPCPSAFTSpecies` use for the same purpose, needed so the generic
structure-dispatched `initialize_profiles` methods in `src/structure/structure.jl`/
`two_phase.jl`/`morphology.jl` — which size arrays via `sum(species.nbeads)` — work
unmodified for SCFT systems too), `levels`/`i_groups`/`n_intergroups` (tree-traversal
data `DiscreteGaussianChainPropagator` needs — see `compute_levels`, `src/utils/base.jl`
— copied straight off the *expanded* model's `GroupParam` here in `get_species`, since
`SCFTSystem` deliberately keeps only the *original*, unexpanded `model` (see
`expand_groups`'s docstring), so the propagator has no other way to reach this
per-node-instance bond data at construction/`propagate!` time; degenerates to a simple
linear chain traversal when `mol_structure` happens to be unbranched), `ensemble`/
`n_molecules` (per molecule type), `molecule_bulk_density` (per molecule type —
`structure.ρbulk` unchanged, the *total* target bulk density of each molecule type, e.g.
`rho0` for a pure melt; used directly in the grand-canonical prefactor formulas in
`compute_densities!`/`free_energy`), and `bulk_density` (per **species**, e.g. one value
each for "A"/"B" in a diblock — computed once here in `get_species` by splitting
`molecule_bulk_density` across each molecule type's segments, weighted by segment count
for grand-canonical types or by `n_molecules`/(box volume) for canonical types.
Precomputing this at construction time, rather than recomputing it on every call the way
the old `compute_bulk_densities(system; V_eff)` runtime function did, is what lets
`structure.ρbulk` be supplied "ahead of time" exactly like DFT-family models require —
see `compute_bulk_densities(system::SCFTSystem)` (`src/models/SCFT/scft.jl`), now a
trivial `system.species.bulk_density` accessor).
"""
struct SCFTSpecies <: DFTSpecies
    sequence::Vector{Vector{Int}}
    nbeads::Vector{Int}
    levels::Vector{Int}
    i_groups::Vector{Vector{Int}}
    n_intergroups::Vector{Matrix{Int}}
    ensemble::Vector{Symbol}
    n_molecules::Vector{Float64}
    molecule_bulk_density::Vector{Float64}
    bulk_density::Vector{Float64}
end
export SCFTSpecies

"""
    expand_groups(model::SCFTModel, mol_structure)

Build a per-instance-expanded `GroupParam` for `model` (one flattened group per chain
position, e.g. `"A_1","A_2","A_3","B_1","B_2"` for a `"AAABB"` chain) — the same shape
`HeterogcPCPSAFT`/`SAFTgammaMie` produce via Clapeyron's own generic `expand_groups`
(`src/utils/expand_model.jl`), but with `i_groups[i]` built by walking each component's
actual chain positions (via `get_connectivity`, proven to return species names in exact
left-to-right order for an unbranched string) rather than Clapeyron's own ascending-global-
index extraction — which discards chain-position order (all "A" instances get consecutive
global indices, then all "B" instances, *regardless of interleaving* — fatal for a
discrete-Gaussian-chain propagator that needs the exact sequence; see
`project_scft_clapeyron_eos` memory for why the generic version was ruled out).

Global flattened indices are still assigned **type-contiguous** (`"A_*"` block, then
`"B_*"` block, matching Clapeyron's own convention) so that
[`expand_params`](@ref)/`expand_model` can reuse Clapeyron's already-generic parameter
expansion unmodified — only the *order in which each component's `i_groups[i]` lists its
instances* differs from the generic version, not the underlying global numbering scheme.

Validates that `mol_structure`'s composition matches `model.groups.n_flattenedgroups`
(the composition already implied by `grouplist`); branched `mol_structure` (produced by
`custom_structure`'s `"A(B)C"` syntax) is accepted — `DiscreteGaussianChainPropagator`
walks the resulting bond tree via `SCFTSpecies.levels` (see `compute_levels`,
`src/utils/base.jl`), the same tree-traversal convention `TangentHSPropagator` uses.
Note `custom_structure`'s parser only supports single-character species names.

Returns `(expanded_groups::GroupParam, ngroups_k::Vector{Int})` (matching Clapeyron's own
`expand_groups`'s return shape, for `expand_params`).
"""
function expand_groups(model::SCFTLatticeFluidModel, mol_structure::Dict{String,<:MolStructure})
    species_list = model.groups.flattenedgroups
    ngroup_types = length(species_list)
    ncomp = length(model.components)
    name_to_idx0 = Dict(s => j for (j,s) in enumerate(species_list))

    names_per_comp = Vector{Vector{String}}(undef, ncomp)
    bondmat_per_comp = Vector{Matrix{Int}}(undef, ncomp)
    for (i, comp) in enumerate(model.components)
        _, names, bond_mat = get_connectivity(model, mol_structure[comp])
        counts = zeros(Int, ngroup_types)
        for nm in names
            counts[name_to_idx0[nm]] += 1
        end
        @assert counts == round.(Int, model.groups.n_flattenedgroups[i]) "mol_structure for $comp doesn't match its group composition"
        names_per_comp[i] = names
        bondmat_per_comp[i] = bond_mat
    end

    # Type-contiguous global flattened index numbering (matches Clapeyron's own
    # expand_groups convention, so expand_params can be reused unmodified).
    ngroups_k = zeros(Int, ngroup_types)
    for i in 1:ncomp
        ngroups_k .+= round.(Int, model.groups.n_flattenedgroups[i])
    end
    ngroups = sum(ngroups_k)
    flattenedgroups = String[]
    for j in 1:ngroup_types
        append!(flattenedgroups, species_list[j]*"_".*string.(1:ngroups_k[j]))
    end
    letter_base = cumsum([0; ngroups_k[1:end-1]])

    # Per-component NODE-ORDER i_groups: walk each component's actual group/node
    # positions in `get_connectivity`'s parse order (a tree pre-order for branched
    # `mol_structure`, and a strict left-to-right chain order in the linear case) and
    # assign the next unused global instance for that position's letter — unlike
    # Clapeyron's own ascending-index extraction.
    letter_running = zeros(Int, ngroup_types)
    i_groups = Vector{Vector{Int}}(undef, ncomp)
    groups = Vector{Vector{String}}(undef, ncomp)
    n_groups = Vector{Vector{Int}}(undef, ncomp)
    for i in 1:ncomp
        seq_global = Int[]
        for nm in names_per_comp[i]
            j = name_to_idx0[nm]
            letter_running[j] += 1
            push!(seq_global, letter_base[j] + letter_running[j])
        end
        i_groups[i] = seq_global
        groups[i] = flattenedgroups[seq_global]
        n_groups[i] = ones(Int, length(seq_global))
    end

    n_flattenedgroups = [zeros(Int, ngroups) for _ in 1:ncomp]
    for i in 1:ncomp
        n_flattenedgroups[i][i_groups[i]] .= 1
    end

    n_intergroups = Matrix{Int}[]
    for i in 1:ncomp
        mat = zeros(Int, ngroups, ngroups)
        bond_mat = bondmat_per_comp[i]
        idx = i_groups[i]
        n = length(idx)
        for a in 1:n, b in 1:n
            mat[idx[a], idx[b]] = bond_mat[a,b]
        end
        push!(n_intergroups, mat)
    end

    expanded_groups = GroupParam(model.components, groups, model.groups.grouptype,
                                  n_groups, n_intergroups, i_groups,
                                  flattenedgroups, n_flattenedgroups, model.groups.sourcecsvs)
    return expanded_groups, ngroups_k
end


"""
    expand_model(model::SCFTModel, mol_structure)

The per-node-instance expansion of `model` (one flattened group per chain position, see
[`expand_groups`](@ref)), mirroring `HeterogcPCPSAFT`/`SAFTgammaMie`'s
`expand_model(model, mol_structure)`. Only `groups` is expanded: chain representations and
interactions stay indexed by the original species, which is all SCFT needs — the
expanded model is a transient used by [`get_species`](@ref) to read chain order, and
`SCFTSystem` keeps the original model.
"""
function expand_model(model::SCFTModel, mol_structure::Dict{String,<:MolStructure})
    expanded_groups, _ = expand_groups(model, mol_structure)
    return SCFTModel(expanded_groups.components, expanded_groups, model.rho0, model.chains,
                     model.interactions, model.L_max, model.idealmodel, model.references)
end

"""
    get_species(model::SCFTModel, structure::DFTStructure; ensemble, n_molecules)

Build the [`SCFTSpecies`](@ref) for `model`, matching `HeterogcPCPSAFT`'s
`get_species(model, structure)` shape. `model` is assumed already expanded (see
[`expand_model`](@ref)) — chain order comes directly from `model.groups.i_groups`
(already in exact chain-position order by construction), mapped back down to the
original unique-species-letter index space (first-seen letter order in the expanded
model's `flattenedgroups` reproduces the pre-expansion model's own letter order exactly,
since `expand_groups` iterates original species-index order when building its
type-contiguous blocks).

`structure.ρbulk` supplies each molecule type's **target total bulk density** directly
(read straight off `structure`, like DFT-family models do) — `molecule_bulk_density[i] =
ρbulk[i]` unchanged, used directly for `:grand_canonical` molecule types in the solve's
own prefactor formulas. The per-**species** split (`bulk_density`, e.g. distinct values
for "A"/"B" in a diblock) is computed once, right here, rather than repeatedly at
solve/init time: `V_eff` (needed for `:canonical` molecule types, `n_molecules/V_eff`) is
the exact geometric box volume from `structure`'s bounds — this is not an approximation,
it's the same value `effective_volume`/the default `:trapz` quadrature already use for
the whole domain (integrating a constant is exact for any sane quadrature rule), so
nothing is lost by fixing it at construction time instead of waiting on a solve-time
quadrature choice. `ensemble`/`n_molecules` are kept as keyword arguments since
`HeterogcPCPSAFT` has no equivalent concept.
"""
function get_species(model::SCFTLatticeFluidModel, structure::DFTStructure;
                      ensemble::Vector{Symbol} = fill(:grand_canonical, length(model.components)),
                      n_molecules::AbstractVector = zeros(length(model.components)))
    letters = _group_letter.(model.groups.flattenedgroups)
    unique_letters = unique(letters)
    letter_idx = Dict(l => i for (i, l) in enumerate(unique_letters))

    sequence = [[letter_idx[letters[k]] for k in ig] for ig in model.groups.i_groups]
    nbeads = length.(unique.(sequence))
    levels = compute_levels(model)

    molecule_bulk_density = Float64.(structure.ρbulk)
    n_molecules_f = Float64.(n_molecules)

    nd = dimension(structure)
    V_eff = prod(ntuple(d -> ((lb, ub) = bounds(structure, d); ub - lb), nd))

    bulk_density = zeros(Float64, length(unique_letters))
    for i in eachindex(sequence)
        seq = sequence[i]
        Ni = length(seq)
        density_per_segment = ensemble[i] == :canonical ? n_molecules_f[i] / V_eff : molecule_bulk_density[i] / Ni
        for α in seq
            bulk_density[α] += density_per_segment
        end
    end

    return SCFTSpecies(sequence, nbeads, levels, model.groups.i_groups, model.groups.n_intergroups,
                        ensemble, n_molecules_f, molecule_bulk_density, bulk_density)
end

#= ---------------------------------------------------------------------------------
   Propagator selection from the per-species chain representations
--------------------------------------------------------------------------------- =#

function get_propagator(model::SCFTModel, species::SCFTSpecies, structure::DFTStructure,
                        backend::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    used = unique(reduce(vcat, species.sequence))
    oriented = [is_oriented(model.chains[α]) for α in used]
    # All-Gaussian systems use the cheaper orientation-free propagator; any
    # orientation-resolved species needs the orientation grid, and WLCPropagator handles
    # Gaussian beads in the same chain through Gaussian bonds (see its docstring).
    if !any(oriented)
        return DiscreteGaussianChainPropagator(model, species, structure, backend, FP)
    else
        return WLCPropagator(model, species, structure, backend, FP; L_max=model.L_max)
    end
end

#= ---------------------------------------------------------------------------------
   Bulk thermodynamics: sums over the interaction submodels
--------------------------------------------------------------------------------- =#

"""
    bulk_free_energy_density(model::SCFTModel, ρ::AbstractVector)

Isotropic bulk interaction free-energy density (units of `k_BT` per volume) at species
densities `ρ`: `Σ_int bulk_f(int, ρ, rho0)`. Excludes chain translational/mixing entropy
(counted by SCFT through `Q`).
"""
bulk_free_energy_density(model::SCFTModel, ρ::AbstractVector) =
    sum(int -> bulk_f(int, ρ, model.rho0), model.interactions; init = zero(eltype(ρ)))

"""
    compute_bulk_fields(model::SCFTModel, bulk_densities)

Bulk (uniform) mean field of each species, `w_α = ∂f_bulk/∂ρ_α`, obtained by automatic
differentiation of [`bulk_free_energy_density`](@ref) — so it is consistent with every
interaction's `bulk_f` by construction.
"""
function compute_bulk_fields(model::SCFTModel, bulk_densities::AbstractVector{T}) where {T<:AbstractFloat}
    f = ρ -> bulk_free_energy_density(model, ρ)
    return Clapeyron.ForwardDiff.gradient(f, collect(bulk_densities))
end

function a_res(model::SCFTLatticeFluidModel, V, T, z)
    Σz = sum(z)
    n_species = sum(z[i] .* model.groups.n_flattenedgroups[i] for i in eachindex(z))
    ρ = n_species ./ V
    return V * bulk_free_energy_density(model, ρ) / Σz
end

# Placeholders: ClassicalDFT always evaluates a_res at a prescribed density, never through
# Clapeyron's own volume/phase solvers, so these matter only for standalone bootstrapping.
lb_volume(model::SCFTLatticeFluidModel, T, z) =
    sum(z[i]*sum(model.groups.n_flattenedgroups[i]) for i in eachindex(z)) / (2*model.rho0)
T_scale(model::SCFTLatticeFluidModel, z) = one(eltype(z))

