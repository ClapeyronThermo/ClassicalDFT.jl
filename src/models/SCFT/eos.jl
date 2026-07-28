import Clapeyron: a_res, lb_volume, T_scale
using Clapeyron: EoSParam, EoSModel, LatticeFluidModel, IdealModel, BasicIdeal,
                 SingleParam, PairParam, GroupParam, init_model

"""
    SCFTLatticeFluidParam

- `b`: statistical segment length, indexed by species (`groups.flattenedgroups`). Not
  used by `a_res` — present for bookkeeping/the propagator, the same relationship `Mw`
  has to `a_res` in Clapeyron's own `SanchezLacombeParam`.
- `chi`: pairwise Flory interaction parameter, symmetric with zero diagonal, indexed by
  species (`groups.flattenedgroups`).
"""
struct SCFTLatticeFluidParam <: EoSParam
    b::SingleParam{Float64}
    chi::PairParam{Float64}
end

abstract type SCFTLatticeFluidModel <: LatticeFluidModel end

"""
    SCFTLatticeFluid(grouplist, b, chi; rho0, kappa, idealmodel=BasicIdeal, references=String[])

A compressible Flory-Huggins-Helfand lattice-fluid `EoSModel` for SCFT bulk thermodynamics. 
Chain order and ensemble/count bookkeeping for a particular SCFT calculation live on `SCFTSpecies` (`get_species`), not here — those describe a specific calculation's setup, not the bulk EoS.

`grouplist` is Clapeyron's own group-contribution format (`Vector{Tuple{String,Vector{Pair{String,Int}}}}`), e.g. `[("diblock", ["A"=>4,"B"=>4]), ("solvent", ["S"=>1])]` — `components` are molecule types (chains and solvents, unified: a solvent is just an `N=1` molecule type), `b`/`chi` are indexed by species (`groups.flattenedgroups`, the union of group names across `grouplist`).

`a_res` is defined so that, for `z` indexed by `components` (molecule-type mole amounts, matching Clapeyron's convention), `VT_chemical_potential_res(model, V, T, z)_i / (R̄*T)` equals `Σ_α n_flattenedgroups[i][α] · w_α(ρ)`, i.e. molecule type `i`'s residual chemical potential is the sum of its constituent segments' potentials — where `w_α` is exactly `src/models/SCFT/scft.jl`'s `compute_fields!`/`compute_bulk_fields` mean field. 
It deliberately omits any chain translational/mixing entropy contribution, since that entropy is already counted once by cDFT's SCFT free energy via the single-chain partition function `Q` (`src/models/SCFT/scft.jl`'s `compute_partition_functions`/`free_energy`). 
Do not add an entropy term here without also removing the corresponding double-count from wherever `Q` is used.
"""
struct SCFTLatticeFluid{I<:IdealModel} <: SCFTLatticeFluidModel
    components::Vector{String}
    groups::GroupParam
    params::SCFTLatticeFluidParam
    rho0::Float64
    kappa::Float64
    idealmodel::I
    references::Vector{String}
end

function SCFTLatticeFluid(grouplist, b::AbstractVector, chi::AbstractMatrix;
                           rho0::Real, kappa::Real, idealmodel = BasicIdeal, references = String[])
    groups = GroupParam(grouplist)
    params = SCFTLatticeFluidParam(
        SingleParam("b", groups.flattenedgroups, Float64.(b)),
        PairParam("chi", groups.flattenedgroups, Float64.(chi)))
    ideal = init_model(idealmodel, groups.components, String[], false)
    return SCFTLatticeFluid(groups.components, groups, params, Float64(rho0), Float64(kappa), ideal, references)
end
export SCFTLatticeFluid

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
    expand_groups(model::SCFTLatticeFluid, mol_structure)

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
function expand_groups(model::SCFTLatticeFluid, mol_structure::Dict{String,<:MolStructure})
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
    expand_model(model::SCFTLatticeFluid, mol_structure)

Build the fully expanded `SCFTLatticeFluid` (one flattened group per chain position),
mirroring `HeterogcPCPSAFT`/`SAFTgammaMie`'s `expand_model(model, mol_structure)` call
pattern. See [`expand_groups`](@ref) for the chain-order-preserving group expansion;
parameter expansion (`b`/`chi`, replicated per instance) reuses Clapeyron's already-generic
`expand_params` (`src/utils/expand_model.jl`) directly — its `SingleParameter`/
`PairParameter` branches are the only ones `SCFTLatticeFluidParam` hits (no association
sites, so the unused `sites` argument is `nothing`).

Used internally by `get_species` to derive chain order cleanly; `SCFTSystem`
continues to store the **original**, unexpanded model (see `SCFTSystem`'s constructor) —
SCFT's mean-field density fields are per species-letter, not per chain-position instance,
so the expanded model is a transient, not a replacement for the system's own `model`.
"""
function expand_model(model::SCFTLatticeFluid, mol_structure::Dict{String,<:MolStructure})
    expanded_groups, ngroups_k = expand_groups(model, mol_structure)
    expanded_params = expand_params(model.params, expanded_groups, nothing, ngroups_k)
    return SCFTLatticeFluid(expanded_groups.components, expanded_groups, expanded_params,
                             model.rho0, model.kappa, model.idealmodel, model.references)
end

"""
    get_species(model::SCFTLatticeFluid, structure::DFTStructure; ensemble, n_molecules)

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
function get_species(model::SCFTLatticeFluid, structure::DFTStructure;
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

#=
    get_propagator(model::SCFTLatticeFluid, species::SCFTSpecies, structure::DFTStructure, backend, FP=Float64)

Build the `DiscreteGaussianChainPropagator` for `model`, matching the generic
`get_propagator(model, species, structure, backend, FP)` dispatch every other DFT-family
model uses (e.g. `HeterogcPCPSAFT`'s `TangentHSPropagator`).
=#
function get_propagator(model::SCFTLatticeFluid, species::SCFTSpecies, structure::DFTStructure,
                         backend::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    return DiscreteGaussianChainPropagator(model, species, structure, backend, FP)
end

function a_res(model::SCFTLatticeFluid, V, T, z)
    Σz = sum(z)
    nspecies = length(model.groups.flattenedgroups)
    n_species = sum(z[i] .* model.groups.n_flattenedgroups[i] for i in eachindex(z))
    ρ = n_species ./ V
    ρtot = sum(ρ)
    chi = model.params.chi.values
    rho0, kappa = model.rho0, model.kappa

    f = zero(V + T + first(z))
    for α in 1:nspecies, β in 1:nspecies
        f += chi[α,β]*ρ[α]*ρ[β]
    end
    f /= 2*rho0
    f += (kappa/(2*rho0^2))*(ρtot - rho0)^2

    return V*f/Σz
end

# Placeholders: cDFT always calls a_res/VT_chemical_potential_res at an explicitly
# prescribed density, never through Clapeyron's own volume/phase solvers, so these
# only matter if this model is ever used standalone for solver bootstrapping.
lb_volume(model::SCFTLatticeFluid, T, z) =
    sum(z[i]*sum(model.groups.n_flattenedgroups[i]) for i in eachindex(z)) / (2*model.rho0)
T_scale(model::SCFTLatticeFluid, z) = one(eltype(z))

#=
    length_scale(model::SCFTLatticeFluid)

The largest statistical segment length `b` across every species — the SCFT analogue of
`length_scale(model::SAFTModel) = maximum(model.params.sigma.values)`, used the same way
(e.g. for choosing grid bounds, and by the Makie plotting recipe's axis normalization).
=#
length_scale(model::SCFTLatticeFluid) = maximum(model.params.b.values)
