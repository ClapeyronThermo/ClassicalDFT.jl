"""
    free_energy(system::DFTSystem, ρ)

Obtain the total free energy of the system for a given profile. This is done by summing the ideal and residual free energies.

The output is a scalar of units J.
"""
function free_energy(system::AbstractcDFTSystem, ρ)
    return F_ideal(system,ρ)+F_res(system,ρ)
end

onevec(model) = Clapeyron.FillArrays.Ones(length(model))

macro chain(component, args...)
    quote
        if hasfield(typeof(model), :groups)
            model.groups.i_groups[$(component)]
        else
            $(component)
        end
    end |> esc
end

"""
    compute_levels(model)

BFS tree depth ("level") for every bonded group/node in `model`, using
`model.groups.i_groups`/`model.groups.n_intergroups`. Within each component, the
highest-degree group is picked as the root (`level = 1`); levels then increase by 1 per
BFS layer outward. Returns a `Vector{Int}` sized `sum(length.(model.groups.groups))`
(one entry per flattened group/node across all components), globally indexed exactly
like `model.groups.i_groups`'s entries.

Shared by group-contribution `DFTSpecies` constructors (`SAFTgammaMieSpecies`,
`gcPCPSAFTSpecies`, `SCFTSpecies`) whose propagator (`TangentHSPropagator`,
`DiscreteGaussianChainPropagator`) needs a tree traversal order.
"""
function compute_levels(model::EoSModel)
    nbeads = length.(model.groups.groups)
    levels = zeros(Int, sum(nbeads))

    for i in @comps
        i_groups = model.groups.i_groups[i]
        bond_mat = Int.(model.groups.n_intergroups[i]) .> 0
        nbonds = sum(bond_mat,dims=2)[:]
        is_leaf = nbonds .== 1
        i_root = i_groups[findfirst(nbonds[i_groups] .== maximum(nbonds[i_groups]))]
        levels[i_root] = 1

        idx_current_level = i_root
        is_bonded = bond_mat[idx_current_level,:]
        k = 1
        while any(levels[i_groups] .== 0)
            levels[is_bonded] .= k+1
            idx_next_level = findall(levels .== k+1 .&& .!(is_leaf))
            is_bonded = (sum(bond_mat[idx_next_level,:],dims=1)[:].==1 .&& levels.==0)
            k+=1
        end
    end
    return levels
end

function get_chain_idx(model::EoSModel,i,j,a,b)
    return get_chain_idx(model.sites,i,j,a,b)
end

function get_chain_idx(param::SiteParam, i::Int64, j::Int64, a::Int64, b::Int64)
    if isnothing(param.site_translator)
        return i,j
    else
        site_translator::Vector{Vector{NTuple{2,Int}}} = param.site_translator
        k,_ = site_translator[i][a]
        l,_ = site_translator[j][b]
        return k,l
    end
end