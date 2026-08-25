function expand_model(model::MODEL,mol_structure::Dict{String,<:MolStructure} = Dict{String,MolStructure}()) where MODEL <: EoSModel
    if !Clapeyron.has_groups(model)
        return model
    end

    # Expand the groups
    grouparam,ngroups_k = expand_groups(model, mol_structure)

    # Expand the sites
    siteparams = expand_sites(model,grouparam,ngroups_k)

    #expand the parameters
    eosparams = expand_params(model.params, grouparam, siteparams, ngroups_k)

    return new_model = MODEL(model.components,
                                grouparam,
                                siteparams,
                                eosparams,
                                model.idealmodel,
                                model.assoc_options,
                                model.references)
end

function expand_model(model,::Nothing)
    if !Clapeyron.has_groups(model)
        return model
    end
    return expand_model(model,Dict{String,MolStructure}())
end

function expand_groups(model, mol_structure::Dict{String,<:MolStructure} = Dict{String,MolStructure}())
    nspecies = length(model)

    # Expand the groups
    ngroup_types = length(model.groups.flattenedgroups)
    ngroups = Int(sum(sum,model.groups.n_flattenedgroups))

    ngroups_k = zeros(Int64,ngroup_types)
    n_groups = Vector{Int64}[]
    for i in 1:nspecies
        ngroups_k .+= model.groups.n_flattenedgroups[i]
        push!(n_groups, ones(Int64,Int(sum(model.groups.n_groups[i]))))
    end
    flattenedgroups = String[]
    n_flattenedgroups = [zeros(Int64,ngroups) for i in 1:nspecies]

    for i in 1:ngroup_types
        append!(flattenedgroups, model.groups.flattenedgroups[i]*"_".*string.(Int.(1:ngroups_k[i])))
        k=Int(sum(ngroups_k[1:i-1]))
        for j in 1:nspecies
            n_flattenedgroups[j][k+1:k+Int(model.groups.n_flattenedgroups[j][i])] .= 1
            k+=Int(model.groups.n_flattenedgroups[j][i])
        end
    end

    groups = Vector{String}[]
    i_groups = Vector{Int}[]
    for i in 1:nspecies
        append!(groups, [flattenedgroups[n_flattenedgroups[i].==1]])
        append!(i_groups,[deleteat!(n_flattenedgroups[i].*(1:ngroups),findall(n_flattenedgroups[i].==0))])
    end

    n_intergroups = Matrix{Int64}[]
    for i in 1:nspecies
        _n_intergroups = zeros(Int64,ngroups,ngroups)
        if length(n_groups[i]) > 1
            if haskey(mol_structure, model.components[i])
                _, group_names, bondmat = get_connectivity(model, mol_structure[model.components[i]])
            else
                _, group_names, bondmat = get_connectivity(model, model.components[i])
            end
            _groups = getindex.(split.(groups[i],"_"),1)
            ngroup_types_i = length(model.groups.groups[i])
            for k in 1:ngroup_types_i
                idx_group_i_1 = findall(_groups.==model.groups.groups[i][k])
                idx_group_i_2 = findall(group_names.==model.groups.groups[i][k])
                for l in 1:ngroup_types_i
                    idx_group_j_1 = findall(_groups.==model.groups.groups[i][l])
                    idx_group_j_2 = findall(group_names.==model.groups.groups[i][l])
                    if !isempty(idx_group_i_2) && !isempty(idx_group_j_2)
                        _n_intergroups[i_groups[i][idx_group_i_1],i_groups[i][idx_group_j_1]] = bondmat[idx_group_i_2,idx_group_j_2]
                    end
                end
            end
            push!(n_intergroups, _n_intergroups)
        else
            push!(n_intergroups, _n_intergroups)
        end
    end


    grouparam = GroupParam(model.components,
                    groups,
                    model.groups.grouptype,
                    n_groups,
                    n_intergroups,
                    i_groups,
                    flattenedgroups,
                    n_flattenedgroups,
                    model.groups.sourcecsvs)

    return grouparam,ngroups_k

end

expand_sites(model::EoSModel, groups, ngroups_k) = expand_sites(Clapeyron.getsites(model),model.groups,groups,ngroups_k)

function expand_sites(siteparam,oldgroups,groups,ngroups_k)
    nspecies = length(groups.components)

    # Expand the groups
    ngroup_types = length(oldgroups.flattenedgroups)
    # Expand the sites 
    assoc_groups = deepcopy(split.(siteparam.flattenedsites,"/"))
    assoc_sites = [assoc_groups[i][2] for i in 1:length(assoc_groups)]
    assoc_groups = [assoc_groups[i][1] for i in 1:length(assoc_groups)]

    n_sites_per_group = deepcopy(sum(siteparam.n_flattenedsites))

    flattenedsites = String[]

    n_sites_per_group_expanded = Int64[]
    
    for i in 1:length(assoc_groups)
        group_idx = findfirst(oldgroups.flattenedgroups.==assoc_groups[i])
        n_sites_per_group[i] /= ngroups_k[group_idx]
        append!(flattenedsites, oldgroups.flattenedgroups[group_idx]*"_".*string.(1:ngroups_k[group_idx]).*"/".*assoc_sites[i])
        append!(n_sites_per_group_expanded, n_sites_per_group[i]*ones(Int64,ngroups_k[group_idx]))
    end

    nsites = length(flattenedsites)
    assoc_sites = unique(assoc_sites)
    sites = []
    i_sites = Vector{Int64}[]
    i_flattenedsites = Vector{Int64}[]
    n_sites = Vector{Int64}[]  
    n_flattenedsites = Vector{Int64}[]  
    site_translator = Vector{Tuple{Int64,Int64}}[]
    k = 1
    for i in 1:nspecies
        sites_per_species = String[]
        _i_flattenedsites = zeros(Int64,nsites)
        _i_sites = Int64[]
        _n_flattenedsites = zeros(Int64,nsites)
        _n_sites = Int64[]
        _site_translator = Tuple{Int64,Int64}[]
        l = 1
        for j in 1:nsites
            group_name = split(flattenedsites[j],"/")[1]
            site_name = split(flattenedsites[j],"/")[2]
            if group_name in groups.groups[i]
                push!(sites_per_species, flattenedsites[j])
                push!(_i_sites, j)
                push!(_n_sites, n_sites_per_group_expanded[j])
                group_idx = findfirst(groups.flattenedgroups.==group_name)
                site_idx = findfirst(assoc_sites.==site_name)
                push!(_site_translator,(group_idx,site_idx))
                _i_flattenedsites[j] = l
                _n_flattenedsites[j] = n_sites_per_group_expanded[j]
                l+=1
            end
        end
        # _i_flattenedsites[k:k+length(sites_per_species)-1] = 1:length(sites_per_species)
        push!(sites, sites_per_species)
        push!(i_sites,_i_sites)
        push!(i_flattenedsites,_i_flattenedsites)
        push!(n_sites,_n_sites)
        push!(n_flattenedsites,_n_flattenedsites)
        push!(site_translator,_site_translator)
        k += length(sites_per_species)
    end

    siteparams = SiteParam(groups.components,
                           sites,
                           Clapeyron.pack_vectors(n_sites),
                           i_sites,
                           flattenedsites,
                           n_flattenedsites,
                           i_flattenedsites,
                           siteparam.sourcecsvs,
                           site_translator)

    # Expand the parameters
end

function expand_params(params::PARAM, groups, sites, ngroups_k) where PARAM
    # Expand the parameters
    params_names = fieldnames(PARAM)
    nparams = fieldcount(PARAM)
    ngroup_types = length(ngroups_k)
    ngroups = length(groups.n_flattenedgroups[1])
    nspecies = length(groups.components)
    newparams = []
    for i in 1:nparams
        param = getfield(params,i)
        name = param.name
        if param.components == groups.components && !(param isa AssocParam) && !(param isa Clapeyron.MixedGCSegmentParam)
            push!(newparams,param)
        elseif param isa Clapeyron.SingleParameter
            values = zeros(Float64,ngroups)
            ismissingvalues = zeros(Bool,ngroups)
            k = 1
            for j in 1:ngroup_types
                values[k:k-1+ngroups_k[j]] .= param.values[j]
                ismissingvalues[k:k-1+ngroups_k[j]] .= param.ismissingvalues[j]
                k+=ngroups_k[j]
            end
            push!(newparams, SingleParam(name,
                                        groups.flattenedgroups,
                                        values,
                                        ismissingvalues,
                                        param.sourcecsvs,
                                        param.sources))
                                    
        elseif param isa Clapeyron.PairParameter
            values = zeros(Float64,ngroups,ngroups)
            ismissingvalues = zeros(Bool,ngroups,ngroups)
            k = 1
            for a in 1:ngroup_types
                l = sum(ngroups_k[1:a-1])+1
                for b in a:ngroup_types
                    values[k:k-1+ngroups_k[a],l:l-1+ngroups_k[b]] .= param.values[a,b]
                    ismissingvalues[k:k-1+ngroups_k[a],l:l-1+ngroups_k[b]] .= param.ismissingvalues[a,b]
                    values[l:l-1+ngroups_k[b],k:k-1+ngroups_k[a]] .= param.values[a,b]
                    ismissingvalues[l:l-1+ngroups_k[b],k:k-1+ngroups_k[a]] .= param.ismissingvalues[a,b]
                    l+=ngroups_k[b]
                end
                k+=ngroups_k[a]
            end
            push!(newparams, PairParam(name,
                                        groups.flattenedgroups,
                                        values,
                                        ismissingvalues,
                                        param.sourcecsvs,
                                        param.sources))
        elseif param isa AssocParam
            if iszero(length((param.values.values))) || iszero(length(sites.n_sites.v))
                push!(newparams, param)
            else
                values = Float64[]
                inner_indices = Tuple{Int64,Int64}[]
                outer_indices = Tuple{Int64,Int64}[]
                n_interaction = length(param.values.values)
                assoc_groups = Vector{String}[]
                assoc_sites  = Vector{String}[]
                for i in 1:nspecies
                    push!(assoc_groups,first.(split.(first.(split.(sites.sites[i],"/")),"_")))
                    push!(assoc_sites, getindex.(split.(sites.sites[i],"/"),2))
                end
                for i in 1:n_interaction
                    value = param.values.values[i]                    
                    id_species_1, id_species_2, id_site_1, id_site_2 = Clapeyron.idx_to_ijab(param.values, i)
                    group_type_1,site_type_1 = split(param.sites[id_species_1][id_site_1],"/")
                    group_type_2,site_type_2 = split(param.sites[id_species_2][id_site_2],"/")

                    inner_idx_1 = findall(assoc_groups[id_species_1].==group_type_1 .&& assoc_sites[id_species_1].==site_type_1)
                    inner_idx_2 = findall(assoc_groups[id_species_2].==group_type_2 .&& assoc_sites[id_species_2].==site_type_2)

                    for i in 1:length(inner_idx_1)
                        for j in 1:length(inner_idx_2)
                            push!(values,value)
                            push!(inner_indices,(inner_idx_1[i],inner_idx_2[j]))
                            push!(outer_indices,(id_species_1,id_species_2))
                        end
                    end
                end

                TT = eltype(values)
                values = Clapeyron.Compressed4DMatrix(values,outer_indices,inner_indices)
                # println(components)
                push!(newparams, AssocParam{TT}(name,
                                            groups.components,
                                            values,
                                            sites.sites,
                                            param.sourcecsvs,
                                            param.sources))
            end
        elseif param isa Clapeyron.MixedGCSegmentParam
            TT = typeof(param.values[1][1])
            push!(newparams,Clapeyron.MixedGCSegmentParam{TT}(groups))
        end
    end
    return PARAM(newparams...)
end
