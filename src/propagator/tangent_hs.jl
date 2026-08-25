function TangentHSPropagator(model::EoSModel,species::DFTSpecies,structure::DFTStructure,device::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    ngrid = structure.ngrid
    nbeads = sum(species.nbeads)
    nd = dimension(structure)
    L = length_scale(model)
    Ω = allocate(device,Complex{FP},ngrid...,nbeads,nbeads)
    ω = structure_ω(structure, device, FP)
    ω = Adapt.adapt(device, ω)
    ω = _scaled_ω(ω, L, FP)
    for i in @comps
        l = 1
        for j in @chain(i)
            for k in @chain(i)[l:end]
                R = FP((species.size[j] + species.size[k])/L*π)
                
                ω̄ = dropdims(sqrt.(sum(abs2, ω, dims=nd+1)), dims=nd+1)  # lives on same backend as ω
                ω̄R   = ω̄ .* R  


                mask = ω̄ .== 0
                selectdim(selectdim(Ω, nd+1, j), nd+1, k) .= ifelse.(mask,
                        1,                  # ω̄=0 case
                        sin.(ω̄R)./ω̄R        # ω̄≠0 case
                    )
                selectdim(selectdim(Ω, nd+1, k), nd+1, j) .= selectdim(selectdim(Ω, nd+1, j), nd+1, k)
            end
            l += 1
        end
    end

    return TangentHSPropagator(Ω)
end

"""
    TangentHSPropagator(model, species, structure::DFTStructure{N,Union{Cylindrical,Spherical}}, device, FP)

Spherical/cylindrical (QDHT-based) counterpart of the Cartesian `TangentHSPropagator`
constructor above. Reuses the same tangent-sphere kernel formula, substituting
`ω̄ = structure_ω(structure,...).ω̄` for the Cartesian `ω̄ = sqrt.(sum(abs2,ω,dims=nd+1))`
and dropping the `ω̄=0` branch (QDHT never samples the origin in k-space).
"""
function TangentHSPropagator(model::EoSModel,species::DFTSpecies,structure::Union{DFTStructByCoord{Cylindrical},DFTStructByCoord{Spherical}},device::Backend, ::Type{FP}=Float64) where {FP<:AbstractFloat}
    ngrid = structure.ngrid
    nbeads = sum(species.nbeads)
    nd = dimension(structure)
    L = length_scale(model)
    ω̄ = _scaled_ω(structure_ω(structure, device, FP), L, FP).ω̄
    Ω = allocate(device,FP,ngrid...,nbeads,nbeads)
    
    σ = sinc.(ω̄ ./ maximum(ω̄))

    for i in @comps
        l = 1
        for j in @chain(i)
            for k in @chain(i)[l:end]
                R = (species.size[j] + species.size[k])/L*π

                val = @. sin(ω̄*R)/ω̄ / R * σ

                selectdim(selectdim(Ω, nd+1, j), nd+1, k) .= val
                selectdim(selectdim(Ω, nd+1, k), nd+1, j) .= val
            end
            l += 1
        end
    end

    return TangentHSPropagator(Ω)
end

function preallocate_propagator(system::AbstractcDFTSystem,propagator::TangentHSPropagator,ρ,backend::Backend)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    FP = eltype(ρ)
    Gcα = allocate(backend, FP, size(ρ)..., sum(system.species.nbeads))
    Gcα .= 1
    Gp = allocate(backend, FP, size(ρ)...)
    Gp .= 1
    CT = transform_eltype(system.structure, FP)
    buf = similar(selectdim(ρ,nd+1,1), CT)
    scratch = allocate(backend, FP, ngrid...)
    plan, iplan = build_transform(system.structure, buf, length(ngrid), backend)
    return Gcα, Gp, buf, plan, iplan, scratch
end


function propagate!(system::AbstractcDFTSystem, propagate::TangentHSPropagator, ρ, δfδρ_res, Gcα, Gp, buf, P, iP, scratch)
    nd = dimension(system)
    model = system.model
    structure = system.structure
    ngrid = structure.ngrid
    species = system.species
    nbeads = sum(system.species.nbeads)

    map = propagate.map

    levels = species.levels
    for i in @comps
        if system.species.nbeads[i] !== 1
            n_intergroups = model.groups.n_intergroups[i] .== 1
            i_groups = model.groups.i_groups[i]
            n_levels = maximum(levels[i_groups])

            i_root = i_groups[levels[i_groups].==1][1]
            is_leaf = sum(n_intergroups,dims=1).==1 .&& (levels.!=1)'
            # Bottom-up pass: compute Gcα
            for L in n_levels:-1:1
                i_group_level = i_groups[findall(levels[i_groups].==L)]
                for k in i_group_level
                    k_children = findall(n_intergroups[k,:] .&& levels.==L+1)
                    if !is_leaf[k]
                        for α in k_children
                            β = findall(n_intergroups[α,:] .&& levels.==L+2)
                            buf .= exp.(-selectdim(δfδρ_res, nd+1, α))
                            for β_k in β
                                buf .*= selectdim(selectdim(Gcα, nd+1, α), nd+1, β_k)
                            end
                            convolve!(selectdim(selectdim(Gcα,nd+1,k),nd+1,α), buf, selectdim(selectdim(map,nd+1,k),nd+1,α), P, iP, buf)
                        end
                    end
                end
            end

            # Top-down pass: compute Gp
            for L in 1:n_levels
                i_group_level = i_groups[findall(levels[i_groups].==L)]
                for k in i_group_level
                    if k == i_root
                        selectdim(Gp,nd+1,k) .= 1
                    else
                        l = findall(n_intergroups[k,:] .&& levels.==L-1)[1]

                        α = findall(n_intergroups[l,:] .&& levels.==L)
                        α = α[α.!=k]

                        buf .= exp.(-selectdim(δfδρ_res, nd+1, l)) .* selectdim(Gp, nd+1, l)
                        for α_k in α
                            buf .*= selectdim(selectdim(Gcα, nd+1, l), nd+1, α_k)
                        end
                        convolve!(selectdim(Gp,nd+1,k), buf, selectdim(selectdim(map,nd+1,l),nd+1,k), P, iP, buf)
                    end
                end
            end
        end
    end

    # Final update: subtract chain-bonding contributions from δfδρ_res
    for i in @comps
        for j in @chain(i)
            if system.species.nbeads[i] != 1
                α_vec = findall(model.groups.n_intergroups[i][j,:] .== 1 .&& species.levels .> species.levels[j])
            else
                α_vec = (j,)
            end
            scratch .= log.(selectdim(Gp, nd+1, j))
            for α_k in α_vec
                scratch .+= log.(selectdim(selectdim(Gcα, nd+2, α_k), nd+1, j))
            end
            selectdim(δfδρ_res, nd+1, j) .-= scratch
        end
    end
end