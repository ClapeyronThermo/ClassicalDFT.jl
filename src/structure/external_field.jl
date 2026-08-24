include("external_fields/steele.jl")
# include("external_fields/lj.jl")

evaluate_external_field!(system::DGTSystem, ρ, δfδρ_res, ::Nothing) = nothing

function evaluate_external_field!(system::AbstractClassicalDFTSystem, ρ, δfδρ_res, cache_external)
    structure = system.structure
    ngrid = structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)

    external_fields = system.external_field
    if isnothing(external_fields)
        return nothing
    end
    for i in 1:length(external_fields)
        evaluate_external_field!(structure, external_fields[i], system.model, ρ, δfδρ_res, cache_external[i]...)
    end
end

function initialize_profiles!(system::AbstractClassicalDFTSystem,external_field::ExternalFieldModel, ρ)
    z = get_coords(system.structure)
    nd = dimension(system.structure)    
    Vext = Adapt.adapt(typeof(ρ),evaluate_external_field!(system.structure, external_field, system.model, ρ, ρ, z))
    ρ .*= exp.(-Vext./10)
    return ρ
end