import Clapeyron: ElectrolyteModel

include("electrostatic_potential.jl")
include("DH.jl")
include("LS.jl")


function ElectrolyteDFTSystem(model::ElectrolyteModel, structure::DFTStructure, external_field::ExternalFieldModel, options::DFTOptions = DFTOptions())
    species = get_species(model.neutralmodel, structure)
    ion_species = get_species(model.ionmodel, model.neutralmodel, model.charge, structure)

    FP = fptype(options)
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), temperature, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / temperature

    species.chempot_res .= μres

    fields = get_fields(model.neutralmodel, species, structure, options.device, FP)
    fields_ion = get_fields((model.ionmodel,FP(length_scale(model))), ion_species, structure, options.device, FP)

    typed_fields = tuple(fields..., fields_ion...)

    propagator = get_propagator(model.neutralmodel, species, structure, options.device, FP)

    external_field = [external_field, ElectrostaticPotential(model, structure, options.device, FP)]

    NF = compute_field_len(typed_fields,dimension(structure))
    chunksize = Val{NF}()
    return ElectrolyteDFTSystem(model, species, ion_species, structure, typed_fields, external_field, propagator, options,chunksize)
end

function ElectrolyteDFTSystem(model::ElectrolyteModel, structure::DFTStructure, options::DFTOptions = DFTOptions())
    device = options.device
    FP = fptype(options)
    species = get_species(model.neutralmodel, structure)
    ion_species = get_species(model.ionmodel, model.neutralmodel, model.charge, structure)

    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), temperature, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / temperature

    species.chempot_res .= μres

    fields = get_fields(model.neutralmodel, species, structure, device, FP)
    fields_ion = get_fields((model.ionmodel,FP(length_scale(model))), ion_species, structure, device, FP)
    
    typed_fields = tuple(fields..., fields_ion...)

    propagator = get_propagator(model.neutralmodel, species, structure, device, FP)

    external_field = [ElectrostaticPotential(model, structure, device, FP)]

    NF = compute_field_len(typed_fields,dimension(structure))
    chunksize = Val{NF}()
    return ElectrolyteDFTSystem(model, species, ion_species, structure, typed_fields, external_field, propagator, options,chunksize)
end

function f_res(system::ElectrolyteDFTSystem, model::ElectrolyteModel,n)
    n_neutral = @view(n[1:end-1,:])
    n_ion = @view(n[end,:])
    return f_res(system,model.neutralmodel,n_neutral) + f_res(system,model.ionmodel,n_ion)
end

length_scale(model::ElectrolyteModel) = length_scale(model.neutralmodel)

# ── Enzyme/KA kernel path for ElectrolyteDFTSystem ──────────────────────────

function preallocate_params(system::ElectrolyteDFTSystem)
    model   = system.model
    n_model = model.neutralmodel
    nd      = dimension(system)

    # Drop the last field (ion) to build a proxy neutral DFTSystem
    n_fields   = Base.front(system.fields)
    NF_neutral = compute_field_len(n_fields, nd)
    n_sys = DFTSystem(n_model, system.species, system.structure,
                      n_fields, nothing, system.propagator,
                      system.options, Val{NF_neutral}())
    neutral_params, nc = preallocate_params(n_sys)

    # Ion model params — dispatch to specific ion model implementation
    ion_params = preallocate_params(system, model.ionmodel)

    return merge(neutral_params, ion_params), nc
end

@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: ElectrolyteModel}
    MN = fieldtype(M, :neutralmodel)
    MI = fieldtype(M, :ionmodel)
    f_res(MN, kk, out, n, params, T, Val(NC), Val(ND))
    f_res(MI, kk, out, n, params, T, Val(NC), Val(ND))
    return nothing
end