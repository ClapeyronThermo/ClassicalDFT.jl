import Clapeyron: PeTS, d_pets, PeTS_A, PeTS_B

"""
    PeTS(components::Vector{String})

The PeTS equation of state developed by Langenbach (2017). This is an unpublished approach which uses a Weighted Density Functional approach and does not use a chain propagator.

The bulk model can be obtained from Clapeyron. 
"""
PeTS

struct PeTSSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function get_species(model::PeTSModel,structure::DFTStructure)
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    T̄ = temperature./diagvalues(model.params.epsilon.values)
    size = d_pets.(T̄).*diagvalues(model.params.sigma.values)
    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), temperature, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / temperature
    nc = length(model)
    return PeTSSpecies(ones(Int64,nc),size,ρbulk,μres)
end

function length_scale(model::PeTSModel)
    return maximum(model.params.sigma.values)
end

function get_fields(model::PeTSModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}) where FP<:AbstractFloat
    nc = length(model)
    ψ = 1.21
    ngrid = structure.ngrid

    # Reduced units: same scheme as PCSAFT.jl's get_fields — ψ is already a fixed
    # constant (not derived from a d/σ ratio), so unlike SAFTVRMie/SAFTgammaMie/COFFEE
    # there's no raw-ratio-first subtlety here.
    L = length_scale(model)
    ω = structure_ω(structure, device, FP)
    d = species.size ./ L

    return (SWeightedDensity(:∫ρdz,0.5*d,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,0.5*d,ω,ngrid,device,model),
            VWeightedDensity(:∫ρzdz,0.5*d,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,ψ*d,ω,ngrid,device,model))
end

function get_propagator(model::PeTSModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    return IdealPropagator()
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
Pointwise residual free energy for PeTS: FMT hard-sphere + Barker–Henderson perturbation.

Field layout (for ND spatial dimensions):
  1        : ∫ρdz  with 0.5*d  → used to build n₀, n₁, n₂ FMT moments
  2        : ∫ρz²dz with 0.5*d → n₃ (volume fraction)
  3..2+ND  : ∫ρzdz with 0.5*d  → vector nᵥ
  3+ND     : ∫ρz²dz with ψ*d   → ρ̄ for perturbation term
"""
@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: PeTSModel}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(1))
    res_pert = f_pert(M, kk, n, params, T, Val(NC), Val(ND))
    out[kk] = res_hs + res_pert
    return nothing
end

"""
PeTS Barker–Henderson perturbation contribution at grid point `kk`.
"""
@inline function f_pert(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: PeTSModel}
    FP  = eltype(n)
    HSd = params.HSd
    m   = params.m
    σ   = params.sigma
    ϵ   = params.epsilon

    # Bare `π` (an Irrational) promotes to Float64 when combined with an Int literal or
    # negation before ever touching an FP value (`π/6`, `-2*π`, `-π*x` are all Float64;
    # `x*π`/`π*x` for x::FP stay FP) — see PCSAFT.jl's f_disp for the verified examples.
    ψ     = FP(1.21)
    T̄     = T / ϵ[1]
    idx_ρ̄ = 3 + ND
    ρ̃     = zero(FP); η_sum = zero(FP); ∑ρ̄ = zero(FP)
    @inbounds @unroll for i in 1:NC
        ρ̄zi  = n[kk, idx_ρ̄, i] * 3 / (4*ψ*ψ*ψ*HSd[i]*HSd[i]*HSd[i]*π)
        ρ̃    += ρ̄zi * m[i] * σ[i]*σ[i]*σ[i]
        η_sum += ρ̄zi * m[i] * HSd[i]*HSd[i]*HSd[i]
        ∑ρ̄   += ρ̄zi
    end
    η     = π * η_sum / 6
    I1    = evalpoly(η, params.PeTS_A)
    I2    = evalpoly(η, params.PeTS_B)
    ã1    = -2*(π*ρ̃*I1 / T̄)
    inv_g = 1 / (1 + 2*η*(4 - η) / (1 - η)^4)
    ã2    = -(π*ρ̃*I2*inv_g) / (T̄*T̄)
    return (ã1 + ã2) * ∑ρ̄
end

function preallocate_params(system::DFTSystem{<:PeTSModel})
    backend = system.options.device
    FP      = fptype(system.options)
    model   = system.model
    PeTS_A_fp = map(FP, PeTS_A)
    PeTS_B_fp = map(FP, PeTS_B)

    # Reduced units: divide every length-dimensioned parameter by L so it matches the
    # `get_fields`-side kernel rescaling. See PCSAFT.jl's `get_fields`/`preallocate_params`
    # docstrings for the full picture.
    L           = length_scale(model)
    HSd_local   = system.species.size ./ L
    sigma_local = diagvalues(model.params.sigma.values) ./ L

    params = (;
        HSd     = adapt_to_device(backend, FP, HSd_local),
        m       = adapt_to_device(backend, FP, model.params.segment.values),
        sigma   = adapt_to_device(backend, FP, sigma_local),
        epsilon = adapt_to_device(backend, FP, diagvalues(model.params.epsilon.values)),
        PeTS_A  = PeTS_A_fp,
        PeTS_B  = PeTS_B_fp,
    )
    nc = length(system.model)
    return params, nc
end
