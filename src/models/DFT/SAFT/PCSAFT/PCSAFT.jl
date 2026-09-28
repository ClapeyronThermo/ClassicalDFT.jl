using Clapeyron: PCSAFTModel

"""
    PCSAFT(components::Vector{String})

The PC-SAFT equation of state developed by Gross and Sadowski (2001). Our DFT implementation follows the work of Sauer and Gross (2017) which uses a Weighted Density Functional approach and does not use a chain propagator. The only additional information required in `PCSAFTSpecies` is the bead size at a given temperature.

The bulk model can be obtained from Clapeyron. 
"""
PCSAFT

struct PCSAFTSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

"""
    _scaled_ω(ω, L, ::Type{FP})

Rescale a `structure_ω` result by the reduced-units length scale `L`, matching the
`width → width/L` rescale applied to weighted-density kernel widths elsewhere in
`get_fields` — together these leave the `ω̄*R` trig argument inside each kernel's closed
form exactly invariant (see `length_scale`/reduced-units design notes on `PCSAFTSpecies`).
Dispatches on whether `structure_ω` returned a plain (Cartesian FFT) array or a
`RadialFrequency` (spherical/cylindrical QDHT) wrapper, since the latter can't be scaled
by a bare broadcast multiply.
"""
_scaled_ω(ω::AbstractArray, L, ::Type{FP}) where FP<:AbstractFloat = ω .* FP(L)
_scaled_ω(ω::RadialFrequency, L, ::Type{FP}) where FP<:AbstractFloat = RadialFrequency(ω.Q, ω.ω̄ .* FP(L))

"""
    get_fields(model::EoSModel, species::DFTSpecies, structure::DFTStructure)

For a given `model`, obtain all of the fields that will be needed to perform the DFT calculation. This function should return a vector of `DFTField`s.

Builds every weighted-density kernel in **reduced units**: lengths (`sigma`, `HSd`-derived
widths) are divided by `L = length_scale(model)`. `ω`-rescaling and the `density_scale`/
`N_A*L^3` compensation now happen internally inside `SWeightedDensity`/`VWeightedDensity`
(they take `model` directly and compute `L` themselves) — together these keep every
intermediate weighted-density value near O(1) regardless of floating-point precision. See
`preallocate_params` for the matching `HSd`/`sigma` rescale applied to the parameters
consumed directly (unconvolved) by `f_hc`/`f_disp`, and `F_res` for the compensating
`1/L^3` applied to the returned scalar free energy.
"""
function get_fields(model::PCSAFTModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}) where FP<:AbstractFloat
    nc = length(model)
    ngrid = structure.ngrid
    #f = [ngrid[i]/(structure.bounds[i,2]-structure.bounds[i,1]) for i in 1:length(ngrid)]
    L = length_scale(model)
    ω = structure_ω(structure, device, FP)
    ψ = 1.3862
    d = species.size ./ L
    return (SWeightedDensity(:ρ,zeros(nc),ω,ngrid,device,model),
            SWeightedDensity(:∫ρdz,0.5*d,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,0.5*d,ω,ngrid,device,model),
            VWeightedDensity(:∫ρzdz,0.5*d,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,d,ω,ngrid,device,model),
            SWeightedDensity(:∫ρdz,d,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,d .* ψ,ω,ngrid,device,model))
end


function get_species(model::PCSAFTModel,structure::DFTStructure)
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    size = d(model,1e-3,temperature,ρbulk)
    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), temperature, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / temperature
    nc = length(model)
    return PCSAFTSpecies(ones(Int64,nc),size,ρbulk,μres)
end


function get_propagator(model::PCSAFTModel, species::DFTSpecies, structure::DFTStructure, backend::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    return IdealPropagator()
end

function length_scale(model::SAFTModel)
    return maximum(model.params.sigma.values)
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────
# These constants are GPU-safe (no heap allocation, no Clapeyron calls).

const PCSAFT_CORR1 = (
    (0.9105631445, -0.3084016918, -0.0906148351),
    (0.6361281449,  0.1860531159,  0.4527842806),
    (2.6861347891, -2.5030047259,  0.5962700728),
    (-26.547362491, 21.419793629, -1.7241829131),
    (97.759208784, -65.25588533,  -4.1302112531),
    (-159.59154087, 83.318680481,  13.77663187),
    (91.297774084, -33.74692293,  -8.6728470368)
)
const PCSAFT_CORR2 = (
    (0.7240946941, -0.5755498075,  0.0976883116),
    (2.2382791861,  0.6995095521, -0.2557574982),
    (-4.0025849485, 3.892567339,  -9.155856153),
    (-21.003576815,-17.215471648,  20.642075974),
    (26.855641363, 192.67226447,  -38.804430052),
    (206.55133841,-161.82646165,   93.626774077),
    (-355.60235612,-165.20769346, -29.666905585)
)

"""
Pointwise residual free energy for PC-SAFT, written in Enzyme/KernelAbstractions-compatible style.
`out[kk]` accumulates the scalar integrand at grid point `kk`.
All model parameters are unpacked from `params` (a NamedTuple of device-adapted arrays).
"""
@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: PCSAFTModel}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_hc  = f_hc(M, kk, n, params, T, Val(NC), Val(ND))
    res_disp, _, _ = f_disp(M, kk, n, params, T, Val(NC), Val(ND))
    res_assoc = _assoc_or_zero(M, kk, n, params, T, Val(NC), Val(ND))
    out[kk] = res_hs + res_hc + res_disp + res_assoc
    return nothing
end

"""
PC-SAFT hard-chain contribution at grid point `kk`.
Field layout assumed: field 1 = ρ (unweighted), field 4+ND = ρ̄hc, field 5+ND = λ.
"""
@inline function f_hc(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: PCSAFTModel}
    FP  = eltype(n)
    m   = params.m
    HSd = params.HSd
    idx_ζ = 4 + ND;  idx_λ = 5 + ND
    ζ₃=zero(FP); ζ₂=zero(FP)
    @inbounds @unroll for i in 1:NC
        mi=m[i]; di=HSd[i]; ρ̄hci=n[kk, idx_ζ, i]
        ζ₃ += mi * ρ̄hci;  ζ₂ += mi * ρ̄hci / di
    end
    ζ₃ /= 8;  ζ₂ /= 8
    inv1ζ₃ = one(FP)/(one(FP)-ζ₃)
    res_hc = zero(FP)
    @inbounds @unroll for i in 1:NC
        ρi  = n[kk, 1, i]
        λ   = n[kk, idx_λ, i] / (2*HSd[i])
        ydd = inv1ζ₃ + FP(1.5)*HSd[i]*ζ₂*inv1ζ₃*inv1ζ₃ +
              FP(0.5)*HSd[i]*HSd[i]*ζ₂*ζ₂*inv1ζ₃*inv1ζ₃*inv1ζ₃
        res_hc += -ρi * (m[i]-1) * Base.log(abs(ydd*λ/ρi))
    end
    return res_hc
end

"""
PC-SAFT dispersion contribution at grid point `kk`. Field 6+ND is the dispersion density.
Returns `(res_disp, m̄, ηd)` — m̄ and ηd are reused by PCP-SAFT for the polar term.
"""
@inline function f_disp(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: PCSAFTModel}
    FP     = eltype(n)
    m      = params.m
    HSd    = params.HSd
    sigma  = params.sigma
    epsilon = params.epsilon
    # `-2*π` (Int*Irrational, negated) promotes to Float64 before ever touching an FP
    # value, unlike `π*x`/`x*π` for x::FP — see line below (`res_disp`) where this
    # otherwise silently makes the return value Float64.
    _π     = FP(π)
    ψ      = FP(1.3862)
    idx_ρz = 6 + ND
    factor = 3 / (4*ψ*ψ*ψ*_π)

    ρ̄z_sum=zero(FP); m̄_top=zero(FP); η_sum=zero(FP)
    @inbounds @unroll for i in 1:NC
        ρ̄zi = n[kk, idx_ρz, i] * factor / (HSd[i]*HSd[i]*HSd[i])
        ρ̄z_sum += ρ̄zi
        m̄_top  += ρ̄zi * m[i]
        η_sum  += m[i] * ρ̄zi * HSd[i]*HSd[i]*HSd[i]
    end
    m̄  = m̄_top / ρ̄z_sum
    ηd = η_sum * _π / 6

    m2ϵσ3_1=zero(FP); m2ϵσ3_2=zero(FP)
    @inbounds @unroll for i in 1:NC
        ρzi = n[kk, idx_ρz, i] * factor / (HSd[i]*HSd[i]*HSd[i])
        @inbounds @unroll for j in 1:NC
            ρzj  = n[kk, idx_ρz, j] * factor / (HSd[j]*HSd[j]*HSd[j])
            cij  = ρzi * ρzj * m[i] * m[j] * sigma[i,j]*sigma[i,j]*sigma[i,j]
            eT   = epsilon[i,j] / T
            m2ϵσ3_1 += cij * eT
            m2ϵσ3_2 += cij * eT * eT
        end
    end
    ηd2    = ηd*ηd
    m1ηd   = one(FP)-ηd
    ηd4    = m1ηd*m1ηd*m1ηd*m1ηd
    inv1ηd = one(FP)/(one(FP)-ηd)
    inv2ηd = one(FP)/(2-ηd)
    C₁     = one(FP) + m̄*(8*ηd-2*ηd2)/ηd4 +
              (one(FP)-m̄)*(20*ηd-27*ηd2+12*(ηd*ηd2)-2*(ηd2*ηd2)) *
              inv1ηd*inv1ηd*inv2ηd*inv2ηd
    I₁     = I_lite(PCSAFT_CORR1, m̄, ηd)
    I₂     = I_lite(PCSAFT_CORR2, m̄, ηd)
    res_disp = -2*(_π*I₁*m2ϵσ3_1) - _π*m̄*I₂*m2ϵσ3_2 / C₁
    return res_disp, m̄, ηd
end

@inline function I_lite(corr, m̄, η)
    FP = typeof(m̄)
    m2 = (m̄ - 1) / m̄
    m3 = m2 * (m̄ - 2) / m̄
    c1 = corr[1]; a1 = FP(c1[1]) + m2*FP(c1[2]) + m3*FP(c1[3])
    c2 = corr[2]; a2 = FP(c2[1]) + m2*FP(c2[2]) + m3*FP(c2[3])
    c3 = corr[3]; a3 = FP(c3[1]) + m2*FP(c3[2]) + m3*FP(c3[3])
    c4 = corr[4]; a4 = FP(c4[1]) + m2*FP(c4[2]) + m3*FP(c4[3])
    c5 = corr[5]; a5 = FP(c5[1]) + m2*FP(c5[2]) + m3*FP(c5[3])
    c6 = corr[6]; a6 = FP(c6[1]) + m2*FP(c6[2]) + m3*FP(c6[3])
    c7 = corr[7]; a7 = FP(c7[1]) + m2*FP(c7[2]) + m3*FP(c7[3])
    return evalpoly(η, (a1, a2, a3, a4, a5, a6, a7))
end

function preallocate_params(system::DFTSystem{<:PCSAFTModel})
    backend = system.options.device
    FP      = fptype(system.options)
    model   = system.model

    # Reduced units: divide every length-dimensioned parameter by L so it matches the
    # `get_fields`-side kernel rescaling. See `get_fields`'s docstring for the full picture.
    L           = length_scale(model)
    HSd_local   = system.species.size ./ L
    sigma_local = model.params.sigma.values ./ L

    base = (;
        HSd     = adapt_to_device(backend, FP, HSd_local),
        m       = adapt_to_device(backend, FP, model.params.segment.values),
        sigma   = adapt_to_device(backend, FP, sigma_local),
        epsilon = adapt_to_device(backend, FP, model.params.epsilon.values),
    )

    nn = Clapeyron.assoc_pair_length(model)
    if nn > 0
        (assoc_icomp_v, assoc_jcomp_v, assoc_isite_v, assoc_jsite_v,
         assoc_eps_v, assoc_kap_v, assoc_sig3_v, assoc_dij_v,
         n_sites_flat_v, n_sites_cumsum_v, total_sites
        ) = pack_assoc_params(model, HSd_local, sigma_local)

        # Integer index arrays stored as NTuples so Enzyme treats them as
        # truly-immutable Const data (avoids EnzymeRuntimeActivityError from
        # Const Vector aliasing). Padded to fixed max sizes (20 pairs, 20 sites,
        # 11 cumsum entries for ≤10 components).
        nc_model         = length(model)
        ia_global_v      = [n_sites_cumsum_v[assoc_icomp_v[p]] + assoc_isite_v[p] for p in 1:nn]
        jb_global_v      = [n_sites_cumsum_v[assoc_jcomp_v[p]] + assoc_jsite_v[p] for p in 1:nn]
        n_ia_v           = [n_sites_flat_v[ia_global_v[p]] for p in 1:nn]
        n_jb_v           = [n_sites_flat_v[jb_global_v[p]] for p in 1:nn]
        assoc_icomp_t    = ntuple(p -> assoc_icomp_v[p],    Val(nn))
        assoc_jcomp_t    = ntuple(p -> assoc_jcomp_v[p],    Val(nn))
        assoc_isite_t    = ntuple(p -> assoc_isite_v[p],    Val(nn))
        assoc_jsite_t    = ntuple(p -> assoc_jsite_v[p],    Val(nn))
        assoc_ia_global_t = ntuple(p -> ia_global_v[p],     Val(nn))
        assoc_jb_global_t = ntuple(p -> jb_global_v[p],     Val(nn))
        assoc_n_ia_t      = ntuple(p -> n_ia_v[p],          Val(nn))
        assoc_n_jb_t      = ntuple(p -> n_jb_v[p],          Val(nn))
        n_sites_flat_t   = ntuple(j -> n_sites_flat_v[j],   Val(total_sites))
        n_sites_cumsum_t = ntuple(i -> n_sites_cumsum_v[i], Val(nc_model + 1))

        assoc = (;
            has_assoc       = true,
            assoc_n_pairs   = Val(nn),
            assoc_n_sites   = Val(total_sites),
            assoc_icomp     = assoc_icomp_t,
            assoc_jcomp     = assoc_jcomp_t,
            assoc_isite     = assoc_isite_t,
            assoc_jsite     = assoc_jsite_t,
            assoc_ia_global = assoc_ia_global_t,
            assoc_jb_global = assoc_jb_global_t,
            assoc_n_ia      = assoc_n_ia_t,
            assoc_n_jb      = assoc_n_jb_t,
            assoc_eps       = adapt_to_device(backend, FP, assoc_eps_v),
            assoc_kap       = adapt_to_device(backend, FP, assoc_kap_v),
            assoc_sig3      = adapt_to_device(backend, FP, assoc_sig3_v),
            assoc_dij       = adapt_to_device(backend, FP, assoc_dij_v),
            n_sites_flat    = n_sites_flat_t,
            n_sites_cumsum  = n_sites_cumsum_t,
            total_sites,
        )
        params = merge(base, assoc)
    else
        params = merge(base, (; has_assoc = false))
    end

    nc = length(model)
    return params, nc
end
