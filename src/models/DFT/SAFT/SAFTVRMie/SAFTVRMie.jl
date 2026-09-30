using Clapeyron: SAFTVRMieModel
using Clapeyron: aS_1, B, KHS, Cλ, f123456
using Clapeyron: KHS_fdf, aS_1_fdf, B_fdf, g_HS
using Clapeyron: SAFTVRMieconsts

"""
    SAFTVRMie(components::Vector{String})

The SAFT-VR Mie equation of state (Lafitte et al., 2013), which uses a variable-range Mie potential for the segment-segment interactions. As with `PCSAFT`, our DFT implementation uses a Weighted Density Functional approach and does not use a chain propagator.

The bulk model can be obtained from Clapeyron.
"""
SAFTVRMie

struct SAFTVRMieSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function get_fields(model::SAFTVRMieModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}) where FP<:AbstractFloat
    nc = length(model)
    ngrid = structure.ngrid

    # ψ is a dimensionless Barker-Henderson-style shape factor derived from the RAW
    # (unscaled) diameter/sigma ratio — it must stay L-invariant, so compute it before
    # any reduced-units rescaling is applied below. See PCSAFT.jl's `get_fields` docstring
    # for the overall reduced-units scheme.
    λ_r = diagvalues(model.params.lambda_r.values)
    λ_a = diagvalues(model.params.lambda_a.values)
    σ   = diagvalues(model.params.sigma.values)
    C = @. λ_r / (λ_r - λ_a) * (λ_r / λ_a)^(λ_a / (λ_r - λ_a))
    x = species.size ./ σ
    ψ = @. cbrt(3*C*x^3*(x^-λ_a/(λ_a-3)-x^-λ_r/(λ_r-3)))

    L = length_scale(model)
    ω = structure_ω(structure, device, FP)
    d = species.size ./ L

    return (SWeightedDensity(:ρ,zeros(nc),ω,ngrid,device,model),
            SWeightedDensity(:∫ρdz,0.5*d,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,0.5*d,ω,ngrid,device,model),
            VWeightedDensity(:∫ρzdz,0.5*d,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,d,ω,ngrid,device,model),
            SWeightedDensity(:∫ρdz,d,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,d .* ψ,ω,ngrid,device,model))
end

function get_species(model::SAFTVRMieModel,structure::DFTStructure)
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    size = d(model,1e-3,temperature,ρbulk)

    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), temperature, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / temperature
    nc = length(model)
    return SAFTVRMieSpecies(ones(Int64,nc),size,ρbulk,μres)
end

function get_propagator(model::SAFTVRMieModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    return IdealPropagator()
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────
# GPU-safe constants: row-tuple form of the SAFTγMie A matrix for ζeff.
const SAFTVRMIE_A = (
    ( 0.81096,    1.7888,   -37.578,   92.284),
    ( 1.02050,  -19.341,   151.26,   -463.50),
    (-1.90570,   22.845,  -228.14,   973.92),
    ( 1.08850,   -6.1962,  106.98,  -677.64),
)

"""
Pointwise residual free energy for SAFT-VR Mie: FMT hard-sphere + chain + dispersion + association.

Field layout (same as PCSAFTModel):
  1        : ρ (unweighted)
  2        : ∫ρdz  with 0.5*d → n₀, n₁, n₂
  3        : ∫ρz²dz with 0.5*d → n₃
  4..3+ND  : ∫ρzdz with 0.5*d → nᵥ
  4+ND     : ∫ρz²dz with d    → ρ̄hc  (chain)
  5+ND     : ∫ρdz  with d    → λ    (chain)
  6+ND     : ∫ρz²dz with d*ψ → ρ̄z   (dispersion)
"""
@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: SAFTVRMieModel}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_chain = f_chain(M, kk, n, params, T, Val(NC), Val(ND))
    res_disp  = f_disp(M, kk, n, params, T, Val(NC), Val(ND), Val(6+ND))
    res_assoc = _assoc_or_zero(M, kk, n, params, T, Val(NC), Val(ND))
    out[kk] = res_hs + res_chain + res_disp + res_assoc
    return nothing
end

"""
SAFT-VR Mie chain contribution (gMie contact value) at grid point `kk`.
"""
@inline function f_chain(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: SAFTVRMieModel}
    HSd     = params.HSd
    m_seg   = params.m
    σ       = params.sigma
    ϵ       = params.epsilon
    λr_mat  = params.lambda_r_t
    λa_mat  = params.lambda_a_t
    A       = params.A
    ϕ       = params.phi
    FP      = eltype(n)
    # See SAFTgammaMie.jl's f_chain for why bare `π` needs this: `4*π` (Int*Irrational)
    # promotes to Float64 before ever touching an FP value, unlike `π*x` for x::FP.
    idx_ζ = 4+ND;  idx_λ = 5+ND

    ρS_c = zero(FP)
    @inbounds @unroll for i in 1:NC
        ρS_c += n[kk,idx_ζ,i] * 3/(4*(π*HSd[i]^3)) * m_seg[i]
    end
    kρS_c = ρS_c * π/6/8

    ζ_Xc = zero(FP);  σ3_xc = zero(FP)
    @inbounds @unroll for i in 1:NC
        di   = HSd[i]
        ρ̄hci = n[kk,idx_ζ,i] * 3/(4*(π*di^3))
        x_Si = ρ̄hci * m_seg[i] / ρS_c
        @inbounds @unroll for j in 1:NC
            dj   = HSd[j]
            ρ̄hcj = n[kk,idx_ζ,j] * 3/(4*(π*dj^3))
            x_Sj = ρ̄hcj * m_seg[j] / ρS_c
            σ3_xc += x_Si*x_Sj*σ[i,j]^3
            ζ_Xc  += kρS_c*x_Si*x_Sj*(di+dj)^3
        end
    end
    ζstc = σ3_xc * ρS_c * π/6

    _KHSc, _∂KHSc = _KHS_fdf_kernel(ρS_c, ζ_Xc)

    res_chain = zero(FP)
    @inbounds @unroll for i in 1:NC
        di   = HSd[i]
        ρ̄hci = n[kk,idx_ζ,i] * 3/(4*(π*di^3))
        x_Si = ρ̄hci * m_seg[i] / ρS_c

        λa = λa_mat[i][i];  λr = λr_mat[i][i]
        _C = _Cλ_kernel(λa, λr)
        x0 = σ[i,i] / di
        ϵii = ϵ[i,i]

        aS1_a,  dS1_a  = _aS1_fdf_kernel(λa,   ζ_Xc, A)
        aS1_r,  dS1_r  = _aS1_fdf_kernel(λr,   ζ_Xc, A)
        B_a,    dB_a   = _B_fdf_kernel(λa,      x0, ζ_Xc)
        B_r,    dB_r   = _B_fdf_kernel(λr,      x0, ζ_Xc)
        aS1_2a, dS1_2a = _aS1_fdf_kernel(2*λa,  ζ_Xc, A)
        aS1_2r, dS1_2r = _aS1_fdf_kernel(2*λr,  ζ_Xc, A)
        aS1_ar, dS1_ar = _aS1_fdf_kernel(λa+λr, ζ_Xc, A)
        B_2a,   dB_2a  = _B_fdf_kernel(2*λa,    x0, ζ_Xc)
        B_2r,   dB_2r  = _B_fdf_kernel(2*λr,    x0, ζ_Xc)
        B_ar,   dB_ar  = _B_fdf_kernel(λa+λr,   x0, ζ_Xc)

        ∂a1ρS = _C*(x0^λa*(dS1_a+dB_a) - x0^λr*(dS1_r+dB_r))
        g1_   = 3*∂a1ρS - _C*(λa*x0^λa*(aS1_a+B_a) - λr*x0^λr*(aS1_r+B_r))

        α   = _C*(1/(λa-3) - 1/(λr-3))
        f1,f2,f3,f4,f5,f6 = _f123456_kernel(α, ϕ)
        θ   = exp(ϵii/T) - 1
        γc  = 10*(-tanh(10*(FP(0.57)-α))+1)*ζstc*θ*exp(-FP(6.7)*ζstc-8*ζstc^2)

        cb2a = x0^(2*λa)*(aS1_2a+B_2a)
        cbar = x0^(λa+λr)*(aS1_ar+B_ar)
        cb2r = x0^(2*λr)*(aS1_2r+B_2r)
        ∂a2ρS = _C*_C/2*(
            ρS_c*_∂KHSc*(cb2a - 2*cbar + cb2r)
          + _KHSc*(x0^(2*λa)*(dS1_2a+dB_2a)
                 - 2*x0^(λa+λr)*(dS1_ar+dB_ar)
                 + x0^(2*λr)*(dS1_2r+dB_2r))
        )
        gMCA2 = 3*∂a2ρS - _KHSc*_C*_C*(λr*cb2r - (λa+λr)*cbar + λa*cb2a)
        g2_ = (1+γc)*gMCA2

        gHS   = _gHS_kernel(x0, ζ_Xc)
        gMie  = gHS * exp(ϵii/T * g1_/gHS + (ϵii/T)^2 * g2_/gHS)

        ρhci  = n[kk,1,i]
        λfld  = n[kk,idx_λ,i] / (2*di)
        res_chain += ρhci * Base.log(abs(gMie*λfld/ρhci)) * (m_seg[i]-1)
    end
    return -res_chain
end

@inline function _Cλ_kernel(λa, λr)
    return (λr / (λr - λa)) * (λr / λa)^(λa / (λr - λa))
end

# ζeff(λ, ζ_X) using SAFTVRMIE_A (row-tuples, FP-typed from preallocate_params)
@inline function _ζeff_kernel(λ, ζ_X, A)
    li = 1/λ;  li2 = li*li;  li3 = li2*li
    aλ1 = A[1][1] + A[1][2]*li + A[1][3]*li2 + A[1][4]*li3
    aλ2 = A[2][1] + A[2][2]*li + A[2][3]*li2 + A[2][4]*li3
    aλ3 = A[3][1] + A[3][2]*li + A[3][3]*li2 + A[3][4]*li3
    aλ4 = A[4][1] + A[4][2]*li + A[4][3]*li2 + A[4][4]*li3
    ζ2 = ζ_X*ζ_X;  ζ3 = ζ2*ζ_X;  ζ4 = ζ3*ζ_X
    return aλ1*ζ_X + aλ2*ζ2 + aλ3*ζ3 + aλ4*ζ4
end

@inline function _aS1_kernel(λ, ζ_X, A)
    ζeff = _ζeff_kernel(λ, ζ_X, A)
    return -1/(λ-3) * (1 - ζeff/2) / (1-ζeff)^3
end

# Returns (aS_1, d(ρS·aS_1)/dρS)
@inline function _aS1_fdf_kernel(λ, ζ_X, A)
    li = 1/λ;  li2 = li*li;  li3 = li2*li
    aλ1 = A[1][1] + A[1][2]*li + A[1][3]*li2 + A[1][4]*li3
    aλ2 = A[2][1] + A[2][2]*li + A[2][3]*li2 + A[2][4]*li3
    aλ3 = A[3][1] + A[3][2]*li + A[3][3]*li2 + A[3][4]*li3
    aλ4 = A[4][1] + A[4][2]*li + A[4][3]*li2 + A[4][4]*li3
    ζ2 = ζ_X*ζ_X;  ζ3 = ζ2*ζ_X;  ζ4 = ζ3*ζ_X
    ζeff = aλ1*ζ_X + aλ2*ζ2 + aλ3*ζ3 + aλ4*ζ4
    # ∂ζeff/∂ζ_X * ζ_X  (= ρS·∂ζeff/∂ρS since ζ_X ∝ ρS)
    dζeff_ζ = aλ1*ζ_X + 2*aλ2*ζ2 + 3*aλ3*ζ3 + 4*aλ4*ζ4
    ζeff3   = (1-ζeff)^3
    ζeffm1  = 1 - ζeff/2
    ζf      = ζeffm1 / ζeff3
    λf      = -1 / (λ-3)
    f       = λf * ζf
    dζf     = (3*ζeffm1*(1-ζeff)^2 - ζeff3/2) / (ζeff3*ζeff3)
    df      = λf * (ζf + dζeff_ζ * dζf)
    return f, df
end

@inline function _B_kernel(λ, x_0, ζ_X)
    x3λ = x_0^(3-λ)
    ζX3 = (1-ζ_X)^3
    I   = (1 - x3λ) / (λ-3)
    J   = (1 - (λ-3)*x_0^(4-λ) + (λ-4)*x3λ) / ((λ-3)*(λ-4))
    return I*(1-ζ_X/2)/ζX3 - 9*J*ζ_X*(ζ_X+1)/(2*ζX3)
end

# Returns (B, d(ρS·B)/dρS)
@inline function _B_fdf_kernel(λ, x_0, ζ_X)
    x3λ = x_0^(3-λ)
    ζX2 = (1-ζ_X)^2;  ζX3 = ζX2*(1-ζ_X);  ζX6 = ζX3*ζX3
    I   = (1 - x3λ) / (λ-3)
    J   = (1 - (λ-3)*x_0^(4-λ) + (λ-4)*x3λ) / ((λ-3)*(λ-4))
    f   = I*(1-ζ_X/2)/ζX3 - 9*J*ζ_X*(ζ_X+1)/(2*ζX3)
    df  = f + ζ_X*(
              (3*(1-ζ_X/2)*ζX2 - ζX3/2)*I/ζX6
            - 9*J*((1+2*ζ_X)*ζX3 + ζ_X*(1+ζ_X)*3*ζX2)/(2*ζX6)
          )
    return f, df
end

@inline function _KHS_kernel(ζ_X)
    return (1-ζ_X)^4 / evalpoly(ζ_X, (1, 4, 4, -4, 1))
end

# Returns (KHS, dKHS/dρS)
@inline function _KHS_fdf_kernel(ρS, ζ_X)
    ζX4   = (1-ζ_X)^4
    denom  = evalpoly(ζ_X, (1, 4, 4, -4, 1))
    ddenom = evalpoly(ζ_X, (4, 8, -12, 4))
    f    = ζX4 / denom
    ρdf  = -ζ_X*(4*(1-ζ_X)^3*denom + ζX4*ddenom) / (denom*denom)
    return f, ρdf / ρS
end

@inline function _gHS_kernel(x_0, ζ_X)
    ζX3 = (1-ζ_X)^3
    k0 = -Base.log(1-ζ_X) + evalpoly(ζ_X,(0,42,-39,9,-2))/(6*ζX3)
    k1 = evalpoly(ζ_X,(0,-12,6,0,1)) / (2*ζX3)
    k2 = -3*ζ_X*ζ_X / (8*(1-ζ_X)^2)
    k3 = evalpoly(ζ_X,(0,3,3,0,-1)) / (6*ζX3)
    return exp(k0 + x_0*(k1 + x_0*(k2 + x_0*k3)))
end

# ϕ: 7 tuples of 6 Float64 — GPU-safe constant (same as SAFTVRMieconsts.ϕ)
const SAFTVRMIE_PHI = (
    ( 7.5365557, -359.440,  1550.9, -1.199320, -1911.2800,  9236.9),
    (-37.604630,  1825.60, -5070.1,  9.063632,  21390.175,-129430.0),
    ( 71.745953, -3168.00,  6534.6,-17.94820,  -51320.700, 357230.0),
    (-46.835520,  1884.20, -3288.7, 11.34027,   37064.540,-315530.0),
    ( -2.4679820,  -0.82376,-2.7171, 20.52142,   1103.7420,  1390.2),
    ( -0.5027200,  -3.19350, 2.0883,-56.63770,  -3264.6100, -4518.2),
    (  8.0956883,   3.70900, 0.0000, 40.53683,   2556.1810,  4241.6),
)

@inline function _f123456_kernel(α, ϕ)
    T   = typeof(α)
    fa1=zero(T);fa2=zero(T);fa3=zero(T);fa4=zero(T);fa5=zero(T);fa6=zero(T)
    fb1=zero(T);fb2=zero(T);fb3=zero(T);fb4=zero(T);fb5=zero(T);fb6=zero(T)
    αi = one(T)
    @unroll for i in 1:4
        p = ϕ[i]
        fa1+=p[1]*αi; fa2+=p[2]*αi; fa3+=p[3]*αi
        fa4+=p[4]*αi; fa5+=p[5]*αi; fa6+=p[6]*αi
        αi *= α
    end
    αi = α
    @unroll for i in 5:7
        p = ϕ[i]
        fb1+=p[1]*αi; fb2+=p[2]*αi; fb3+=p[3]*αi
        fb4+=p[4]*αi; fb5+=p[5]*αi; fb6+=p[6]*αi
        αi *= α
    end
    return (fa1/(1+fb1), fa2/(1+fb2), fa3/(1+fb3),
            fa4/(1+fb4), fa5/(1+fb5), fa6/(1+fb6))
end

"""
SAFT-VR Mie dispersion contribution at grid point `kk`.
- `meff[i]`: effective segment count (m*S for SAFTγMie, m otherwise)
- `IDX_ρz`: field index for the dispersion weighted density
  - 6+ND for SAFTVRMie and SAFTgammaMie
  - 4+ND for COFFEE
Used by SAFTVRMieModel, SAFTgammaMieModel, COFFEEModel.
"""
@inline function f_disp(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}, ::Val{IDX_ρz}) where {NC, ND, IDX_ρz, M}
    meff     = params.meff
    HSd      = params.HSd
    sigma    = params.sigma
    epsilon  = params.epsilon
    lambda_r = params.lambda_r_t
    lambda_a = params.lambda_a_t
    psi_eff  = params.psi_eff
    A        = params.A
    phi      = params.phi
    FP = eltype(n)
    ρS_d = zero(FP)
    @inbounds @unroll for i in 1:NC
        ρS_d += n[kk, IDX_ρz, i] * 3/(4*(π*psi_eff[i]^3)) * meff[i]
    end
    kρS_d = ρS_d * π/6/8

    ζ_Xd=zero(FP);  σ3_xd=zero(FP)
    @inbounds @unroll for i in 1:NC
        di   = HSd[i]
        ρ̄zi  = n[kk, IDX_ρz, i] * 3/(4*(π*psi_eff[i]^3))
        x_Si = ρ̄zi * meff[i] / ρS_d
        @inbounds @unroll for j in 1:NC
            dj   = HSd[j]
            ρ̄zj  = n[kk, IDX_ρz, j] * 3/(4*(π*psi_eff[j]^3))
            x_Sj = ρ̄zj * meff[j] / ρS_d
            σ3_xd += x_Si*x_Sj*sigma[i,j]^3
            ζ_Xd  += kρS_d*x_Si*x_Sj*(di+dj)^3
        end
    end
    ζstd  = σ3_xd * ρS_d * π/6
    ζst5d = ζstd^5;  ζst8d = ζstd^8
    KHSd  = _KHS_kernel(ζ_Xd)

    a₁=zero(FP);  a₂=zero(FP);  a₃=zero(FP)
    @inbounds @unroll for i in 1:NC
        di   = HSd[i]
        ρ̄zi  = n[kk, IDX_ρz, i] * 3/(4*(π*psi_eff[i]^3))
        x_Si = ρ̄zi * meff[i] / ρS_d
        λa=lambda_a[i][i]; λr=lambda_r[i][i]; σii=sigma[i,i]; ϵii=epsilon[i,i]
        _C=_Cλ_kernel(λa,λr);  x0=σii/di;  dij3=di^3
        aS1_a=_aS1_kernel(λa,   ζ_Xd,A); B_a=_B_kernel(λa,   x0,ζ_Xd)
        aS1_r=_aS1_kernel(λr,   ζ_Xd,A); B_r=_B_kernel(λr,   x0,ζ_Xd)
        a1ij = 2*(π*ϵii)*dij3*_C*ρS_d*(x0^λa*(aS1_a+B_a)-x0^λr*(aS1_r+B_r))
        aS1_2a=_aS1_kernel(2*λa, ζ_Xd,A); B_2a=_B_kernel(2*λa, x0,ζ_Xd)
        aS1_2r=_aS1_kernel(2*λr, ζ_Xd,A); B_2r=_B_kernel(2*λr, x0,ζ_Xd)
        aS1_ar=_aS1_kernel(λa+λr,ζ_Xd,A); B_ar=_B_kernel(λa+λr,x0,ζ_Xd)
        α=_C*(1/(λa-3)-1/(λr-3))
        f1,f2,f3,f4,f5,f6 = _f123456_kernel(α,phi)
        χ = f1*ζstd+f2*ζst5d+f3*ζst8d
        a2ij = π*KHSd*(1+χ)*ρS_d*ϵii^2*dij3*_C^2*(
               x0^(2*λa)*(aS1_2a+B_2a)
             - 2*x0^(λa+λr)*(aS1_ar+B_ar)
             + x0^(2*λr)*(aS1_2r+B_2r))
        a3ij = -ϵii^3*f4*ζstd*exp(f5*ζstd+f6*ζstd^2)
        a₁ += a1ij*x_Si*x_Si;  a₂ += a2ij*x_Si*x_Si;  a₃ += a3ij*x_Si*x_Si
        @inbounds @unroll for j in 1:NC
            if j != i
                dj    = HSd[j]
                ρ̄zj   = n[kk, IDX_ρz, j] * 3/(4*(π*psi_eff[j]^3))
                x_Sj  = ρ̄zj * meff[j] / ρS_d
                λa2=lambda_a[i][j]; λr2=lambda_r[i][j]; σij=sigma[i,j]; ϵij=epsilon[i,j]
                _C2=_Cλ_kernel(λa2,λr2); dij2=(di+dj)/2; dij3_2=dij2^3; x0ij=σij/dij2
                aS1_a2=_aS1_kernel(λa2,    ζ_Xd,A); B_a2=_B_kernel(λa2,    x0ij,ζ_Xd)
                aS1_r2=_aS1_kernel(λr2,    ζ_Xd,A); B_r2=_B_kernel(λr2,    x0ij,ζ_Xd)
                a1ij2 = 2*(π*ϵij)*dij3_2*_C2*ρS_d*(x0ij^λa2*(aS1_a2+B_a2)-x0ij^λr2*(aS1_r2+B_r2))
                aS1_2a2=_aS1_kernel(2*λa2, ζ_Xd,A); B_2a2=_B_kernel(2*λa2, x0ij,ζ_Xd)
                aS1_2r2=_aS1_kernel(2*λr2, ζ_Xd,A); B_2r2=_B_kernel(2*λr2, x0ij,ζ_Xd)
                aS1_ar2=_aS1_kernel(λa2+λr2,ζ_Xd,A); B_ar2=_B_kernel(λa2+λr2,x0ij,ζ_Xd)
                α2=_C2*(1/(λa2-3)-1/(λr2-3))
                f1_2,f2_2,f3_2,f4_2,f5_2,f6_2 = _f123456_kernel(α2,phi)
                χ2 = f1_2*ζstd+f2_2*ζst5d+f3_2*ζst8d
                a2ij2 = π*KHSd*(1+χ2)*ρS_d*ϵij^2*dij3_2*_C2^2*(
                        x0ij^(2*λa2)*(aS1_2a2+B_2a2)
                      - 2*x0ij^(λa2+λr2)*(aS1_ar2+B_ar2)
                      + x0ij^(2*λr2)*(aS1_2r2+B_2r2))
                a3ij2 = -ϵij^3*f4_2*ζstd*exp(f5_2*ζstd+f6_2*ζstd^2)
                a₁ += a1ij2*x_Si*x_Sj
                a₂ += a2ij2*x_Si*x_Sj
                a₃ += a3ij2*x_Si*x_Sj
            end
        end
    end
    return ρS_d * (a₁/T + a₂/(T*T) + a₃/(T*T*T))
end

# SAFTVRMie-specific association strength: Δ = expm1(ε_assoc/T) * κ * I(Tr, ρr)
# where I is an 11×11 polynomial in (Tr=T/ε_Mie, ρr=ρS*σ³_x).
# Overrides the default g_hs _assoc_delta via more specific M <: SAFTVRMieModel dispatch.
@inline function _assoc_delta(p, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix,
                               ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: SAFTVRMieModel}
    FP = eltype(n)
    p > n_pairs && return zero(FP)

    # Compute ρS from individual n₃ fields (field index 3 = F2+1)
    ρS = zero(FP)
    @inbounds @unroll for k in 1:NC
        ρS += n[kk, 3, k] * 6 / (π * params.HSd[k]^3) * params.m[k]
    end

    # σ³_x double loop: x_Sk = ρ̄k * m[k] / ρS
    σ3_x = zero(FP)
    @inbounds @unroll for k in 1:NC
        ρ̄k  = n[kk, 3, k] * 6 / (π * params.HSd[k]^3)
        xSk = ρ̄k * params.m[k] / ρS
        σ3_x += xSk * xSk * params.sigma[k,k]^3
        @inbounds @unroll for l in 1:(k-1)
            ρ̄l  = n[kk, 3, l] * 6 / (π * params.HSd[l]^3)
            xSl = ρ̄l * params.m[l] / ρS
            σ3_x += 2 * xSk * xSl * params.sigma[k,l]^3
        end
    end
    ρr = ρS * σ3_x

    ic = _nti(params.assoc_icomp, p)
    jc = _nti(params.assoc_jcomp, p)
    Tr = T / params.epsilon[ic, jc]

    # I(Tr, ρr): c stored as NTuple{11, NTuple{11, FP}} (row = n-index, col = m-index)
    I_val = zero(FP)
    ρrn   = one(FP)
    @unroll for ni in 0:10
        row = _nti(params.VRMie_c, ni + 1)
        Trm = one(FP)
        @unroll for mi in 0:10
            I_val += _nti(row, mi + 1) * Trm * ρrn
            Trm *= Tr
        end
        ρrn *= ρr
    end

    return (exp(params.assoc_eps[p] / T)-1) * params.assoc_kap[p] * I_val
end

function preallocate_params(system::DFTSystem{<:SAFTVRMieModel})
    backend = system.options.device
    FP      = fptype(system.options)
    model   = system.model
    nc = length(model)
    lr = model.params.lambda_r.values
    la = model.params.lambda_a.values
    lambda_r_t = ntuple(i -> ntuple(j -> FP(lr[i,j]), nc), nc)
    lambda_a_t = ntuple(i -> ntuple(j -> FP(la[i,j]), nc), nc)
    A_fp  = ntuple(i -> ntuple(j -> FP(SAFTVRMIE_A[i][j]),   4), 4)
    phi_fp = ntuple(i -> ntuple(j -> FP(SAFTVRMIE_PHI[i][j]), 6), 7)

    # Reduced units: divide every length-dimensioned parameter by L so it matches the
    # `get_fields`-side kernel rescaling. `psi_eff` (system.fields[end].width) needs no
    # separate treatment — it already reads the rescaled width back from `get_fields`.
    # See PCSAFT.jl's `get_fields`/`preallocate_params` docstrings for the full picture.
    L           = length_scale(model)
    HSd_local   = system.species.size ./ L
    sigma_local = model.params.sigma.values ./ L

    base = (;
        HSd        = adapt_to_device(backend, FP, HSd_local),
        m          = adapt_to_device(backend, FP, model.params.segment.values),
        meff       = adapt_to_device(backend, FP, model.params.segment.values),
        sigma      = adapt_to_device(backend, FP, sigma_local),
        epsilon    = adapt_to_device(backend, FP, model.params.epsilon.values),
        lambda_r_t = lambda_r_t,
        lambda_a_t = lambda_a_t,
        psi_eff    = adapt_to_device(backend, FP, system.fields[end].width),
        A          = A_fp,
        phi        = phi_fp,
    )

    nn = Clapeyron.assoc_pair_length(model)
    if nn > 0
        (assoc_icomp_v, assoc_jcomp_v, assoc_isite_v, assoc_jsite_v,
         assoc_eps_v, assoc_kap_v, assoc_sig3_v, assoc_dij_v,
         n_sites_flat_v, n_sites_cumsum_v, total_sites
        ) = pack_assoc_params(model, HSd_local, sigma_local)

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

        # Pack I(Tr,ρr) polynomial as NTuple{11,NTuple{11,FP}}: row=n-index, col=m-index
        c_mat   = SAFTVRMieconsts.c
        VRMie_c = ntuple(ni -> ntuple(mi -> FP(c_mat[ni, mi]), 11), 11)

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
            assoc_kap       = adapt_to_device(backend, FP, assoc_kap_v ./ L^3),
            assoc_sig3      = adapt_to_device(backend, FP, assoc_sig3_v ./ L^3),
            assoc_dij       = adapt_to_device(backend, FP, assoc_dij_v),
            n_sites_flat    = n_sites_flat_t,
            n_sites_cumsum  = n_sites_cumsum_t,
            total_sites,
            VRMie_c,
        )
        params = merge(base, assoc)
    else
        params = merge(base, (; has_assoc = false))
    end

    return params, length(model)
end
