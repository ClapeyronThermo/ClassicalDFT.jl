import Clapeyron: COFFEEModel, Iμμ, ∫∫∫Odξ₁dξ₂dγ12, ∫∫∫dξ₁dξ₂dγ12, ∫odr, COFFEEconsts
import Clapeyron: pcp_segment, pcp_sigma, pcp_epsilon, pcp_dipole2

"""
    COFFEE(components::Vector{String})

The COFFEE equation of state developed by Langenbach (2017). This is an unpublished approach which uses a Weighted Density Functional approach and does not use a chain propagator.

The bulk model can be obtained from Clapeyron. 
"""
COFFEE

function get_fields(model::COFFEEModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}) where FP<:AbstractFloat
    nc = length(model)
    ψ = 1.3862
    ngrid = structure.ngrid

    # ψ1 is a dimensionless Barker-Henderson-style shape factor derived from the RAW
    # (unscaled) diameter/sigma ratio — it must stay L-invariant, so compute it before
    # any reduced-units rescaling is applied below (ψ, unlike ψ1, is already a fixed
    # constant). See PCSAFT.jl's `get_fields` docstring for the overall reduced-units
    # scheme.
    λ_r = diagvalues(model.params.lambda_r.values)
    λ_a = diagvalues(model.params.lambda_a.values)
    σ   = diagvalues(model.params.sigma.values)
    C = @. λ_r / (λ_r - λ_a) * (λ_r / λ_a)^(λ_a / (λ_r - λ_a))
    x = species.size ./ σ
    ψ1 = @. cbrt(3*C*x^3*(x^-λ_a/(λ_a-3)-x^-λ_r/(λ_r-3)))

    L = length_scale(model)
    ω = structure_ω(structure, device, FP)
    d = species.size ./ L

    return (SWeightedDensity(:∫ρdz,0.5*d,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,0.5*d,ω,ngrid,device,model),
            VWeightedDensity(:∫ρzdz,0.5*d,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,ψ*d,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,ψ1.*d,ω,ngrid,device,model))
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: COFFEEModel}
    res_hs, n₀, n₂, n₃₃, nv2_1, nv2_2, nv2_3 =
        f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(1))
    res_disp = f_disp(M, kk, n, params, T, Val(NC), Val(ND), Val(4+ND))
    res_ff = f_ff(M, kk, n, params, T, Val(NC), Val(ND))
    res_nf = f_nf(M, kk, n, params, T, n₀, n₂, n₃₃, nv2_1, nv2_2, nv2_3, Val(NC), Val(ND))
    out[kk] = res_hs + res_disp + res_ff + res_nf
    return nothing
end

"""
Pointwise residual free energy for COFFEE:
FMT hard-sphere + SAFT-VR Mie dispersion + far-field polar (f_ff) + near-field polar (f_nf).
No chain (IdealPropagator).

Field layout (5 fields):
  1        : ∫ρdz  with 0.5*d → n₀, n₁, n₂ (FMT + f_nf)
  2        : ∫ρz²dz with 0.5*d → n₃ (FMT + f_nf η)
  3..2+ND  : ∫ρzdz with 0.5*d → nᵥ (FMT + f_nf)
  3+ND     : ∫ρz²dz with ψ*d (ψ=1.3862) → far-field polar density
  4+ND     : ∫ρz²dz with ψ1*d → dispersion density (per-component ψ1)
"""
@inline function f_ff(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: COFFEEModel}
    FP    = eltype(n)
    # Bare `π` (an Irrational) silently promotes to Float64 in many combinations that
    # don't touch an FP value first — e.g. `π/6`, `-π*x`, and `-4*π*π/3` are ALL Float64
    # (verified directly; even `-π*x` for x::Float32, despite `π*x` alone staying
    # Float32). Binding `π = FP(π)` once and using it everywhere below avoids having to
    # reason about which specific orderings are safe.
    HSd   = params.HSd
    m_seg = params.m
    pcp_ϵ = params.pcp_epsilon
    pcp_σ = params.pcp_sigma
    dip2  = params.dipole2
    ψff   = FP(1.3862)
    idx_ff = 3 + ND
    cb2   = params.coffee_b2;  cc2 = params.coffee_c2
    cb3   = params.coffee_b3;  cc3 = params.coffee_c3

    ∑ρ̄_ff = zero(FP);  η_ff = zero(FP)
    @inbounds @unroll for i in 1:NC
        di    = HSd[i]
        ρ̄i_ff = n[kk,idx_ff,i] * 3/(4*ψff^3*π*di*di*di)
        ∑ρ̄_ff += ρ̄i_ff
        η_ff  += m_seg[i] * ρ̄i_ff * di*di*di
    end
    η_ff *= π/6

    has_polar = false
    @inbounds @unroll for i in 1:NC
        if !iszero(dip2[i]); has_polar = true; break; end
    end

    res_polar = zero(FP)
    if has_polar
        _A₂ = zero(FP)
        @inbounds @unroll for i in 1:NC
            d2i = dip2[i]; if iszero(d2i); continue; end
            ρ̄i = n[kk,idx_ff,i] * 3/(4*ψff^3*π*HSd[i]^3)
            xi = ρ̄i / ∑ρ̄_ff
            J2ii = _J2_coffee_kernel(pcp_ϵ[i,i], η_ff, T, cb2, cc2)
            _A₂ += xi*xi * d2i*d2i / (pcp_σ[i,i]*pcp_σ[i,i]*pcp_σ[i,i]) * J2ii
            @inbounds @unroll for j in (i+1):NC
                d2j = dip2[j]; if iszero(d2j); continue; end
                ρ̄j = n[kk,idx_ff,j] * 3/(4*ψff^3*π*HSd[j]^3)
                xj = ρ̄j / ∑ρ̄_ff
                σij3 = pcp_σ[i,j]*pcp_σ[i,j]*pcp_σ[i,j]
                J2ij = _J2_coffee_kernel(pcp_ϵ[i,j], η_ff, T, cb2, cc2)
                _A₂ += 2*xi*xj * d2i*d2j / σij3 * J2ij
            end
        end
        _A₂ *= -π * ∑ρ̄_ff / (T*T)

        if abs(_A₂) > 0
            _A₃ = zero(FP)
            @inbounds @unroll for i in 1:NC
                d2i = dip2[i]; if iszero(d2i); continue; end
                ρ̄i = n[kk,idx_ff,i] * 3/(4*ψff^3*π*HSd[i]^3)
                xi  = ρ̄i / ∑ρ̄_ff
                a3i = xi * d2i / pcp_σ[i,i]
                J3iii = _J3_coffee_kernel(pcp_ϵ[i,i], η_ff, T, cb3, cc3)
                _A₃ += a3i*a3i*a3i * J3iii
                @inbounds @unroll for j in (i+1):NC
                    d2j = dip2[j]; if iszero(d2j); continue; end
                    ρ̄j = n[kk,idx_ff,j] * 3/(4*ψff^3*π*HSd[j]^3)
                    xj   = ρ̄j / ∑ρ̄_ff
                    σij⁻¹ = 1 / pcp_σ[i,j]
                    a3iij = xi*d2i*σij⁻¹;  a3ijj = xj*d2j*σij⁻¹
                    a3j   = xj*d2j/pcp_σ[j,j]
                    J3iij = _J3_coffee_kernel(pcp_ϵ[i,j], η_ff, T, cb3, cc3)
                    J3ijj = _J3_coffee_kernel(pcp_ϵ[i,j], η_ff, T, cb3, cc3)
                    _A₃ += 3*a3iij*a3ijj*(a3i*J3iij + a3j*J3ijj)
                    @inbounds @unroll for kk2 in (j+1):NC
                        d2k = dip2[kk2]; if iszero(d2k); continue; end
                        ρ̄k = n[kk,idx_ff,kk2] * 3/(4*ψff^3*π*HSd[kk2]^3)
                        xk  = ρ̄k / ∑ρ̄_ff
                        J3ijk = _J3_coffee_kernel(pcp_ϵ[i,j], η_ff, T, cb3, cc3)
                        _A₃ += 6*xi*xj*xk * d2i*d2j*d2k *
                                σij⁻¹/(pcp_σ[i,kk2]*pcp_σ[j,kk2]) * J3ijk
                    end
                end
            end
            _A₃ *= -4/3 * ∑ρ̄_ff*∑ρ̄_ff*π*π / (T*T*T)
            denom_p = _A₂ - _A₃
            res_polar = ∑ρ̄_ff * _A₂*_A₂ / denom_p
        end
    end
    return res_polar
end

# GPU-safe inline helpers for COFFEE polar terms

@inline function _J2_coffee_kernel(ϵij, η, T, b2, c2)
    eT = ϵij / T
    c0 = b2[1] + c2[1]*eT;  c1 = b2[2] + c2[2]*eT
    c2v= b2[3] + c2[3]*eT;  c3 = b2[4] + c2[4]*eT
    c4 = b2[5] + c2[5]*eT
    return evalpoly(η, (c0, c1, c2v, c3, c4))
end

@inline function _J3_coffee_kernel(ϵij, η, T, b3, c3)
    eT = ϵij / T
    c0 = b3[1] + c3[1]*eT;  c1 = b3[2] + c3[2]*eT
    c2v= b3[3] + c3[3]*eT;  c3v= b3[4] + c3[4]*eT
    return evalpoly(η, (c0, c1, c2v, c3v))
end

@inline function f_nf(::Type{M}, kk, n, params, T, n₀, n₂, n₃₃, nv2_1, nv2_2, nv2_3, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: COFFEEModel}
    HSd       = params.HSd
    nf_d      = params.nf_d
    nf_mu2    = params.nf_mu2
    nf_mu     = sqrt(abs(nf_mu2))
    nf_eps    = params.nf_epsilon
    nf_sigrat = params.nf_sigma_ratio
    corr_I    = params.coffee_corr_I
    xi_x      = params.xi_x;  xi_w = params.xi_w
    gamma_x   = params.gamma_x;  gamma_w = params.gamma_w
    FP        = typeof(n₀)

    ρ̄_nf   = n₃₃ * 6/π * nf_sigrat
    T̄_nf   = T / nf_eps
    Iμμval = _Iμμ_kernel(ρ̄_nf, T̄_nf, nf_mu, corr_I)

    coeff = FP(-24/19) * nf_mu2 / T̄_nf * Iμμval
    Q_nf = zero(FP)
    @inbounds for i in 1:25
        ξ1i = xi_x[i]; w1i = xi_w[i]
        @inbounds for j in 1:25
            ξ2j = xi_x[j]; w2j = xi_w[j]
            @inbounds for kk3 in 1:20
                γk  = gamma_x[kk3]; wγk = gamma_w[kk3]
                Iv  = _∫odr_kernel(nf_d, ξ1i, ξ2j, γk)
                Q_nf += w1i * w2j * wγk * exp(coeff * Iv)
            end
        end
    end

    nv2sq_nf = nv2_1*nv2_1 + nv2_2*nv2_2 + nv2_3*nv2_3
    di1  = HSd[1]
    ξ_nf = 1 - nv2sq_nf / (n₂*n₂)
    g_hs_nf = 1/(1-n₃₃) + di1*ξ_nf*n₂/(4*(1-n₃₃)^2) +
              di1*di1/4*n₂*n₂*ξ_nf/(18*(1-n₃₃)^3)

    return FP(19π/12) * n₀ * ρ̄_nf * g_hs_nf * Base.log(π/Q_nf*4)
end

@inline function _Iμμ_kernel(ρ̄, T̄, μ, a)
    return a[1] + a[2]/T̄ + a[3]/(T̄*T̄) + ρ̄*(
        a[4] + a[5]*μ + a[6]/T̄ + a[7]/(T̄*T̄) + a[8]*ρ̄ + a[9]*ρ̄*ρ̄ + a[10]*ρ̄*μ/T̄)
end

@inline function _∫odr_kernel(d, ξ1, ξ2, γ12)
    if iszero(d)
        s1 = sqrt(abs(1 - ξ1*ξ1));  s2 = sqrt(abs(1 - ξ2*ξ2))
        cosΘ = 2*ξ1*ξ2 - s1*s2*cos(γ12)
        return -cosΘ * typeof(ξ1)(5/18)
    else
        s1 = sqrt(abs(1 - ξ1*ξ1));  s2 = sqrt(abs(1 - ξ2*ξ2))
        A12 = ξ1*ξ2 + s1*s2*cos(γ12)
        a_  = 3*ξ1*ξ2 - A12
        b_  = d*(ξ1 - ξ2)*(A12 - 3)
        c_  = -d*d*(A12*A12 - 4*A12 + 3)
        f_  = 2*d*(ξ1 - ξ2)
        f2  = f_*f_;  f3 = f2*f_
        g_  = -2*d*d*(A12 - 1)
        g2  = g_*g_
        inv_3f24g2 = 1 / (3*(f2 - 4*g_)*(f2 - 4*g_))
        p1         = 1 + f_ + g_; p1h = p1 * sqrt(p1)
        p2         = 9 + 6*f_ + 4*g_; p2h = p2 * sqrt(p2)
        term1 = (1/p1h) * (
                2*c_*(2+f_)*(-8-8*f_+f2-12*g_)
              + 2*b_*(3*f3+8*g2+2*f2*(6+g_)+4*f_*(2+3*g_))
              - 2*a_*(3*f3+8*g_+4*f_*g_*(3+2*g_)+2*f2*(1+6*g_)))
        term2 = (1/p2h) * 4 * (
              - 4*c_*(3+f_)*(-12*f_+f2-6*(3+2*g_))
              - 2*b_*(9*f3+16*g2+18*f_*(3+2*g_)+f2*(54+4*g_))
              + a_*(27*f3+108*g_+9*f2*(3+8*g_)+4*f_*g_*(27+8*g_)))
        return -inv_3f24g2 * (term1 + term2)
    end
end

function preallocate_params(system::DFTSystem{<:COFFEEModel})
    backend = system.options.device
    FP      = fptype(system.options)
    model   = system.model
    σ_diag  = diagvalues(model.params.sigma.values)
    ϵ_diag  = diagvalues(model.params.epsilon.values)
    # nf_mu2_val/nf_d_val/nf_sig3 are dimensionless ratios of raw (unscaled) lengths —
    # they must keep referencing σ_diag/system.species.size directly, NOT the reduced
    # HSd_local/sigma_local below, or they'd stop being L-invariant.
    nf_mu2_val = pcp_dipole2(model)[1] / ϵ_diag[1] / σ_diag[1]^3
    nf_d_val   = model.params.shift[1] / σ_diag[1]
    nf_sig3    = (σ_diag[1] / system.species.size[1])^3

    nc = length(model)
    lr = model.params.lambda_r.values
    la = model.params.lambda_a.values
    lambda_r_t = ntuple(i -> ntuple(j -> FP(lr[i,j]), nc), nc)
    lambda_a_t = ntuple(i -> ntuple(j -> FP(la[i,j]), nc), nc)
    A_fp   = ntuple(i -> ntuple(j -> FP(SAFTVRMIE_A[i][j]),   4), 4)
    phi_fp = ntuple(i -> ntuple(j -> FP(SAFTVRMIE_PHI[i][j]), 6), 7)
    _conv_tup(tup) = map(FP, tup)

    # Reduced units: divide every length-dimensioned parameter fed directly into f_res
    # (as opposed to through a weighted-density kernel) by L. See PCSAFT.jl's
    # `get_fields`/`preallocate_params` docstrings for the full picture.
    L               = length_scale(model)
    HSd_local       = system.species.size ./ L
    sigma_local     = model.params.sigma.values ./ L
    pcp_sigma_local = pcp_sigma(model) ./ L

    params = (;
        HSd         = adapt_to_device(backend, FP, HSd_local),
        m           = adapt_to_device(backend, FP, model.params.segment.values),
        meff        = adapt_to_device(backend, FP, model.params.segment.values),
        sigma       = adapt_to_device(backend, FP, sigma_local),
        epsilon     = adapt_to_device(backend, FP, model.params.epsilon.values),
        lambda_r_t  = lambda_r_t,
        lambda_a_t  = lambda_a_t,
        psi_eff     = adapt_to_device(backend, FP, system.fields[end].width),
        A           = A_fp,
        phi         = phi_fp,
        pcp_m       = adapt_to_device(backend, FP, pcp_segment(model)),
        pcp_sigma   = adapt_to_device(backend, FP, pcp_sigma_local),
        pcp_epsilon = adapt_to_device(backend, FP, pcp_epsilon(model)),
        # dip2_i*dip2_j is combined directly with reduced pcp_σ³ in f_ff (unlike
        # nf_mu2_val above, which is a self-contained raw ratio) — needs the matching
        # /L^3, mirroring PPCSAFT.jl's f_polar fix.
        dipole2     = adapt_to_device(backend, FP, pcp_dipole2(model) ./ L^3),
        coffee_b2   = _conv_tup(COFFEEconsts.corr_b2),
        coffee_c2   = _conv_tup(COFFEEconsts.corr_c2),
        coffee_b3   = _conv_tup(COFFEEconsts.corr_b3),
        coffee_c3   = _conv_tup(COFFEEconsts.corr_c3),
        coffee_corr_I = _conv_tup(COFFEEconsts.corr_I),
        nf_d        = FP(nf_d_val),
        nf_mu2      = FP(nf_mu2_val),
        nf_epsilon  = FP(ϵ_diag[1]),
        nf_sigma_ratio = FP(nf_sig3),
        xi_x        = _conv_tup(COFFEEconsts.xi_quadrature[1]),
        xi_w        = _conv_tup(COFFEEconsts.xi_quadrature[2]),
        gamma_x     = _conv_tup(COFFEEconsts.gamma_quadrature[1]),
        gamma_w     = _conv_tup(COFFEEconsts.gamma_quadrature[2]),
    )
    nc = length(model)
    return params, nc
end