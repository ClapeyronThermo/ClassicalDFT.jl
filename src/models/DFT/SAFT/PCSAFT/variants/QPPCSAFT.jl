using Clapeyron: QPCPSAFTModel, pcp_sigma, pcp_dipole2, pcp_epsilon, pcp_segment

"""
    QPCPSAFT(components::Vector{String})

The quadrupolar polar PC-SAFT (QPCP-SAFT) equation of state, which extends `PCPSAFT` with quadrupole-quadrupole and dipole-quadrupole polar terms in addition to the dipole-dipole term. As with `PCSAFT`, our DFT implementation uses a Weighted Density Functional approach and does not use a chain propagator.

The bulk model can be obtained from Clapeyron.
"""
QPCPSAFT

const QQ_consts = (
    corr_a = 
    ((1.2378308, 1.2854109,	1.7942954),
    (2.4355031,	-11.465615,	0.7695103),
    (1.6330905,	22.086893,	7.2647923),
    (-1.6118152, 7.4691383,	94.486699),
    (6.9771185,	-17.197772,	-77.148458)),

    corr_b = 
    ((0.4542718, -0.813734, 6.8682675),
    (-4.5016264, 10.06403, -5.1732238),
    (3.5858868,	-10.876631, -17.240207),
    (0., 0., 0.),
    (0., 0., 0.)),

    corr_c =
    ((-0.5000437, 2.0002094, 3.1358271),
    (6.5318692, -6.7838658, 7.2475888),
    (-16.01478, 20.383246, 3.0759478),
    (14.42597, -10.895984, 0.),
    (0., 0., 0.))    
)

const DQ_consts = (
    corr_a =
    ((0.697095, -0.6734593, 0.6703408),
    (-0.6335541, -1.4258991, -4.3384718),
    (2.945509, 4.1944139, 7.2341684),
    (-1.4670273, 1.0266216, 0.)),

    corr_b =
    ((-0.4840383, 0.6765101, -1.1675601),
    (1.9704055, -3.0138675, 2.1348843),
    (-2.1185727, 0.4674266, 0.),
    (0.,	0.,	0.)),

    corr_c =
    ((0.7846431, -2.072202),
    (3.3427, -5.863904),
    (0.4689111, -0.1764887),
    (0., 0.))
)

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
Pointwise residual free energy for QPCP-SAFT: HS + HC + disp + polar (DD + QQ + DQ).

Field layout identical to PCP-SAFT / PC-SAFT:
  1        : ρ (unweighted)
  2        : ∫ρdz  with 0.5*d → n₀, n₁, n₂
  3        : ∫ρz²dz with 0.5*d → n₃
  4..3+ND  : ∫ρzdz with 0.5*d → vector nᵥ
  4+ND     : ∫ρz²dz with d    → ρ̄hc
  5+ND     : ∫ρdz  with d    → λ
  6+ND     : ∫ρz²dz with d*ψ → ρ̄z  (disp + polar)
"""
@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: QPCPSAFTModel}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_hc  = f_hc(M, kk, n, params, T, Val(NC), Val(ND))
    res_disp, m̄, ηd = f_disp(M, kk, n, params, T, Val(NC), Val(ND))
    res_polar = f_polar(M, kk, n, params, T, m̄, ηd, Val(NC), Val(ND))
    res_assoc = _assoc_or_zero(M, kk, n, params, T, Val(NC), Val(ND))
    out[kk] = res_hs + res_hc + res_disp + res_polar + res_assoc
    return nothing
end

"""
QPCP-SAFT polar term at grid point `kk`: Padé sum of DD + QQ + DQ contributions.
"""
@inline function f_polar(::Type{M}, kk, n, params, T, m̄, ηd, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: QPCPSAFTModel}
    FP    = eltype(n)
    pcp_m = params.pcp_m
    pcp_ϵ = params.pcp_epsilon
    pcp_σ = params.pcp_sigma
    dip2  = params.dipole2
    quad2 = params.quadrupole2
    ca_dd = params.dd_a;  cb_dd = params.dd_b;  cc_dd = params.dd_c
    ca_qq = params.qq_a;  cb_qq = params.qq_b;  cc_qq = params.qq_c
    ca_dq = params.dq_a;  cb_dq = params.dq_b;  cc_dq = params.dq_c

    # `-π*x` (Int/Irrational combos before touching an FP value) promotes to Float64,
    # unlike `x*π`/`π*x` for x::FP — see PCSAFT.jl's f_disp. Note `FP(4)*π`/`FP(9)*π`-
    # style patterns elsewhere in this function are already safe (FP touches π first).
    _π     = FP(π)
    ψ      = FP(1.3862)
    idx_ρz = 6 + ND
    factor = 3 / (4*ψ*ψ*ψ*_π)
    ∑ρ̄_p  = zero(FP)
    @inbounds @unroll for i in 1:NC
        ∑ρ̄_p += n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
    end

    has_dipole = false;  has_quad = false
    @inbounds @unroll for i in 1:NC
        if !iszero(dip2[i]);  has_dipole = true; end
        if !iszero(quad2[i]); has_quad   = true; end
    end

    res_polar = zero(FP)

    # ── DD contribution ───────────────────────────────────────────────────────
    if has_dipole
        _A₂_dd = zero(FP)
        @inbounds @unroll for i in 1:NC
            dip2_i = dip2[i]
            if iszero(dip2_i); continue; end
            ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
            xᵢ = ρ̄zi_i / ∑ρ̄_p
            @inbounds @unroll for j in 1:NC
                dip2_j = dip2[j]
                if iszero(dip2_j); continue; end
                ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                xⱼ = ρ̄zi_j / ∑ρ̄_p
                σij3 = pcp_σ[i,j]*pcp_σ[i,j]*pcp_σ[i,j]
                _J2_ij = _J2_kernel(pcp_m[i], pcp_m[j], pcp_ϵ[i,j], ηd, T, ca_dd, cb_dd)
                _A₂_dd += xᵢ * xⱼ * dip2_i * dip2_j / σij3 * _J2_ij
            end
        end
        _A₂_dd *= -_π * ∑ρ̄_p / (T*T)

        if !iszero(_A₂_dd)
            _A₃_dd = zero(FP)
            @inbounds @unroll for i in 1:NC
                dip2_i = dip2[i]
                if iszero(dip2_i); continue; end
                ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
                xᵢ = ρ̄zi_i / ∑ρ̄_p
                @inbounds @unroll for j in 1:NC
                    dip2_j = dip2[j]
                    if iszero(dip2_j); continue; end
                    ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                    xⱼ = ρ̄zi_j / ∑ρ̄_p
                    @inbounds @unroll for k in 1:NC
                        dip2_k = dip2[k]
                        if iszero(dip2_k); continue; end
                        ρ̄zi_k = n[kk, idx_ρz, k] * factor / (params.HSd[k]*params.HSd[k]*params.HSd[k])
                        xₖ = ρ̄zi_k / ∑ρ̄_p
                        _J3_ijk = _J3_kernel(pcp_m[i], pcp_m[j], pcp_m[k], ηd, cc_dd)
                        _A₃_dd += xᵢ*xⱼ*xₖ * dip2_i*dip2_j*dip2_k /
                                  (pcp_σ[i,j]*pcp_σ[i,k]*pcp_σ[j,k]) * _J3_ijk
                    end
                end
            end
            _A₃_dd *= -FP(4)*π*π/3 * ∑ρ̄_p*∑ρ̄_p / (T*T*T)
            denom_dd = _A₂_dd - _A₃_dd
            res_polar += ∑ρ̄_p * _A₂_dd*_A₂_dd / denom_dd
        end
    end

    # ── QQ contribution ───────────────────────────────────────────────────────
    if has_quad
        _A₂_qq = zero(FP)
        @inbounds @unroll for i in 1:NC
            quad2_i = quad2[i]
            if iszero(quad2_i); continue; end
            ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
            xᵢ = ρ̄zi_i / ∑ρ̄_p
            @inbounds @unroll for j in 1:NC
                quad2_j = quad2[j]
                if iszero(quad2_j); continue; end
                ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                xⱼ = ρ̄zi_j / ∑ρ̄_p
                σij_sq = pcp_σ[i,j]*pcp_σ[i,j]
                σij7   = σij_sq*σij_sq*σij_sq*pcp_σ[i,j]
                _J2_ij = _J2_qq_kernel(pcp_m[i], pcp_m[j], pcp_ϵ[i,j], ηd, T, ca_qq, cb_qq)
                _A₂_qq += xᵢ * xⱼ * quad2_i * quad2_j / σij7 * _J2_ij
            end
        end
        _A₂_qq *= -(FP(9)/16) * π * ∑ρ̄_p / (T*T)

        if !iszero(_A₂_qq)
            _A₃_qq = zero(FP)
            @inbounds @unroll for i in 1:NC
                quad2_i = quad2[i]
                if iszero(quad2_i); continue; end
                ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
                xᵢ = ρ̄zi_i / ∑ρ̄_p
                @inbounds @unroll for j in 1:NC
                    quad2_j = quad2[j]
                    if iszero(quad2_j); continue; end
                    ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                    xⱼ = ρ̄zi_j / ∑ρ̄_p
                    σij3 = pcp_σ[i,j]*pcp_σ[i,j]*pcp_σ[i,j]
                    @inbounds @unroll for k in 1:NC
                        quad2_k = quad2[k]
                        if iszero(quad2_k); continue; end
                        ρ̄zi_k = n[kk, idx_ρz, k] * factor / (params.HSd[k]*params.HSd[k]*params.HSd[k])
                        xₖ = ρ̄zi_k / ∑ρ̄_p
                        σik3 = pcp_σ[i,k]*pcp_σ[i,k]*pcp_σ[i,k]
                        σjk3 = pcp_σ[j,k]*pcp_σ[j,k]*pcp_σ[j,k]
                        _J3_ijk = _J3_qq_kernel(pcp_m[i], pcp_m[j], pcp_m[k], ηd, cc_qq)
                        _A₃_qq += xᵢ*xⱼ*xₖ * quad2_i*quad2_j*quad2_k /
                                  (σij3*σik3*σjk3) * _J3_ijk
                    end
                end
            end
            _A₃_qq *= FP(9)*π*π/16 * ∑ρ̄_p*∑ρ̄_p / (T*T*T)
            denom_qq = _A₂_qq - _A₃_qq
            res_polar += ∑ρ̄_p * _A₂_qq*_A₂_qq / denom_qq
        end
    end

    # ── DQ cross contribution ─────────────────────────────────────────────────
    if has_dipole && has_quad
        _A₂_dq = zero(FP)
        @inbounds @unroll for i in 1:NC
            dip2_i = dip2[i]
            if iszero(dip2_i); continue; end
            ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
            xᵢ = ρ̄zi_i / ∑ρ̄_p
            @inbounds @unroll for j in 1:NC
                quad2_j = quad2[j]
                if iszero(quad2_j); continue; end
                ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                xⱼ = ρ̄zi_j / ∑ρ̄_p
                σij_sq = pcp_σ[i,j]*pcp_σ[i,j]
                σij5   = σij_sq*σij_sq*pcp_σ[i,j]
                _J2_ij = _J2_dq_kernel(pcp_m[i], pcp_m[j], pcp_ϵ[i,j], ηd, T, ca_dq, cb_dq)
                _A₂_dq += xᵢ * xⱼ * dip2_i * quad2_j / σij5 * _J2_ij
            end
        end
        _A₂_dq *= -(FP(9)/4) * π * ∑ρ̄_p / (T*T)

        if !iszero(_A₂_dq)
            _A₃_dq = zero(FP)
            @inbounds @unroll for i in 1:NC
                dip2_i = dip2[i]
                if iszero(dip2_i); continue; end
                ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
                xᵢ = ρ̄zi_i / ∑ρ̄_p
                @inbounds @unroll for j in 1:NC
                    if iszero(dip2[j]) && iszero(quad2[j]); continue; end
                    ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                    xⱼ = ρ̄zi_j / ∑ρ̄_p
                    contrib_j = pcp_σ[j,j] * dip2[j] + FP(1.19374) / pcp_σ[j,j] * quad2[j]
                    @inbounds @unroll for k in 1:NC
                        quad2_k = quad2[k]
                        if iszero(quad2_k); continue; end
                        ρ̄zi_k = n[kk, idx_ρz, k] * factor / (params.HSd[k]*params.HSd[k]*params.HSd[k])
                        xₖ = ρ̄zi_k / ∑ρ̄_p
                        σij2 = pcp_σ[i,j]*pcp_σ[i,j]
                        σik2 = pcp_σ[i,k]*pcp_σ[i,k]
                        σjk2 = pcp_σ[j,k]*pcp_σ[j,k]
                        _J3_ijk = _J3_dq_kernel(pcp_m[i], pcp_m[j], pcp_m[k], ηd, cc_dq)
                        _A₃_dq += xᵢ * xⱼ * xₖ * pcp_σ[i,i] /
                                    (pcp_σ[k,k] * σij2*σik2*σjk2) *
                                    dip2_i * quad2_k * contrib_j * _J3_ijk
                    end
                end
            end
            _A₃_dq *= -∑ρ̄_p*∑ρ̄_p / (T*T*T)
            denom_dq = _A₂_dq - _A₃_dq
            res_polar += ∑ρ̄_p * _A₂_dq*_A₂_dq / denom_dq
        end
    end

    return res_polar
end

"""
QQ J₂ integral kernel: no min(m̄,2) clamping (unlike DD), 5 terms.
"""
@inline function _J2_qq_kernel(mᵢ, mⱼ, ϵᵢⱼ, η, T, corr_a, corr_b)
    FP     = typeof(η)
    ϵT     = ϵᵢⱼ / T
    m̄      = sqrt(mᵢ * mⱼ)
    m1     = 1 - 1/m̄
    m2     = m1 * (1 - 2/m̄)
    result = zero(FP)
    ηn     = one(FP)
    @unroll for n in 0:4
        a0, a1, a2 = corr_a[n+1]
        b0, b1, b2 = corr_b[n+1]
        result += (a0 + a1*m1 + a2*m2 + (b0 + b1*m1 + b2*m2)*ϵT) * ηn
        ηn *= η
    end
    return result
end

"""
DQ J₂ integral kernel: no min clamping, 4 terms only (DQ_consts has 4 entries).
"""
@inline function _J2_dq_kernel(mᵢ, mⱼ, ϵᵢⱼ, η, T, corr_a, corr_b)
    FP     = typeof(η)
    ϵT     = ϵᵢⱼ / T
    m̄      = sqrt(mᵢ * mⱼ)
    m1     = 1 - 1/m̄
    m2     = m1 * (1 - 2/m̄)
    result = zero(FP)
    ηn     = one(FP)
    @unroll for n in 0:3
        a0, a1, a2 = corr_a[n+1]
        b0, b1, b2 = corr_b[n+1]
        result += (a0 + a1*m1 + a2*m2 + (b0 + b1*m1 + b2*m2)*ϵT) * ηn
        ηn *= η
    end
    return result
end

"""
QQ J₃ integral kernel: no min clamping, 5 terms, includes m2.
"""
@inline function _J3_qq_kernel(mᵢ, mⱼ, mₖ, η, corr_c)
    FP     = typeof(η)
    m̄      = exp(Base.log(mᵢ * mⱼ * mₖ)/3)
    m1     = 1 - 1/m̄
    m2     = m1 * (1 - 2/m̄)
    result = zero(FP)
    ηn     = one(FP)
    @unroll for n in 0:4
        c0, c1, c2 = corr_c[n+1]
        result += (c0 + c1*m1 + c2*m2) * ηn
        ηn *= η
    end
    return result
end

"""
DQ J₃ integral kernel: no min clamping, 4 terms, no m2 (DQ_consts.corr_c has only (c0,c1) per entry).
"""
@inline function _J3_dq_kernel(mᵢ, mⱼ, mₖ, η, corr_c)
    FP     = typeof(η)
    m̄      = exp(Base.log(mᵢ * mⱼ * mₖ)/3)
    m1     = 1 - 1/m̄
    result = zero(FP)
    ηn     = one(FP)
    @unroll for n in 0:3
        c0, c1 = corr_c[n+1]
        result += (c0 + c1*m1) * ηn
        ηn *= η
    end
    return result
end

function preallocate_params(system::DFTSystem{<:QPCPSAFTModel})
    backend = system.options.device
    FP      = fptype(system.options)
    model   = system.model
    dd_a_fp = ntuple(i -> ntuple(j -> FP(DD_consts.corr_a[i][j]), 3), 5)
    dd_b_fp = ntuple(i -> ntuple(j -> FP(DD_consts.corr_b[i][j]), 3), 5)
    dd_c_fp = ntuple(i -> ntuple(j -> FP(DD_consts.corr_c[i][j]), 3), 5)
    qq_a_fp = ntuple(i -> ntuple(j -> FP(QQ_consts.corr_a[i][j]), 3), 5)
    qq_b_fp = ntuple(i -> ntuple(j -> FP(QQ_consts.corr_b[i][j]), 3), 5)
    qq_c_fp = ntuple(i -> ntuple(j -> FP(QQ_consts.corr_c[i][j]), 3), 5)
    dq_a_fp = ntuple(i -> ntuple(j -> FP(DQ_consts.corr_a[i][j]), 3), 4)
    dq_b_fp = ntuple(i -> ntuple(j -> FP(DQ_consts.corr_b[i][j]), 3), 4)
    dq_c_fp = ntuple(i -> ntuple(j -> FP(DQ_consts.corr_c[i][j]), 2), 4)

    # Reduced units: divide every length-dimensioned parameter by L (raised to however
    # many powers of length it physically carries) so it matches the `get_fields`-side
    # kernel rescaling (inherited from PCSAFT.jl). dipole2/quadrupole2 are stored in
    # units of K·Å³/K·Å⁵ respectively (Clapeyron's PCPSAFT.jl/QPCPSAFT.jl: dipole2 =
    # μ²/m/k_B*(...), quadrupole2 = Q²/m/k_B*(...), Q being in D·Å vs μ's D — the extra
    # Å² gives quadrupole2 two more powers of length than dipole2). See PCSAFT.jl's
    # `get_fields`/`preallocate_params` docstrings for the full picture.
    L               = length_scale(model)
    HSd_local       = system.species.size ./ L
    sigma_local     = model.params.sigma.values ./ L
    pcp_sigma_local = pcp_sigma(model) ./ L

    base = (;
        HSd         = adapt_to_device(backend, FP, HSd_local),
        m           = adapt_to_device(backend, FP, model.params.segment.values),
        sigma       = adapt_to_device(backend, FP, sigma_local),
        epsilon     = adapt_to_device(backend, FP, model.params.epsilon.values),
        pcp_m       = adapt_to_device(backend, FP, pcp_segment(model)),
        pcp_sigma   = adapt_to_device(backend, FP, pcp_sigma_local),
        pcp_epsilon = adapt_to_device(backend, FP, pcp_epsilon(model)),
        dipole2     = adapt_to_device(backend, FP, pcp_dipole2(model) ./ L^3),
        quadrupole2 = adapt_to_device(backend, FP, model.params.quadrupole2.values ./ L^5),
        dd_a = dd_a_fp, dd_b = dd_b_fp, dd_c = dd_c_fp,
        qq_a = qq_a_fp, qq_b = qq_b_fp, qq_c = qq_c_fp,
        dq_a = dq_a_fp, dq_b = dq_b_fp, dq_c = dq_c_fp,
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

    return params, length(model)
end