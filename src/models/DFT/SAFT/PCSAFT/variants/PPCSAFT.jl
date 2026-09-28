using Clapeyron: PCPSAFTModel, pcp_sigma, pcp_dipole, pcp_dipole2, pcp_epsilon, pcp_segment

"""
    PCPSAFT(components::Vector{String})

The polar PC-SAFT (PCP-SAFT) equation of state, which augments PC-SAFT with a dipole-dipole polar term (Gross and Vrabec, 2006). As with `PCSAFT`, our DFT implementation uses a Weighted Density Functional approach and does not use a chain propagator.

The bulk model can be obtained from Clapeyron.
"""
PCPSAFT

const DD_consts = (
    corr_a =
    ((0.3043504,0.9534641,-1.161008),
    (-0.1358588,-1.8396383,4.5258607),
    (1.4493329,2.013118,0.9751222),
    (0.3556977,-7.3724958,-12.281038),
    (-2.0653308,8.2374135,5.9397575)),

    corr_b =
    ((0.2187939,-0.5873164,3.4869576),
    (-1.1896431,1.2489132,-14.915974),
    (1.1626889,-0.508528,15.372022),
    (0.,0.,0.),
    (0.,0.,0.)),

    corr_c =
    ((-0.0646774,-0.9520876,-0.6260979),
    (0.1975882,2.9924258,1.2924686),
    (-0.8087562,-2.3802636,1.6542783),
    (0.6902849,-0.2701261,-3.4396744),
    (0.,0.,0.))
)

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
Pointwise residual free energy for PCP-SAFT: identical to PC-SAFT (HS + HC + disp) with
an additional dipole–dipole polar term (A₂²/(A₂−A₃) Padé approximant).

Field layout (same as PCSAFTModel):
  1        : ρ (unweighted)
  2        : ∫ρdz  with 0.5*d → n₀, n₁, n₂
  3        : ∫ρz²dz with 0.5*d → n₃
  4..3+ND  : ∫ρzdz with 0.5*d → vector nᵥ
  4+ND     : ∫ρz²dz with d    → ρ̄hc
  5+ND     : ∫ρdz  with d    → λ
  6+ND     : ∫ρz²dz with d*ψ → ρ̄z  (disp + polar)
"""
@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: PCPSAFTModel}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_hc  = f_hc(M, kk, n, params, T, Val(NC), Val(ND))
    res_disp, m̄, ηd = f_disp(M, kk, n, params, T, Val(NC), Val(ND))
    res_polar = f_polar(M, kk, n, params, T, m̄, ηd, Val(NC), Val(ND))
    res_assoc = _assoc_or_zero(M, kk, n, params, T, Val(NC), Val(ND))
    out[kk] = res_hs + res_hc + res_disp + res_polar + res_assoc
    return nothing
end

"""
PCP-SAFT dipole–dipole polar term (Padé: A₂²/(A₂−A₃)) at grid point `kk`.
Takes `m̄` and `ηd` from f_disp output.
"""
@inline function f_polar(::Type{M}, kk, n, params, T, m̄, ηd, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: PCPSAFTModel}
    FP    = eltype(n)
    pcp_m = params.pcp_m
    pcp_ϵ = params.pcp_epsilon
    pcp_σ = params.pcp_sigma
    dip2  = params.dipole2
    ca    = params.dd_a
    cb    = params.dd_b
    cc    = params.dd_c

    has_polar = false
    @inbounds @unroll for i in 1:NC
        if !iszero(dip2[i]); has_polar = true; break; end
    end

    res_polar = zero(FP)
    if has_polar
        # `-π*x` and `-4*π*π/3` (Int/Irrational combos before touching an FP value)
        # promote to Float64, unlike `x*π`/`π*x` for x::FP — see PCSAFT.jl's f_disp.
        _π      = FP(π)
        ψ       = FP(1.3862)
        idx_ρz  = 6 + ND
        factor  = 3 / (4*ψ*ψ*ψ*_π)
        ∑ρ̄_p = zero(FP)
        @inbounds @unroll for i in 1:NC
            ∑ρ̄_p += n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
        end

        _A₂ = zero(FP)
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
                _J2_ij = _J2_kernel(pcp_m[i], pcp_m[j], pcp_ϵ[i,j], ηd, T, ca, cb)
                _A₂ += xᵢ * xⱼ * dip2_i * dip2_j / σij3 * _J2_ij
            end
        end
        _A₂ *= -_π * ∑ρ̄_p / (T*T)

        if abs(_A₂) > 0
            _A₃ = zero(FP)
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
                        _J3_ijk = _J3_kernel(pcp_m[i], pcp_m[j], pcp_m[k], ηd, cc)
                        _A₃ += xᵢ*xⱼ*xₖ * dip2_i*dip2_j*dip2_k /
                               (pcp_σ[i,j]*pcp_σ[i,k]*pcp_σ[j,k]) * _J3_ijk
                    end
                end
            end
            _A₃ *= -4*_π*_π/3 * ∑ρ̄_p*∑ρ̄_p / (T*T*T)

            denom_p = _A₂ - _A₃
            res_polar = ∑ρ̄_p * _A₂*_A₂ / denom_p
        end
    end

    return res_polar
end

@inline function _J2_kernel(mᵢ, mⱼ, ϵᵢⱼ, η, T, corr_a, corr_b)
    FP  = typeof(η)
    ϵT  = ϵᵢⱼ / T
    m̄   = min(sqrt(mᵢ * mⱼ), 2)
    m1  = 1 - 1/m̄
    m2  = m1 * (1 - 2/m̄)
    result = zero(FP)
    ηn = one(FP)
    @unroll for n in 0:4
        a0, a1, a2 = corr_a[n+1]
        b0, b1, b2 = corr_b[n+1]
        result += (a0 + a1*m1 + a2*m2 + (b0 + b1*m1 + b2*m2)*ϵT) * ηn
        ηn *= η
    end
    return result
end

@inline function _J3_kernel(mᵢ, mⱼ, mₖ, η, corr_c)
    FP  = typeof(η)
    m̄   = min(exp(Base.log(mᵢ * mⱼ * mₖ)/3), 2)
    m1  = 1 - 1/m̄
    m2  = m1 * (1 - 2/m̄)
    result = zero(FP)
    ηn = one(FP)
    @unroll for n in 0:4
        c0, c1, c2 = corr_c[n+1]
        result += (c0 + c1*m1 + c2*m2) * ηn
        ηn *= η
    end
    return result
end

function preallocate_params(system::DFTSystem{<:PCPSAFTModel})
    backend = system.options.device
    FP      = fptype(system.options)
    model   = system.model
    dd_a_fp = ntuple(i -> ntuple(j -> FP(DD_consts.corr_a[i][j]), 3), 5)
    dd_b_fp = ntuple(i -> ntuple(j -> FP(DD_consts.corr_b[i][j]), 3), 5)
    dd_c_fp = ntuple(i -> ntuple(j -> FP(DD_consts.corr_c[i][j]), 3), 5)

    # Reduced units: divide every length-dimensioned parameter by L so it matches the
    # `get_fields`-side kernel rescaling (inherited from PCSAFT.jl). See PCSAFT.jl's
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
        dipole2     = adapt_to_device(backend, FP, pcp_dipole2(model) ./ (L*L*L)),
        dd_a        = dd_a_fp,
        dd_b        = dd_b_fp,
        dd_c        = dd_c_fp,
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