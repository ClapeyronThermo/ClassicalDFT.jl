using Clapeyron: HeterogcPCPSAFT, pcp_segment, pcp_sigma, pcp_epsilon, pcp_dipole2

"""
    HeterogcPCPSAFT(components::Vector{String})

The heterosegmented group-contribution polar PC-SAFT equation of state. Unlike `HomogcPCPSAFT`, each group in a molecule can carry distinct segment properties, so the DFT implementation expands the model into individual bonded beads (see `expand_model`) and uses a `TangentHSPropagator` chain propagator to enforce connectivity between them.

The bulk model can be obtained from Clapeyron.
"""
HeterogcPCPSAFT

struct gcPCPSAFTSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    levels::Vector{Int64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function get_species(model::HeterogcPCPSAFT,structure::DFTStructure)
    (pressure,temperature) = structure.conditions
    z = structure.ρbulk
    v = 1/sum(z)

    HSd = d(model,1e-3,temperature,z)
    μres = Clapeyron.VT_chemical_potential_res(model, v, temperature, z/sum(z)) / Clapeyron.R̄ / temperature
    nbeads = length.(model.groups.groups)

    levels = compute_levels(model)
    return gcPCPSAFTSpecies(nbeads,HSd,levels,structure.ρbulk,μres)
end

function get_fields(model::HeterogcPCPSAFT, species::DFTSpecies, structure::DFTStructure, backend::Backend, ::Type{FP}) where FP<:AbstractFloat
    nb = sum(species.nbeads)
    ngrid = structure.ngrid
    L = length_scale(model)
    ω = structure_ω(structure, backend, FP)
    d = species.size ./ L
    ψ = 1.5357
    return (SWeightedDensity(:ρ,zeros(nb),ω,ngrid,backend,model),
            SWeightedDensity(:∫ρdz,0.5*d,ω,ngrid,backend,model),
            SWeightedDensity(:∫ρz²dz,0.5*d,ω,ngrid,backend,model),
            VWeightedDensity(:∫ρzdz,0.5*d,ω,ngrid,backend,model),
            SWeightedDensity(:∫ρz²dz,d,ω,ngrid,backend,model),
            SWeightedDensity(:∫ρz²dz,d .* ψ,ω,ngrid,backend,model))

end

function get_propagator(model::HeterogcPCPSAFT, species::DFTSpecies, structure::DFTStructure, backend::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    return TangentHSPropagator(model, species, structure, backend, FP)
end

# function  Δ(model::HeterogcPCPSAFT, T, n, n₃, nᵥ)
#     ϵ_assoc = model.params.epsilon_assoc.values
#     κ = model.params.bondvol.values
#     σ = model.params.sigma.values
#     Δout = 
    
#     for (idx,(i,j),(a,b)) in indices(κ)
#         k,l = get_chain_idx(model,i,j,a,b)
#         gkl = @f(g_hs,k,l,_data)
#         Δout[idx] = gkl*σ[k,l]^3*(exp(ϵ_assoc[i,j][a,b]/T)-1)*κ[i,j][a,b]
#     end
#     return Δout
# end
function length_scale(model::HeterogcPCPSAFT)
    return maximum(model.params.sigma.values)
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
Pointwise residual free energy for HeterogcPCPSAFT:
FMT hard-sphere + hard-chain bonding (Wertheim topology) + PCSAFT-style dispersion.
Polar term is NOT included in the Enzyme kernel (requires per-component aggregation).
Chain connectivity is handled by TangentHSPropagator.

Field layout (6 fields total):
  1        : ρ (unweighted)
  2        : ∫ρdz  with 0.5*d → n₀, n₁, n₂
  3        : ∫ρz²dz with 0.5*d → n₃
  4..3+ND  : ∫ρzdz with 0.5*d → nᵥ
  4+ND     : ∫ρz²dz with d    → ρ̄hc (bonding)
  5+ND     : ∫ρz²dz with d*ψ → ρ̄z  (dispersion, ψ=1.5357)

NC = total number of groups (sum of nbeads per component).
"""
@inline function f_res(::Type{M}, kk, out, n, params, T,
                       ::Val{NC}, ::Val{ND}) where {NC, ND, M <: HeterogcPCPSAFT}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_hc    = f_hc(M, kk, n, params, T, Val(NC), Val(ND))
    res_disp  = f_disp(M, kk, n, params, T, Val(NC), Val(ND))
    res_assoc = _assoc_or_zero(M, kk, n, params, T, Val(NC), Val(ND))
    out[kk] = res_hs + res_disp + res_hc + res_assoc
    return nothing
end

@inline function f_hc(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: HeterogcPCPSAFT}
    HSd   = params.HSd
    m_seg = params.m
    bond_k  = params.bond_k
    bond_l  = params.bond_l

    FP    = eltype(n)
    idx_ζ = 4 + ND
    ζ₃ = zero(FP); ζ₂ = zero(FP)
    @inbounds for i in 1:NC
        mi = m_seg[i]; di = HSd[i]; ρ̄hci = n[kk, idx_ζ, i]
        ζ₃ += mi * ρ̄hci
        ζ₂ += mi * ρ̄hci / di
    end
    ζ₃ /= 8; ζ₂ /= 8
    inv1ζ₃ = 1 / (1 - ζ₃)

    return _f_hc_bonds(n, bond_k, bond_l, HSd, kk, ζ₂, inv1ζ₃)
end

# Bond-loop helper: NB is a WHERE-clause type param so the loop bound is always
# compile-time inside this function, regardless of how autodiff_deferred specialises.
@inline function _f_hc_bonds(n, bond_k::NTuple{NB, Int}, bond_l::NTuple{NB, Int},
                               HSd, kk, ζ₂, inv1ζ₃) where NB
    FP     = typeof(ζ₂)
    res_hc = zero(FP)
    @inbounds for ib in 1:NB
        k = _nti(bond_k, ib); l = _nti(bond_l, ib)
        dk = HSd[k]; dl = HSd[l]
        r_HSd = dk * dl / (dk + dl)
        ζ₂_ov3 = ζ₂ * inv1ζ₃
        yᵈᵈ = inv1ζ₃ + 3*r_HSd*ζ₂_ov3*inv1ζ₃ + 2*r_HSd^2*ζ₂_ov3^2*inv1ζ₃
        ρhck = n[kk, 1, k]
        res_hc -= ρhck / 2 * Base.log(abs(yᵈᵈ))
    end
    return res_hc
end

@inline function f_disp(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: HeterogcPCPSAFT}
    HSd              = params.HSd
    m_seg            = params.m
    σ                = params.sigma
    ϵ                = params.epsilon
    nbeads_for_group = params.nbeads_for_group

    FP      = eltype(n)
    # `π/6*x` and `-2*π*x` (Int/Irrational combos before touching an FP value) promote
    # to Float64, unlike `x*π`/`π*x` for x::FP — see PCSAFT.jl's f_disp for the same
    # fix and the verified examples.
    _π      = FP(π)
    ψ       = FP(1.5357)
    idx_ρz  = 5 + ND
    factor  = 3 / (4*ψ*ψ*ψ*_π)
    ρ̄_tot   = zero(FP); m̄_num = zero(FP); η_sum = zero(FP)
    @inbounds for i in 1:NC
        di  = HSd[i]
        ρ̄i  = n[kk, idx_ρz, i] * factor / (di*di*di)
        m̄_num += m_seg[i] * ρ̄i
        η_sum  += m_seg[i] * ρ̄i * di*di*di
        ρ̄_tot  += ρ̄i / _nti(nbeads_for_group, i)
    end
    m̄  = m̄_num / ρ̄_tot
    ηd = _π * η_sum /6

    m2ϵσ3_1 = zero(FP); m2ϵσ3_2 = zero(FP)
    @inbounds for i in 1:NC
        di   = HSd[i]
        ρ̄i   = n[kk, idx_ρz, i] * factor / (di*di*di)
        @inbounds for j in 1:NC
            dj   = HSd[j]
            ρ̄j   = n[kk, idx_ρz, j] * factor / (dj*dj*dj)
            cij  = ρ̄i * ρ̄j * m_seg[i] * m_seg[j] * σ[i,j]*σ[i,j]*σ[i,j]
            eT   = ϵ[i,j] / T
            m2ϵσ3_1 += cij * eT
            m2ϵσ3_2 += cij * eT * eT
        end
    end
    ηd2    = ηd*ηd
    ηd4    = (1-ηd)^4
    inv1ηd = 1/(1-ηd)
    inv2ηd = 1/(2-ηd)
    C₁     = 1 + m̄*(8*ηd-2*ηd2)/ηd4 +
              (1-m̄)*(20*ηd-27*ηd2+12*(ηd*ηd2)-2*(ηd2*ηd2)) *
              inv1ηd*inv1ηd*inv2ηd*inv2ηd
    I₁     = I_lite(PCSAFT_CORR1, m̄, ηd)
    I₂     = I_lite(PCSAFT_CORR2, m̄, ηd)
    # println(m2ϵσ3_1, ", ", m2ϵσ3_2)
    return -2*_π*I₁*m2ϵσ3_1 - _π*m̄*I₂*m2ϵσ3_2 / C₁
end

function preallocate_params(system::DFTSystem{<:HeterogcPCPSAFT})
    backend  = system.options.device
    FP       = fptype(system.options)
    model    = system.model
    nc_spec  = length(model)
    nbeads   = system.species.nbeads
    nc_groups = sum(nbeads)

    bond_k_list = Int[]
    bond_l_list = Int[]
    for i in 1:nc_spec
        i_groups        = model.groups.i_groups[i]
        n_intergroups_i = model.groups.n_intergroups[i]
        for k in i_groups
            for l in findall(n_intergroups_i[k,:] .== 1)
                push!(bond_k_list, k)
                push!(bond_l_list, l)
            end
        end
    end

    nbeads_for_group = Vector{FP}(undef, nc_groups)
    for i in 1:nc_spec
        nbi = FP(nbeads[i])
        for k in model.groups.i_groups[i]
            nbeads_for_group[k] = nbi
        end
    end

    n_bonds = length(bond_k_list)
    bond_k_t           = ntuple(ib -> bond_k_list[ib], n_bonds)
    bond_l_t           = ntuple(ib -> bond_l_list[ib], n_bonds)
    nbeads_for_group_t = ntuple(i  -> nbeads_for_group[i], nc_groups)

    # Reduced units: divide every length-dimensioned parameter by L so it matches the
    # `get_fields`-side kernel rescaling. See PCSAFT.jl's `get_fields`/`preallocate_params`
    # docstrings for the full picture.
    L           = length_scale(model)
    HSd_local   = system.species.size ./ L
    sigma_local = system.model.params.sigma.values ./ L

    base = (;
        HSd              = adapt_to_device(backend, FP, HSd_local),
        m                = adapt_to_device(backend, FP, system.model.params.segment.values),
        sigma            = adapt_to_device(backend, FP, sigma_local),
        epsilon          = adapt_to_device(backend, FP, system.model.params.epsilon.values),
        nbeads_for_group = nbeads_for_group_t,
        bond_k           = bond_k_t,
        bond_l           = bond_l_t,
    )

    nn = Clapeyron.assoc_pair_length(model)
    if nn > 0
        (assoc_icomp_v, assoc_jcomp_v, _ispec, _jspec,
         assoc_isite_v, assoc_jsite_v,
         assoc_eps_v, assoc_kap_v, assoc_sig3_v, assoc_dij_v,
         n_sites_flat_v, n_sites_cumsum_v, total_sites
        ) = pack_assoc_params_gc(model, HSd_local, sigma_local)

        ia_global_v       = [n_sites_cumsum_v[assoc_icomp_v[p]] + assoc_isite_v[p] for p in 1:nn]
        jb_global_v       = [n_sites_cumsum_v[assoc_jcomp_v[p]] + assoc_jsite_v[p] for p in 1:nn]
        n_ia_v            = [n_sites_flat_v[ia_global_v[p]] for p in 1:nn]
        n_jb_v            = [n_sites_flat_v[jb_global_v[p]] for p in 1:nn]
        assoc_icomp_t    = ntuple(p -> assoc_icomp_v[p],    Val(nn))
        assoc_jcomp_t    = ntuple(p -> assoc_jcomp_v[p],    Val(nn))
        assoc_isite_t    = ntuple(p -> assoc_isite_v[p],    Val(nn))
        assoc_jsite_t    = ntuple(p -> assoc_jsite_v[p],    Val(nn))
        assoc_ia_global_t = ntuple(p -> ia_global_v[p],     Val(nn))
        assoc_jb_global_t = ntuple(p -> jb_global_v[p],     Val(nn))
        assoc_n_ia_t      = ntuple(p -> n_ia_v[p],          Val(nn))
        assoc_n_jb_t      = ntuple(p -> n_jb_v[p],          Val(nn))
        n_sites_flat_t   = ntuple(j -> n_sites_flat_v[j],   Val(total_sites))
        n_sites_cumsum_t = ntuple(i -> n_sites_cumsum_v[i], Val(nc_groups + 1))

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

    return params, nc_groups
end