using Clapeyron: SAFTgammaMieModel
using Clapeyron: d_gc_av

"""
    SAFTgammaMie(components::Vector{String})

The SAFT-γ Mie group-contribution equation of state (Papaioannou et al., 2014). Since groups can carry heterogeneous segment properties, the DFT implementation expands the model into individual bonded beads (see `expand_model`) and uses a `TangentHSPropagator` chain propagator to enforce connectivity between them.

The bulk model can be obtained from Clapeyron.
"""
SAFTgammaMie

struct SAFTgammaMieSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    levels::Vector{Int64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function get_species(model::SAFTgammaMie,structure::DFTStructure)
    (pressure,temperature) = structure.conditions
    ρbulk = structure.ρbulk
    HSd = d(model,1e-3,temperature,ones(length(model.groups.flattenedgroups)))

    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), temperature, ρbulk./sum(ρbulk)) / Clapeyron.R̄ / temperature
    nbeads = length.(model.groups.groups)

    levels = compute_levels(model)
    return SAFTgammaMieSpecies(nbeads,HSd,levels,ρbulk,μres)
end

function get_fields(model::SAFTgammaMieModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}) where FP<:AbstractFloat
    nb = sum(species.nbeads)
    ngrid = structure.ngrid
    nd = dimension(structure)

    # ψ is a dimensionless Barker-Henderson-style shape factor derived from the RAW
    # (unscaled) diameter/sigma ratio — it must stay L-invariant, so compute it before
    # any reduced-units rescaling is applied below. See PCSAFT.jl's `get_fields` docstring
    # for the overall reduced-units scheme.
    d   = species.size
    λ_r = diagvalues(model.params.lambda_r.values)
    λ_a = diagvalues(model.params.lambda_a.values)
    σ   = diagvalues(model.params.sigma.values)
    C = @. λ_r / (λ_r - λ_a) * (λ_r / λ_a)^(λ_a / (λ_r - λ_a))
    x = d ./ σ
    ψ = @. cbrt(3*C*(1/(λ_a-3)-1/(λ_r-3)))

    L = length_scale(model)
    ω = structure_ω(structure, device, FP)
    d_local = species.size ./ L

    return (SWeightedDensity(:ρ,zeros(nb),ω,ngrid,device,model),
            SWeightedDensity(:∫ρdz,0.5*d_local,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,0.5*d_local,ω,ngrid,device,model),
            VWeightedDensity(:∫ρzdz,0.5*d_local,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,d_local,ω,ngrid,device,model),
            SWeightedDensity(:∫ρdz,d_local,ω,ngrid,device,model),
            SWeightedDensity(:∫ρz²dz,d_local .* ψ,ω,ngrid,device,model))
end

function get_propagator(model::SAFTgammaMieModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    return TangentHSPropagator(model, species, structure, device, FP)
end

function expand_model(model::SAFTgammaMieModel,
        mol_structure::Dict{String,<:MolStructure} = Dict{String,MolStructure}())

    nspecies = length(model)

    #Expand groups
    grouparam,ngroups_k = expand_groups(model, mol_structure)
    
    #Expand the sites 
    siteparams = expand_sites(model, grouparam, ngroups_k)
    params_old,vrparams_old = model.params,model.vrmodel.params
    PARAM = typeof(params_old)
    
    oldparams = PARAM(params_old.segment,params_old.shapefactor,params_old.lambda_a,params_old.lambda_r,params_old.sigma,params_old.epsilon,vrparams_old.epsilon_assoc,vrparams_old.bondvol,params_old.mixed_segment)
    #Expand the parameters
    eosparams = expand_params(oldparams, grouparam, siteparams, ngroups_k)

    #compute mixed segment
    Clapeyron.mix_segment!(eosparams.mixed_segment,grouparam,eosparams.shapefactor.values,eosparams.segment.values).values
    vreosparam_type = typeof(model.vrmodel.params)
    vreosparams = vreosparam_type(model.vrmodel.params.Mw,
                                  model.vrmodel.params.segment,
                                  model.vrmodel.params.sigma,
                                  model.vrmodel.params.lambda_a,
                                  model.vrmodel.params.lambda_r,
                                  model.vrmodel.params.epsilon,
                                  eosparams.epsilon_assoc,
                                  eosparams.bondvol)

    vrmodel = SAFTVRMie(model.vrmodel.components,
                        siteparams,
                        vreosparams,
                        model.vrmodel.idealmodel,
                        model.vrmodel.assoc_options,
                        model.vrmodel.references
                        )

    return new_model = SAFTgammaMie(model.components,
                                grouparam,
                                siteparams,
                                eosparams,
                                model.idealmodel,
                                vrmodel,
                                model.epsilon_mixing,
                                model.assoc_options,
                                model.references)
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────


@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: SAFTgammaMieModel}
    res_hs, = f_hs(n, params.meff, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_disp  = f_disp(M, kk, n, params, T, Val(NC), Val(ND), Val(6+ND))
    res_chain = f_chain(M, kk, n, params, T, Val(NC), Val(ND))
    res_assoc = _assoc_or_zero(M, kk, n, params, T, Val(NC), Val(ND))
    out[kk] = res_hs + res_chain + res_disp + res_assoc
    return nothing
end

"""
Pointwise residual free energy for SAFTγMie: FMT hard-sphere (with shapefactor) +
chain (groups aggregated to species level, gMie from vrmodel params) +
SAFT-VR Mie dispersion (with effective m*S segments).

Field layout (same as SAFTVRMieModel):
  1        : ρ (unweighted)
  2        : ∫ρdz  with 0.5*d → n₀, n₁, n₂
  3        : ∫ρz²dz with 0.5*d → n₃
  4..3+ND  : ∫ρzdz with 0.5*d → nᵥ
  4+ND     : ∫ρz²dz with d    → ρ̄hc  (for TangentHSPropagator chain)
  5+ND     : ∫ρdz  with d     → λ    (for TangentHSPropagator chain)
  6+ND     : ∫ρz²dz with d*ψ → ρ̄z   (dispersion)

NC here is the total number of groups (sum of nbeads per component).
"""
@inline function f_chain(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: SAFTgammaMieModel}
    HSd    = params.HSd
    meff   = params.meff
    σ      = params.sigma
    A      = params.A
    ϕ      = params.phi

    idx_ζ_c = 4 + ND
    nbeads_c = params.nbeads_comp
    HSd_s    = params.HSd_species
    m_s      = params.m_species
    σ_s      = params.sigma_species
    ϵ_s      = params.epsilon_species
    λr_s     = params.lambda_r_species_t
    λa_s     = params.lambda_a_species_t
    nc_s     = length(nbeads_c)

    FP    = eltype(n)
    # Bare `π` (an Irrational) promotes to Float64 when combined with an Int literal
    # BEFORE ever touching an FP value (e.g. `4*π` alone is Float64, so `4*π*x` for
    # x::Float32 stays Float64 — confirmed: `typeof(4*π*Float32(1)) == Float64` but
    # `typeof(π*Float32(1)*4) == Float32`). Inside a runtime-conditional loop (see
    # `_chain_bead_sum` below) this makes the accumulator's type branch-dependent — a
    # genuine Union{Float32,Float64} that trips Enzyme's strict type analysis. Binding
    # `π = FP(π)` once and using it everywhere below sidesteps the whole ordering trap.
    sg_idx = params.species_group_idx
    ρS_c  = zero(FP)
    @inbounds for s in 1:nc_s
        nb_s     = _nti(nbeads_c, s)
        sg_idx_s = _nti(sg_idx, s)      # NTuple{MNB,Int} — Const to Enzyme
        ρ̄hc_s   = _chain_bead_sum(sg_idx_s, n, HSd, kk, idx_ζ_c, nb_s)
        ρS_c    += (ρ̄hc_s / nb_s) * m_s[s]
    end
    kρS_c = ρS_c * π/6/8

    ρhc_gc_total = zero(FP)
    @inbounds for kg in 1:NC
        ρhc_gc_total += n[kk, 1, kg]
    end
    m̄_gc = zero(FP)
    @inbounds for kg in 1:NC
        z_gc_kg = n[kk, 1, kg] / ρhc_gc_total
        m̄_gc += z_gc_kg * meff[kg]
    end
    m̄inv_gc = 1/m̄_gc

    ζ_Xc = zero(FP);  σ3_xc = zero(FP)
    @inbounds for i in 1:NC
        z_gc_i = n[kk, 1, i] / ρhc_gc_total
        x_Si_c = z_gc_i * meff[i] * m̄inv_gc
        di_c   = HSd[i]
        @inbounds for j in 1:NC
            z_gc_j = n[kk, 1, j] / ρhc_gc_total
            x_Sj_c = z_gc_j * meff[j] * m̄inv_gc
            dj_c   = HSd[j]
            σ3_xc += x_Si_c*x_Sj_c*σ[i,j]^3
            ζ_Xc  += kρS_c*x_Si_c*x_Sj_c*(di_c+dj_c)^3
        end
    end
    ζstc = σ3_xc * ρS_c * π/6
    _KHSc, _∂KHSc = _KHS_fdf_kernel(ρS_c, ζ_Xc)

    res_chain = zero(FP)
    @inbounds for s in 1:nc_s
        nb_s     = _nti(nbeads_c, s)
        sg_idx_s = _nti(sg_idx, s)          # NTuple{MNB,Int} — Const to Enzyme
        ρhc_s    = _chain_ρhc_s(sg_idx_s, n, kk, nb_s)
        ρhc_s   /= nb_s

        di_s = HSd_s[s]
        λa_c = _nti(_nti(λa_s, s), s)       # use _nti to avoid jl_get_nth_field_checked on GPU
        λr_c = _nti(_nti(λr_s, s), s)
        _Cc  = _Cλ_kernel(λa_c, λr_c)
        x0c  = σ_s[s,s] / di_s
        ϵiic = ϵ_s[s,s]

        aS1c_a,  dS1c_a  = _aS1_fdf_kernel(λa_c,      ζ_Xc, A)
        aS1c_r,  dS1c_r  = _aS1_fdf_kernel(λr_c,      ζ_Xc, A)
        Bc_a,    dBc_a   = _B_fdf_kernel(λa_c,    x0c, ζ_Xc)
        Bc_r,    dBc_r   = _B_fdf_kernel(λr_c,    x0c, ζ_Xc)
        aS1c_2a, dS1c_2a = _aS1_fdf_kernel(2*λa_c,    ζ_Xc, A)
        aS1c_2r, dS1c_2r = _aS1_fdf_kernel(2*λr_c,    ζ_Xc, A)
        aS1c_ar, dS1c_ar = _aS1_fdf_kernel(λa_c+λr_c, ζ_Xc, A)
        Bc_2a,   dBc_2a  = _B_fdf_kernel(2*λa_c,  x0c, ζ_Xc)
        Bc_2r,   dBc_2r  = _B_fdf_kernel(2*λr_c,  x0c, ζ_Xc)
        Bc_ar,   dBc_ar  = _B_fdf_kernel(λa_c+λr_c,x0c,ζ_Xc)

        ∂a1ρSc = _Cc*(x0c^λa_c*(dS1c_a+dBc_a) - x0c^λr_c*(dS1c_r+dBc_r))
        g1c    = 3*∂a1ρSc - _Cc*(λa_c*x0c^λa_c*(aS1c_a+Bc_a) - λr_c*x0c^λr_c*(aS1c_r+Bc_r))

        αc  = _Cc*(1/(λa_c-3) - 1/(λr_c-3))
        f1c,f2c,f3c,f4c,f5c,f6c = _f123456_kernel(αc, ϕ)
        θc  = exp(ϵiic/T) - 1
        γcc = 10*(-tanh(10*(FP(0.57)-αc))+1)*ζstc*θc*exp(-FP(6.7)*ζstc-8*ζstc^2)

        cb2ac = x0c^(2*λa_c)*(aS1c_2a+Bc_2a)
        cbarc = x0c^(λa_c+λr_c)*(aS1c_ar+Bc_ar)
        cb2rc = x0c^(2*λr_c)*(aS1c_2r+Bc_2r)
        ∂a2ρSc = _Cc*_Cc/2*(
            ρS_c*_∂KHSc*(cb2ac - 2*cbarc + cb2rc)
          + _KHSc*(x0c^(2*λa_c)*(dS1c_2a+dBc_2a)
                 - 2*x0c^(λa_c+λr_c)*(dS1c_ar+dBc_ar)
                 + x0c^(2*λr_c)*(dS1c_2r+dBc_2r))
        )
        gMCA2c = 3*∂a2ρSc - _KHSc*_Cc*_Cc*(λr_c*cb2rc - (λa_c+λr_c)*cbarc + λa_c*cb2ac)
        g2c    = (1+γcc)*gMCA2c

        gHSc  = _gHS_kernel(x0c, ζ_Xc)
        gMiec = gHSc * exp(ϵiic/T * g1c/gHSc + (ϵiic/T)^2 * g2c/gHSc)

        ms = m_s[s]
        res_chain += ρhc_s * Base.log(abs(gMiec)) * (ms - 1)
    end
    return -res_chain
end

# Chain loop helpers: NCS / MNB come from NTuple type params so loop bounds are always
# compile-time inside these functions, regardless of how autodiff_deferred specialises.

@inline function _chain_bead_sum(sg_idx_s::NTuple{MNB, Int}, n, HSd, kk, field_idx, nb_s) where MNB
    FP  = eltype(n)
    acc = zero(FP)
    @inbounds for j in 1:MNB        # MNB from WHERE clause → compile-time
        if j <= nb_s                 # nb_s is runtime but MNB (bound) is compile-time
            kg = _nti(sg_idx_s, j)
            dg = HSd[kg]
            acc += n[kk, field_idx, kg] * 3/(4*(π*dg^3))
        end
    end
    return acc
end

@inline function _chain_ρhc_s(sg_idx_s::NTuple{MNB, Int}, n, kk, nb_s) where MNB
    FP  = eltype(n)
    acc = zero(FP)
    @inbounds for j in 1:MNB        # MNB from WHERE clause → compile-time
        j <= nb_s || continue
        acc += n[kk, 1, _nti(sg_idx_s, j)]
    end
    return acc
end

# Model-specific Wertheim Δ for SAFTgammaMie: VR Mie polynomial I(Tr, ρr).
# Uses params.meff (= m.*S) for ρS and params.assoc_ispec/jspec + params.epsilon_species
# (nc_spec × nc_spec VR Mie ε at species level) for the reduced temperature Tr.
@inline function _assoc_delta(p, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix,
                               ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: SAFTgammaMieModel}
    FP = eltype(n)
    p > n_pairs && return zero(FP)
    ρS = zero(FP)
    @inbounds for k in 1:NC
        ρS += n[kk, 3, k] * 6 / (π * params.HSd[k]^3) * params.meff[k]
    end
    σ3_x = zero(FP)
    @inbounds for k in 1:NC
        ρ̄k  = n[kk, 3, k] * 6 / (π * params.HSd[k]^3)
        xSk = ρ̄k * params.meff[k] / ρS
        σ3_x += xSk * xSk * params.sigma[k,k]^3
        @inbounds for l in 1:(k-1)
            ρ̄l  = n[kk, 3, l] * 6 / (π * params.HSd[l]^3)
            xSl = ρ̄l * params.meff[l] / ρS
            σ3_x += 2 * xSk * xSl * params.sigma[k,l]^3
        end
    end
    ρr = ρS * σ3_x
    is = _nti(params.assoc_ispec, p)
    js = _nti(params.assoc_jspec, p)
    Tr = T / params.epsilon_species[is, js]
    I_val = zero(FP); ρrn = one(FP)
    for ni in 0:10
        row = _nti(params.VRMie_c, ni + 1); Trm = one(FP)
        for mi in 0:10
            I_val += _nti(row, mi + 1) * Trm * ρrn; Trm *= Tr
        end
        ρrn *= ρr
    end
    return (exp(params.assoc_eps[p] / T) - 1) * params.assoc_kap[p] * I_val
end

function preallocate_params(system::DFTSystem{<:SAFTgammaMieModel})
    backend = system.options.device
    FP      = fptype(system.options)
    model   = system.model
    T_val  = system.structure.conditions[2]
    x_val  = system.structure.ρbulk ./ sum(system.structure.ρbulk)
    HSd_sp = d_gc_av(model, 1e-3, T_val, x_val, system.species.size)

    m_vals = model.params.segment.values
    S_vals = model.params.shapefactor.values
    meff   = m_vals .* S_vals

    nc_g = sum(system.species.nbeads)
    lr_g = model.params.lambda_r.values
    la_g = model.params.lambda_a.values
    lambda_r_t = ntuple(i -> ntuple(j -> FP(lr_g[i,j]), nc_g), nc_g)
    lambda_a_t = ntuple(i -> ntuple(j -> FP(la_g[i,j]), nc_g), nc_g)
    ncs = size(model.vrmodel.params.lambda_r.values, 1)
    lr_s = model.vrmodel.params.lambda_r.values
    la_s = model.vrmodel.params.lambda_a.values
    lambda_r_species_t = ntuple(i -> ntuple(j -> FP(lr_s[i,j]), ncs), ncs)
    lambda_a_species_t = ntuple(i -> ntuple(j -> FP(la_s[i,j]), ncs), ncs)
    nc_s_v   = length(system.species.nbeads)
    nbeads_v = system.species.nbeads
    max_nb   = maximum(nbeads_v)
    species_group_idx = ntuple(s -> ntuple(j -> (j <= nbeads_v[s] ? model.groups.i_groups[s][j] : 0), max_nb), nc_s_v)
    A_fp   = ntuple(i -> ntuple(j -> FP(SAFTVRMIE_A[i][j]),   4), 4)
    phi_fp = ntuple(i -> ntuple(j -> FP(SAFTVRMIE_PHI[i][j]), 6), 7)

    # Reduced units: divide the GROUP-level length parameters by L so they match the
    # `get_fields`-side kernel rescaling. `psi_eff` needs no separate treatment (reads
    # the rescaled width back from `get_fields`). The SPECIES-level params below
    # (HSd_species, sigma_species, etc., from model.vrmodel) are deliberately left
    # unscaled — they only ever combine with each other as ratios/energies (see
    # `f_chain`/`_assoc_delta`), never directly with the L-rescaled group-level HSd or a
    # bare weighted density. See PCSAFT.jl's `get_fields`/`preallocate_params`
    # docstrings for the full picture.
    L           = length_scale(model)
    HSd_local   = system.species.size ./ L
    sigma_local = model.params.sigma.values ./ L

    base = (;
        HSd                = adapt_to_device(backend, FP, HSd_local),
        m                  = adapt_to_device(backend, FP, m_vals),
        S                  = adapt_to_device(backend, FP, S_vals),
        meff               = adapt_to_device(backend, FP, meff),
        sigma              = adapt_to_device(backend, FP, sigma_local),
        epsilon            = adapt_to_device(backend, FP, model.params.epsilon.values),
        lambda_r_t         = lambda_r_t,
        lambda_a_t         = lambda_a_t,
        psi_eff            = adapt_to_device(backend, FP, system.fields[end].width),
        A                  = A_fp,
        phi                = phi_fp,
        nbeads_comp        = ntuple(i -> system.species.nbeads[i], nc_s_v),
        HSd_species        = adapt_to_device(backend, FP, HSd_sp),
        m_species          = adapt_to_device(backend, FP, model.vrmodel.params.segment.values),
        sigma_species      = adapt_to_device(backend, FP, model.vrmodel.params.sigma.values),
        epsilon_species    = adapt_to_device(backend, FP, model.vrmodel.params.epsilon.values),
        lambda_r_species_t = lambda_r_species_t,
        lambda_a_species_t = lambda_a_species_t,
        species_group_idx  = species_group_idx,
    )

    nn = Clapeyron.assoc_pair_length(model)
    if nn > 0
        nc_model = sum(system.species.nbeads)
        (assoc_icomp_v, assoc_jcomp_v, assoc_ispec_v, assoc_jspec_v,
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
        assoc_ispec_t    = ntuple(p -> assoc_ispec_v[p],    Val(nn))
        assoc_jspec_t    = ntuple(p -> assoc_jspec_v[p],    Val(nn))
        assoc_isite_t    = ntuple(p -> assoc_isite_v[p],    Val(nn))
        assoc_jsite_t    = ntuple(p -> assoc_jsite_v[p],    Val(nn))
        assoc_ia_global_t = ntuple(p -> ia_global_v[p],     Val(nn))
        assoc_jb_global_t = ntuple(p -> jb_global_v[p],     Val(nn))
        assoc_n_ia_t      = ntuple(p -> n_ia_v[p],          Val(nn))
        assoc_n_jb_t      = ntuple(p -> n_jb_v[p],          Val(nn))
        n_sites_flat_t   = ntuple(j -> n_sites_flat_v[j],   Val(total_sites))
        n_sites_cumsum_t = ntuple(i -> n_sites_cumsum_v[i], Val(nc_model + 1))

        c_mat   = SAFTVRMieconsts.c
        VRMie_c = ntuple(ni -> ntuple(mi -> FP(c_mat[ni, mi]), 11), 11)

        assoc = (;
            has_assoc       = true,
            assoc_n_pairs   = Val(nn),
            assoc_n_sites   = Val(total_sites),
            assoc_icomp     = assoc_icomp_t,
            assoc_jcomp     = assoc_jcomp_t,
            assoc_ispec     = assoc_ispec_t,
            assoc_jspec     = assoc_jspec_t,
            assoc_isite     = assoc_isite_t,
            assoc_jsite     = assoc_jsite_t,
            assoc_ia_global = assoc_ia_global_t,
            assoc_jb_global = assoc_jb_global_t,
            assoc_n_ia      = assoc_n_ia_t,
            assoc_n_jb      = assoc_n_jb_t,
            assoc_eps       = adapt_to_device(backend, FP, assoc_eps_v),
            # SAFTgammaMie's _assoc_delta (VR-Mie polynomial I(Tr,ρr)) uses assoc_kap
            # directly with no compensating σ³ factor (unlike the PCSAFT-family default
            # _assoc_delta, where assoc_sig3 already carries the /L^3 via sigma_local) —
            # so assoc_kap needs its own explicit /L^3 here to stay consistent with the
            # L³-inflated n0/xi it's multiplied against. assoc_sig3 is unused by this
            # model's _assoc_delta but rescaled too for consistency.
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

    nc = sum(system.species.nbeads)
    return params, nc
end
