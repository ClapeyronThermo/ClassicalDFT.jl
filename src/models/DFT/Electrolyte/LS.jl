import Clapeyron: LS, LSNeutral, LSIon, LS_SIGMA

#=
LS-theory spatial DFT functional. `LS <: Clapeyron.ElectrolyteModel`, so it
uses `ElectrolyteDFTSystem` directly -- no bespoke system struct, composite
`f_res`, or `preallocate_model`/`_energy_scale` override needed here: all of
those are already generic over any `M<:ElectrolyteModel`
(`models/DFT/Electrolyte/base.jl`, `models/models.jl`). Only the two
submodels' own dispatches (this file) are LS-specific, exactly mirroring
this directory's `DH.jl` for a Debye-Hückel ion model.
=#

# ── Neutral half: FMT hard-sphere + per-bond TPT1 chain term ────────────────
# Mirrors `hetero_gcPPCSAFT.jl`, minus the PC-SAFT dispersion term LS theory
# has no equivalent of.

struct LSDFTNeutralSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    levels::Vector{Int64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

"""
    get_species(model::LSNeutral, structure)

One bead per `model.groups` entry, all diameter `LS_SIGMA` (LS theory's
beads are strictly hard, so unlike PCSAFT's Barker-Henderson `d(model,...)`,
no temperature-dependent softening integral is needed). `chempot_res` is a
placeholder overwritten by `ElectrolyteDFTSystem`'s own constructor with the
*composite* `LS` model's chemical potential (`species.chempot_res .= μres`).
"""
function get_species(model::LSNeutral, structure::DFTStructure)
    nbeads = length.(model.groups.groups)
    nc_groups = sum(nbeads)
    size = fill(LS_SIGMA, nc_groups)
    levels = compute_levels(model)
    μres = zeros(length(nbeads))
    return LSDFTNeutralSpecies(nbeads, size, levels, structure.ρbulk, μres)
end

"""
    get_fields(model::LSNeutral, species, structure, device, FP)

Field layout (5 fields, reduced units -- lengths divided by `L=length_scale(model)`,
matching PCSAFT.jl/hetero_gcPPCSAFT.jl's convention):
  1        : ρ (unweighted)               -- used by the per-bond chain term
  2        : ∫ρdz  with 0.5*d → n₀,n₁,n₂  -- FMT
  3        : ∫ρz²dz with 0.5*d → n₃       -- FMT
  4..3+ND  : ∫ρzdz with 0.5*d → nᵥ        -- FMT
  4+ND     : ∫ρz²dz with d    → ρ̄hc       -- chain term ζ₂/ζ₃
No dispersion/polar field: LS theory has neither.
"""
function get_fields(model::LSNeutral, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}) where FP<:AbstractFloat
    nb = sum(species.nbeads)
    ngrid = structure.ngrid
    L = length_scale(model)
    ω = structure_ω(structure, device, FP)
    d = species.size ./ L
    return (SWeightedDensity(:ρ, zeros(nb), ω, ngrid, device, model),
            SWeightedDensity(:∫ρdz, 0.5*d, ω, ngrid, device, model),
            SWeightedDensity(:∫ρz²dz, 0.5*d, ω, ngrid, device, model),
            VWeightedDensity(:∫ρzdz, 0.5*d, ω, ngrid, device, model),
            SWeightedDensity(:∫ρz²dz, d, ω, ngrid, device, model))
end

"""
    get_propagator(model::LSNeutral, species, structure, device, FP)

Full `TangentHSPropagator`: bead *position* along the specific sequence is
the whole point of this model, so every real bond is tracked individually
(never `IdealPropagator`/a folded bulk-homopolymer treatment).
"""
function get_propagator(model::LSNeutral, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    return TangentHSPropagator(model, species, structure, device, FP)
end

length_scale(model::LSNeutral) = LS_SIGMA

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
Pointwise residual free energy for `LSNeutral`: FMT hard-sphere + per-bond
chain term (every bond uses `y_hs`, i.e. the Γ_MSA=0 limit -- see
`Clapeyron.LSNeutral.a_res`, the bulk model this reduces to in the uniform
limit). Field layout matches `get_fields` above; `F2=2` (n₀/n₁/n₂ source is
field 2), so this is a direct call into `f_hs`/`_f_hc_bonds` (the same
functions `HeterogcPCPSAFT` uses) -- no reimplementation.
"""
@inline function f_res(::Type{M}, kk, out, n, params, T,
                        ::Val{NC}, ::Val{ND}) where {NC, ND, M <: LSNeutral}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_bond = f_bond_neutral(M, kk, n, params, T, Val(NC), Val(ND))
    out[kk] = res_hs + res_bond
    return nothing
end

@inline function f_bond_neutral(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: LSNeutral}
    HSd = params.HSd
    m_seg = params.m
    bond_k = params.bond_k
    bond_l = params.bond_l

    FP = eltype(n)
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

"""
    preallocate_params(system::DFTSystem{<:LSNeutral})

Bond list + reduced-units `HSd`, exactly `hetero_gcPPCSAFT.jl`'s pattern
(minus dispersion's `epsilon`/`nbeads_for_group`, which LS theory has no use
for; `m` is all-ones since each group is a single hard sphere). Loops over
every component (`model.groups.i_groups[c]`/`n_intergroups[c]`), not just
component 1, so a free-ion component is handled by the same code path as
the chain -- it simply contributes no bonds (its own `n_intergroups[c]` is
identically zero by construction).
"""
function preallocate_params(system::DFTSystem{<:LSNeutral})
    device = system.options.device
    FP = fptype(system.options)
    model = system.model
    nc_groups = sum(system.species.nbeads)

    bond_k_list = Int[]
    bond_l_list = Int[]
    for c in 1:length(model)
        i_groups_c = model.groups.i_groups[c]
        n_intergroups_c = model.groups.n_intergroups[c]
        for k in i_groups_c
            for l in findall(n_intergroups_c[k, :] .== 1)
                push!(bond_k_list, k)
                push!(bond_l_list, l)
            end
        end
    end

    n_bonds = length(bond_k_list)
    bond_k_t = ntuple(ib -> bond_k_list[ib], n_bonds)
    bond_l_t = ntuple(ib -> bond_l_list[ib], n_bonds)

    L = length_scale(model)
    HSd_local = system.species.size ./ L

    params = (;
        HSd = adapt_to_device(device, FP, HSd_local),
        m = adapt_to_device(device, FP, ones(nc_groups)),
        bond_k = bond_k_t,
        bond_l = bond_l_t,
    )
    return params, nc_groups
end

# ── Charge-induced half: restricted-primitive-model MSA + per-bond ±Δ(r) ────
# Mirrors this directory's `DH.jl`, substituting LS theory's closed-form
# Γ_MSA=(-1+sqrt(1+2κ_MSA))/2 (exact for equal-sized beads) for DH's
# Padé-approximant χ(x) term (needed there for possibly-unequal ion sizes).

struct LSDFTIonSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    charges::Vector{Float64}
    size::Vector{Float64}
    levels::Vector{Int64}
    bulk_density::Vector{Float64}
end

"""
    get_species(model::LSIon, neutralmodel::LSNeutral, charges::Vector{Int}, structure)

One entry per bead (same `model.groups` as `neutralmodel`); charges come
straight from `model.params.Z.values` (already per-bead for an `expand=true`
model) -- the `charges` argument is accepted only to match `get_species`'s
general `(ionmodel, neutralmodel, charges, structure)` calling convention
(`DH.jl`'s own signature), and is asserted consistent rather than used, since
LS's own `LSIon` already carries this information internally.

`levels` (via `compute_levels`) is carried here too even though the
composite `ElectrolyteDFTSystem` constructor never uses it (propagation
happens once, via the neutral species' own `levels`, shared across
neutral+ion fields) -- it's needed for a standalone `DFTSystem{<:LSIon}`
built with a real `TangentHSPropagator` (rather than the default
`IdealPropagator`) to work at all, e.g. for isolated testing of the
electrostatic term on a bonded (not just single-ion) sequence.
"""
function get_species(model::LSIon, neutralmodel::LSNeutral, charges::Vector{Int}, structure::DFTStructure)
    Z = Float64.(model.params.Z.values)
    @assert charges == round.(Int, Z) "get_species(::LSIon,...): supplied charges do not match model.params.Z"
    nc_groups = length(Z)
    size = fill(LS_SIGMA, nc_groups)
    levels = compute_levels(model)
    return LSDFTIonSpecies(ones(Int64, nc_groups), Z, size, levels, structure.ρbulk)
end

"""
    get_fields(tup::Tuple{<:LSIon,FP}, species, structure, device, FP)

One extra `∫ρdz` field (on top of `LSDFTNeutral`'s 5), smoothed over a
`(σ/2 + 1/κ_MSA_bulk)` width -- exactly `DH.jl`'s `get_fields` pattern, with
`κ_MSA_bulk` from LS theory's own (not DH's) screening-length formula. `L`
must be the *neutral* model's `length_scale` (shared with `LSDFTNeutral`'s
fields) since only one global `_energy_scale`/`L^3` correction applies to
the combined `F_res` -- see `DH.jl`'s own matching note.
"""
function get_fields(tup::Tuple{<:LSIon,FP}, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}) where FP<:AbstractFloat
    ionmodel, L = tup
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    ngrid = structure.ngrid
    p = ionmodel.params
    v = 1 / sum(ρbulk)

    ρ★ = Clapeyron.ls_group_densities(ionmodel.groups, v, ρbulk)
    ϵr = dielectric_constant(ionmodel.RSPmodel, v, temperature, ρbulk)
    lB = e_c^2 / (4π * ϵ_0 * ϵr * LS_SIGMA * k_B * temperature)
    Zvals = p.Z.values
    κ_MSA_bulk = sqrt(4π * lB * sum(ρ★ .* abs.(Zvals) .* Zvals .^ 2))

    ω = structure_ω(structure, device, FP)
    # κ_MSA_bulk is dimensionless (built from lB and ρ★, both already
    # reduced), so 1/κ_MSA_bulk is ALREADY a σ-reduced length -- unlike
    # LS_SIGMA/2 (a raw physical length that still needs the /L reduction).
    width = fill(LS_SIGMA / (2L) + 1 / κ_MSA_bulk, length(Zvals))
    return (SWeightedDensity(:∫ρdz, width, ω, ngrid, device, L),)
end

function get_fields(model::LSIon, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}) where FP<:AbstractFloat
    L = FP(length_scale(model))
    return get_fields((model, L), species, structure, device, FP)
end

"""
Ions carry no independent connectivity of their own -- the beads' chain
connectivity is entirely handled by the neutral half's `TangentHSPropagator`,
acting on the same physical beads. Mirrors `DH.jl`'s `get_propagator`.
"""
function get_propagator(model::LSIon, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    return IdealPropagator()
end

length_scale(model::LSIon) = LS_SIGMA

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
    _ls_ion_params(model::LSIon, FP, width_vec, L, NF_neutral)

Mirrors `DH.jl`'s `preallocate_params(system::ElectrolyteDFTSystem, model::DHModel)`:
precomputes the bulk dielectric constant (`ConstRSP` only, so this is exact
-- no per-point recomputation needed), and per-bead `Z`/smoothing-`width`
tuples.

Additionally builds the per-bond charge-combination sign tuple
(`ion_bond_sign`) for the closed-form `±Δ(r)` chain-term correction: `+1`
for a same-charge bonded pair, `-1` for unlike-charge, `0` if either bead is
neutral. Mirrors `preallocate_params(::DFTSystem{<:LSNeutral})`'s
`bond_k`/`bond_l` construction exactly (each physical bond appears twice,
once per direction), so `f_bond_ion`'s `-ρ/2·sign·Δ` sum collapses to the
bulk closed form `ρ_chain·(1+2Npm+Nnc-N)·Δ` in the uniform limit.
"""
function _ls_ion_params(model::LSIon, FP, width_vec, L, NF_neutral)
    Z_vec = Float64.(model.params.Z.values)
    nc = length(Z_vec)

    Z_t = ntuple(i -> FP(Z_vec[i]), nc)
    w_t = ntuple(i -> FP(width_vec[i]), nc)

    bond_k_list = Int[]
    sign_list = Int[]
    for c in 1:length(model)
        i_groups_c = model.groups.i_groups[c]
        n_intergroups_c = model.groups.n_intergroups[c]
        for k in i_groups_c
            for l in findall(n_intergroups_c[k, :] .== 1)
                push!(bond_k_list, k)
                Zk, Zl = Z_vec[k], Z_vec[l]
                push!(sign_list, (Zk == 0 || Zl == 0) ? 0 : (Zk * Zl > 0 ? 1 : -1))
            end
        end
    end
    n_bonds = length(bond_k_list)
    bond_k_t = ntuple(ib -> bond_k_list[ib], n_bonds)
    bond_sign_t = ntuple(ib -> FP(sign_list[ib]), n_bonds)

    return (;
        ls_Z = Z_t,
        ls_width = w_t,
        ls_L = FP(L),
        ls_nf_neutral = Val(NF_neutral),
        ion_bond_k = bond_k_t,
        ion_bond_sign = bond_sign_t,
    )
end

function preallocate_params(system::ElectrolyteDFTSystem, model::LSIon)
    nd = dimension(system)
    FP = fptype(system.options)
    NF_neutral = compute_field_len(Base.front(system.fields), nd)
    temperature = system.structure.conditions[2]
    ρbulk_ion = system.ion_species.bulk_density
    eps_r = FP(dielectric_constant(model.RSPmodel, 1 / sum(ρbulk_ion), temperature, ρbulk_ion))
    L = length_scale(system.model)
    width_vec = last(system.fields).width

    base = _ls_ion_params(model, FP, width_vec, L, NF_neutral)
    return merge((; ls_eps_r = eps_r), base)
end

"""
    preallocate_params(system::DFTSystem{<:LSIon})

Standalone counterpart to the `ElectrolyteDFTSystem`-composite method above,
for testing/debugging `LSIon`'s electrostatic term in isolation (no neutral
model, no `TangentHSPropagator` -- `get_propagator(::LSIon,...)` returns
`IdealPropagator`, so a `DFTSystem{<:LSIon}` never invokes chain-bonding
machinery at all). The ion field is field 1 directly (`ls_nf_neutral=Val(0)`),
since there is no neutral field prepended.
"""
function preallocate_params(system::DFTSystem{<:LSIon})
    model = system.model
    FP = fptype(system.options)
    temperature = system.structure.conditions[2]
    ρbulk_ion = system.species.bulk_density
    eps_r = FP(dielectric_constant(model.RSPmodel, 1 / sum(ρbulk_ion), temperature, ρbulk_ion))
    L = length_scale(model)
    width_vec = system.fields[1].width

    base = _ls_ion_params(model, FP, width_vec, L, 0)
    params = merge((; ls_eps_r = eps_r), base)
    return params, length(model.params.Z.values)
end

@inline function f_res(::Type{M}, kk, out, n, params, T,
                        ::Val{NC}, ::Val{ND}) where {NC, ND, M <: LSIon}
    out[kk] += f_el_and_bond_ion(M, kk, n, params, T, Val(NC), params.ls_nf_neutral)
    return nothing
end

"""
GPU/Enzyme-compatible local MSA electrostatic free energy plus the per-bond
`±Δ(r)` chain correction at grid point `kk`.

`NF_NEUTRAL` locates the ion field (`F_ion = NF_NEUTRAL+1`) in `n`, exactly
like `f_dh`. Unlike `f_dh` -- which un-inflates `n[]` by `_NA = N_A*L^3`
because its physically-scaled χ(x)/κ formula runs in true SI units -- LS's
electrostatic formulas (`Γ_MSA`, `κ_MSA`) run entirely in the same
`ρ★=N_Azσ³/V`-reduced units the FMT/chain terms already use: `n[kk,F_ion,i]/(wi*2)`
and `n[kk,1,k]` already equal that reduced `ρ★` directly -- no `_NA`
conversion needed, matching the neutral model's convention.
"""
@inline function f_el_and_bond_ion(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{NF_NEUTRAL}) where {M, NC, NF_NEUTRAL}
    FP = eltype(n)
    F_ion = NF_NEUTRAL + 1
    ε_r = params.ls_eps_r
    _ec = FP(e_c); _kB = FP(k_B); _ϵ0 = FP(ϵ_0)
    lB = _ec * _ec / (4 * FP(π) * _ϵ0 * ε_r * LS_SIGMA * _kB * T)

    I = zero(FP)
    @inbounds for i in 1:NC
        Zi = _nti(params.ls_Z, i)
        wi = _nti(params.ls_width, i)
        ρi = n[kk, F_ion, i] / (wi * 2)
        I += ρi * abs(Zi) * Zi * Zi
    end
    κ_MSA = sqrt(max(4 * FP(π) * lB * I, zero(FP)))
    Γ_MSA = (-1 + sqrt(1 + 2κ_MSA)) / 2
    fel = -Γ_MSA * Γ_MSA * Γ_MSA * (FP(2) / 3 + Γ_MSA) / FP(π)

    Δ = lB * (1 - 1 / (1 + Γ_MSA)^2)
    res_bond = f_bond_ion(n, params.ion_bond_k, params.ion_bond_sign, kk, Δ)

    return fel + res_bond
end

@inline function f_bond_ion(n, bond_k::NTuple{NB, Int}, bond_sign::NTuple{NB}, kk, Δ) where NB
    FP = typeof(Δ)
    res = zero(FP)
    @inbounds for ib in 1:NB
        k = _nti(bond_k, ib)
        s = _nti(bond_sign, ib)
        ρk = n[kk, 1, k]
        res -= ρk / 2 * s * Δ
    end
    return res
end
