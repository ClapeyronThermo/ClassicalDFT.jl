import Clapeyron: getsites
import Enzyme.EnzymeRules

# ── GPU/Enzyme-compatible association support ───────────────────────────────

"""
    pack_assoc_params(model, HSd, sigma=model.params.sigma.values) -> NamedTuple

Pack Clapeyron's sparse association parameters into flat GPU-compatible vectors.
Returns flat vectors for each association pair and per-site data.

`sigma` defaults to the model's raw (physical-units) pair diameters, but callers that have
rescaled `HSd` into some other length unit (e.g. PC-SAFT's reduced-units path, which divides
`HSd` by `length_scale(model)`) must pass the identically-rescaled `sigma` too, so `assoc_sig3`
and `assoc_dij` stay dimensionally consistent with the `HSd`-derived weighted densities
`f_assoc` consumes them alongside.
"""
function pack_assoc_params(model, HSd, sigma=model.params.sigma.values)
    sites = Clapeyron.getsites(model)
    n_sites_per_comp = [length(sites.n_sites[i]) for i in 1:length(model)]
    total_sites      = sum(n_sites_per_comp)
    n_sites_cumsum   = cumsum([0; n_sites_per_comp])  # length NC+1

    assoc_icomp = Int[];   assoc_jcomp = Int[]
    assoc_isite = Int[];   assoc_jsite = Int[]
    assoc_eps   = Float64[]; assoc_kap = Float64[]
    assoc_sig3  = Float64[]; assoc_dij = Float64[]

    eps_vals = model.params.epsilon_assoc.values
    kap_vals = model.params.bondvol.values

    for idx in 1:length(eps_vals.values)
        i, j, a, b = idx_to_ijab(eps_vals, idx)
        push!(assoc_icomp, i); push!(assoc_jcomp, j)
        push!(assoc_isite, a); push!(assoc_jsite, b)
        push!(assoc_eps, eps_vals.values[idx])
        push!(assoc_kap, kap_vals.values[idx])
        push!(assoc_sig3, sigma[i,j]^3)
        di = HSd[i]; dj = HSd[j]
        push!(assoc_dij, (di*dj)/(di+dj))
    end

    # n_sites_flat[global_site_idx] = number of sites of that type
    n_sites_flat = Int[]
    for i in 1:length(model)
        ns_i = sites.n_sites[i]
        for a in 1:length(ns_i)
            push!(n_sites_flat, ns_i[a])
        end
    end

    return (assoc_icomp, assoc_jcomp, assoc_isite, assoc_jsite,
            assoc_eps, assoc_kap, assoc_sig3, assoc_dij,
            n_sites_flat, n_sites_cumsum, total_sites)
end

"""
    pack_assoc_params_gc(model, HSd, sigma=model.params.sigma.values)

Like `pack_assoc_params` but for group-contribution models (HeterogcPCPSAFT, SAFTgammaMie)
where the `(i,j)` component indices from `epsilon_assoc` are *species* (molecular) indices
rather than group/bead indices.  Uses `getsites(model).site_translator[i][a][1]` to map each (species i, site a) to
the global bead/group index, then builds `n_sites_cumsum` and `n_sites_flat` over BEADS.

Returns bead indices in `assoc_icomp/jcomp` (for n₀ indexing in `f_assoc`) and species
indices in `assoc_ispec/jspec` (needed by SAFTgammaMie `_assoc_delta` for Tr via
`params.epsilon_species`).

`sigma` defaults to the model's raw pair diameters; see `pack_assoc_params` for why
callers that rescale `HSd` must pass an identically-rescaled `sigma`.
"""
function pack_assoc_params_gc(model, HSd, sigma=model.params.sigma.values)
    sites    = Clapeyron.getsites(model)
    eps_vals = model.params.epsilon_assoc.values
    kap_vals = model.params.bondvol.values

    assoc_icomp = Int[]; assoc_jcomp = Int[]
    assoc_ispec = Int[]; assoc_jspec = Int[]
    assoc_isite = Int[]; assoc_jsite = Int[]
    assoc_eps   = Float64[]; assoc_kap = Float64[]
    assoc_sig3  = Float64[]; assoc_dij  = Float64[]

    nc_groups       = length(HSd)
    bead_site_count = zeros(Int, nc_groups)
    bead_local_idx  = Dict{Tuple{Int,Int}, Int}()
    site_to_bead    = Dict{Tuple{Int,Int}, Int}()

    for idx in 1:length(eps_vals.values)
        i, j, a, b = idx_to_ijab(eps_vals, idx)
        k = sites.site_translator[i][a][1]
        l = sites.site_translator[j][b][1]
        if !haskey(site_to_bead, (i, a))
            site_to_bead[(i, a)] = k
            bead_site_count[k] += 1
            bead_local_idx[(i, a)] = bead_site_count[k]
        end
        if !haskey(site_to_bead, (j, b))
            site_to_bead[(j, b)] = l
            bead_site_count[l] += 1
            bead_local_idx[(j, b)] = bead_site_count[l]
        end
    end

    n_sites_cumsum = cumsum([0; bead_site_count])
    n_sites_flat   = zeros(Int, max(1, sum(bead_site_count)))
    for ((i, a), k) in site_to_bead
        gidx = n_sites_cumsum[k] + bead_local_idx[(i, a)]
        n_sites_flat[gidx] = sites.n_sites[i][a]
    end
    total_sites = sum(bead_site_count)

    for idx in 1:length(eps_vals.values)
        i, j, a, b = idx_to_ijab(eps_vals, idx)
        k = sites.site_translator[i][a][1]
        l = sites.site_translator[j][b][1]
        push!(assoc_icomp, k);  push!(assoc_jcomp, l)
        push!(assoc_ispec, i);  push!(assoc_jspec, j)
        push!(assoc_isite, bead_local_idx[(i, a)])
        push!(assoc_jsite, bead_local_idx[(j, b)])
        push!(assoc_eps,   eps_vals.values[idx])
        push!(assoc_kap,   kap_vals.values[idx])
        push!(assoc_sig3,  sigma[k, l]^3)
        dk = HSd[k]; dl = HSd[l]
        push!(assoc_dij, (dk*dl)/(dk+dl))
    end

    return (assoc_icomp, assoc_jcomp, assoc_ispec, assoc_jspec,
            assoc_isite, assoc_jsite,
            assoc_eps, assoc_kap, assoc_sig3, assoc_dij,
            n_sites_flat, n_sites_cumsum, total_sites)
end

# Dispatch helper: zero-cost when model has no association (hasfield is compile-time constant)
@inline _assoc_or_zero(M, kk, n, params::P, T, vNC, vND) where P =
    hasfield(P, :assoc_eps) ? f_assoc(M, kk, n, params, T, vNC, vND,
                                      params.assoc_n_pairs, params.assoc_n_sites) : zero(eltype(n))

# ── GPU/Enzyme-compatible kernel helpers ────────────────────────────────────

# GPU-safe NTuple indexing with a runtime integer.
# Julia's default NTuple getindex with a runtime Int calls jl_get_nth_field_checked,
# which is unsupported in PTX kernels.  This @generated helper expands to an explicit
# ifelse chain so the GPU compiler sees only conditional moves — no dynamic dispatch.
@generated function _nti(t::NTuple{N,T}, i::Int) where {N,T}
    ex = :(t[1])
    for k in 2:N
        ex = :(ifelse(i == $k, t[$k], $ex))
    end
    ex
end

# Default Wertheim association strength Δ: g_hs-based formula.
# Used by PCSAFT, PCPSAFT, QPCPSAFT, HomogcPCPSAFT (all same Δ formula).
# Model-specific overrides (e.g. SAFTVRMieModel) use a more specific M dispatch.
# The n, kk, NC, ND, M arguments are unused here but required for a uniform signature.
@inline function _assoc_delta(p, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix,
                               ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M}
    FP = typeof(n3_mix)
    p > n_pairs && return zero(FP)
    dij_p  = params.assoc_dij[p]
    inv1n3 = one(FP) / (one(FP) - n3_mix)
    g_hs   = inv1n3 + FP(0.5)*dij_p*xi_mix*n2_mix*inv1n3^2 +
             dij_p*dij_p*n2_mix*n2_mix*xi_mix*(inv1n3^3)/18
    return g_hs * params.assoc_sig3[p] * (exp(params.assoc_eps[p]/T)-1) * params.assoc_kap[p]
end

# ── @generated tuple-construction helpers ────────────────────────────────────
#
# f_assoc(::Val{NP}) is a regular @inline (not @generated) so Enzyme can inline
# and trace it without hitting the runtime_generic_fwd JIT path. But closures
# (ntuple ... do) box captured mutable variables on the GPU heap. We avoid
# closures entirely by delegating tuple construction to these @generated helpers,
# which generate flat, closure-free, GPU-safe code. Mutated scalars (n3_mix etc.)
# are passed as arguments — not captured — so no boxing occurs.

# Per-component number density n₀ᵢ = ρᵢ/dᵢ = n[kk,2,i]/HSd[i]
@generated function _assoc_n0(n, kk, params, ::Val{NC}) where NC
    stmts = [:($(Symbol("_n0_$i")) = n[kk, 2, $i] / params.HSd[$i]) for i in 1:NC]
    return quote
        $(stmts...)
        tuple($([ Symbol("_n0_$i") for i in 1:NC ]...))
    end
end

# Per-component geometry factor ξᵢ = 1 − |nv2ᵢ|²/n2ᵢ²  (FV=4 always)
@generated function _assoc_xi(n, kk, params, ::Val{NC}, ::Val{ND}) where {NC, ND}
    FP     = eltype(n)
    z      = zero(FP)
    o      = one(FP)
    two_pi = FP(2π)
    stmts = Expr[]
    for i in 1:NC
        n2v  = Symbol("_xi_n2_$i")
        sqv  = Symbol("_xi_sq_$i")
        xiv  = Symbol("_xi_$i")
        push!(stmts, :($n2v = π * n[kk, 2, $i] * params.HSd[$i]))
        push!(stmts, :($sqv = $z))
        if ND >= 1
            ndv = Symbol("_xi_nd_$(i)_1")
            push!(stmts, :($ndv = -$(two_pi) * n[kk, 4, $i]))
            push!(stmts, :($sqv += $ndv * $ndv))
        end
        if ND >= 2
            ndv = Symbol("_xi_nd_$(i)_2")
            push!(stmts, :($ndv = -$(two_pi) * n[kk, 5, $i]))
            push!(stmts, :($sqv += $ndv * $ndv))
        end
        if ND >= 3
            ndv = Symbol("_xi_nd_$(i)_3")
            push!(stmts, :($ndv = -$(two_pi) * n[kk, 6, $i]))
            push!(stmts, :($sqv += $ndv * $ndv))
        end
        push!(stmts, :($xiv = $o - $sqv / ($n2v * $n2v)))
    end
    return quote
        $(stmts...)
        tuple($([ Symbol("_xi_$i") for i in 1:NC ]...))
    end
end

# ── Diagnostics: mixture-level ξ, not otherwise exposed outside f_assoc ─────
#
# Plain (non-`@generated`, non-AD-traced) duplicate of the ξ_mix formula inlined in
# both f_assoc methods below — kept as an independent copy rather than refactored into
# a shared helper so it carries zero risk of touching the Enzyme-differentiated path.
# For diagnostic use (e.g. `xi_profile`) only.
function _xi_mix_at(n, kk, params, ::Val{NC}, ::Val{ND}) where {NC, ND}
    F2  = 2; FV = 4
    FP  = eltype(n)
    _2π = FP(2π)
    n2_mix = zero(FP); nv2sq_mix = zero(FP)
    @inbounds for i in 1:NC
        n2_mix += π * n[kk, F2, i] * params.m[i] * params.HSd[i]
    end
    if ND >= 1
        nv2d = zero(FP)
        @inbounds for i in 1:NC; nv2d -= _2π * n[kk, FV,   i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    if ND >= 2
        nv2d = zero(FP)
        @inbounds for i in 1:NC; nv2d -= _2π * n[kk, FV+1, i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    if ND >= 3
        nv2d = zero(FP)
        @inbounds for i in 1:NC; nv2d -= _2π * n[kk, FV+2, i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    return one(FP) - nv2sq_mix / (n2_mix * n2_mix)
end

"""
    xi_profile(system::DFTSystem, ρ) -> (; xi_mix, xi_groups, assoc_group_idx)

Compute the Rosenfeld/Wertheim FMT anisotropy factor `ξ = 1 - |n̄v2|²/n̄2²` at every
grid point of a converged (or trial) density profile `ρ`, for diagnostic/plotting
purposes. For the *true* continuous weighted densities `ξ ∈ [0,1]`; grid-discretization
error can push it outside that range (most visibly near sharp density features), which
is what this is for inspecting.

Returns a named tuple:
- `xi_mix`: `Vector`/`Array` of length `ngrid`, the single mixture-level ξ that feeds
  every association pair's bonding strength Δ (the quantity used inside `_assoc_delta`).
- `xi_groups`: `ngrid × NC` array, the per-field/per-group ξ (one column per bead type
  in `system.model`, e.g. one column per SAFTγMie group) — the same quantity `f_assoc`
  computes via `_assoc_xi` for each site-bearing group individually.
- `assoc_group_idx`: the field/group indices (`1:NC`, matching `xi_groups`' columns)
  that actually participate in at least one association pair — i.e. the columns of
  `xi_groups` worth plotting, without having to know the model's group ordering.

Errors if `system.model` has no association sites (`ξ` is only defined/used there).
Read-only: does not mutate `ρ` or `system`, and does not touch the Enzyme-differentiated
`f_res`/`f_assoc` code path at all (this recomputes ξ independently — see `_xi_mix_at`).
"""
function xi_profile(system::DFTSystem, ρ)
    n, _, fft_buf, in_buf, out_buf, plan, iplan, params, _, _, nc, nd, _ = preallocate_model(system, ρ)
    hasproperty(params, :has_assoc) && params.has_assoc || error(
        "xi_profile: system.model has no association sites — ξ is not defined/used.")
    assoc_group_idx = sort(unique(vcat(collect(params.assoc_icomp), collect(params.assoc_jcomp))))
    evaluate_field!(system, ρ, fft_buf, in_buf, out_buf, plan, iplan)
    copyto!(n, fft_buf)

    ngrid = system.structure.ngrid
    FP = eltype(n)
    xi_mix    = allocate(system.options.device, FP, ngrid...)
    xi_groups = allocate(system.options.device, FP, ngrid..., nc)
    for kk in CartesianIndices(ngrid)
        xi_mix[kk] = _xi_mix_at(n, kk, params, Val(nc), Val(nd))
        xi_i = _assoc_xi(n, kk, params, Val(nc), Val(nd))
        for i in 1:nc
            xi_groups[kk, i] = xi_i[i]
        end
    end
    return (; xi_mix, xi_groups, assoc_group_idx)
end

# NP explicit Δ evaluations; n3_mix/n2_mix/xi_mix passed as args to avoid boxing
@generated function _assoc_delta_vals(n, params, T, kk, n3_mix, n2_mix, xi_mix,
                                       ::Val{NC}, ::Val{ND}, ::Val{NP}, ::Type{M}) where {NC, ND, NP, M}
    stmts = [:($(Symbol("_Δ$p")) = _assoc_delta($p, $NP, n, params, T, kk,
                                                  n3_mix, n2_mix, xi_mix,
                                                  Val($NC), Val($ND), M))
             for p in 1:NP]
    return quote
        $(stmts...)
        tuple($([ Symbol("_Δ$p") for p in 1:NP ]...))
    end
end

# Initial X₀ = (0.9, …, 0.9) with NS elements, typed as FP
@generated function _assoc_X0(::Val{NS}, ::Type{FP}) where {NS, FP<:AbstractFloat}
    v = FP(0.9)
    :(tuple($(fill(v, NS)...)))
end

# ── @generated successive-substitution step ─────────────────────────────────
#
# Generates exactly NS explicit site updates with literal pair-index p — no
# closures, no runtime NTuple indexing for the outer pair loop. GPU-safe.
@generated function _assoc_SS_step(X::NTuple{NS,<:AbstractFloat},
                                    n0::NTuple{NC,<:AbstractFloat},
                                    xi::NTuple{NC,<:AbstractFloat},
                                    Δ_vals::NTuple{NP,<:AbstractFloat},
                                    params) where {NS, NC, NP}
    FP = eltype(X)
    z  = zero(FP)
    o  = one(FP)
    h  = FP(0.5)
    stmts = Expr[]
    for s in 1:NS
        denom_parts = Expr[]
        for p in 1:NP
            push!(denom_parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                params.assoc_n_jb[$p] *
                _nti(n0, params.assoc_jcomp[$p]) * _nti(xi, params.assoc_jcomp[$p]) *
                Δ_vals[$p] * _nti(X, params.assoc_jb_global[$p]),
                $z)))
            push!(denom_parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                params.assoc_n_ia[$p] *
                _nti(n0, params.assoc_icomp[$p]) * _nti(xi, params.assoc_icomp[$p]) *
                Δ_vals[$p] * _nti(X, params.assoc_ia_global[$p]),
                $z)))
        end
        denom_expr = reduce((a, b) -> :($a + $b), denom_parts)
        new_x = Symbol("new_x_$s")
        blend = Symbol("blend_$s")
        push!(stmts, :($new_x = $o / ($o + $denom_expr)))
        push!(stmts, :($blend = $h * (X[$s] + $new_x)))
    end
    result = :(tuple($([ Symbol("blend_$s") for s in 1:NS ]...)))
    return quote
        $(stmts...)
        $result
    end
end

# ── @generated safeguarded Newton-Raphson step ──────────────────────────────
#
# Builds the NS×NS Wertheim Jacobian in flat scalar code (no closures, GPU-safe),
# solves J*δX = F via Gaussian elimination (no pivoting; J is diagonally dominant
# near the fixed point since J_{ss} = 1 + C_s ≥ 1), then applies the update.
# If X_s - δX_s ∉ (0,1) for any site, that site falls back to the unblended SS
# update — same safeguard as Clapeyron's assoc_matrix_solve_general.
# Called FROM the regular @inline f_assoc so Enzyme traces f_assoc directly.
@generated function _assoc_newton_step(X::NTuple{NS,<:AbstractFloat},
                                        n0::NTuple{NC,<:AbstractFloat},
                                        xi::NTuple{NC,<:AbstractFloat},
                                        Δ_vals::NTuple{NP,<:AbstractFloat},
                                        params) where {NS, NC, NP}
    FP = eltype(X)
    z  = zero(FP)
    o  = one(FP)
    stmts = Expr[]
    cnt = Ref(0)
    new_sym() = (cnt[] += 1; Symbol("_nt_$(cnt[])"))

    # Step A: per-site denominator C_s and unblended SS fallback x_ss_s
    C_syms  = [new_sym() for _ in 1:NS]
    ss_syms = [new_sym() for _ in 1:NS]
    for s in 1:NS
        parts = Expr[]
        for p in 1:NP
            push!(parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                params.assoc_n_jb[$p] *
                _nti(n0, params.assoc_jcomp[$p]) * _nti(xi, params.assoc_jcomp[$p]) *
                Δ_vals[$p] * _nti(X, params.assoc_jb_global[$p]), $z)))
            push!(parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                params.assoc_n_ia[$p] *
                _nti(n0, params.assoc_icomp[$p]) * _nti(xi, params.assoc_icomp[$p]) *
                Δ_vals[$p] * _nti(X, params.assoc_ia_global[$p]), $z)))
        end
        C_expr = reduce((a, b) -> :($a + $b), parts)
        push!(stmts, :($(C_syms[s]) = $C_expr))
        push!(stmts, :($(ss_syms[s]) = $o / ($o + $(C_syms[s]))))
    end

    # Step B: residuals F_s = X_s*(1 + C_s) - 1
    F_syms = [new_sym() for _ in 1:NS]
    for s in 1:NS
        push!(stmts, :($(F_syms[s]) = X[$s] * ($o + $(C_syms[s])) - $o))
    end

    # Step C: Jacobian J_{st} = (1+C_s)*δ_{st} + X_s * ∂C_s/∂X_t
    J_sym = [new_sym() for s in 1:NS, t in 1:NS]
    for s in 1:NS, t in 1:NS
        k_parts = Expr[]
        for p in 1:NP
            push!(k_parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                ifelse(params.assoc_jb_global[$p] == $t,
                    params.assoc_n_jb[$p] *
                    _nti(n0, params.assoc_jcomp[$p]) * _nti(xi, params.assoc_jcomp[$p]) *
                    Δ_vals[$p], $z), $z)))
            push!(k_parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                ifelse(params.assoc_ia_global[$p] == $t,
                    params.assoc_n_ia[$p] *
                    _nti(n0, params.assoc_icomp[$p]) * _nti(xi, params.assoc_icomp[$p]) *
                    Δ_vals[$p], $z), $z)))
        end
        K_st  = reduce((a, b) -> :($a + $b), k_parts)
        diag  = s == t ? :($o + $(C_syms[s])) : :($z)
        push!(stmts, :($(J_sym[s,t]) = $diag + X[$s] * $K_st))
    end

    # Step D: Gaussian elimination on augmented matrix [J | F]
    J_aug = Matrix{Symbol}(undef, NS, NS + 1)
    for s in 1:NS
        for t in 1:NS; J_aug[s,t] = J_sym[s,t]; end
        J_aug[s, NS+1] = F_syms[s]
    end
    for k in 1:NS
        for i in k+1:NS
            f = new_sym()
            push!(stmts, :($f = $(J_aug[i,k]) / $(J_aug[k,k])))
            for j in k+1:NS+1
                t = new_sym()
                push!(stmts, :($t = $(J_aug[i,j]) - $f * $(J_aug[k,j])))
                J_aug[i,j] = t
            end
        end
    end

    # Step E: back substitution → δX
    δX = Vector{Symbol}(undef, NS)
    for k in NS:-1:1
        rhs = J_aug[k, NS+1]
        for j in k+1:NS
            t = new_sym()
            push!(stmts, :($t = $rhs - $(J_aug[k,j]) * $(δX[j])))
            rhs = t
        end
        δX[k] = new_sym()
        push!(stmts, :($(δX[k]) = $rhs / $(J_aug[k,k])))
    end

    # Step F: accept Newton if X_s - δX_s ∈ (0,1), else fall back to SS
    res = [new_sym() for _ in 1:NS]
    for s in 1:NS
        cand = new_sym()
        push!(stmts, :($cand = X[$s] - $(δX[s])))
        push!(stmts, :($(res[s]) = ifelse($cand > $z,
            ifelse($cand < $o, $cand, $(ss_syms[s])),
            $(ss_syms[s]))))
    end

    return quote
        $(stmts...)
        tuple($(res...))
    end
end

# ── Wertheim fixed-point solver (IFT rule registered below) ─────────────────
#
# Extracted from f_assoc so that EnzymeRules.forward can intercept the
# gradient of X* w.r.t. (n0, xi, Δ_vals) via the Implicit Function Theorem.
# Enzyme differentiates _assoc_n0/_assoc_xi/_assoc_delta_vals → (n0,xi,Δ)
# normally; the IFT rule fires here and replaces the iteration-based gradient
# with a single solve: δX = -J⁻¹ · (∂F/∂n0·δn0 + ∂F/∂xi·δxi + ∂F/∂Δ·δΔ).
@inline function _assoc_solve(n0::NTuple{NC,<:AbstractFloat}, xi::NTuple{NC,<:AbstractFloat},
                               Δ_vals::NTuple{NP,<:AbstractFloat}, params,
                               ::Val{NS}) where {NC, NP, NS}
    FP = eltype(n0)
    X = _assoc_X0(Val(NS), FP)
    for _ in 1:5*NS; X = _assoc_SS_step(X, n0, xi, Δ_vals, params); end
    for _ in 1:5*NS; X = _assoc_newton_step(X, n0, xi, Δ_vals, params); end
    X
end

# ── @generated IFT tangent: δX = -J⁻¹ · δF ─────────────────────────────────
#
# Computes the IFT directional derivative of X* w.r.t. perturbations
# (δn0, δxi, δΔ) of the solver inputs.  Entirely flat scalar code (GPU-safe).
#
# J_{st}   = ∂F_s/∂X_t at the fixed point X* (same Jacobian as Newton step).
# δF_s     = X*_s · (∂C_s/∂n0·δn0 + ∂C_s/∂xi·δxi + ∂C_s/∂Δ·δΔ) — analytical.
# Solve    : J · δX = -δF  via Gaussian elimination (no pivoting, J diag-dom).
@generated function _assoc_ift_tangent(
        X_star::NTuple{NS,<:AbstractFloat},
        n0::NTuple{NC,<:AbstractFloat},  xi::NTuple{NC,<:AbstractFloat},  Δ_vals::NTuple{NP,<:AbstractFloat},
        dn0::NTuple{NC,<:AbstractFloat}, dxi::NTuple{NC,<:AbstractFloat}, dΔ::NTuple{NP,<:AbstractFloat},
        params) where {NS, NC, NP}
    FP = eltype(X_star)
    z  = zero(FP)
    o  = one(FP)
    stmts = Expr[]
    cnt = Ref(0)
    new_sym() = (cnt[] += 1; Symbol("_ift_$(cnt[])"))

    # Step A: C_s (denominator sum) for the Jacobian diagonal 1+C_s
    C_syms = [new_sym() for _ in 1:NS]
    for s in 1:NS
        parts = Expr[]
        for p in 1:NP
            push!(parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                params.assoc_n_jb[$p] *
                _nti(n0, params.assoc_jcomp[$p]) * _nti(xi, params.assoc_jcomp[$p]) *
                Δ_vals[$p] * _nti(X_star, params.assoc_jb_global[$p]), $z)))
            push!(parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                params.assoc_n_ia[$p] *
                _nti(n0, params.assoc_icomp[$p]) * _nti(xi, params.assoc_icomp[$p]) *
                Δ_vals[$p] * _nti(X_star, params.assoc_ia_global[$p]), $z)))
        end
        push!(stmts, :($(C_syms[s]) = $(reduce((a,b)->:($a+$b), parts))))
    end

    # Step B: Jacobian J_{st} = (1+C_s)·δ_{st} + X*_s · ∂C_s/∂X_t
    J_sym = [new_sym() for s in 1:NS, t in 1:NS]
    for s in 1:NS, t in 1:NS
        k_parts = Expr[]
        for p in 1:NP
            push!(k_parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                ifelse(params.assoc_jb_global[$p] == $t,
                    params.assoc_n_jb[$p] *
                    _nti(n0, params.assoc_jcomp[$p]) * _nti(xi, params.assoc_jcomp[$p]) *
                    Δ_vals[$p], $z), $z)))
            push!(k_parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                ifelse(params.assoc_ia_global[$p] == $t,
                    params.assoc_n_ia[$p] *
                    _nti(n0, params.assoc_icomp[$p]) * _nti(xi, params.assoc_icomp[$p]) *
                    Δ_vals[$p], $z), $z)))
        end
        K_st = reduce((a,b)->:($a+$b), k_parts)
        diag = s == t ? :($o + $(C_syms[s])) : :($z)
        push!(stmts, :($(J_sym[s,t]) = $diag + X_star[$s] * $K_st))
    end

    # Step C: IFT residual δF_s = X*_s · δC_s
    # δC_s = Σ_p [ia[p]==s: n_jb·X*[jb]·(δn0[jc]·xi[jc] + n0[jc]·δxi[jc])·Δ + n_jb·n0[jc]·xi[jc]·X*[jb]·δΔ]
    #             [jb[p]==s: symmetric with ia↔jb, icomp↔jcomp]
    dF_syms = [new_sym() for _ in 1:NS]
    for s in 1:NS
        parts = Expr[]
        for p in 1:NP
            push!(parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                params.assoc_n_jb[$p] * _nti(X_star, params.assoc_jb_global[$p]) *
                (_nti(dn0, params.assoc_jcomp[$p]) * _nti(xi,  params.assoc_jcomp[$p]) * Δ_vals[$p]
               + _nti(n0,  params.assoc_jcomp[$p]) * _nti(dxi, params.assoc_jcomp[$p]) * Δ_vals[$p]
               + _nti(n0,  params.assoc_jcomp[$p]) * _nti(xi,  params.assoc_jcomp[$p]) * dΔ[$p]),
                $z)))
            push!(parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                params.assoc_n_ia[$p] * _nti(X_star, params.assoc_ia_global[$p]) *
                (_nti(dn0, params.assoc_icomp[$p]) * _nti(xi,  params.assoc_icomp[$p]) * Δ_vals[$p]
               + _nti(n0,  params.assoc_icomp[$p]) * _nti(dxi, params.assoc_icomp[$p]) * Δ_vals[$p]
               + _nti(n0,  params.assoc_icomp[$p]) * _nti(xi,  params.assoc_icomp[$p]) * dΔ[$p]),
                $z)))
        end
        dC = reduce((a,b)->:($a+$b), parts)
        push!(stmts, :($(dF_syms[s]) = X_star[$s] * $dC))
    end

    # Step D: Gaussian elimination on augmented matrix [J | -δF]
    # Initialise RHS as -δF so back-substitution directly gives δX = -J⁻¹δF.
    J_aug = Matrix{Symbol}(undef, NS, NS + 1)
    for s in 1:NS
        for t in 1:NS; J_aug[s,t] = J_sym[s,t]; end
        neg_dF = new_sym()
        push!(stmts, :($neg_dF = -($(dF_syms[s]))))
        J_aug[s, NS+1] = neg_dF
    end
    for k in 1:NS
        for i in k+1:NS
            f = new_sym()
            push!(stmts, :($f = $(J_aug[i,k]) / $(J_aug[k,k])))
            for j in k+1:NS+1
                t = new_sym()
                push!(stmts, :($t = $(J_aug[i,j]) - $f * $(J_aug[k,j])))
                J_aug[i,j] = t
            end
        end
    end

    # Step E: back substitution → δX
    δX = Vector{Symbol}(undef, NS)
    for k in NS:-1:1
        rhs = J_aug[k, NS+1]
        for j in k+1:NS
            t = new_sym()
            push!(stmts, :($t = $rhs - $(J_aug[k,j]) * $(δX[j])))
            rhs = t
        end
        δX[k] = new_sym()
        push!(stmts, :($(δX[k]) = $rhs / $(J_aug[k,k])))
    end

    return quote
        $(stmts...)
        tuple($(δX...))
    end
end

# Reverse-mode IFT: given cotangent seed = ∂L/∂X*, compute ∂L/∂(n0,xi,Δ).
# Solve J^T·λ = seed (note transpose — J is NOT symmetric), then:
#   ∂L/∂θ_k = -∑_s λ_s · ∂F_s/∂θ_k
@generated function _assoc_ift_cotangents(
        X_star::NTuple{NS,<:AbstractFloat},
        n0::NTuple{NC,<:AbstractFloat}, xi::NTuple{NC,<:AbstractFloat}, Δ_vals::NTuple{NP,<:AbstractFloat},
        seed::NTuple{NS,<:AbstractFloat},
        params) where {NS, NC, NP}
    FP = eltype(X_star)
    z  = zero(FP)
    o  = one(FP)
    stmts = Expr[]
    cnt = Ref(0)
    new_sym() = (cnt[] += 1; Symbol("_ct_$(cnt[])"))

    # Step A: C_s (same as _assoc_ift_tangent — diagonal of J)
    C_syms = [new_sym() for _ in 1:NS]
    for s in 1:NS
        parts = Expr[]
        for p in 1:NP
            push!(parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                params.assoc_n_jb[$p] *
                _nti(n0, params.assoc_jcomp[$p]) * _nti(xi, params.assoc_jcomp[$p]) *
                Δ_vals[$p] * _nti(X_star, params.assoc_jb_global[$p]), $z)))
            push!(parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                params.assoc_n_ia[$p] *
                _nti(n0, params.assoc_icomp[$p]) * _nti(xi, params.assoc_icomp[$p]) *
                Δ_vals[$p] * _nti(X_star, params.assoc_ia_global[$p]), $z)))
        end
        push!(stmts, :($(C_syms[s]) = $(reduce((a,b)->:($a+$b), parts))))
    end

    # Step B: all NS² J_{s,t} entries (same as _assoc_ift_tangent)
    J_sym = [new_sym() for s in 1:NS, t in 1:NS]
    for s in 1:NS, t in 1:NS
        k_parts = Expr[]
        for p in 1:NP
            push!(k_parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                ifelse(params.assoc_jb_global[$p] == $t,
                    params.assoc_n_jb[$p] *
                    _nti(n0, params.assoc_jcomp[$p]) * _nti(xi, params.assoc_jcomp[$p]) *
                    Δ_vals[$p], $z), $z)))
            push!(k_parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                ifelse(params.assoc_ia_global[$p] == $t,
                    params.assoc_n_ia[$p] *
                    _nti(n0, params.assoc_icomp[$p]) * _nti(xi, params.assoc_icomp[$p]) *
                    Δ_vals[$p], $z), $z)))
        end
        K_st = reduce((a,b)->:($a+$b), k_parts)
        diag = s == t ? :($o + $(C_syms[s])) : :($z)
        push!(stmts, :($(J_sym[s,t]) = $diag + X_star[$s] * $K_st))
    end

    # Step C: Gaussian elimination on [J^T | seed]
    # J^T[t,s] = J[s,t] → augmented row t is [J[1,t], J[2,t], ..., J[NS,t] | seed[t]]
    J_aug = Matrix{Symbol}(undef, NS, NS + 1)
    for t in 1:NS
        for s in 1:NS; J_aug[t,s] = J_sym[s,t]; end
        seed_t = new_sym()
        push!(stmts, :($seed_t = seed[$t]))
        J_aug[t, NS+1] = seed_t
    end
    for k in 1:NS
        for i in k+1:NS
            f = new_sym()
            push!(stmts, :($f = $(J_aug[i,k]) / $(J_aug[k,k])))
            for j in k+1:NS+1
                t = new_sym()
                push!(stmts, :($t = $(J_aug[i,j]) - $f * $(J_aug[k,j])))
                J_aug[i,j] = t
            end
        end
    end

    # Step D: back-substitution → λ (adjoint variable)
    λ = Vector{Symbol}(undef, NS)
    for k in NS:-1:1
        rhs = J_aug[k, NS+1]
        for j in k+1:NS
            tmp = new_sym()
            push!(stmts, :($tmp = $rhs - $(J_aug[k,j]) * $(λ[j])))
            rhs = tmp
        end
        λ[k] = new_sym()
        push!(stmts, :($(λ[k]) = $rhs / $(J_aug[k,k])))
    end

    # Step E: dn0_i = -∑_{p,s} [ia[p]==s && jcomp[p]==i] n_jb·xi[jc]·Δ[p]·X*[jb]·λ[s]·X*[s]
    #                         + [jb[p]==s && icomp[p]==i] n_ia·xi[ic]·Δ[p]·X*[ia]·λ[s]·X*[s]
    dn0_syms = [new_sym() for _ in 1:NC]
    for i in 1:NC
        parts = Expr[]
        for p in 1:NP, s in 1:NS
            push!(parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                ifelse(params.assoc_jcomp[$p] == $i,
                    params.assoc_n_jb[$p] * _nti(xi, params.assoc_jcomp[$p]) *
                    Δ_vals[$p] * _nti(X_star, params.assoc_jb_global[$p]) *
                    $(λ[s]) * X_star[$s], $z), $z)))
            push!(parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                ifelse(params.assoc_icomp[$p] == $i,
                    params.assoc_n_ia[$p] * _nti(xi, params.assoc_icomp[$p]) *
                    Δ_vals[$p] * _nti(X_star, params.assoc_ia_global[$p]) *
                    $(λ[s]) * X_star[$s], $z), $z)))
        end
        push!(stmts, :($(dn0_syms[i]) = -($(reduce((a,b)->:($a+$b), parts)))))
    end

    # Step F: dxi_i — same structure, n0↔xi
    dxi_syms = [new_sym() for _ in 1:NC]
    for i in 1:NC
        parts = Expr[]
        for p in 1:NP, s in 1:NS
            push!(parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                ifelse(params.assoc_jcomp[$p] == $i,
                    params.assoc_n_jb[$p] * _nti(n0, params.assoc_jcomp[$p]) *
                    Δ_vals[$p] * _nti(X_star, params.assoc_jb_global[$p]) *
                    $(λ[s]) * X_star[$s], $z), $z)))
            push!(parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                ifelse(params.assoc_icomp[$p] == $i,
                    params.assoc_n_ia[$p] * _nti(n0, params.assoc_icomp[$p]) *
                    Δ_vals[$p] * _nti(X_star, params.assoc_ia_global[$p]) *
                    $(λ[s]) * X_star[$s], $z), $z)))
        end
        push!(stmts, :($(dxi_syms[i]) = -($(reduce((a,b)->:($a+$b), parts)))))
    end

    # Step G: dΔ_p — sum over s for each pair p
    dΔ_syms = [new_sym() for _ in 1:NP]
    for p in 1:NP
        parts = Expr[]
        for s in 1:NS
            push!(parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                params.assoc_n_jb[$p] * _nti(n0, params.assoc_jcomp[$p]) *
                _nti(xi, params.assoc_jcomp[$p]) *
                _nti(X_star, params.assoc_jb_global[$p]) * $(λ[s]) * X_star[$s], $z)))
            push!(parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                params.assoc_n_ia[$p] * _nti(n0, params.assoc_icomp[$p]) *
                _nti(xi, params.assoc_icomp[$p]) *
                _nti(X_star, params.assoc_ia_global[$p]) * $(λ[s]) * X_star[$s], $z)))
        end
        push!(stmts, :($(dΔ_syms[p]) = -($(reduce((a,b)->:($a+$b), parts)))))
    end

    return quote
        $(stmts...)
        (tuple($(dn0_syms...)), tuple($(dxi_syms...)), tuple($(dΔ_syms...)))
    end
end

"""
Wertheim association free energy density at grid point `kk`.

The `::Val{1}` specialisation uses the analytical quadratic formula — pure scalar
arithmetic, fully GPU-safe with Enzyme forward mode. Dispatch via `params.assoc_n_pairs`
which is stored as `Val(nn)` in preallocate_params.

The `::Val{NP}` general method runs `5*NS` relaxed SS warm-up then `5*NS` Newton steps.
`f_assoc` is a regular `@inline` (not `@generated`) so Enzyme can inline and trace it
without hitting `runtime_generic_fwd`. NTuple construction uses `@generated` helpers
(`_assoc_n0`, `_assoc_xi`, `_assoc_delta_vals`, `_assoc_X0`, `_assoc_newton_step`) so
no closures appear in `f_assoc` itself — closures that capture mutated locals cause GPU
heap allocation.

`_assoc_delta` is dispatched on `::Type{M}` so each model can supply its own Δ formula.
"""
@inline function f_assoc(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND},
                          ::Val{1}, ::Val{NS}) where {NC, ND, NS, M}
    F2    = 2
    FV    = F2 + 2
    FP    = eltype(n)
    _2π   = FP(2π)

    # Mixture FMT densities
    n2_mix    = zero(FP)
    n3_mix    = zero(FP)
    nv2sq_mix = zero(FP)
    @inbounds for i in 1:NC
        nim = n[kk, F2, i] * params.m[i]
        n2_mix += π * nim * params.HSd[i]
        n3_mix += n[kk, F2+1, i] * params.m[i]
    end
    if ND >= 1
        nv2d = zero(FP)
        @inbounds for i in 1:NC; nv2d -= _2π * n[kk, FV,   i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    if ND >= 2
        nv2d = zero(FP)
        @inbounds for i in 1:NC; nv2d -= _2π * n[kk, FV+1, i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    if ND >= 3
        nv2d = zero(FP)
        @inbounds for i in 1:NC; nv2d -= _2π * n[kk, FV+2, i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    xi_mix = one(FP) - nv2sq_mix / (n2_mix*n2_mix)

    # ── Analytical 1-pair solution ─────────────────────────────────────────
    ic  = params.assoc_icomp[1];  jc  = params.assoc_jcomp[1]

    Δ_val = _assoc_delta(1, 1, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)

    n0_ic    = n[kk, F2, ic] / params.HSd[ic]
    n2_ic    = π * n[kk, F2, ic] * params.HSd[ic]
    nv2sq_ic = zero(FP)
    if ND >= 1; nd = -_2π*n[kk, FV,   ic]; nv2sq_ic += nd*nd; end
    if ND >= 2; nd = -_2π*n[kk, FV+1, ic]; nv2sq_ic += nd*nd; end
    if ND >= 3; nd = -_2π*n[kk, FV+2, ic]; nv2sq_ic += nd*nd; end
    xi_ic = one(FP) - nv2sq_ic / (n2_ic*n2_ic)

    n0_jc    = n[kk, F2, jc] / params.HSd[jc]
    n2_jc    = π * n[kk, F2, jc] * params.HSd[jc]
    nv2sq_jc = zero(FP)
    if ND >= 1; nd = -_2π*n[kk, FV,   jc]; nv2sq_jc += nd*nd; end
    if ND >= 2; nd = -_2π*n[kk, FV+1, jc]; nv2sq_jc += nd*nd; end
    if ND >= 3; nd = -_2π*n[kk, FV+2, jc]; nv2sq_jc += nd*nd; end
    xi_jc = one(FP) - nv2sq_jc / (n2_jc*n2_jc)

    # Precomputed site counts (NP=1 → literal index 1)
    n_ia = params.assoc_n_ia[1]
    n_jb = params.assoc_n_jb[1]

    kia  = n_ia * n0_ic * xi_ic * Δ_val
    kjb  = n_jb * n0_jc * xi_jc * Δ_val
    _b   = one(FP) - kia + kjb
    X_ia = 2 / (_b + sqrt(_b*_b + 4*kia))
    X_jb = one(FP) / (one(FP) + kia * X_ia)
    h    = FP(0.5)

    return n0_ic * xi_ic * n_ia * (Base.log(abs(X_ia)) - h*X_ia + h) +
           n0_jc * xi_jc * n_jb * (Base.log(abs(X_jb)) - h*X_jb + h)
end

@inline function f_assoc(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND},
                          ::Val{NP}, ::Val{NS}) where {NC, ND, NP, NS, M}
    F2  = 2; FV = 4
    FP  = eltype(n)
    _2π = FP(2π)

    # Mixture FMT densities (straight-line loops, no closures)
    n2_mix = zero(FP); n3_mix = zero(FP); nv2sq_mix = zero(FP)
    @inbounds for i in 1:NC
        nim = n[kk, F2, i] * params.m[i]
        n2_mix += π * nim * params.HSd[i]
        n3_mix += n[kk, F2+1, i] * params.m[i]
    end
    if ND >= 1
        nv2d = zero(FP)
        @inbounds for i in 1:NC; nv2d -= _2π * n[kk, FV,   i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    if ND >= 2
        nv2d = zero(FP)
        @inbounds for i in 1:NC; nv2d -= _2π * n[kk, FV+1, i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    if ND >= 3
        nv2d = zero(FP)
        @inbounds for i in 1:NC; nv2d -= _2π * n[kk, FV+2, i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    xi_mix = one(FP) - nv2sq_mix / (n2_mix * n2_mix)

    # Per-component n₀ and ξ via @generated helpers (no closures, no boxing)
    n0     = _assoc_n0(n, kk, params, Val(NC))
    xi     = _assoc_xi(n, kk, params, Val(NC), Val(ND))
    # n3_mix/n2_mix/xi_mix passed as args (not captured) → no boxing
    Δ_vals = _assoc_delta_vals(n, params, T, kk, n3_mix, n2_mix, xi_mix,
                                Val(NC), Val(ND), Val(NP), M)

    # _assoc_solve runs 5*NS SS + 5*NS Newton; IFT EnzymeRule provides exact gradient.
    X = _assoc_solve(n0, xi, Δ_vals, params, Val(NS))

    # Accumulate f_assoc = Σᵢ Σₐ n₀ᵢ ξᵢ nᵢₐ (ln Xᵢₐ - Xᵢₐ/2 + 1/2)
    # Regular for-loops + _nti: no closures, GPU-safe
    h   = FP(0.5)
    res = zero(FP)
    for i in 1:NC
        base_i = _nti(params.n_sites_cumsum, i)
        ns_i   = _nti(params.n_sites_cumsum, i + 1) - base_i
        n0i    = _nti(n0, i)
        xii    = _nti(xi, i)
        for a in 1:ns_i
            s        = base_i + a
            n_ia_val = _nti(params.n_sites_flat, s)
            X_val    = _nti(X, s)
            res     += n0i * xii * n_ia_val * (Base.log(abs(X_val)) - h*X_val + h)
        end
    end
    res
end

# ── Enzyme IFT rules for _assoc_solve ─────────────────────────────────────────
#
# These replace iteration-based gradient propagation through SS+Newton with the
# exact IFT derivative δX = -J⁻¹·(∂F/∂n0·δn0 + ∂F/∂xi·δxi + ∂F/∂Δ·δΔ).
# Both params and vNS are always Const in kernel calls; only n0/xi/Δ_vals carry
# tangents (derived from the single active input n).

# Single forward mode (Width=1) — used by δf_fwd_kernel!
function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    func::EnzymeRules.Const{typeof(_assoc_solve)},
    ::Type{<:Duplicated},
    n0::Union{EnzymeRules.Const, Duplicated},
    xi::Union{EnzymeRules.Const, Duplicated},
    Δ_vals::Union{EnzymeRules.Const, Duplicated},
    params::EnzymeRules.Const,
    vNS::EnzymeRules.Const)
    X_star = _assoc_solve(n0.val, xi.val, Δ_vals.val, params.val, vNS.val)
    dn0  = n0     isa EnzymeRules.Const ? map(zero, n0.val)     : n0.dval
    dxi  = xi     isa EnzymeRules.Const ? map(zero, xi.val)     : xi.dval
    dΔ   = Δ_vals isa EnzymeRules.Const ? map(zero, Δ_vals.val) : Δ_vals.dval
    δX = _assoc_ift_tangent(X_star, n0.val, xi.val, Δ_vals.val, dn0, dxi, dΔ, params.val)
    Duplicated(X_star, δX)
end

# Batch forward mode (Width=W) — used by δf_fwd_batch_kernel! (default ad_mode)
function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{W},
    func::EnzymeRules.Const{typeof(_assoc_solve)},
    ::Type{<:BatchDuplicated},
    n0::Union{EnzymeRules.Const, BatchDuplicated},
    xi::Union{EnzymeRules.Const, BatchDuplicated},
    Δ_vals::Union{EnzymeRules.Const, BatchDuplicated},
    params::EnzymeRules.Const,
    vNS::EnzymeRules.Const) where W
    X_star     = _assoc_solve(n0.val, xi.val, Δ_vals.val, params.val, vNS.val)
    dn0_batch  = n0     isa EnzymeRules.Const ? ntuple(_ -> map(zero, n0.val),     Val(W)) : n0.dval
    dxi_batch  = xi     isa EnzymeRules.Const ? ntuple(_ -> map(zero, xi.val),     Val(W)) : xi.dval
    dΔ_batch   = Δ_vals isa EnzymeRules.Const ? ntuple(_ -> map(zero, Δ_vals.val), Val(W)) : Δ_vals.dval
    δX_tuple = ntuple(Val(W)) do k
        _assoc_ift_tangent(X_star, n0.val, xi.val, Δ_vals.val,
                           dn0_batch[k], dxi_batch[k], dΔ_batch[k], params.val)
    end
    BatchDuplicated(X_star, δX_tuple)
end

# Reverse mode — augmented_primal stores X* on tape; reverse applies IFT cotangents.
# NTuple is immutable → Enzyme uses Active (not Duplicated) for n0/xi/Δ_vals/return.
# Cotangents are returned from reverse (Active convention), not accumulated into .dval.

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::EnzymeRules.Const{typeof(_assoc_solve)},
    ::Type{RT},
    n0::Union{EnzymeRules.Const, Active},
    xi::Union{EnzymeRules.Const, Active},
    Δ_vals::Union{EnzymeRules.Const, Active},
    params::EnzymeRules.Const,
    vNS::EnzymeRules.Const) where RT
    X_star = _assoc_solve(n0.val, xi.val, Δ_vals.val, params.val, vNS.val)
    primal = EnzymeRules.needs_primal(config) ? X_star : nothing
    # NTuple return is immutable → no shadow needed
    EnzymeRules.AugmentedReturn(primal, nothing, X_star)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::EnzymeRules.Const{typeof(_assoc_solve)},
    dret::Active,
    tape,
    n0::Union{EnzymeRules.Const, Active},
    xi::Union{EnzymeRules.Const, Active},
    Δ_vals::Union{EnzymeRules.Const, Active},
    params::EnzymeRules.Const,
    vNS::EnzymeRules.Const)
    X_star = tape
    seed   = dret.val
    dn0_c, dxi_c, dΔ_c = _assoc_ift_cotangents(
        X_star, n0.val, xi.val, Δ_vals.val, seed, params.val)
    return (
        n0     isa EnzymeRules.Const ? nothing : dn0_c,
        xi     isa EnzymeRules.Const ? nothing : dxi_c,
        Δ_vals isa EnzymeRules.Const ? nothing : dΔ_c,
        nothing,
        nothing,
    )
end
