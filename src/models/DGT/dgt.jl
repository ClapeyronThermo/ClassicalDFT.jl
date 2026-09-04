preallocate_external_potential(system::DGTSystem, ρ) = nothing
preallocate_propagator(system::DGTSystem, ρ)         = nothing

struct DGTSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

"""
Pointwise residual free energy for DGT, written in Enzyme-compatible style.
`n[kk,1,i]` = smoothed density ρ̄ᵢ; `n[kk,1+d,i]` = gradient component d of ρ̄ᵢ.
The bulk EoS term calls Clapeyron.a_res with the model stored in Const(params);
the gradient correction is inlined arithmetic over the κ matrix.

Reduced units: `:ρ`/`:∇ρ` (base.jl's `DGTSystem` constructors) are built with
`density_scale=L`, so `n[kk,1,i]`/`n[kk,1+d,i]` are `L^3`-inflated versions of the true
density/gradient (via `evaluate_field!`'s `NA=N_A*L^3`). `z[i] = n[kk,1,i]` (no `/N_A`
division — `z` carries the raw, `N_A*L^3`-inflated value) combined with `params.dgt_V =
N_A*L^3` reproduces `a_res`'s exact original value unchanged: `a_res(V,T,z)` is
homogeneous degree-0 (intensive) in `(V,z)` jointly, so `a_res(N_A*L^3, T, n[kk,1,:]) ==
a_res(1.0, T, n[kk,1,:]/N_A)` exactly, for any `L`. `∑ρ = Σn[kk,1,i]` is now `L^3`-inflated
(no code change needed there), so `bulk_term` ends up `L^3` times its original value.
`params.kappa` is pre-divided by `L^3` in `preallocate_params` to compensate `grad_term`'s
`n[...]^2` picking up `L^6` (both gradient components are each `L^3`-inflated) — bringing
`grad_term` back down to the same `L^3` inflation as `bulk_term`, so the whole `out[kk]`
comes out uniformly `L^3`-inflated, matching `_energy_scale(::DGTSystem)`.
"""
@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: DGTSystem}
    κ = params.kappa

    ∑ρ = zero(eltype(n))
    for i in 1:NC
        ∑ρ += n[kk, 1, i]
    end

    z = Vector{Float64}(undef, NC)
    for i in 1:NC
        z[i] = n[kk, 1, i]
    end
    bulk_term = a_res(params.model, params.dgt_V, T, z) * ∑ρ

    grad_term = zero(eltype(n))
    for i in 1:NC
        for d in 1:ND
            grad_term += κ[i, i] * n[kk, 1+d, i]^2 / 2
        end
        for j in i+1:NC
            for d in 1:ND
                grad_term += κ[i, j] * n[kk, 1+d, i] * n[kk, 1+d, j]
            end
        end
    end

    out[kk] = bulk_term + grad_term / T
    return nothing
end

include("gradients/const.jl")
include("gradients/zuo_stenby.jl")

function preallocate_params(system::DGTSystem)
    T  = system.structure.conditions[2]
    ρ̄  = system.species.bulk_density
    κ  = kappa(system.gradient, system.model, T, ρ̄)

    # Reduced units: L must match base.jl's DGTSystem constructors (density_scale=L on
    # the :ρ/:∇ρ fields) — see f_res's docstring for the full derivation.
    L = length_scale(system.model)
    κ_reduced = κ ./ L^3
    dgt_V = N_A * L^3

    return (; kappa = κ_reduced, model = system.model, dgt_V = dgt_V), length(system.model)
end

function preallocate_model(system::DGTSystem, ρ)
    nf    = length_fields(system)
    ngrid = system.structure.ngrid
    nd    = length(ngrid)
    nb    = size(ρ, nd+1)

    n       = allocate(CPU(), Float64, ngrid..., nf, nb)
    δf      = allocate(CPU(), Float64, ngrid..., nf, nb);  fill!(δf, 0.0)
    fft_buf = allocate(CPU(), Float64, ngrid..., nf, nb)
    in_buf  = allocate(CPU(), ComplexF64, ngrid...)
    out_buf = similar(in_buf)
    tmp     = similar(in_buf)
    plan    = plan_fft!(tmp, 1:nd; num_threads = Threads.nthreads())
    iplan   = inv(plan)
    f_val   = allocate(CPU(), Float64, ngrid...)
    δf_val  = allocate(CPU(), Float64, ngrid...);  fill!(δf_val, 1.0)

    params, nc = preallocate_params(system)

    # Mirrors preallocate_model(::DFTSystem,...)/(::ElectrolyteDFTSystem,...)
    # (src/models/models.jl) — δFδρ_res! destructures fwd_cache unconditionally whenever
    # system.options.ad_mode is :forward or :forward_batch (the DFTOptions default), so
    # DGT needs the same dn_seeds/df_outs/batch construction, not just `nothing`.
    if system.options.ad_mode === :forward || system.options.ad_mode === :forward_batch
        batch = nf * nc
        dn_seeds = ntuple(Val(batch)) do k
            f_idx = (k - 1) ÷ nc + 1
            c_idx = (k - 1) % nc + 1
            seed = allocate(CPU(), Float64, ngrid..., nf, nc)
            fill!(seed, 0)
            fill!(selectdim(selectdim(seed, nd+1, f_idx), nd+1, c_idx), 1)
            seed
        end
        df_outs   = ntuple(_ -> allocate(CPU(), Float64, ngrid...), Val(batch))
        fwd_cache = (dn_seeds, df_outs, Val(batch))
    else
        fwd_cache = nothing
    end

    _warmup_cpu_kernel!(system, CPU(), n, δf, f_val, δf_val, params, nc, nd, fwd_cache)

    return n, δf, fft_buf, in_buf, out_buf, plan, iplan, params, f_val, δf_val, nc, nd, fwd_cache
end
