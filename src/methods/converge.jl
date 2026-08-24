#wrapper for the AASol constructor that uses the keyword arguments in converge!

function __AASol(;maxit, beta, tol, anderson_start, anderson_m, verbose)
    return AASol(anderson_m;
            beta=beta, rtol=tol, atol=tol, maxit=maxit,
            picard_maxit=maxit, picard_beta=beta,
            picard_rtol=anderson_start, picard_atol=anderson_start,
            verbose=verbose)
end

#dispatches for specific defaults
AASol(system::Union{DFTSystem,DGTSystem,ElectrolyteDFTSystem};maxit=10000,beta=1e-2,tol=1e-4,anderson_start=1e-1,anderson_m=5,verbose=false,kwargs...) = __AASol(;maxit, beta, tol, anderson_start, anderson_m, verbose)
AASol(system::SCFTSystem;maxit=5000,beta=1e-2,tol=1e-6,anderson_start=1e-2,anderson_m=5,verbose=false,kwargs...) = __AASol(;maxit, beta, tol, anderson_start, anderson_m, verbose)


"""

    abstract type ClassicalDFTProblem{S} end
    ClassicalDFTProblem(system::S;kwargs...) where S

Abstract type for all types of ClassicalDFT.jl problems.
A `ClassicalDFTProblem` is just a wrapper of a `AbstractClassicalDFTSystem`, along with different options relevant to the system during the convergence phase, and independent of the solver type used.
"""
abstract type ClassicalDFTProblem{S} end

#=
IF we are ever going to solve this system in other ways other than fixed point,
The option should be here.
=#

struct DFTProblem{S,C} <: ClassicalDFTProblem{S}
    system::S
    log_interval::Int
    save_interval::Int
    save_callback::C
end

"""

    DFTProblem{S} <: ClassicalDFTProblem{S}
    DFTProblem(system::S;kwargs...) where S <: Union{DFTSystem,DGTSystem,ElectrolyteDFTSystem}


struct problem type for all DFT and DGT systems.

## Keyword arguments
- `log_interval::Int = 0`: Log the free energy every N iterations (0 = never).
- `save_interval::Int = 0`: Call `save_callback` every N iterations (0 = never).
- `save_callback = nothing`: a function of the form `f(iter, ρ_array)` called at each save interval.
"""
function DFTProblem(system;
            log_interval   :: Int  = 0,
            save_interval  :: Int  = 0,
            save_callback   ::C    = nothing,
            kwargs...) where C
    @assert log_interval >= 0
    @assert save_interval >= 0
    save_interval > 0 && (@assert save_callback != nothing)
    return DFTProblem(system,log_interval,save_interval,save_callback)
end

struct SCFTProblem{S,C} <: ClassicalDFTProblem{S}
    system::S
    quadrature::Symbol
    log_interval::Int
    save_interval::Int
    save_callback::C
end

"""

    SCFTProblem{S} <: ClassicalDFTProblem{S}
    SCFTProblem(system::S;kwargs...) where S <: SCFTSystem

struct problem type for all SCFT systems.

## Keyword arguments
- `log_interval::Int = 0`: Log the free energy every N iterations (0 = never).
- `save_interval::Int = 0`: Call `save_callback` every N iterations (0 = never).
- `save_callback = nothing`: a function of the form `f(iter, ρ_array, w_array)` called at each save interval.
- `quadrature::Symbol = :trapz`: Quadrature rule for partition-function integrals, the available values are:
  - `:trapz`   — periodic composite trapezoidal (uniform weights, spectral convergence for smooth periodic functions, works for any N). Recommended for periodic SCFT domains.
  - `:simpson` — composite Simpson rule (O(h⁴), requires odd N per dimension).
"""
function SCFTProblem(system;
            quadrature     :: Symbol = :trapz,
            log_interval   :: Int    = 0,
            save_interval  :: Int    = 0,
            save_callback   ::C      = nothing,
            kwargs...) where C
    @assert log_interval >= 0
    @assert save_interval >= 0
    save_interval > 0 && (@assert save_callback != nothing)
    return SCFTProblem(system,quadrature,log_interval,save_interval,save_callback)
end

ClassicalDFTProblem(system::Union{DFTSystem,DGTSystem,ElectrolyteDFTSystem};kwargs...) = DFTProblem(system;kwargs...)
ClassicalDFTProblem(system::SCFTSystem;kwargs...) = SCFTProblem(system;kwargs...)

"""
    get_new_profile!(system, ρ, δfδρ_res, caches)

One DFT-family fixed-point step: given the current density profile `ρ`:
- evaluate the functional derivative `δfδρ_res` (bulk residual `δFδρ_res!` + external field `evaluate_external_field!` + propagator chain contributions `propagate!`)
- assemble the new log-density-space field guess into `caches.ln_Gx` from the per-species chemical-potential balance (plus the electrostatic constant-potential correction for electrolyte systems).
`caches` is a named tuple of buffers preallocated once outside the iteration loop: `cache_model, cache_external, cache_propagator, ln_Gx`.
"""
function get_new_profile!(system::Union{DFTSystem,DGTSystem,ElectrolyteDFTSystem}, ρ, δfδρ_res, caches)
    (; cache_model, cache_external, cache_propagator, ln_Gx) = caches
    FP = fptype(system.options)
    nd = dimension(system)
    species = system.species
    model = system.model

    δFδρ_res!(system, ρ, δfδρ_res, cache_model...)

    evaluate_external_field!(system, ρ, δfδρ_res, cache_external)

    propagate!(system, ρ, δfδρ_res, cache_propagator)

    chem_pot_res_dens = FP.(species.chempot_res .+ log.(species.bulk_density))

    for i in @comps
        chem_pot_res_dens_i = chem_pot_res_dens[i]
        for k in @chain(i)
            if system.species.nbeads[i] != 1
                α = findall(model.groups.n_intergroups[i][k,:] .== 1 .&& species.levels .> species.levels[k])
            else
                α = k
            end

            # All operations stay vectorized on GPU
            selectdim(ln_Gx, nd+1, k) .=  chem_pot_res_dens_i .-
                                        selectdim(δfδρ_res, nd+1, k)
        end
    end

    if any(typeof.(system.external_field) .<: ElectrostaticPotentialModel)
        ep_model = filter(x -> x isa ElectrostaticPotentialModel, system.external_field)[1]
        Z = model.charge

        psi_c = find_ψ_const(system.structure, ep_model, model, exp.(ln_Gx))/k_B/system.structure.conditions[2]
        for i in @comps
            for k in @chain(i)
                selectdim(ln_Gx,nd+1,k) .-= psi_c*Z[k]
            end
        end
    end

    clamp!(ln_Gx, -100, 100)

    return nothing
end

"""
    converge!(system::AbstractClassicalDFTSystem,ρ::AbstractArray,solver;kwargs...)
    converge!(system::AbstractClassicalDFTSystem,ρ::AbstractArray;kwargs...)
    converge!(prob::ClassicalDFTProblem,ρ::AbstractArray;kwargs...)


For a given system, converge the profiles using the `solver` method. Convergence is achieved by solving the generic equation:

```julia
ρi = ρi_bulk*exp(β(μi_res - δFδρ_res))
```

For stability purposes, the the DFT system has be reformulated as:

```julia
ln(ρi) = ln(ρi_bulk) + β(μi_res - δFδρ_res)
```

On SCFT systems, the mean-field potential `w` is iterated instead, and the the density is calculated as a function of `w`.

If only keyword arguments are used, then the problem is solved via the Anderson acceleration routine [aasol](@ref)), The rest of keyword arguments are passed to [`ClassicalDFTProblem`](@ref)

## Anderson arguments
- `maxit::Int`: Maximum Anderson-phase iterations (also used as the Picard-phase cap, since the Picard phase is expected to exit via `anderson_start` well before either cap in practice).
- `beta::AbstractFloat`: Mixing coefficient for both the Picard warmup and Anderson phases.
- `tol::AbstractFloat`: Convergence tolerance (used as both `aasol`'s `rtol` and `atol`).
- `anderson_start::AbstractFloat`: Switch from Picard to Anderson once the residual norm drops below this (used as both `aasol`'s `picard_rtol` and `picard_atol`).
- `anderson_m::Int`: Anderson history length (number of past iterates kept). The default of `0` recovers pure (damped) Picard iteration throughout — the historical default behavior. Set to a positive integer to enable real Anderson acceleration.
- `verbose::Bool = false`: Print `aasol`'s per-iteration convergence summary, plus a final convergence/free-energy summary line.

## Default anderson values
- `DFTSystem`           : `maxit = 10000,beta = 1e-2, tol = 1e-4,anderson_start = 1e-1, anderson_m = 5,verbose = false`
- `SCFTSystem`          : `maxit = 5000,beta = 1e-2, tol = 1e-6,anderson_start = 1e-2, anderson_m = 5,verbose = false`
- `DGTSytem`            : same as `DFTSystem`
- `ElectrolyteDFTSystem`: same as `DFTSystem`

## Problem arguments (passed to [`ClassicalDFTProblem(system,kwargs...)`](@ref ClassicalDFT.ClassicalDFTProblem))
- `log_interval::Int = 0`: Log the free energy every N iterations (0 = never).
- `save_interval::Int = 0`: Call `save_callback` every N iterations (0 = never).
- `save_callback = nothing`: a function of the form `f(iter, ρ_array)` called at each save interval. SCFT systems require the function form `f(iter, ρ_array, w_array)` instead.
- `quadrature::Symbol = :trapz`: Quadrature rule for partition-function integrals, only used in SCFT systems. the available values are:
  - `:trapz`   — periodic composite trapezoidal (uniform weights, spectral convergence for smooth periodic functions, works for any N). Recommended for periodic SCFT domains.
  - `:simpson` — composite Simpson rule (O(h⁴), requires odd N per dimension).

Returns `nothing` (the input density is modified in-place).

## Manual API

instead of using just the keywords, one can have more granularity by instantiating the solver and problem separately:

```julia
solver = AASol(5,rtol = 1e-2,atol = 1e-3,picard_atol = 1e-1,picard_rtol = 1e-1,beta = 0.5, beta_picard = 0.9)
prob = ClassicalDFTProblem(system,log_interval = 5)
converge!(prob,solver,ρ)
```
"""
function converge! end

#convenience methods
converge!(system::AbstractClassicalDFTSystem,ρ::AbstractArray;kwargs...) = converge!(ClassicalDFTProblem(system;kwargs...),AASol(system;kwargs...),ρ)
converge!(system::AbstractClassicalDFTSystem,ρ::AbstractArray,solver;kwargs...) = converge!(ClassicalDFTProblem(system;kwargs...),solver,ρ)

converge!(prob::ClassicalDFTProblem,ρ::AbstractArray,solver) = converge!(prob,solver,ρ)
converge!(prob::ClassicalDFTProblem,ρ::AbstractArray;kwargs...) = converge!(prob,AASol(prob.system;kwargs...),ρ)

function converge!(prob::DFTProblem{S}, method::AASol, ρ::AbstractArray) where S
    system = prob.system
    FP = fptype(system.options)
    #ngrid = system.structure.ngrid
    #nd = dimension(system)
    #nbeads = size(ρ,nd+1)

    δfδρ_res, cache_model, cache_external, cache_propagator = ClassicalDFT.preallocate(system, ρ)
    ln_Gx = similar(ρ)
    caches = (; cache_model, cache_external, cache_propagator, ln_Gx)

    iter_count = Ref(0)
    ρ_vec = @view ρ[:]
    # aasol's fixed-point map: given the current log-density (flattened), advance one
    # DFT step and return the new log-density guess (flattened).
    function GFix!(ln_G, ln_x)
        ρ_vec .= exp.(ln_x)
        #ρᵢ .= exp.(reshape(ln_x, (ngrid..., nbeads)))
        get_new_profile!(prob.system, ρ, δfδρ_res, caches)
        GFix_logger(prob,iter_count,ρ)
        copyto!(ln_G, vec(ln_Gx))
        return ln_G
    end

    ln_X0 = vec(log.(ρ))
    result = aasol(GFix!, ln_X0, method)
    ρ_vec .= exp.(result.solution)

    if method.verbose
        H = free_energy(system, ρ)
        err = isempty(result.history) ? NaN : result.history[end]
        msg = result.idid ? "DFT converged" : "DFT did not converge"
        @info "$msg after $(iter_count[]) iterations: err = $(round(err; sigdigits=3)) | F = $(round(H; sigdigits=6))"
    end
end

function GFix_logger(prob::DFTProblem{S}, iter_count, ρ) where S
    system = prob.system
    iter_count[] += 1
    it = iter_count[]
    if prob.log_interval > 0 && it % prob.log_interval == 0
        H = free_energy(system, ρ)
        @info "DFT iter $(lpad(iter_count[], 5)) | F = $(round(H; sigdigits=6))"
    end
    if prob.save_interval > 0 && it % prob.save_interval == 0 && prob.save_callback !== nothing
        prob.save_callback(it, Array(ρ))
    end
end

"""
    get_new_profile!(system::SCFTSystem, ρ, w, caches)

One SCFT fixed-point step.

Given the current mean-field potential `w`:
- propagate the chains
- compute single-chain partition functions
- assemble the new density profile `ρ`
- evaluate the new field guess `caches.w_new` from that density.

`caches` is a named tuple of buffers preallocated once outside the iteration loop:
`w_new, w_bulk, dz, cache_external, q_in, q_out, buf_r, buf_c, child_buf, P, iP, weights, V_eff, exp_field, inv_exp_field, scratch`.

Returns `Q` (one partition function per molecule type).
"""
function get_new_profile!(system::SCFTSystem, ρ, w, caches)
    (; w_new, w_bulk, dz, cache_external, q_in, q_out, buf_r, buf_c, child_buf, P, iP,
       weights, V_eff, exp_field, inv_exp_field, scratch) = caches
    nd = dimension(system)
    FT = eltype(ρ)

    # exp_field[α] = exp(w_bulk[α] - w_α); reused by propagate!,
    # compute_partition_functions, and compute_densities! below.
    for α in eachindex(exp_field)
        w_α = selectdim(w, nd+1, α)
        @. exp_field[α]     = exp(w_bulk[α] - w_α)
        @. inv_exp_field[α] = one(FT) / exp_field[α]
    end

    cache_propagator = (q_in, q_out, buf_r, buf_c, child_buf, P, iP)
    propagate!(system, ρ, w, cache_propagator;
              w_bulk=w_bulk, exp_field=exp_field)
    Q = compute_partition_functions(system, w, w_bulk, q_in, dz;
                               weights=weights, V_eff=V_eff, exp_field=exp_field)
    compute_densities!(system, w, w_bulk, q_in, q_out, Q, ρ;
                       V_eff=V_eff, exp_field=exp_field, inv_exp_field=inv_exp_field)
    compute_fields!(system, ρ, w_new; scratch=scratch)
    # No-ops today (system.external_field is always nothing — see SCFTSystem's docstring);
    # mirrors the generic get_new_profile!'s δFδρ_res! → evaluate_external_field! →
    # propagate! ordering (bulk field first, external field added on top) so a future
    # field-type implementation for SCFTLatticeFluid only needs to add the dispatch, not
    # touch this loop.
    evaluate_external_field!(system, ρ, w_new, cache_external)

    return Q
end

function converge!(prob::SCFTProblem{S}, method::AASol, ρ::AbstractArray) where S
    system = prob.system
    dz = structure_dz(system.structure)
    FT = eltype(ρ)
    # Universal preallocation entry point (src/base/devices.jl), same as the generic
    # DFT-family converge! above: dispatches to preallocate_model,
    # preallocate_external_potential, and preallocate_propagator. SCFT's main field
    # buffer `w` is the returned δfδρ_res; `quadrature` is forwarded to preallocate_model
    # since it's a solve-time choice, not a system invariant.
    w, cache_model, cache_external, cache_propagator = preallocate(system, ρ; quadrature = prob.quadrature)
    (; w_new, weights, V_eff, w_bulk, scratch, exp_field, inv_exp_field) = cache_model
    q_in, q_out, buf_r, buf_c, child_buf, P, iP = cache_propagator

    compute_fields!(system, ρ, w; scratch = scratch)

    # Bundle the buffers preallocated above into a single cache passed to
    # get_new_profile! every iteration.
    caches = (; w_new, w_bulk, dz, cache_external, q_in, q_out, buf_r, buf_c, child_buf, P, iP,
              weights, V_eff, exp_field, inv_exp_field, scratch)

    iter_count = Ref(0)
    Q_last = Ref{Vector{FT}}(FT[])

    # aasol's fixed-point map: given the current field (flattened), advance one SCFT
    # step and return the new field (flattened). ρ/w_new/Q_last are mutated as a side
    # effect, matching get_new_profile!'s own in-place contract.
    function GFix!(G, xin)
        copyto!(w, reshape(xin, size(w)))
        Qᵢ = get_new_profile!(system, ρ, w, caches)
        Q_last[] = Qᵢ
        GFix_logger(prob, iter_count, ρ, (Qᵢ, w, V_eff, w_bulk))
        copyto!(G, vec(w_new))
        return G
    end

    result = aasol(GFix!, vec(copy(w)), method)

    w .= reshape(result.solution, size(w))
    # ρ already holds the density from the last GFix! call — aasol evaluates the
    # fixed-point map once more at `solution` to populate `functionval`, so ρ is
    # already consistent with the converged w.
    converged = result.idid
    err = isempty(result.history) ? FT(NaN) : FT(result.history[end])

    if method.verbose
        H = free_energy(system, ρ, w, Q_last[]; V_eff=V_eff, w_bulk=w_bulk)
        msg = converged ? "SCFT converged" : "SCFT did not converge"
        @info "$msg after $(iter_count[]) iterations: err = $(round(err; sigdigits=3)) | F = $(round(H; sigdigits=6))"
    end
end

function GFix_logger(prob::SCFTProblem, iter_count, ρ, logger_cache::C) where C
    system = prob.system
    Q, w, V_eff, w_bulk = logger_cache
    iter_count[] += 1
    it = iter_count[]
    if prob.log_interval > 0 && it % prob.log_interval == 0
        H = free_energy(system, ρ, w, Q; V_eff=V_eff, w_bulk=w_bulk)
        @info "SCFT iter $(lpad(it, 5)) | F = $(round(H; sigdigits=6))"
    end

    if prob.save_interval > 0 && it % prob.save_interval == 0 && prob.save_callback !== nothing
        prob.save_callback(it, Array(ρ), Array(w))
    end
end

export converge!, ClassicalDFTProblem, DFTProblem, SCFTProblem