import Clapeyron: a_res

_kernel_type(system::AbstractcDFTSystem) = typeof(system.model)
_kernel_type(system::DGTSystem)          = typeof(system)

"""
    length_scale(model::EoSModel)
    length_scale(L::Real) = L

Obtains the maximum length scale in the model and helps define the dimensions of the DFT system. This is typically equal to the size of the largest bead.
For real numbers is just the identity function.
"""
function length_scale end
length_scale(L::Real) = L

"""
    length_scale(model::Clapeyron.ABCubicModel)

The DGT analogue for cubic (van der Waals-family: PR, SRK, RK, ...) equations of state,
which have no SAFT-style `sigma` bead diameter to fall back on. `b` (the EoS covolume) is a
per-mole excluded volume, so `(b/N_A)^(1/3)` converts it to a per-molecule length on the same
footing as `length_scale(model::SAFTModel) = maximum(model.params.sigma.values)` -- verified
to land in the same physical ballpark (both ~3-5 Å for ethanol via PCSAFT vs. PR).
"""
length_scale(model::Clapeyron.ABCubicModel) = (maximum(model.params.b.values)/N_A)^(1/3)

"""
    get_species(model::EoSModel, structure::DFTStructure)

For a given `model` and `structure`, define the relevant parameters for each species. These structs will contain additional information not present by default in the inital `model`, such as the bead size, the number of beads and the connectivity of the beads.
"""
function get_species end

"""
    get_propagator(model::EoSModel, species::DFTSpecies, structure::DFTStructure)

For a given `model`, return the relevant propagator structure.
"""
function get_propagator end

include("BasicIdeal.jl")
include("DFT/dft.jl")
include("DGT/dgt.jl")
include("SCFT/scft.jl")

"""
    _energy_scale(system)

Compensating factor for `F_res`'s integrated scalar, for systems whose `get_fields`/
`preallocate_params` build their weighted-density kernels in reduced (length-scale-divided)
units (PC-SAFT and all its PCSAFTModel-hierarchy variants — PCPSAFT, QPCPSAFT,
pharmaPCSAFT, HomogcPCPSAFT, HeterogcPCPSAFT — plus SAFT-VR Mie, SAFTγMie, COFFEE, and PeTS
— see PCSAFT.jl's `get_fields` docstring). Reduced units make the per-point integrand `L^3`
times too large (a pure change-of-variables artifact, exact for any `ρ`), so `F_res`
divides by this. Defaults to `1.0` (no-op) for every other system. Every PCSAFTModel
subtype shares this one `DFTSystem{<:Clapeyron.PCSAFTModel}` dispatch — no per-variant
override needed, but that also means any *new* PCSAFTModel-hierarchy variant added later
must get matching reduced-units treatment in its own `get_fields`/`preallocate_params`, or
it will silently inherit this correction while still computing raw-unit kernels.

Dispatches on the *system* type, not just the model, so that `DGTSystem` — which wraps the
same Clapeyron models (e.g. `DGTSystem{<:PCSAFTModel}` in `test_dgt.jl`) but builds its own,
structurally different reduced-units treatment (`dgt.jl`'s `f_res`/`preallocate_params`: no
weighted-density convolution kernels, just `density_scale=L` on the `:ρ`/`:∇ρ` smoothing
fields plus a `κ/L^3` rescale — see `dgt.jl`'s `f_res` docstring) — can't inherit a
`DFTSystem`-specific per-model-type correction via shared model-type dispatch.
`DGTSystem{<:PCSAFTModel}` is a distinct outer type from `DFTSystem{<:PCSAFTModel}`, so it
never matches the `DFTSystem{<:...}`-dispatched methods above; it gets its own blanket
override below instead (every `DGTSystem` uses reduced units the same way, regardless of
which Clapeyron model it wraps, unlike `DFTSystem`'s per-model-type kernel differences).
"""
_energy_scale(system) = 1.0
_energy_scale(system::DFTSystem{<:Clapeyron.PCSAFTModel})       = length_scale(system.model)^3
_energy_scale(system::DFTSystem{<:Clapeyron.SAFTVRMieModel})    = length_scale(system.model)^3
_energy_scale(system::DFTSystem{<:Clapeyron.SAFTgammaMieModel}) = length_scale(system.model)^3
_energy_scale(system::DFTSystem{<:Clapeyron.COFFEEModel})       = length_scale(system.model)^3
_energy_scale(system::DFTSystem{<:Clapeyron.PeTSModel})         = length_scale(system.model)^3
_energy_scale(system::DGTSystem)                                = length_scale(system.model)^3

"""
    F_res(system::DFTSystem, ρ)

Residual free energy for DFT systems via the Enzyme/KA kernel path.
Runs the same δf_rev_kernel!/δf_fwd_kernel! used by δFδρ_res! and integrates the primal f_val.
"""
function F_res(system::Union{DFTSystem, DGTSystem}, ρ)
    δfδρ_res, cache_model, _, _ = preallocate(system, ρ)
    δFδρ_res!(system, ρ, δfδρ_res, cache_model...)
    f_val = cache_model[9]
    return ∫(f_val, system.structure) / _energy_scale(system)
end

"""
    F_res(system::AbstractcDFTSystem, ρ)

Residual free energy fallback for systems that use the old scalar f_res path (e.g. Electrolyte).
"""
function F_res(system::AbstractcDFTSystem, ρ)
    ngrid  = system.structure.ngrid
    model  = system.model

    δfδρ_res, cache_model, cache_external, cache_propagator = preallocate(system, ρ)
    n       = cache_model[1]
    fft_buf = cache_model[3]
    in_buf  = cache_model[4]
    out_buf = cache_model[5]
    P       = cache_model[6]
    iP      = cache_model[7]

    evaluate_field!(system, ρ, fft_buf, in_buf, out_buf, P, iP)
    copyto!(n, Adapt.adapt(typeof(n), fft_buf))

    ϕ = similar(ρ, ngrid...)
    ϕ .= 0
    for kk in CartesianIndices(ngrid)
        k = Tuple(kk)
        ϕ[k...] = f_res(system, model, @view(n[k...,:,:]))
    end
    return ∫(ϕ, system.structure)
end

"""
    δFδρ_res!(system, ρ, δfδρ_res, n, δf, fft_buf, in_buf, out_buf, P, iP,
              params, f_val, δf_val, nc, nd[, fwd_cache])

Enzyme/KernelAbstractions-based functional derivative evaluation. Runs on CPU or GPU
depending on `system.options.device`. When `system.options.ad_mode` is `:forward` or
`:forward_batch`, `fwd_cache` must be the `(dn_seeds, df_outs, Val(BATCH))` tuple from
`preallocate_model`.
"""
function δFδρ_res!(system::AbstractcDFTSystem, ρ, δfδρ_res,
                   n, δf, fft_buf, in_buf, out_buf, P, iP,
                   params, f_val, δf_val, nc, nd, fwd_cache = nothing)
    backend     = system.options.device
    ngrid       = system.structure.ngrid
    NF          = size(n, ndims(n)-1)
    NB          = size(n, ndims(n))
    temperature = convert(fptype(system.options), system.structure.conditions[2])
    M           = _kernel_type(system)

    evaluate_field!(system, ρ, fft_buf, in_buf, out_buf, P, iP)
    synchronize(backend)
    copyto!(n, fft_buf)

    if system.options.ad_mode === :forward_batch
        dn_seeds, df_outs, BATCH_val = fwd_cache
        fill!(δf, 0)
        kernel_fwd_batch = δf_fwd_batch_kernel!(backend)
        kernel_fwd_batch(df_outs, n, f_val, dn_seeds, params, temperature,
                         Val(NF), Val(NB), Val(nc), Val(nd), BATCH_val, M,
                         ndrange = ngrid)
        synchronize(backend)
        for f_idx in 1:NF, c_idx in 1:nc
            k = (f_idx - 1) * nc + c_idx
            selectdim(selectdim(δf, nd+1, f_idx), nd+1, c_idx) .= df_outs[k]
        end
    elseif system.options.ad_mode === :forward
        dn_seeds, df_outs, _ = fwd_cache
        fill!(δf, 0)
        kernel_fwd = δf_fwd_kernel!(backend)
        for k in eachindex(dn_seeds)
            fill!(df_outs[k], 0)
            kernel_fwd(df_outs[k], n, f_val, dn_seeds[k], params, temperature,
                       Val(NF), Val(NB), Val(nc), Val(nd), M,
                       ndrange = ngrid)
        end
        synchronize(backend)
        for f_idx in 1:NF, c_idx in 1:nc
            k = (f_idx - 1) * nc + c_idx
            selectdim(selectdim(δf, nd+1, f_idx), nd+1, c_idx) .= df_outs[k]
        end
    elseif system.options.ad_mode === :reverse
        fill!(δf_val, 1)
        fill!(δf, 0)
        kernel_rev = δf_rev_kernel!(backend)
        kernel_rev(δf, n, f_val, δf_val, params, temperature,
                   Val(NF), Val(NB), Val(nc), Val(nd), M,
                   ndrange = ngrid)
        synchronize(backend)
    else
        error("Unknown ad_mode $(system.options.ad_mode); expected :forward, :forward_batch, or :reverse")
    end

    copyto!(fft_buf, δf)
    integrate_field!(system, fft_buf, δfδρ_res, in_buf, P, iP)
end

"""
    δFδρ_res(system::DFTSystem, ρ)

Obtain the functional derivatives of the residual free energy for each component / bead.
Output is a 2D array `(ngrid, nb)`, normalised by `kB*T`.
"""
function δFδρ_res(system::AbstractcDFTSystem, ρ)
    δfδρ_res, cache_model, cache_external, cache_propagator = preallocate(system, ρ)
    δFδρ_res!(system, ρ, δfδρ_res, cache_model...)
    evaluate_external_field!(system, ρ, δfδρ_res, cache_external)
    propagate!(system, ρ, δfδρ_res, cache_propagator)
    return δfδρ_res
end

"""
    preallocate_model(system::DFTSystem, ρ)

Generic Enzyme/KernelAbstractions preallocator for all DFT models.
Allocates shared buffers, then delegates model-specific parameter assembly
to `preallocate_params(system)` which returns `(params, nc)`.
"""
function preallocate_model(system::DFTSystem, ρ)
    backend = system.options.device
    FP      = fptype(system.options)
    nf      = length_fields(system)
    ngrid   = system.structure.ngrid
    nd      = length(ngrid)
    nb      = size(ρ, nd + 1)

    n       = allocate(backend, FP, ngrid..., nf, nb)
    δf      = allocate(backend, FP, ngrid..., nf, nb)
    fill!(δf, 0)
    fft_buf = allocate(backend, FP, ngrid..., nf, nb)

    CT      = transform_eltype(system.structure, FP)
    in_buf  = allocate(backend, CT, ngrid...)
    out_buf = similar(in_buf)
    tmp     = similar(in_buf)
    plan, iplan = build_transform(system.structure, tmp, nd, backend)

    f_val  = allocate(backend, FP, ngrid...)
    δf_val = allocate(backend, FP, ngrid...)
    fill!(δf_val, 1)

    params, nc = preallocate_params(system)

    if system.options.ad_mode === :forward || system.options.ad_mode === :forward_batch
        batch = nf * nc
        dn_seeds = ntuple(Val(batch)) do k
            f_idx = (k - 1) ÷ nc + 1
            c_idx = (k - 1) % nc + 1
            seed = allocate(backend, FP, ngrid..., nf, nc)
            fill!(seed, 0)
            fill!(selectdim(selectdim(seed, nd+1, f_idx), nd+1, c_idx), 1)
            seed
        end
        df_outs   = ntuple(_ -> allocate(backend, FP, ngrid...), Val(batch))
        fwd_cache = (dn_seeds, df_outs, Val(batch))
    else
        fwd_cache = nothing
    end

    _warmup_cpu_kernel!(system, backend, n, δf, f_val, δf_val, params, nc, nd, fwd_cache)

    return n, δf, fft_buf, in_buf, out_buf, plan, iplan, params, f_val, δf_val, nc, nd, fwd_cache
end

function preallocate_model(system::ElectrolyteDFTSystem, ρ)
    backend = system.options.device
    FP      = fptype(system.options)
    nf      = length_fields(system)
    ngrid   = system.structure.ngrid
    nd      = length(ngrid)
    nb      = size(ρ, nd + 1)

    n       = allocate(backend, FP, ngrid..., nf, nb)
    δf      = allocate(backend, FP, ngrid..., nf, nb)
    fill!(δf, 0)
    fft_buf = allocate(backend, FP, ngrid..., nf, nb)

    CT      = transform_eltype(system.structure, FP)
    in_buf  = allocate(backend, CT, ngrid...)
    out_buf = similar(in_buf)
    tmp     = similar(in_buf)
    plan, iplan = build_transform(system.structure, tmp, nd, backend)

    f_val  = allocate(backend, FP, ngrid...)
    δf_val = allocate(backend, FP, ngrid...)
    fill!(δf_val, 1)

    params, nc = preallocate_params(system)

    # Electrolyte uses its own f_res path; forward-mode batch not yet supported
    if system.options.ad_mode === :forward || system.options.ad_mode === :forward_batch
        batch = nf * nc
        dn_seeds = ntuple(Val(batch)) do k
            f_idx = (k - 1) ÷ nc + 1
            c_idx = (k - 1) % nc + 1
            seed = allocate(backend, FP, ngrid..., nf, nc)
            fill!(seed, 0)
            fill!(selectdim(selectdim(seed, nd+1, f_idx), nd+1, c_idx), 1)
            seed
        end
        df_outs   = ntuple(_ -> allocate(backend, FP, ngrid...), Val(batch))
        fwd_cache = (dn_seeds, df_outs, Val(batch))
    else
        fwd_cache = nothing
    end

    _warmup_cpu_kernel!(system, backend, n, δf, f_val, δf_val, params, nc, nd, fwd_cache)

    return n, δf, fft_buf, in_buf, out_buf, plan, iplan, params, f_val, δf_val, nc, nd, fwd_cache
end

"""
    preallocate_model(system::SCFTSystem, ρ; quadrature::Symbol=:trapz)

SCFT's `preallocate_model` counterpart to the generic DFT-family `preallocate_model`
(`src/models/models.jl`), called by the universal `preallocate(system, ρ)`
(`src/base/devices.jl`). Bundles everything `get_new_profile!`/`converge!` need each
iteration besides the propagator buffers and the main field array `δfδρ_res` (SCFT's `w`):
a second field buffer `w_new`, quadrature weights/`V_eff`, the bulk field `w_bulk`, a
`compute_fields!` scratch buffer, and per-species `exp_field`/`inv_exp_field` caches.

`quadrature` is a solve-time choice (not a system invariant), so it's accepted as a keyword
here and forwarded from `preallocate(system, ρ; quadrature=...)`.
"""
function preallocate_model(system::SCFTSystem, ρ; quadrature::Symbol=:trapz)
    nd  = dimension(system)
    nspecies = length(system.model.groups.flattenedgroups)
    dz  = structure_dz(system.structure)

    w_new = similar(ρ)

    weights = if quadrature == :trapz
        trapz_weights(system.structure.ngrid, dz, system.options)
    elseif quadrature == :simpson
        simpson_weights(system.structure.ngrid, dz, system.options)
    else
        error("Unknown quadrature rule: $quadrature. Use :trapz or :simpson.")
    end
    V_eff = sum(weights)

    w_bulk = compute_bulk_fields(system.model, compute_bulk_densities(system))

    scratch       = similar(selectdim(w_new, nd+1, 1))
    exp_field     = [similar(selectdim(w_new, nd+1, 1)) for _ in 1:nspecies]
    inv_exp_field = [similar(selectdim(w_new, nd+1, 1)) for _ in 1:nspecies]

    return (; w_new, weights, V_eff, w_bulk, scratch, exp_field, inv_exp_field)
end

function length_scales(model::EoSModel)
    if hasfield(typeof(model.params), :sigma)
        return diagvalues(model.params.sigma.values)
    elseif hasfield(typeof(model.params), :b)
        return diagvalues(cbrt.(model.params.b.values / N_A))
    elseif hasfield(typeof(model.params), :lb_volume)
        return cbrt.(model.params.lb_volume.values / N_A)
    else
        error("No length scale defined in model")
    end
end

"""
    δf_rev_kernel!(backend)

KernelAbstractions kernel that applies Enzyme reverse-mode AD to `f_res` at each grid point.
Dispatches to the correct `f_res` implementation via `::Type{M}`.
Runs identically on CPU and GPU; the backend is selected at call time.
"""
@kernel function δf_rev_kernel!(
    δf, n, f_val, δf_val, params,
    temperature, ::Val{NF}, ::Val{NB}, ::Val{NC}, ::Val{ND}, ::Type{M}
) where {NF, NB, NC, ND, M}
    kk = @index(Global, Cartesian)
    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Reverse), Const(f_res), Const,
        Const(M), Const(kk),
        Duplicated(f_val, δf_val),
        Duplicated(n, δf),
        Const(params), Const(temperature), Const(Val(NC)), Const(Val(ND))
    )
end

"""
    δf_fwd_kernel!(backend)

KernelAbstractions kernel that applies Enzyme forward-mode AD to `f_res` at each grid point.
Called once per (field, component) direction; `dn` carries the unit-vector seed.
`δf_val[kk]` receives the directional derivative ∂f_res/∂n[kk, f, c].
Uses no per-thread intermediate storage, avoiding GPU stack overflow for large NC.
"""
@kernel function δf_fwd_kernel!(
    δf_val, n, f_val, dn, params,
    temperature, ::Val{NF}, ::Val{NB}, ::Val{NC}, ::Val{ND}, ::Type{M}
) where {NF, NB, NC, ND, M}
    kk = @index(Global, Cartesian)
    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Forward), Const(f_res), Const,
        Const(M), Const(kk),
        Duplicated(f_val, δf_val),
        Duplicated(n, dn),
        Const(params), Const(temperature), Const(Val(NC)), Const(Val(ND))
    )
end

"""
    δf_fwd_batch_kernel!(backend)

Batch forward-mode kernel: computes all NF×NC directional derivatives of `f_res`
in a single Enzyme call using `BatchDuplicated`. Each of the BATCH seeds in `dn_tuple`
has 1.0 at one specific (field, component) position; the corresponding `df_tuple[k][kk]`
receives ∂f_res(n[kk,:])/∂n[kk, f_k, c_k].
"""
@kernel function δf_fwd_batch_kernel!(
    df_tuple, n, f_val, dn_tuple, params,
    temperature, ::Val{NF}, ::Val{NB}, ::Val{NC}, ::Val{ND}, ::Val{BATCH}, ::Type{M}
) where {NF, NB, NC, ND, BATCH, M}
    kk = @index(Global, Cartesian)
    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Forward), Const(f_res), Const,
        Const(M), Const(kk),
        BatchDuplicated(f_val, df_tuple),
        BatchDuplicated(n,     dn_tuple),
        Const(params), Const(temperature), Const(Val(NC)), Const(Val(ND))
    )
end

"""
    _warmup_cpu_kernel!(system, backend, n, δf, f_val, δf_val, params, nc, nd, fwd_cache)

Force the Enzyme/KernelAbstractions AD kernel matching `system.options.ad_mode` to compile
once via a tiny single-block launch (`ndrange=(1,)`, which KernelAbstractions' CPU backend
always runs on a single thread/task) *before* any real, full-grid, multi-threaded launch.

LLVM's ORC JIT compiles a kernel lazily on its first invocation. If `Threads.nthreads() > 1`
and the real (large) launch is that first invocation, KernelAbstractions' `Threads.@spawn`-based
CPU backend has multiple threads race into the *same* not-yet-compiled closure simultaneously
— which deadlocks in LLVM's lazy-compilation trampolines rather than compiling once and
sharing the result. Confirmed via a live `lldb` backtrace on a hung process: the main thread
parked in `@sync`'s `wait()`, a worker thread blocked in `llvm::orc::LocalTrampolinePool::reenter`
waiting on a future that never resolves, and the actual compile-worker thread itself stuck.
Reproduced with a single-component associating system (e.g. `PCSAFT(["water"])`) above
KernelAbstractions' 1024-point single-workgroup grain size, under `-t 4`; never triggered by
a short-to-compile non-associating functional, since the race window is too small to hit in
practice. Warming up on a single block first compiles the closure before any thread can race
on it. `n` only needs to hold values in-domain for `f_res` (not physically meaningful — it's
fully overwritten by `evaluate_field!` before any real use), and the warm-up output is
discarded. A no-op when there's only one thread to race in the first place.
"""
function _warmup_cpu_kernel!(system, backend, n, δf, f_val, δf_val, params, nc, nd, fwd_cache)
    (backend isa CPU && Threads.nthreads() > 1) || return nothing

    NF = size(n, ndims(n) - 1)
    NB = size(n, ndims(n))
    temperature = convert(fptype(system.options), system.structure.conditions[2])
    M = _kernel_type(system)
    fill!(n, eltype(n)(1e-6))

    ad_mode = system.options.ad_mode
    if ad_mode === :forward_batch
        dn_seeds, df_outs, BATCH_val = fwd_cache
        kernel = δf_fwd_batch_kernel!(backend)
        kernel(df_outs, n, f_val, dn_seeds, params, temperature,
            Val(NF), Val(NB), Val(nc), Val(nd), BATCH_val, M; ndrange = (1,))
    elseif ad_mode === :forward
        dn_seeds, df_outs, _ = fwd_cache
        kernel = δf_fwd_kernel!(backend)
        kernel(df_outs[1], n, f_val, dn_seeds[1], params, temperature,
            Val(NF), Val(NB), Val(nc), Val(nd), M; ndrange = (1,))
    elseif ad_mode === :reverse
        fill!(δf_val, 1)
        kernel = δf_rev_kernel!(backend)
        kernel(δf, n, f_val, δf_val, params, temperature,
            Val(NF), Val(NB), Val(nc), Val(nd), M; ndrange = (1,))
        fill!(δf, 0)
    end
    synchronize(backend)
    return nothing
end

export length_scale
export expand_groups, expand_model
export get_species, get_propagator
