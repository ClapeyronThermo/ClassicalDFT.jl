import KernelAbstractions: Backend, get_backend, synchronize


"""
    DFTOptions(device, ad_mode= :forward; precision::Type{FP}=Float64)

A struct which includes all the settings that need to be set for the convergence algorithms and devices used

## Arguments:

- `device`: Specification of either CPU (pinned or un-pinned) or GPU devices. (unpinned CPU by default)
- `ad_mode`: One of:
  - `:forward` (unbatched — one Enzyme `Duplicated` forward-mode call per (field, component) direction)
  - `:forward_batch` (default on most backends — one Enzyme `BatchDuplicated` forward-mode call covering all directions at once; fastest where it compiles, but the resulting kernel can overflow some GPU shader compilers, e.g. Apple's Metal AGX backend, for non-trivial free-energy terms)
  - `:reverse` (one adjoint pass; the cheapest choice in general, since `f_res` returns a single scalar and reverse-mode cost is independent of the number of directional derivatives needed — also the only mode presently proven to compile on Metal for dispersion-level complexity, hence the default there via `ext/MetalcDFTExt.jl`).
- `precision`: The floating-point type (e.g. `Float64`, `Float32`) used to allocate and run the DFT calculation, retrievable via `fptype(options)`.

## Example usage:
```julia
julia> options = DFTOptions()

julia> using ThreadPinning

julia> options = DFTOptions(CPU(4, [0,1,12,13]))  # pins this process's threads to core IDs [0,1,12,13]; run julia with `-t 4` to actually get 4 threads

julia> options = DFTOptions(CPU(); precision = Float32)
```
"""
struct DFTOptions{D, FP<:AbstractFloat}
    device::D
    ad_mode::Symbol   # :forward_batch (default on most backends), :forward, or :reverse

    function DFTOptions(device::D, ad_mode::Symbol = :forward; precision::Type{FP} = Float64) where {D, FP<:AbstractFloat}
        return new{D,FP}(device, ad_mode)
    end
end

DFTOptions() = DFTOptions(CPU(; static=true))
DFTOptions(device::Backend; ad_mode::Symbol = :forward, precision::Type{FP} = Float64) where FP<:AbstractFloat =
    DFTOptions(device, ad_mode; precision)

"""
    fptype(options::DFTOptions)

Return the floating-point type (`Float64` by default) that `options` was configured with via the `precision` keyword, used throughout the DFT calculation to allocate arrays and dispatch kernels at the requested precision.
"""
fptype(::DFTOptions{D,FP}) where {D,FP} = FP

adapt_to_device(backend, ::Type{FP}, arr::AbstractArray) where FP<:AbstractFloat = Adapt.adapt(backend, FP.(arr))

function preallocate(system, ρ; kwargs...)
    backend = system.options.device
    FP = fptype(system.options)

    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)

    δfδρ_res = allocate(backend, FP, ngrid...,nb)

    cache_model = preallocate_model(system, ρ; kwargs...)

    cache_external = preallocate_external_potential(system, ρ)

    cache_propagator = preallocate_propagator(system, ρ)

    return δfδρ_res, cache_model, cache_external, cache_propagator
end


function preallocate_external_potential(system, ρ)
    backend = system.options.device
    ngrid = system.structure.ngrid
    nd = length(ngrid)

    cache_external = Any[]

    external_fields = system.external_field
    if isnothing(external_fields)
        return nothing
    end
    for external_field in external_fields
    
        if external_field isa ElectrostaticPotentialModel
            CT = transform_eltype(system.structure, fptype(system.options))
            Vext = similar(selectdim(ρ, nd+1, 1), CT)
            plan, iplan = build_transform(system.structure, Vext, nd, backend)

            push!(cache_external, (plan, iplan, Vext))
        else
            # Vext = allocate(backend, Float64, system.structure.ngrid...)

            z = get_coords(system.structure)

            Vext = Adapt.adapt(typeof(ρ),evaluate_external_field!(system.structure, external_field, system.model, ρ, ρ, z))

            push!(cache_external, (z,Vext))
        end
    end
    
    return cache_external
end

"""
    preallocate_propagator(system, ρ)

Universal propagator-buffer preallocation entry point, called by `preallocate`
alongside `preallocate_model`/`preallocate_external_potential`.
Dispatches on `system.propagator`'s concrete type (`IdealPropagator`,
`TangentHSPropagator`, `DiscreteGaussianChainPropagator` — see [Propagators](@ref)) to the
matching `preallocate_propagator(system, propagator, ρ, backend)` method, which allocates
whatever forward/backward propagator arrays and FFT plans that propagator needs.
"""
function preallocate_propagator(system, ρ)
    backend = system.options.device
    propagator = system.propagator

    return preallocate_propagator(system, propagator, ρ, backend)
end

export CPU, DFTOptions, fptype
