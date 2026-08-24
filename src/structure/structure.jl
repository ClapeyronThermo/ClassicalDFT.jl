tanh_prof(x,start,stop,shift,coef) = 1/2*(start-stop)*tanh((x-shift)*coef)+1/2*(start+stop)
cos_prof(x,start,stop,shift,coef) = 1/2*(start-stop)*cos((x-shift)*2π)*sqrt((1+coef^2)/(1+coef^2*cos((x-shift)*2π)^2))+1/2*(start+stop)

# include("surface_tension.jl")
# include("interfacial_tension.jl")
include("transforms.jl")
include("two_phase.jl")
include("morphology.jl")
include("external_field.jl")

"""
    initialize_profiles(system::DFTSystem; noise::Real=0.0)

Based on the system specifications, this function will initialize the density profiles for each of the species / beads in the model. The output will be an array of size `(ngrid,nb)` where `ngrid` is the number of grid points used and `nb` is the number of beads.

If `noise` is nonzero, the profile is perturbed by independent, per-grid-point,
per-bead multiplicative noise `ρ *= 1 + noise*U(-1,1)` — useful for seeding an unstable
uniform profile (e.g. inside a miscibility gap) before a Dynamic DFT time evolution, since
a perfectly uniform profile is otherwise an exact (if unstable) fixed point that never
starts phase-separating on its own. Being multiplicative rather than additive, this keeps
the perturbed density strictly positive everywhere for any `noise < 1`, regardless of the
local (possibly near-zero) density. The perturbed profile is then rescaled, at each grid
point, so the grand total across every species/bead matches what the *unperturbed*
profile's grand total already was — independent per-species noise would otherwise drift
the pointwise total away from it. This needs no model-specific knowledge (no `rho0`/
`bulk_density` lookups — the pre-noise profile is already correct by construction, so its
own total is exactly the right target); for most DFT-family models it's a minor nicety
(their own functional derivative already handles compressibility), but for models with a
stiff total-density constraint (e.g. SCFT's incompressibility penalty `κ`) it's
load-bearing — without it, a perturbed initial guess can make the Picard warmup diverge,
since independent per-species noise otherwise violates that constraint outright.

Example:
```julia
julia> model = PCSAFT(["water"])

julia> ρbulk = [molar_density(model,1e5,298.15)]

julia> L = length_scale(model)

julia> structure = Uniform1DCart((1e5, 298.15), ρbulk, [0, 20L], 201)

julia> system = DFTSystem(model, structure)

julia> profiles = initialize_profiles(system)

julia> profiles_perturbed = initialize_profiles(system; noise=0.01)  # for DDFT
```
"""
function initialize_profiles(system::AbstractClassicalDFTSystem; noise::Real=0.0)
    ρ = initialize_profiles(system.model,system.structure,system.species,system.options.device,fptype(system.options))
    if system.external_field == nothing
        # pass
    else
        for i in system.external_field
            if !(i isa ElectrostaticPotentialModel)
                 initialize_profiles!(system, i, ρ)
            end
        end

        if any(typeof.(system.external_field) .<: ElectrostaticPotentialModel)
            ψ = find_ψ_const(system.structure, system.external_field[findfirst(typeof.(system.external_field) .<: ElectrostaticPotentialModel)], system.model, ρ) ./ k_B / system.structure.conditions[2]
            # system.model.charge is a plain Vector{Int64} on Clapeyron's ElectrolyteModel
            # struct, never adapted to the system's device — broadcasting it directly
            # against ψ (GPU-resident, since it's built from ρ) fails GPU compilation
            # ("passing non-bitstype argument": a CPU Vector reachable from the broadcast
            # tree). adapt_to_device moves it to the same backend/FP as ρ, matching the
            # convention used for every other model-derived array throughout this codebase.
            Z = adapt_to_device(system.options.device, fptype(system.options), system.model.charge)
            ρ .*= exp.(-ψ*Z')
        end
    end

    if !iszero(noise)
        nd = dimension(system)
        FP = fptype(system.options)
        ρ_total = sum(ρ, dims=nd+1)
        ξ = adapt_to_device(system.options.device, FP, rand(FP, size(ρ)...))
        ρ .*= 1 .+ FP(noise) .* (2 .* ξ .- 1)
        ρ .*= ρ_total ./ sum(ρ, dims=nd+1)
    end

    # Bounds derived from the working type's own representable range (not a fixed
    # Float64 literal) so this remains safe under `Float32`: several downstream
    # correlations (e.g. PCSAFT's `I_lite`) raise densities/weighted-densities to
    # up to the 6th power, so a value at the clamp boundary must survive that
    # without underflowing to exact zero (→ log(0) → NaN) or overflowing to Inf.
    FP = eltype(ρ)
    lo = floatmin(FP)^(one(FP)/6)
    hi = floatmax(FP)^(one(FP)/6)
    clamp!(ρ, lo, hi)

    return ρ
end

"""
    initialize_profiles(model::EoSModel, structure::Union{Uniform1DCart,...}, species::SCFTSpecies, device, FP)

SCFT-aware override of the generic per-structure `Uniform*Cart` seeding below: reads the already-correct, precomputed per-species bulk density (`species.bulk_density`, from`get_species`) directly, rather than replicating `structure.ρbulk[component]` unsplit across every species of that molecule type (correct for the DFT family's expanded, per-bead-occurrence indexing; wrong for SCFT's per-species-letter aggregation). 
Dispatch on `species::SCFTSpecies` (more specific than the untyped `species` the generic methods below take) is what lets `initialize_profiles(system::SCFTSystem; noise=0.0)` — just the generic `initialize_profiles(system::AbstractClassicalDFTSystem;...)` above, no SCFT-specific override needed — build the correct base profile via this same shared entry point, and (together with `_SCFTUnsupportedStructure`'s guard now living in `SCFTSystem`'s constructor, `src/models/SCFT/scft.jl`, and the grand-total noise renormalization now generic above) is the last piece that made a full SCFT-specific override unnecessary.
`LamellarStack*`/`HexLattice*`/`BCC3DCart`/`Gyroid3DCart` need no equivalent override:
`src/structure/morphology.jl`'s `_fill_morphology!` is already per-species-aware
(`sign[j]` differs per species `j` within a chain), unlike the `Uniform*` functions below.
"""
function _scft_initialize_profiles(structure, species::SCFTSpecies, device, ::Type{FP}) where FP<:AbstractFloat
    nd = dimension(structure)
    bulk = species.bulk_density
    ρ = allocate(device, FP, structure.ngrid..., length(bulk))
    for α in eachindex(bulk)
        selectdim(ρ, nd+1, α) .= FP(bulk[α])
    end
    return ρ
end



# Split into three methods (rather than one combined Union) to exactly match the
# structure-type groupings of the generic methods below — a broader combined Union here
# would be ambiguous with them (neither method strictly more specific than the other
# across both the `structure` and `species` arguments at once).
function initialize_profiles(model::EoSModel, structure::DFTStructure{N,Cartesian,<:UniformGrid}, species::SCFTSpecies, device, ::Type{FP}=Float64) where {N,FP<:AbstractFloat}
    _scft_initialize_profiles(structure, species, device, FP)
end

function initialize_profiles(model::EoSModel,structure::DFTStructure{N,C,<:UniformGrid}, species, device, ::Type{FP}=Float64) where {N,C,FP<:AbstractFloat}
    ngrid = structure.ngrid
    ρbulk = structure.ρbulk
    ρ = allocate(device, FP, ngrid..., sum(species.nbeads))
    for i in @comps
        for j in @chain(i)
            ρj = selectdim(ρ,N+1,j)
            ρj .= FP(ρbulk[i])
        end
    end
    return ρ
end

function get_coords(structure::DFTStructure)
    ngrid = structure.ngrid
    nd = length(ngrid)
    bounds = structure.bounds
    z = [uniform_range(structure,i) |> collect for i in 1:nd]
    Z = zeros(ngrid...,nd)
    for jj in CartesianIndices(ngrid)
        j = Tuple(jj)
        for i in 1:nd
            Z[j...,i] = z[i][j[i]]
        end
    end
    return Z
end

function structure_fftfreq(structure::DFTStructure)
    ngrid = structure.ngrid
    nd = dimension(structure)
    function ff(i)
        lb,ub = bounds(structure,i)
        return ngrid[i]/(ub - lb)
    end
    f = ntuple(ff,nd)
    ω = fftfreq.(ngrid, f)
end

function structure_ω(structure::DFTStructure, device::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    ngrid = structure.ngrid
    nd = dimension(structure)
    ω̂ = structure_fftfreq(structure)  # ntuple of fftfreq vectors, one per dimension

    # Move each 1D frequency vector to GPU, then reshape for broadcasting
    # dim 1: (Nx,1,1,...), dim 2: (1,Ny,1,...), dim 3: (1,1,Nz,...) etc.
    ω_components = ntuple(nd) do i
        vec = adapt(device, Complex{FP}.(ω̂[i]))   # move to GPU
        shape = ntuple(d -> d == i ? ngrid[d] : 1, nd)  # e.g. (Nx,1,1) for i=1
        reshape(vec, shape)                               # ready to broadcast
    end

    ω = allocate(device, Complex{FP}, ngrid..., nd)

    # Fill each α slice via broadcast — no scalar indexing
    for α in 1:nd
        selectdim(ω, nd+1, α) .= ω_components[α]  # broadcasts (Ni,1,1) → (Nx,Ny,Nz)
    end

    return ω
end

function structure_dz(structure::DFTStructure)
    ngrid = structure.ngrid
    nd = dimension(structure)
    function ff(i)
        lb,ub = bounds(structure,i)
        return (ub - lb)/ngrid[i]
    end
    return ntuple(ff,nd)
end

function structure_ω(structure::Union{DFTStructByCoord{Cylindrical},DFTStructByCoord{Spherical}}, device::Backend, ::Type{FP}=Float64) where {FP<:AbstractFloat}
    device isa CPU || error("Spherical/cylindrical coordinate systems are CPU-only for now")
    Q = radial_transform(structure)
    return RadialFrequency{FP,typeof(Q)}(Q, FP.(Q.k ./ (2π)))
end

"""
    structure_r(structure)

The real-space radial grid of a spherical/cylindrical structure, i.e. the (non-uniform,
Bessel-zero-derived) sample points `Q.r` of its underlying `Hankel.QDHT`.
"""
structure_r(structure::Union{DFTStructByCoord{Cylindrical},DFTStructByCoord{Spherical}})= radial_transform(structure).r

function get_coords(structure::Union{DFTStructByCoord{Cylindrical},DFTStructByCoord{Spherical}})
    r = structure_r(structure)
    return reshape(r, length(r), 1)
end

export get_coords, initialize_profiles, structure_r