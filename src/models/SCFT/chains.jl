#=
Chain representations for SCFTModel.

A chain submodel describes the single-segment statistics of ONE species (a monomer type):
which propagator physics its bonds use and the parameters that physics needs. SCFTModel
stores one chain submodel per species (`model.chains[α]`), so a single molecule can mix
representations (e.g. a rod-coil diblock). Mirrors the role of Clapeyron's cubic `alpha`
submodels: swappable components selected at construction, each carrying its own params.
=#

"""
    SCFTChainModel

Abstract supertype of the per-species chain representations used by [`SCFTModel`](@ref):
[`GaussianChain`](@ref), [`WormLikeChain`](@ref), [`RigidRod`](@ref).

Interface: `segment_length(c)`, `persistence_length(c)`, `is_oriented(c)` (whether the
propagator must resolve bond orientation `u ∈ S²` for this species).
"""
abstract type SCFTChainModel end

"""
    GaussianChain(; b=1.0)

Discrete Gaussian (bead-spring) chain segment with statistical segment length `b`.
Propagated by [`DiscreteGaussianChainPropagator`](@ref); no orientation is tracked.
"""
struct GaussianChain <: SCFTChainModel
    b::Float64
end
GaussianChain(; b::Real=1.0) = GaussianChain(Float64(b))

"""
    WormLikeChain(; b=1.0, lp)

Discrete worm-like-chain (bead-rod / Kratky-Porod) segment: fixed bond length `b` and
persistence length `lp` (same units as `b`; bending stiffness `κ = lp/b`). `lp=Inf` is the
exact rigid-rod limit (see also [`RigidRod`](@ref)). Propagated by
[`WLCPropagator`](@ref); bond orientation is tracked on a spherical-harmonic grid whose
truncation degree is the model-level `L_max`.
"""
struct WormLikeChain <: SCFTChainModel
    b::Float64
    lp::Float64
end
WormLikeChain(; b::Real=1.0, lp::Real) = WormLikeChain(Float64(b), Float64(lp))

"""
    RigidRod(; b=1.0)

Rigid-rod segment: a [`WormLikeChain`](@ref) with `lp = Inf` (no bending between
consecutive same-species bonds). Provided as its own submodel for readability.
"""
struct RigidRod <: SCFTChainModel
    b::Float64
end
RigidRod(; b::Real=1.0) = RigidRod(Float64(b))

segment_length(c::SCFTChainModel) = c.b

persistence_length(::GaussianChain) = throw(ArgumentError(
    "GaussianChain has no persistence length; only orientation-resolved chains (WormLikeChain, RigidRod) do."))
persistence_length(c::WormLikeChain) = c.lp
persistence_length(::RigidRod) = Inf

is_oriented(::SCFTChainModel) = true
is_oriented(::GaussianChain) = false

chain_label(::GaussianChain) = "Gaussian"
chain_label(::WormLikeChain) = "worm-like"
chain_label(::RigidRod) = "rigid rod"

# Normalize the user's `chain=` keyword into one chain submodel per species.
_resolve_chains(c::SCFTChainModel, species::Vector{String}) = SCFTChainModel[c for _ in species]

function _resolve_chains(c::AbstractVector, species::Vector{String})
    length(c) == length(species) || throw(ArgumentError(
        "chain vector has $(length(c)) entries but the model has $(length(species)) species $(species)."))
    all(x -> x isa SCFTChainModel, c) || throw(ArgumentError("every entry of a chain vector must be an SCFTChainModel."))
    return SCFTChainModel[x for x in c]
end

function _resolve_chains(c::AbstractDict, species::Vector{String})
    missing_sp = setdiff(species, collect(keys(c)))
    isempty(missing_sp) || throw(ArgumentError("chain Dict is missing an entry for species $(missing_sp)."))
    extra = setdiff(collect(keys(c)), species)
    isempty(extra) || throw(ArgumentError("chain Dict names unknown species $(extra); model species are $(species)."))
    return SCFTChainModel[c[s] for s in species]
end

export SCFTChainModel, GaussianChain, WormLikeChain, RigidRod
