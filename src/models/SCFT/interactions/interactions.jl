#=
Interaction submodels for SCFTModel.

`model.interactions` is a Tuple of concrete SCFTInteractionModel values. Every SCFT
quantity that depends on interactions (mean fields, bulk free energy / bulk fields,
inhomogeneous free energy) is a sum over that tuple, so adding a new interaction means
implementing the interface below for one new type — no edits to the solver or the
propagators. Mirrors the role of Clapeyron's cubic `mixing`/`translation` submodels.

All functions here sit on the solve path or are differentiated through (`bulk_f` →
`bulk_fields`, `a_res`), so — per this codebase's convention — they use no
clamp/min/max guards.
=#

"""
    SCFTInteractionModel

Abstract supertype of the interaction submodels of [`SCFTModel`](@ref).

Interface (units of `k_BT`; `ρ` indexed by species, `rho0` the reference density):

- `bind_interaction(int, species::Vector{String})` → `int` with every species-named input
  resolved to species-indexed arrays (default: `int` unchanged).
- `validate_interaction(int, model)` → throws if `int` is incompatible with the model's
  chain representations (default: no-op).
- `bulk_f(int, ρ::AbstractVector, rho0)` → isotropic bulk free-energy density. Must be
  generic in `eltype(ρ)` (it is differentiated by `bulk_fields`/`a_res`).
- `add_field!(w, int, ρ, ρ_total, rho0, nd)` → adds this interaction's scalar mean field
  `δU/δρ_α` into every species slice `selectdim(w, nd+1, α)` (default: no-op).
- `field_energy(int, ρ, ρ_total, rho0, nd, per_∫, aux)` → this interaction's contribution
  to the SCFT free energy *beyond* the generic `-Σ_α ∫ w_α ρ_α` term (for an interaction
  whose field is part of `w`, that is just its energy `U`). `ρ`/`ρ_total` are host arrays,
  `per_∫` the periodic quadrature, `aux` a NamedTuple of auxiliary solver state
  (e.g. `R_tensor`).
"""
abstract type SCFTInteractionModel end

bind_interaction(int::SCFTInteractionModel, ::Vector{String}) = int
validate_interaction(::SCFTInteractionModel, model) = nothing
add_field!(w, ::SCFTInteractionModel, ρ, ρ_total, rho0, nd) = w
bulk_f(::SCFTInteractionModel, ρ::AbstractVector, rho0) = zero(eltype(ρ))
field_energy(::SCFTInteractionModel, ρ, ρ_total, rho0, nd, per_∫, aux) = 0.0

"""
    _species_matrix(x, species, name; zero_diagonal=false)

Resolve a pairwise parameter given either as an `AbstractMatrix` (rows/columns in
`species` order) or as a `Dict` keyed by species-name pairs (`("A","B") => 1.2` or
`"A"=>"B" => 1.2`; symmetric, unspecified pairs are zero) into a dense symmetric
`Matrix{Float64}`.
"""
function _species_matrix(x::AbstractMatrix, species::Vector{String}, name; zero_diagonal=false)
    n = length(species)
    size(x) == (n, n) || throw(ArgumentError("$name must be $(n)×$(n) (species $(species)), got size $(size(x))."))
    M = Matrix{Float64}(x)
    M ≈ transpose(M) || throw(ArgumentError("$name must be symmetric."))
    zero_diagonal && any(!iszero, [M[i, i] for i in 1:n]) &&
        throw(ArgumentError("$name must have a zero diagonal."))
    return M
end

function _species_matrix(x::AbstractDict, species::Vector{String}, name; zero_diagonal=false)
    idx = Dict(s => i for (i, s) in enumerate(species))
    M = zeros(Float64, length(species), length(species))
    for (k, v) in x
        a, b = k isa Pair ? (k.first, k.second) : (k[1], k[2])
        haskey(idx, a) && haskey(idx, b) || throw(ArgumentError("$name names unknown species in $(k); model species are $(species)."))
        M[idx[a], idx[b]] = v
        M[idx[b], idx[a]] = v
    end
    return _species_matrix(M, species, name; zero_diagonal)
end

# Total density Σ_α ρ_α into `out` (a single species-shaped slice), allocation-free.
function _total_density!(out, ρ, nspecies, nd)
    out .= zero(eltype(out))
    for α in 1:nspecies
        out .+= selectdim(ρ, nd + 1, α)
    end
    return out
end

include("flory_huggins.jl")
include("helfand.jl")
include("maier_saupe.jl")
