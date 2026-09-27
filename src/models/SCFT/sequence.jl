"""
    SequenceDescriptor

The sequence-SCFT descriptor function `f(s)` (Xie & Olsen, *Macromolecules* 2022, 55,
6516-6524), evaluated once at construction on `Nc` contour points and stored densely
(consistent with `chi`/`b` already being precomputed `PairParam`/`SingleParam` objects
rather than re-evaluated during the solve).

- `values::Matrix{Float64}`: `(Nc, d)`, `values[k, :] == f(s_k)`.
- `s::Vector{Float64}`: contour positions in `[0,1]`, for plotting/debugging only — the
  propagator's own contour discretization is `Δs=1` per bond (see [`sequence_chi_b`](@ref)),
  `s` here is purely a normalized coordinate for evaluating `f`.
"""
struct SequenceDescriptor
    values::Matrix{Float64}
    s::Vector{Float64}
end

"""
    contour_points(Nc::Int)

`Nc` evenly-spaced points in `[0,1]`, the contour coordinate `s` used to evaluate a
descriptor function. `Nc==1` returns `[0.0]`.
"""
contour_points(Nc::Int) = Nc == 1 ? [0.0] : collect(range(0.0, 1.0, length=Nc))

_as_vec(x::Real) = [x]
_as_vec(x::AbstractVector) = x

"""
    SequenceDescriptor(f, Nc::Int)

Evaluate an arbitrary user-supplied descriptor function `f(s::Real) -> Real or Vector`
at `Nc` evenly-spaced contour points (see [`contour_points`](@ref)). The descriptor
dimension `d` is inferred from `f`'s return value.
"""
function SequenceDescriptor(f, Nc::Int)
    s = contour_points(Nc)
    f1 = _as_vec(f(s[1]))
    d = length(f1)
    values = Matrix{Float64}(undef, Nc, d)
    values[1, :] .= f1
    for k in 2:Nc
        values[k, :] .= _as_vec(f(s[k]))
    end
    return SequenceDescriptor(values, s)
end

"""
    step_descriptor(fractions, levels, Nc::Int)

The paper's Figure 1a-c block-copolymer encoding: a piecewise-constant descriptor with
`length(fractions)` blocks, block `b` occupying contour fraction `fractions[b]` (must sum
to `1`) and taking descriptor value `levels[b]` (each a `Real` or `Vector`, all the same
dimension). E.g. a symmetric AB diblock: `step_descriptor([0.5,0.5], [[0.0],[1.0]], Nc)`.
"""
function step_descriptor(fractions::AbstractVector{<:Real}, levels, Nc::Int)
    @assert isapprox(sum(fractions), 1.0; atol=1e-8) "fractions must sum to 1, got $(sum(fractions))"
    @assert length(fractions) == length(levels) "fractions and levels must have the same length"
    d = length(_as_vec(levels[1]))
    values = Matrix{Float64}(undef, Nc, d)
    boundaries = round.(Int, Nc .* cumsum(fractions))
    boundaries[end] = Nc  # guard against a cumulative rounding gap at the last block
    lo = 1
    for (b, hi) in enumerate(boundaries)
        values[lo:hi, :] .= reshape(Float64.(_as_vec(levels[b])), 1, d)
        lo = hi + 1
    end
    return SequenceDescriptor(values, contour_points(Nc))
end

# Explicit if/elseif/else, not clamp/min/max, by convention in this codebase for
# functionals near the solve path (see project_scft_clapeyron_eos memory).
function _ramp_t(s::Real, t0::Real, frac::Real)
    t1 = t0 + frac
    if s <= t0
        return 0.0
    elseif s >= t1
        return 1.0
    else
        return (s - t0) / frac
    end
end

"""
    taper_descriptor(f_start, f_end, Nc::Int; taper_fraction=1.0, taper_start=0.0)

The paper's Figure 1d-f tapered-copolymer encoding: constant at `f_start` for
`s <= taper_start`, a linear ramp from `f_start` to `f_end` over the contour fraction
`[taper_start, taper_start+taper_fraction]`, then constant at `f_end` for
`s >= taper_start+taper_fraction`. The default (`taper_fraction=1.0, taper_start=0.0`)
is a linear ramp over the whole chain; a smaller `taper_fraction` matches the paper's
"inverse tapered copolymer" example (Figure 3g), where only part of the chain tapers.
"""
function taper_descriptor(f_start, f_end, Nc::Int; taper_fraction::Real=1.0, taper_start::Real=0.0)
    s = contour_points(Nc)
    f0, f1 = Float64.(_as_vec(f_start)), Float64.(_as_vec(f_end))
    d = length(f0)
    @assert length(f1) == d "f_start and f_end must have the same dimension"
    values = Matrix{Float64}(undef, Nc, d)
    for k in 1:Nc
        t = _ramp_t(s[k], taper_start, taper_fraction)
        values[k, :] .= (1 - t) .* f0 .+ t .* f1
    end
    return SequenceDescriptor(values, s)
end

"""
    sequence_chi_b(descriptor::SequenceDescriptor, b; rho0, gamma=(f1,f2)->sum(abs2, f1 .- f2))

Build the dense `Nc×Nc` sequence-SCFT `chi` matrix and per-node `b` vector from a
[`SequenceDescriptor`](@ref), for consumption by [`SCFTSequenceFluid`](@ref).

`chi[k,k'] = (rho0^2/Nc^2) * gamma(f(s_k), f(s_k'))` (zero diagonal) — the factor that
makes this codebase's existing, unmodified mean field
`w_α(r) = Σ_β (χ_αβ/ρ₀)ρ_β(r) + ...` (`compute_fields!`) reproduce the sequence-SCFT
field equation (Xie & Olsen eq 11) exactly, when every contour node is its own unique
species (see [`SCFTSequenceFluid`](@ref)'s docstring for the full derivation). `gamma` is
the *bare* interaction kernel with no `1/N` factor — this function's own `rho0^2/Nc^2`
prefactor already reconstitutes both the paper's eq-1-internal and eq-11-external `1/N`
factors for the default choice `gamma(f1,f2) = ‖f1-f2‖²` (paper eq 1).

`b` sets the per-node statistical segment length: a constant `Real`, a function
`s::Real -> Real` evaluated at `descriptor.s`, or an explicit `AbstractVector` of length
`Nc`.
"""
function sequence_chi_b(descriptor::SequenceDescriptor, b;
                         rho0::Real, gamma = (f1, f2) -> sum(abs2, f1 .- f2))
    Nc = size(descriptor.values, 1)
    chi = zeros(Float64, Nc, Nc)
    pref = rho0^2 / Nc^2
    @inbounds for k in 1:Nc, kp in (k+1):Nc
        γkk = pref * gamma(view(descriptor.values, k, :), view(descriptor.values, kp, :))
        chi[k, kp] = γkk
        chi[kp, k] = γkk
    end
    b_vec = if b isa Function
        Float64[b(s) for s in descriptor.s]
    elseif b isa Real
        fill(Float64(b), Nc)
    else
        Float64.(b)
    end
    @assert length(b_vec) == Nc "b must have Nc=$Nc entries, got $(length(b_vec))"
    return chi, b_vec
end

"""
    SCFTSequenceFluid(descriptor::SequenceDescriptor, b; rho0, kappa,
                       gamma=(f1,f2)->sum(abs2, f1 .- f2), component="sequence",
                       idealmodel=BasicIdeal, references=String[])

The sequence-SCFT (Xie & Olsen 2022) analogue of [`SCFTLatticeFluid`](@ref): a single
linear chain of `Nc = size(descriptor.values,1)` contour nodes, each its own unique
species (no two nodes share a species index), with `chi`/`b` built from `descriptor` via
[`sequence_chi_b`](@ref).

Deliberately bypasses `custom_structure`/`GroupParam(grouplist)`/`expand_model`: those
are built around a small, named, repeatable chemical alphabet (`custom_structure`'s
parser is one ASCII character per node, capping distinct species at ~52), not a
per-contour-point identity that can run to hundreds of nodes. Instead this builds the
already-"expanded" `GroupParam` directly (one flattenedgroup per node, tridiagonal linear
bond matrix) via Clapeyron's own raw positional `GroupParam` constructor — the exact same
low-level idiom `expand_groups` (`eos.jl`) already uses internally for every other SCFT
model in this codebase, not a new pattern. Because node names contain no `"_"`,
`_group_letter` (`src/structure/morphology.jl`) is the identity map, so
[`get_species`](@ref) naturally treats every node as its own species with zero code
changes, and [`SCFTSequenceSystem`](@ref) can call `get_species`/`get_propagator`
directly on this model with no `expand_model` step.

## Derivation (why this reproduces the paper's equations exactly)

Matching this codebase's existing field equation
```
w_α(r) = Σ_β (χ_αβ/ρ₀) ρ_β(r) + (κ/ρ₀)(ρ₊(r)/ρ₀ − 1)          [compute_fields!]
```
against the paper's eq 11, `ω(r,s) = −Ξ(r) + (βρ₀/N)∫ds' γ(s,s')ψ(r,s')`, discretized
with unit contour spacing (this codebase's existing convention — `Δs=1` per
`DiscreteGaussianChainPropagator` bond, and `free_energy`'s contour sums never apply a
separate quadrature weight either), with `β≡1` (this codebase's `w`/`χ`/`a_res` are
already in `k_BT=1` units), and unique-species-per-node so `ψ*(r,s_{k'}) ≡ ρ_{k'}(r)`
exactly (`compute_densities!`'s species-grouping degenerates to one term per species when
no two nodes share a species index):
```
χ_{k,k'} = (ρ₀²/Nc²) · γ(f(s_k), f(s_k'))
```
— see [`sequence_chi_b`](@ref). This is a direct node-by-node identity valid for any
descriptor curve (block, taper, or fully arbitrary), not a block-copolymer special case.
`free_energy`'s `U_int` term is, by construction, already the functional whose derivative
w.r.t. `ρ_α` reproduces `compute_fields!`'s χ-term — matching the field equation is
therefore sufficient for `free_energy` to be correct too, with no changes needed there.

## Scope (v1)

Exactly one sequence-chain molecule type per system (see
[`SCFTSequenceSystem`](@ref) — the `1/Nc` above is chain-length-specific, so blending
differently-sized sequence chains has no unambiguous shared normalization), linear chains
only, and the Gaussian-thread propagator only (`DiscreteGaussianChainPropagator` — matches
the paper's own canonical treatment; `WLCPropagator` is linear-only anyway, so
orientation-resolved sequence chains are a natural but separate future extension).
"""
function SCFTSequenceFluid(descriptor::SequenceDescriptor, b; rho0::Real, kappa::Real,
                            gamma = (f1, f2) -> sum(abs2, f1 .- f2),
                            component::String = "sequence",
                            idealmodel = BasicIdeal, references::Vector{String} = String[])
    Nc = size(descriptor.values, 1)
    names = ["n$k" for k in 1:Nc]

    bondmat = zeros(Int, Nc, Nc)
    for k in 1:(Nc - 1)
        bondmat[k, k+1] = 1
        bondmat[k+1, k] = 1
    end

    groups = GroupParam([component], [names], :sequence, [ones(Int, Nc)],
                         [bondmat], [collect(1:Nc)], names, [ones(Int, Nc)], String[])

    chi, b_vec = sequence_chi_b(descriptor, b; rho0=rho0, gamma=gamma)
    params = SCFTLatticeFluidParam(SingleParam("b", names, b_vec),
                                    PairParam("chi", names, chi))
    ideal = init_model(idealmodel, [component], String[], false)
    return SCFTLatticeFluid([component], groups, params, Float64(rho0), Float64(kappa),
                             ideal, references)
end

"""
    SCFTSequenceSystem(model::SCFTLatticeFluid, structure::DFTStructure, options=DFTOptions();
                        ensemble=:grand_canonical, n_molecules=0.0, external_field=nothing)

Construct an [`SCFTSystem`](@ref) for a sequence-SCFT `model` built by
[`SCFTSequenceFluid`](@ref). Mirrors `SCFTSystem`'s own constructor exactly, minus the
`expand_model` step — `model` is already "expanded" (one node per species) by
construction, so [`get_species`](@ref)/`get_propagator` are called on it directly. See
[`SCFTSequenceFluid`](@ref)'s docstring for why this is safe and why the general
`mol_structure`-based path isn't used here.

`ensemble`/`n_molecules` are scalar (not `Vector`s, unlike `SCFTSystem`) since v1 supports
exactly one sequence-chain molecule type.
"""
function SCFTSequenceSystem(model::SCFTLatticeFluid, structure::DFTStructure,
                             options::DFTOptions = DFTOptions();
                             ensemble::Symbol = :grand_canonical,
                             n_molecules::Real = 0.0,
                             external_field = nothing)
    @assert length(model.components) == 1 "SCFTSequenceSystem supports exactly one sequence-chain molecule type; got $(length(model.components))."
    @assert length(structure.ρbulk) == 1 "structure.ρbulk must have exactly one entry, got $(length(structure.ρbulk))"
    structure.topology isa TwoPhaseSystem && error(
        "SCFTSequenceSystem does not support TwoPhaseSystem structures (see SCFTSystem's docstring).")

    species = get_species(model, structure; ensemble=[ensemble], n_molecules=[Float64(n_molecules)])
    FP = fptype(options)
    propagator = get_propagator(model, species, structure, options.device, FP)

    normalized_external_field = external_field isa ExternalFieldModel ? [external_field] : external_field

    return SCFTSystem(model, species, structure, propagator, options, normalized_external_field)
end

export SCFTSequenceFluid, SCFTSequenceSystem, SequenceDescriptor,
       step_descriptor, taper_descriptor, sequence_chi_b, contour_points
