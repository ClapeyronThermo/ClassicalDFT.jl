"""
    periodic_interp(field::AbstractArray, structure::DFTStructure, point)

Multilinear interpolation of a scalar field defined on `structure`'s own periodic grid
(`structure.ngrid` points spanning each dimension's `bounds`, spacing `structure_dz`) at
an arbitrary continuous `point` (any indexable length-`dimension(structure)` container),
with periodic wraparound in every dimension — the same convention this package's
FFT-based propagators already assume (`dz = (ub-lb)/ngrid`, so grid index `ngrid+1`
coincides with index `1` shifted by one period, not a separate point one `dz` further
out).

Generic over `dimension(structure) ∈ {1,2,3}`; `field` must have exactly that many
leading dimensions (trailing dimensions, e.g. an orientation-node index, are not
supported here — index those before calling).
"""
function periodic_interp(field::AbstractArray, structure::DFTStructure, point)
    nd = dimension(structure)
    ngrid = structure.ngrid
    dz = structure_dz(structure)
    lo = ntuple(d -> bounds(structure, d)[1], nd)

    xi = ntuple(nd) do d
        mod(point[d] - lo[d], ngrid[d] * dz[d]) / dz[d]
    end
    idx0 = ntuple(d -> floor(Int, xi[d]), nd)
    frac = ntuple(d -> xi[d] - idx0[d], nd)

    val = zero(eltype(field))
    for corner in CartesianIndices(ntuple(_ -> 0:1, nd))
        w = one(eltype(field))
        idxs = ntuple(nd) do d
            c = corner[d]
            w *= c == 0 ? (1 - frac[d]) : frac[d]
            mod(idx0[d] + c, ngrid[d]) + 1
        end
        val += w * field[idxs...]
    end
    return val
end

"""
    legendre_all(x::Real, L_max::Int)

The Legendre polynomials `P_0(x),...,P_{L_max}(x)` via Bonnet's three-term recursion.
Shared helper behind [`bend_kernel_value`](@ref).
"""
function legendre_all(x::Real, L_max::Int)
    P = zeros(typeof(float(x)), L_max + 1)
    P[1] = 1.0
    L_max >= 1 && (P[2] = x)
    for l in 1:(L_max - 1)
        P[l + 2] = ((2l + 1) * x * P[l + 1] - l * P[l]) / (l + 1)
    end
    return P
end

"""
    bend_kernel_value(u1::AbstractVector, u2::AbstractVector, eig_l::AbstractVector)

The exact, isotropic single-bond FRC (freely-rotating-chain) angular transition density
`P(u2|u1) = Σ_l (2l+1)/(4π) · eig_l[l+1] · P_l(u1·u2)` implied by a bond's own
[`bending_eigenvalues`](@ref) `eig_l` — the same Legendre-mode decomposition
`WLCPropagator`'s `expand_eig_to_lm`/spherical-harmonic machinery represents internally,
evaluated here directly in real (not spherical-harmonic) space since the kernel only
depends on the angle between the two bond directions. Properly normalized: `∫dΩ_2
P(u2|u1) = eig_l[1] = 1` for every bond (`bending_eigenvalues`'s own `l=0` mode is always
exactly `1`).

For an exactly rigid bond (`κ=Inf`, every degree exactly `1`), this truncated-series
kernel only *approximates* the true delta function on the sphere — [`sample_chain`](@ref)
special-cases that limit as forced equality (`u_k=u_{k-1}`) instead of sampling from it,
which is both exact and cheaper.
"""
function bend_kernel_value(u1::AbstractVector, u2::AbstractVector, eig_l::AbstractVector)
    cosθ = clamp(dot(u1, u2), -1.0, 1.0)
    P = legendre_all(cosθ, length(eig_l) - 1)
    return sum((2 * (l - 1) + 1) / (4π) * eig_l[l] * P[l] for l in eachindex(eig_l))
end

function _kappa_and_bondlength(b_species, lp_species, s1::Int, s2::Int)
    if s1 == s2
        return lp_species[s1] / b_species[s1], b_species[s1]
    end
    b_bond = sqrt((b_species[s1]^2 + b_species[s2]^2) / 2)
    κ1, κ2 = lp_species[s1] / b_species[s1], lp_species[s2] / b_species[s2]
    κ_bond = isinf(κ1) != isinf(κ2) ? (isinf(κ1) ? κ2 : κ1) : sqrt((κ1^2 + κ2^2) / 2)
    return κ_bond, b_bond
end

function _sample_categorical(rng::AbstractRNG, weights::AbstractVector)
    cw = cumsum(weights)
    return searchsortedfirst(cw, rand(rng) * cw[end])
end

"""
    bead_marginal_density(system::SCFTWLCSystem, q_in, q_out, bead::Int; chain::Int=1)

The (unnormalized) marginal spatial density of one specific bead, `ρ_bead(r) = ∫dΩ
q_in(r,u,bead) q_out(r,u,bead)`, as opposed to [`compute_densities!`](@ref)'s own
`ρ_α(r)`, which pools every bead of species `α` together. Only the *shape* is
meaningful (`q_in`/`q_out`'s overall scale is gauge-dependent — see
[`compute_orientation_moments`](@ref)'s docstring), which is exactly what
[`terminal_orientation_anchor`](@ref) needs: the `argmax` location of this array is that
bead's own most likely position.

Returns an `ngrid`-shaped array for chain type `chain` (default the first/only molecule
type).
"""
function bead_marginal_density(system::SCFTWLCSystem, q_in, q_out, bead::Int; chain::Int=1)
    nd = dimension(system)
    sht = system.propagator.sht
    n_orient = size(sht.u_nodes, 2)
    ngrid = system.structure.ngrid
    ρ_bead = zeros(eltype(q_in[chain]), ngrid...)
    qin_c, qout_c = q_in[chain], q_out[chain]
    for i in 1:n_orient
        ρ_bead .+= selectdim(selectdim(qin_c, nd + 1, i), nd + 1, bead) .*
                   selectdim(selectdim(qout_c, nd + 1, i), nd + 1, bead) .* sht.quad_weight[i]
    end
    return ρ_bead
end

"""
    terminal_orientation_anchor(system::SCFTWLCSystem, q_in, q_out, bead::Int;
                                chain::Int=1, orientation::Symbol=:mode)

The most likely position `r0` for `bead` (`argmax` of [`bead_marginal_density`](@ref)),
paired with either the most likely (`orientation=:mode`) or mean (`orientation=:mean`)
orientation there — a principled starting point for [`sample_chain`](@ref), rather than
a manually chosen `(r0,u0)`.

`:mode` returns the single quadrature-grid orientation node maximizing the local weight
`ψ(u) ∝ q_in(r0,u,bead) q_out(r0,u,bead)`. At a position with no net directional bias
(e.g. by symmetry), `ψ` can be nearly flat, making `:mode`'s specific choice among
several near-tied candidates numerically arbitrary rather than physical — check the
returned `mean_u`'s own magnitude (near `0` there) to tell the two cases apart.

`:mean` instead returns the *nearest quadrature node* to the polar vector `⟨u⟩` (see
[`mean_orientation_field`](@ref)'s docstring for why this is generally not a unit
vector, and thus not itself usable as a bond direction) — the returned `mean_u` is the
raw, un-normalized vector, so its magnitude (`≤1`) tells you how peaked the true
distribution is: near `1` means `u0` is a faithful representative, near `0` means the
orientation there is close to isotropic and `u0` is only a somewhat arbitrary
representative of "no strong preference."

Returns `(r0, u0, mean_u)`: `r0::NTuple` (grid-point coordinates), `u0::Vector{Float64}`
(a unit vector — one of `system.propagator.sht.u_nodes`' own columns), `mean_u` (the raw
polar vector at `r0`, for diagnosing the `:mode` case and reporting alongside `:mean`).
"""
function terminal_orientation_anchor(system::SCFTWLCSystem, q_in, q_out, bead::Int;
                                      chain::Int=1, orientation::Symbol=:mode)
    orientation in (:mode, :mean) || throw(ArgumentError("orientation must be :mode or :mean, got $orientation"))
    nd = dimension(system)
    structure = system.structure
    dz = structure_dz(structure)
    sht = system.propagator.sht
    n_orient = size(sht.u_nodes, 2)
    qin_c, qout_c = q_in[chain], q_out[chain]

    ρ_bead = bead_marginal_density(system, q_in, q_out, bead; chain=chain)
    idx0 = Tuple(argmax(ρ_bead))
    r0 = ntuple(d -> bounds(structure, d)[1] + (idx0[d] - 1) * dz[d], nd)

    ψ = [selectdim(selectdim(qin_c, nd + 1, i), nd + 1, bead)[idx0...] *
         selectdim(selectdim(qout_c, nd + 1, i), nd + 1, bead)[idx0...] * sht.quad_weight[i]
         for i in 1:n_orient]
    ψsum = sum(ψ)
    mean_u = sum(ψ[i] .* sht.u_nodes[:, i] for i in 1:n_orient) ./ ψsum

    u0_idx = if orientation == :mode
        argmax(ψ)
    else
        mean_u_mag = norm(mean_u)
        dir = mean_u_mag > 0 ? mean_u ./ mean_u_mag : [1.0, 0.0, 0.0]
        argmax(dot(sht.u_nodes[:, i], dir) for i in 1:n_orient)
    end
    return r0, sht.u_nodes[:, u0_idx], mean_u
end

"""
    sample_chain(system::SCFTWLCSystem, q_in, q_out, anchor_bead::Int, r0, u0;
                 chain::Int=1, rng::AbstractRNG=Random.default_rng())

Reconstruct one full bead-by-bead conformation `R_1,...,R_N` of chain type `chain`,
consistent with the converged mean field, conditioned on `anchor_bead` sitting at
position `r0` with orientation `u0` (a length-3 unit vector, e.g. from
[`terminal_orientation_anchor`](@ref)) — a genuinely different kind of quantity from
every other function built on `q_in`/`q_out` elsewhere in this package
(`orientation_order_parameter`, `mean_orientation_field`, `compute_densities!`, ...),
all of which are ENSEMBLE-AVERAGED marginals. This is one concrete realization.

Discrete WLC convention: `N` beads, `N-1` bonds, bond `k`'s direction is `u_k`,
`R_{k+1} = R_k + b_k·u_k[1:nd]` (only the first `nd` Cartesian components of `u`
translate position, matching `WLCPropagator`'s own convention for `dimension(system) <
3`; bending always uses the full 3D `u` regardless). Walking away from the anchor in
each direction, every subsequent bond direction is sampled from
`P(u_k|u_{k-1}) ∝ bend_kernel_value(u_{k-1},u_k,eig_l) · q_out(R_k,u_k,k)` (walking
toward increasing bead index) or the `q_in`-mirrored version (walking toward decreasing
bead index) — except across an exactly rigid bond (`κ=Inf`), where that's forced
equality (`u_k=u_{k-1}`) rather than an approximate sample from a truncated-series
kernel (see [`bend_kernel_value`](@ref)'s docstring).

`anchor_bead` need not be a terminal bead — the walk proceeds independently in both
directions from it, `q_out`-driven above `anchor_bead` and `q_in`-driven below it.

Returns a `Vector{Vector{Float64}}` of length-`nd` bead positions, `R[anchor_bead] ==
collect(r0)` exactly.
"""
function sample_chain(system::SCFTWLCSystem, q_in, q_out, anchor_bead::Int, r0, u0;
                       chain::Int=1, rng::AbstractRNG=Random.default_rng())
    nd = dimension(system)
    structure = system.structure
    species = system.species.sequence[chain]
    N = length(species)
    (1 <= anchor_bead <= N) || throw(ArgumentError("anchor_bead=$anchor_bead out of range 1:$N"))
    b_species = system.model.params.b.values
    lp_species = system.model.params.lp.values
    L_max = system.propagator.sht.L_max
    u_nodes = system.propagator.sht.u_nodes
    quad_w = system.propagator.sht.quad_weight
    n_orient = size(u_nodes, 2)
    qin_c, qout_c = q_in[chain], q_out[chain]

    R = Vector{Vector{Float64}}(undef, N)
    U = Vector{Vector{Float64}}(undef, N)
    R[anchor_bead] = collect(Float64, r0)
    U[anchor_bead] = collect(Float64, u0)

    function step!(k::Int, k_prev::Int, qfield)
        s1, s2 = species[min(k, k_prev)], species[max(k, k_prev)]
        if s1 == s2 && isinf(lp_species[s1])
            U[k] = U[k_prev]
            R[k] = R[k_prev] .+ b_species[s1] .* U[k_prev][1:nd]
            return nothing
        end
        κ_bond, b_bond = _kappa_and_bondlength(b_species, lp_species, s1, s2)
        eig_l = bending_eigenvalues(κ_bond, L_max)
        step_sign = k > k_prev ? 1 : -1
        w_arr = zeros(n_orient)
        cands = Vector{Vector{Float64}}(undef, n_orient)
        for i in 1:n_orient
            uc = @view u_nodes[:, i]
            rc = R[k_prev] .+ (step_sign * b_bond) .* uc[1:nd]
            cands[i] = rc
            qval = periodic_interp(selectdim(selectdim(qfield, nd + 1, i), nd + 1, k), structure, rc)
            w_arr[i] = bend_kernel_value(U[k_prev], uc, eig_l) * quad_w[i] * max(qval, 0.0)
        end
        idx = _sample_categorical(rng, w_arr)
        U[k] = u_nodes[:, idx]; R[k] = cands[idx]
        return nothing
    end

    for k in (anchor_bead + 1):N
        step!(k, k - 1, qout_c)
    end
    for k in (anchor_bead - 1):-1:1
        step!(k, k + 1, qin_c)
    end
    return R
end

"""
    plot_chain_conformations(system::SCFTWLCSystem, ρ, chains; kwargs...)

Plot one or more [`sample_chain`](@ref) conformations (`chains`, a `Vector` of that
function's own return type, or a single one) over the converged density profile `ρ`,
handling `dimension(system) ∈ {1,2}` and periodic wraparound (a sampled path can — and
often does — legitimately extend outside the structure's own `[0,L)` box; this tiles
the periodic background to match, rather than clipping). Requires `using Makie` (or a
Makie backend, e.g. `CairoMakie`) — this is a stub until then.
"""
function plot_chain_conformations end

"""
    plot_orientation_field(system::SCFTWLCSystem, q_in, q_out; kwargs...)

Plot [`mean_orientation_field`](@ref)'s `⟨u⟩(r)` as an arrow map over the converged
density profile, handling `dimension(system) ∈ {1,2}`. Requires `using Makie` (or a
Makie backend, e.g. `CairoMakie`) — this is a stub until then.
"""
function plot_orientation_field end
