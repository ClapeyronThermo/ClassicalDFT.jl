"""
    IdealPropagator <: DFTPropagator
    Type used to indicate that the propagator is ideal.

# Description
Ideal propagator for DFT calculations. Assumes all species are represented by a single bead.
"""
struct IdealPropagator <: DFTPropagator end
"""
    TangentHSPropagator <: DFTPropagator
    Type used to indicate that the propagator is a Tangent Hard-Sphere propagator.

# Description
Tangent Hard-Sphere propagator for DFT calculations. Assumes all species are made up of tangentially-bonded hard-sphere beads. Contains:
- `map`: The Fourier transform of the weights used in the propagator.

Uses the algorithm developed by Xu et al. (2009) to handle branching.

# References
1. Xu, X., Cao, D., Zhang, X. and Wang W. (2009). Universal version of density-functional theory for polymers with complex architecture. PHYSICAL REVIEW E, 79, 021805. [doi::10.1103/PhysRevE.79.021805](https://doi.org/10.1103/PhysRevE.79.021805)
"""
struct TangentHSPropagator{M} <: DFTPropagator 
    map::M
end

"""
    DiscreteGaussianChainPropagator <: DFTPropagator

Discrete Gaussian chain propagator for linear or branched-tree polymer chains. Computes
bottom-up (`q_in`) and top-down (`q_out`) propagators using Gaussian transition
probabilities via FFT, swept over each chain's bond tree (see `_dgc_tree_sweep!`,
`src/propagator/discrete_gaussian_chain.jl`) — the same tree-traversal convention
`TangentHSPropagator` uses, driven by `species.levels`/`species.i_groups`/
`species.n_intergroups`.

# Fields
- `kernel_map`: Dictionary mapping species pairs `(i, j)` (sorted) to Fourier-space Gaussian kernels.

Chain length and segment-to-species mapping (`N`/`segment_species`, matching
`TangentHSPropagator`'s minimal-state convention) are not stored here — they're already
available as `length.(system.species.sequence)`/`system.species.sequence` wherever this
propagator is used.
"""
struct DiscreteGaussianChainPropagator{K} <: DFTPropagator
    kernel_map::K
end

"""
    OrientedTree

A chain's bond tree rooted at a chain end (the first node, in node order, with at most
one bond), as used by [`WLCPropagator`](@ref). Every non-root bead has exactly one parent
bond, and its propagators are indexed by that bond's direction `u` (parent → bead).

- `order`: nodes in breadth-first order from the root (parents before children).
- `parent`: parent node of each node (`0` for the root).
- `children`: child nodes of each node.
- `qnode`: a leaf other than the root (the root itself for a one-bead molecule), where
  `Q̃ = (1/V)∫dr∫du q_in` is evaluated.
"""
struct OrientedTree
    order::Vector{Int}
    parent::Vector{Int}
    children::Vector{Vector{Int}}
    qnode::Int
end

"""
    WLCPropagator{S<:SHTPlan,K,B,G} <: DFTPropagator

Discrete worm-like-chain (bead-rod/Kratky-Porod) propagator for linear or branched chains, tracking
bond orientation on S² via a real-spherical-harmonic representation
(`src/utils/spherical_harmonics.jl`) in addition to position. Each bead step splits into
a bending half-step (a kernel diagonal in spherical-harmonic degree `l`, applied via
forward/inverse SHT — the orientation-space analogue of `DiscreteGaussianChainPropagator`
being diagonal in Fourier `|k|`) followed by a translation half-step (a fixed-length shift
along the bond direction, applied as a per-orientation-node complex phase kernel in
Fourier(r)-space, reusing the same `plan_rfft`/`plan_irfft`/`convolve!` R2C infrastructure
`DiscreteGaussianChainPropagator` uses). See `_wlc_tree_sweep!`
(`src/propagator/wlc.jl`) for the full recursion and the derivation of why bending must
come *before* translation (the shift applied to build `q_in[k]`/`q_out[k]` depends on
`k`'s own bond orientation `u`, not the previous node's). Branched chains are swept on
their bond tree rooted at a chain end; branch points are freely jointed (no bending
correlation between the bonds that meet there).

# Fields
- `sht::S`: the shared orientation quadrature grid/SHT operator (one per propagator).
- `trans_kernel::K`: per-bonded-species-pair (sorted) translation phase kernel used by
  `q_in` (shift by `+b_bond·u`), shape `(rfft_ngrid..., n_orient)`.
- `trans_kernel_conj::K`: the complex conjugate of `trans_kernel`, used by `q_out`
  (shift by `-b_bond·u`) — precomputed once at construction rather than conjugating in
  the hot `propagate!` loop.
- `bend_eig::B`: per-bonded-species-pair bending-kernel eigenvalues, already expanded
  from `bending_eigenvalues(κ_bond, L_max)` (indexed by degree `l`) out to a length-`nlm`
  vector matching `sht`'s per-coefficient layout (`expand_eig_to_lm`), so the bending
  step is a single broadcast multiply.
- `gauss_kernel::G`: per-bonded-species-pair Gaussian bond kernel (R2C shape
  `rfft_ngrid`, same formula as `DiscreteGaussianChainPropagator`) for every bond that
  touches a [`GaussianChain`](@ref) species. Empty when every species is
  orientation-resolved.
- `trees::Vector{OrientedTree}`: per chain, the bond tree rooted at a chain end (see
  [`OrientedTree`](@ref)), which fixes the sweep order and each bead's orientation label.

# Mixed (composite) chains
A chain may mix orientation-resolved species ([`WormLikeChain`](@ref), [`RigidRod`](@ref))
with [`GaussianChain`](@ref) species, e.g. a rod-coil block copolymer. A bond is a **rod
bond** (bending + fixed-length translation) only when *both* of its beads are
orientation-resolved; any bond touching a Gaussian bead is a **Gaussian bond** with
`b_bond = √((b_α² + b_β²)/2)`. Every node keeps the orientation-resolved storage layout:
a Gaussian node's propagator is constant in `u`, normalized like a chain start
(`q_in` carries the isotropic `1/4π`, `q_out` does not), so the density, partition-function
and orientation-moment kernels apply unchanged. An orientation-resolved bead entered
through a Gaussian bond starts with an isotropic orientation, exactly like the first bead
of a chain. All-orientation-resolved chains take exactly the pure-WLC code path.
"""
struct WLCPropagator{S<:SHTPlan,K,B,G} <: DFTPropagator
    sht::S
    trans_kernel::K
    trans_kernel_conj::K
    bend_eig::B
    gauss_kernel::G
    trees::Vector{OrientedTree}
end

# |k|² = (2π)²|ν|² on the R2C (half-complex, `plan_rfft`) grid of `structure`: the first
# dimension uses rfftfreq (0..N/2), the rest fftfreq.
function _rfft_k2(structure, ::Type{FP}) where FP
    nd = dimension(structure)
    ngrid = structure.ngrid
    ω̂ = structure_fftfreq(structure)
    lb1, ub1 = bounds(structure, 1)
    rfft_ngrid = (ngrid[1] ÷ 2 + 1, ngrid[2:end]...)
    k2 = zeros(FP, rfft_ngrid...)
    for d in 1:nd
        ν = d == 1 ? rfftfreq(ngrid[1], ngrid[1] / (ub1 - lb1)) : ω̂[d]
        k2 .+= reshape(FP.((2π .* ν) .^ 2), ntuple(i -> i == d ? rfft_ngrid[d] : 1, nd))
    end
    return k2
end

# Fourier transform of a discrete Gaussian bond of statistical segment length b.
_gaussian_bond_kernel(k2, b) = exp.(-k2 .* (b^2 / 6))

# Unique (sorted) species pairs joined by a bond anywhere in any chain, from the bond tree
# `species.i_groups`/`species.n_intergroups` (works for linear and branched chains).
function _bonded_species_pairs(species)
    pairs = Set{Tuple{Int,Int}}()
    for (c, seq) in enumerate(species.sequence)
        ig = species.i_groups[c]
        bonds = species.n_intergroups[c]
        for a in eachindex(seq), b in (a+1):length(seq)
            bonds[ig[a], ig[b]] != 0 && push!(pairs, minmax(seq[a], seq[b]))
        end
    end
    return pairs
end

cache_q_in(cache_propagator::Tuple) = cache_propagator[1]
cache_q_out(cache_propagator::Tuple) = cache_propagator[2]
cache_q_in(cache_propagator::NamedTuple) = cache_propagator.q_in
cache_q_out(cache_propagator::NamedTuple) = cache_propagator.q_out

propagate!(system::DGTSystem, δf_res, ρ, ::Nothing) = nothing

function propagate!(system::AbstractcDFTSystem, δf_res, ρ, cache_propagator)
    if !(system.propagator isa IdealPropagator)
        return propagate!(system, system.propagator, δf_res, ρ, cache_propagator...)
    end
end

include("ideal.jl")
include("tangent_hs.jl")
include("discrete_gaussian_chain.jl")
include("wlc.jl")