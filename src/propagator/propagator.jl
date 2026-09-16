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
    WLCPropagator{S<:SHTPlan,K,B} <: DFTPropagator

Discrete worm-like-chain (bead-rod/Kratky-Porod) propagator for LINEAR chains, tracking
bond orientation on S² via a real-spherical-harmonic representation
(`src/utils/spherical_harmonics.jl`) in addition to position. Each bead step splits into
a bending half-step (a kernel diagonal in spherical-harmonic degree `l`, applied via
forward/inverse SHT — the orientation-space analogue of `DiscreteGaussianChainPropagator`
being diagonal in Fourier `|k|`) followed by a translation half-step (a fixed-length shift
along the bond direction, applied as a per-orientation-node complex phase kernel in
Fourier(r)-space, reusing the same `plan_rfft`/`plan_irfft`/`convolve!` R2C infrastructure
`DiscreteGaussianChainPropagator` uses). See `_wlc_linear_sweep!`
(`src/propagator/wlc.jl`) for the full recursion and the derivation of why bending must
come *before* translation (the shift applied to build `q_in[k]`/`q_out[k]` depends on
`k`'s own bond orientation `u`, not the previous node's). Branching is out of scope for
v1 — this propagator assumes every chain in `species.sequence` is a simple linear chain.

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
"""
struct WLCPropagator{S<:SHTPlan,K,B} <: DFTPropagator
    sht::S
    trans_kernel::K
    trans_kernel_conj::K
    bend_eig::B
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