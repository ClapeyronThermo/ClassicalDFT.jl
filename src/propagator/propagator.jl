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

propagate!(system::DGTSystem, δf_res, ρ, ::Nothing) = nothing

function propagate!(system::AbstractcDFTSystem, δf_res, ρ, cache_propagator)
    if !(system.propagator isa IdealPropagator)
        return propagate!(system, system.propagator, δf_res, ρ, cache_propagator...)
    end
end

include("ideal.jl")
include("tangent_hs.jl")
include("discrete_gaussian_chain.jl")