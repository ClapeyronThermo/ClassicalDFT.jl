"""
    DiscreteGaussianChainPropagator(model, species, structure, device, FP=Float64)

Construct a `DiscreteGaussianChainPropagator` for (linear or branched-tree) polymer
chains with per-species-type segment lengths and automatic junction kernels, matching
the generic `(model, species, structure, device, FP)` constructor signature every other
`DFTPropagator` uses (e.g. `TangentHSPropagator`).

# Arguments
- `model::EoSModel`: Supplies `b_species = model.params.b.values`, the statistical segment
  length for each species type.
- `species::DFTSpecies`: Supplies `species.sequence`, the node-to-species mapping per
  chain (also gives `N = length.(species.sequence)`), and `species.i_groups`/
  `species.n_intergroups`, the per-chain bond tree used to enumerate bonded pairs (see
  `SCFTSpecies`'s docstring — this data can't come from `model.groups` here, since
  `SCFTSystem` keeps only the original, unexpanded `model`). Only used here to build the
  kernel map — not stored on the returned propagator (see its docstring).
- `structure::DFTStructure`: The spatial discretization structure.
- `device::Backend`: Device backend.
- `FP::Type{<:AbstractFloat}`: Float precision (default `Float64`).

For same-species bonds (α, α), the kernel uses `b_α`. For junction bonds (α, β),
the kernel uses `b_αβ = √((b_α² + b_β²) / 2)`.
"""
function DiscreteGaussianChainPropagator(
    model::EoSModel,
    species::DFTSpecies,
    structure::DFTStructure,
    device::Backend,
    ::Type{FP}=Float64
) where FP<:AbstractFloat
    b_species = model.params.b.values
    segment_species = species.sequence
    N = length.(segment_species)
    nchains = length(N)
    @assert length(segment_species) == nchains
    for c in 1:nchains
        @assert length(segment_species[c]) == N[c]
    end

    nd = dimension(structure)
    ngrid = structure.ngrid
    ω̂ = structure_fftfreq(structure)

    # Build |ν|² on the R2C (half-complex) grid.
    # First dimension uses rfftfreq (0..N/2 only), rest use fftfreq.
    # This matches the output layout of plan_rfft, halving the kernel memory and
    # matching the complexity of a real-to-complex FFT (~2× cheaper than C2C).
    lb1, ub1 = bounds(structure, 1)
    ω̂1_rfft  = rfftfreq(ngrid[1], ngrid[1] / (ub1 - lb1))
    rfft_ngrid = (ngrid[1] ÷ 2 + 1, ngrid[2:end]...)

    ν_sq = zeros(rfft_ngrid...)
    # Dimension 1: rfftfreq
    ν_sq .+= reshape(ω̂1_rfft .^ 2, ntuple(d -> d == 1 ? rfft_ngrid[1] : 1, nd))
    # Dimensions 2..nd: standard fftfreq
    for i in 2:nd
        ν_sq .+= reshape(ω̂[i] .^ 2, ntuple(d -> d == i ? rfft_ngrid[d] : 1, nd))
    end

    # Find all unique bonded species pairs across all chains, walking each chain's
    # bond tree (species.i_groups/n_intergroups, copied off the expanded model in
    # get_species — see SCFTSpecies's docstring) rather than assuming linear
    # i-1/i adjacency. Degenerates to the old linear walk when every chain is unbranched.
    bond_pairs = Set{Tuple{Int,Int}}()
    for c in 1:nchains
        seg_spec = segment_species[c]
        ig = species.i_groups[c]
        bonds = species.n_intergroups[c]
        for a in 1:N[c], b in (a+1):N[c]
            if bonds[ig[a], ig[b]] != 0
                push!(bond_pairs, minmax(seg_spec[a], seg_spec[b]))
            end
        end
    end

    # Compute kernel for each unique bond pair in R2C (half-complex) shape.
    # Precision is controlled by FP (set via DFTOptions' precision kwarg; Float64 by default).
    CT = Complex{FP}
    kernel_map = Dict{Tuple{Int,Int}, Array{CT}}()
    for (α, β) in bond_pairs
        if α == β
            b_bond = b_species[α]
        else
            b_bond = sqrt((b_species[α]^2 + b_species[β]^2) / 2)
        end
        kernel_map[(α, β)] = CT.(exp.(-2π^2 * b_bond^2 .* ν_sq ./ 3))
    end

    # Move kernels to device.
    # Use a dummy CPU array of the right shape/type to determine the device array type,
    # which avoids calling first() on an empty dict when nchains == 0 (solvent-only system).
    dummy_cpu = CT.(zeros(rfft_ngrid...))
    device_kernel_map = Dict{Tuple{Int,Int}, typeof(Adapt.adapt(device, dummy_cpu))}()
    for (key, k) in kernel_map
        device_kernel_map[key] = Adapt.adapt(device, k)
    end

    return DiscreteGaussianChainPropagator(device_kernel_map)
end

function preallocate_propagator(system, propagator::DiscreteGaussianChainPropagator, ρ, backend::Backend)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    sequence = system.species.sequence
    nchains = length(sequence)

    # Allocate q_in (bottom-up: subtree rooted at each node) and q_out (top-down:
    # everything on the far side of each node from its own subtree) arrays per chain,
    # one node-indexed slot per entry of species.sequence[c] (node order — see
    # SCFTSpecies's docstring; degenerates to plain chain-position order for a linear
    # chain, so this is the same shape the old q_fwd/q_bwd arrays used).
    # Use a concrete element type (not Vector{Any}) to avoid type-dispatch overhead
    # in the hot propagation loop. When nchains == 0 (solvent-only system), allocate
    # empty typed vectors using a dummy size so downstream code has a concrete type.
    FP = fptype(system.options)
    CT = Complex{FP}
    proto_N = nchains > 0 ? length(sequence[1]) : 1
    q_proto = allocate(backend, FP, ngrid..., proto_N)
    q_in    = Vector{typeof(q_proto)}(undef, nchains)
    q_out   = Vector{typeof(q_proto)}(undef, nchains)
    if nchains > 0
        q_in[1] = q_proto
        q_out[1] = allocate(backend, FP, ngrid..., length(sequence[1]))
        for c in 2:nchains
            q_in[c] = allocate(backend, FP, ngrid..., length(sequence[c]))
            q_out[c] = allocate(backend, FP, ngrid..., length(sequence[c]))
        end
    end

    # R2C (real-to-complex) FFT buffers:
    #   buf_r — real input,  shape ngrid
    #   buf_c — complex output, shape (ngrid[1]÷2+1, ngrid[2:end]...)
    # Using R2C instead of C2C halves the FFT work (~2× faster for real data).
    # child_buf — real scratch (shape ngrid) holding one child/sibling subtree's
    # convolved contribution before it's folded into a node's accumulated product;
    # only needed once a node can have more than one child (branching).
    rfft_ngrid = (ngrid[1] ÷ 2 + 1, ngrid[2:end]...)
    buf_r = allocate(backend, FP, ngrid...)
    buf_c = allocate(backend, CT, rfft_ngrid...)
    child_buf = allocate(backend, FP, ngrid...)

    if backend isa CPU
        P  = plan_rfft(buf_r,  1:nd; num_threads=Threads.nthreads())
        iP = plan_irfft(buf_c, ngrid[1], 1:nd; num_threads=Threads.nthreads())
    else
        P  = plan_rfft(buf_r,  1:nd)
        iP = plan_irfft(buf_c, ngrid[1], 1:nd)
    end

    return q_in, q_out, buf_r, buf_c, child_buf, P, iP
end

"""
    chain_root(species, c)

Local index (within `species.sequence[c]`/`species.i_groups[c]`) of chain `c`'s root
node — the same highest-degree-node BFS root `compute_levels` (`src/utils/base.jl`)
picks. By the sum-product/belief-propagation invariant that makes tree message passing
exact, `Q`/the per-node densities computed from `q_in`/`q_out` do not depend on which
node is chosen as root — this just needs to match the root `_dgc_tree_sweep!` uses.
"""
chain_root(species, c) = findfirst(==(1), @view(species.levels[species.i_groups[c]]))

"""
    _dgc_tree_sweep!(q_in, q_out, child_buf, buf_r, buf_c, P, iP, kernel_map, species, c, seg_spec, ef, nd)

Bottom-up/top-down discrete-Gaussian-chain sweep for one chain `c`'s bond tree
(`species.i_groups[c]`/`species.n_intergroups[c]`/`species.levels`), writing every
node's `q_in`/`q_out` value into `q_in[c]`/`q_out[c]`. `ef(α)` supplies the per-species
field factor (`exp(-w_α)` for the plain propagator, `exp(w_bulk_α - w_α)` for SCFT's
shifted-field variant — the two `propagate!` methods below only differ in this closure).

For node `k` with children `children(k)` (nodes one level further from the root, bonded
to `k`) and, for non-root `k`, parent `p` and siblings `S = children(p) \\ {k}`:
```
q_in[k]  = ef(α(k)) · ∏_{j ∈ children(k)} conv(q_in[j], kernel(α(k),α(j)))
q_out[root] = ef(α(root))
q_out[k] = conv(q_out[p] · ∏_{s ∈ S} conv(q_in[s], kernel(α(p),α(s))), kernel(α(p),α(k))) · ef(α(k))    (k ≠ root)
```
(each sibling's `q_in` must be convolved onto `p`'s position *before* being combined
with `q_out[p]` — it lives at the sibling's own position, not `p`'s, exactly like a
child's `q_in` is convolved onto its parent's position in the `q_in` recursion above).
Each of `q_in[k]`/`q_out[k]` carries exactly one factor of `ef(α(k))` — the same
double-counting convention the old linear `q_fwd`/`q_bwd` arrays used (`q_fwd[s]` and
`q_bwd[N+1-s]` each carried segment `s`'s own field factor once, so their product
double-counts it once; downstream code corrects for this, see `compute_densities!`'s
`inv_exp_field`/`exp(Δw_α)` — unchanged here). A one-child-per-node tree (a linear chain)
makes this recursion collapse exactly onto the old `q_fwd`/`q_bwd` recursion.
"""
function _dgc_tree_sweep!(q_in, q_out, child_buf, buf_r, buf_c, P, iP, kernel_map, species, c, seg_spec, ef, nd)
    ig  = species.i_groups[c]
    lev = species.levels[ig]
    Bc  = species.n_intergroups[c][ig, ig] .!= 0
    n_levels = maximum(lev)
    i_root = findfirst(==(1), lev)

    q_in_c  = q_in[c]
    q_out_c = q_out[c]

    # Bottom-up: leaves → root
    for L in n_levels:-1:1
        for k in findall(==(L), lev)
            dest_k = selectdim(q_in_c, nd+1, k)
            dest_k .= ef(seg_spec[k])
            for j in findall(Bc[k, :] .& (lev .== L + 1))
                bond_key = minmax(seg_spec[k], seg_spec[j])
                convolve!(child_buf, selectdim(q_in_c, nd+1, j), kernel_map[bond_key], P, iP, buf_r, buf_c)
                dest_k .*= child_buf
            end
        end
    end

    # Top-down: root → leaves
    selectdim(q_out_c, nd+1, i_root) .= ef(seg_spec[i_root])
    for L in 1:n_levels-1
        for k in findall(==(L), lev)
            children = findall(Bc[k, :] .& (lev .== L + 1))
            # Convolve each child's q_in onto k's own position once (it lives at the
            # child's position, not k's — same as the bottom-up pass), reused below when
            # building every OTHER child's q_out, instead of recomputing per sibling pair.
            conv_to_k = Vector{typeof(child_buf)}(undef, length(children))
            for (idx, s) in enumerate(children)
                buf_s = similar(child_buf)
                bond_key_s = minmax(seg_spec[k], seg_spec[s])
                convolve!(buf_s, selectdim(q_in_c, nd+1, s), kernel_map[bond_key_s], P, iP, buf_r, buf_c)
                conv_to_k[idx] = buf_s
            end
            for (idx, j) in enumerate(children)
                dest_j = selectdim(q_out_c, nd+1, j)
                dest_j .= selectdim(q_out_c, nd+1, k)
                for (idx2, s) in enumerate(children)
                    idx2 == idx && continue
                    dest_j .*= conv_to_k[idx2]
                end
                bond_key = minmax(seg_spec[k], seg_spec[j])
                convolve!(child_buf, dest_j, kernel_map[bond_key], P, iP, buf_r, buf_c)
                dest_j .= child_buf .* ef(seg_spec[j])
            end
        end
    end
    return nothing
end

function propagate!(system, propagator::DiscreteGaussianChainPropagator, ρ, δfδρ_res, q_in, q_out, buf_r, buf_c, child_buf, P, iP)
    nd = dimension(system)
    species = system.species
    sequence = species.sequence
    nchains = length(sequence)

    ef(α) = exp.(.-selectdim(δfδρ_res, nd+1, α))

    for c in 1:nchains
        seg_spec = sequence[c]
        _dgc_tree_sweep!(q_in, q_out, child_buf, buf_r, buf_c, P, iP, propagator.kernel_map, species, c, seg_spec, ef, nd)

        # Density contribution: modify δfδρ_res. q_in[k]*q_out[k] is the (one-factor
        # double-counted, uncorrected) per-node weight — see _dgc_tree_sweep!'s docstring;
        # this mirrors the old linear code's uncorrected q_fwd[s]*q_bwd[N+1-s] exactly.
        unique_species = unique(seg_spec)
        for α in unique_species
            seg_indices = findall(==(α), seg_spec)
            first_idx = seg_indices[1]
            sum_qq = selectdim(q_in[c], nd+1, first_idx) .* selectdim(q_out[c], nd+1, first_idx)
            for idx in seg_indices[2:end]
                sum_qq = sum_qq .+ selectdim(q_in[c], nd+1, idx) .* selectdim(q_out[c], nd+1, idx)
            end
            selectdim(δfδρ_res, nd+1, α) .-= log.(sum_qq)
        end
    end
end

"""
    propagate!(system::SCFTSystem, ρ, w, cache_propagator; w_bulk, exp_field=nothing)

Run discrete-Gaussian-chain tree sweeps (see `_dgc_tree_sweep!`) using shifted fields
`Δw = w - w_bulk`. `ρ` is accepted but unused — kept purely for positional uniformity
with every other `propagate!` method (`IdealPropagator`, `TangentHSPropagator`), all of
which take `(system, ρ, field, cache_propagator)`.

Shifting by `w_bulk` keeps propagator values near O(1) for near-uniform systems,
avoiding the numerical underflow that occurs when raw fields are large.
The shifted propagator satisfies `q̃(r,s) = q(r,s) * exp(Σ_{t=1}^{s} w_bulk[α(t)])`,
so `Q̃ ≈ 1` for uniform systems (instead of `Q ∼ exp(-N * w_bulk) ≈ 0`).
"""
function propagate!(system::SCFTSystem, ρ, w, cache_propagator;
                    w_bulk, exp_field=nothing)
    q_in, q_out, buf_r, buf_c, child_buf, P, iP = cache_propagator
    nd = dimension(system)
    propagator = system.propagator
    species = system.species
    sequence = species.sequence
    nchains = length(sequence)

    # Helper: return precomputed exp_field[α] if available, else compute on the fly.
    # exp_field[α] = exp(w_bulk[α] - w_α(r))
    ef(α) = exp_field !== nothing ? exp_field[α] :
                exp.(w_bulk[α] .- selectdim(w, nd+1, α))

    for c in 1:nchains
        seg_spec = sequence[c]
        _dgc_tree_sweep!(q_in, q_out, child_buf, buf_r, buf_c, P, iP, propagator.kernel_map, species, c, seg_spec, ef, nd)
    end
end
