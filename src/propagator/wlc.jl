"""
    expand_eig_to_lm(eig_l, sht::SHTPlan)

Broadcast per-degree bending eigenvalues `eig_l` (length `L_max+1`, from
`bending_eigenvalues`) out to a length-`nlm` vector matching `sht`'s per-coefficient
`cos_range`/`sin_range` layout, so a bending step (`wlc_bend!`) is a single broadcast
multiply rather than a loop over degrees.
"""
function expand_eig_to_lm(eig_l::AbstractVector{FP}, sht::SHTPlan) where FP<:AbstractFloat
    eig_lm = zeros(FP, sht.nlm)
    for m in 0:sht.L_max, l in m:sht.L_max
        li = l - m + 1
        eig_lm[sht.cos_range[m+1][li]] = eig_l[l+1]
        m >= 1 && (eig_lm[sht.sin_range[m][li]] = eig_l[l+1])
    end
    return eig_lm
end

"""
    WLCPropagator(model, species, structure, device, FP=Float64; L_max=8)

Construct a `WLCPropagator`, matching the generic `(model, species, structure, device,
FP)` constructor signature every `DFTPropagator` uses.

`model.params.b`/`model.params.lp` give per-species bond length / persistence length;
`κ_α = lp_α/b_α`. For a junction bond `(α,β)`, `b_bond = sqrt((b_α²+b_β²)/2)` (matching
`DiscreteGaussianChainPropagator`'s convention) and `κ_bond = sqrt((κ_α²+κ_β²)/2)`.

Requires `dimension(structure) == 3` — orientation is inherently a 3-component unit
vector, and v1 does not address how it would project onto a reduced-dimensionality
domain — and every chain in `species.sequence` must be linear: node `a` and `b` (in
`species.sequence[c]`'s own order, which SCFT's `expand_groups` already builds in exact
chain-position order — *not* `species.levels`'s BFS-depth-from-highest-degree-root
labeling, which is unrelated to chain position and generally is not `1:N_c` even for an
unbranched chain) must be bonded iff `b == a+1`. Branched `mol_structure` is rejected with
a clear error, since v1 has no sibling-combination logic for the bend-then-translate
recursion.

`L_max` (keyword, default 8) sets both the spherical-harmonic truncation degree and, via
`SHTPlan`, the orientation quadrature grid size — larger `L_max` is needed to resolve
`bend_eig`'s decay for stiffer species (larger `κ_α = lp_α/b_α`, since `κ̂_l` decays over
more degrees `l` as `κ` grows).
"""
function WLCPropagator(
    model::EoSModel,
    species::DFTSpecies,
    structure::DFTStructure,
    device::Backend,
    ::Type{FP}=Float64;
    L_max::Int=8
) where FP<:AbstractFloat
    nd = dimension(structure)
    nd == 3 || error(
        "WLCPropagator only supports 3D spatial domains (dimension(structure) == 3) — " *
        "orientation is inherently a 3-component unit vector; got dimension(structure) = $nd."
    )

    segment_species = species.sequence
    N = length.(segment_species)
    nchains = length(N)
    for c in 1:nchains
        ig = species.i_groups[c]
        bonds = species.n_intergroups[c]
        Nc = N[c]
        for a in 1:Nc, b in (a+1):Nc
            is_bonded = bonds[ig[a], ig[b]] != 0
            is_bonded == (b == a + 1) || error(
                "WLCPropagator only supports LINEAR chains (v1 has no branching " *
                "support) — chain $c is not a simple path in its own chain-position order."
            )
        end
    end

    sht = SHTPlan(L_max, FP)
    n_orient = sht.n_theta * sht.n_phi

    ngrid = structure.ngrid
    ω̂ = structure_fftfreq(structure)
    lb1, ub1 = bounds(structure, 1)
    ω̂1_rfft = rfftfreq(ngrid[1], ngrid[1] / (ub1 - lb1))
    rfft_ngrid = (ngrid[1] ÷ 2 + 1, ngrid[2:end]...)

    ν = ntuple(nd) do i
        vec = i == 1 ? ω̂1_rfft : ω̂[i]
        reshape(FP.(vec), ntuple(d -> d == i ? rfft_ngrid[d] : 1, nd))
    end

    b_species = model.params.b.values
    lp_species = model.params.lp.values

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

    CT = Complex{FP}
    trans_kernel = Dict{Tuple{Int,Int}, Array{CT}}()
    trans_kernel_conj = Dict{Tuple{Int,Int}, Array{CT}}()
    bend_eig = Dict{Tuple{Int,Int}, Vector{FP}}()

    for (α, β) in bond_pairs
        if α == β
            b_bond = FP(b_species[α])
            κ_bond = FP(lp_species[α] / b_species[α])
        else
            b_bond = sqrt(FP(b_species[α]^2 + b_species[β]^2) / 2)
            κα, κβ = lp_species[α] / b_species[α], lp_species[β] / b_species[β]
            κ_bond = sqrt(FP(κα^2 + κβ^2) / 2)
        end

        phase = zeros(CT, rfft_ngrid..., n_orient)
        for i in 1:n_orient
            d = b_bond .* @view sht.u_nodes[:, i]
            arg = zeros(FP, rfft_ngrid...)
            for dim in 1:nd
                arg = arg .+ ν[dim] .* d[dim]
            end
            selectdim(phase, nd+1, i) .= CT.(exp.(2 * FP(pi) * im .* arg))
        end
        trans_kernel[(α, β)] = phase
        trans_kernel_conj[(α, β)] = conj.(phase)

        eig_l = FP.(bending_eigenvalues(κ_bond, L_max))
        bend_eig[(α, β)] = expand_eig_to_lm(eig_l, sht)
    end

    dummy_cpu = zeros(CT, rfft_ngrid..., n_orient)
    KT = typeof(Adapt.adapt(device, dummy_cpu))
    device_trans_kernel = Dict{Tuple{Int,Int}, KT}()
    device_trans_kernel_conj = Dict{Tuple{Int,Int}, KT}()
    device_bend_eig = Dict{Tuple{Int,Int}, typeof(Adapt.adapt(device, zeros(FP, sht.nlm)))}()
    for (key, k) in trans_kernel
        device_trans_kernel[key] = Adapt.adapt(device, k)
    end
    for (key, k) in trans_kernel_conj
        device_trans_kernel_conj[key] = Adapt.adapt(device, k)
    end
    for (key, k) in bend_eig
        device_bend_eig[key] = Adapt.adapt(device, k)
    end

    return WLCPropagator(sht, device_trans_kernel, device_trans_kernel_conj, device_bend_eig)
end

function preallocate_propagator(system, propagator::WLCPropagator, ρ, backend::Backend)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    sequence = system.species.sequence
    nchains = length(sequence)
    sht = propagator.sht
    n_orient = sht.n_theta * sht.n_phi
    nlm = sht.nlm

    FP = fptype(system.options)
    CT = Complex{FP}

    q_in  = [allocate(backend, FP, ngrid..., n_orient, length(seq)) for seq in sequence]
    q_out = [allocate(backend, FP, ngrid..., n_orient, length(seq)) for seq in sequence]

    rfft_ngrid = (ngrid[1] ÷ 2 + 1, ngrid[2:end]...)
    buf_r = allocate(backend, FP, ngrid...)
    buf_c = allocate(backend, CT, rfft_ngrid...)
    child_buf = allocate(backend, FP, ngrid..., n_orient)
    qgrid_buf = allocate(backend, FP, ngrid..., n_orient)
    qlm_buf = allocate(backend, FP, ngrid..., nlm)

    if backend isa CPU
        P  = plan_rfft(buf_r,  1:nd; num_threads=Threads.nthreads())
        iP = plan_irfft(buf_c, ngrid[1], 1:nd; num_threads=Threads.nthreads())
    else
        P  = plan_rfft(buf_r,  1:nd)
        iP = plan_irfft(buf_c, ngrid[1], 1:nd)
    end

    return (; q_in, q_out, buf_r, buf_c, child_buf, qgrid_buf, qlm_buf, P, iP)
end

"""
    wlc_translate!(dest, src, kernel, buf_r, buf_c, P, iP, nd)

Fixed-bond-length translation half-step: for each orientation node `i`,
`dest[...,i] = conv(src[...,i], kernel[...,i])` via the R2C `convolve!`
(`src/utils/integrals.jl`) — `n_orient` independent FFT-pair applications, since
translation mixes position `r` but not orientation `u`. `kernel` is
`propagator.trans_kernel[bond_key]` (shift by `+b_bond·u`, used by `q_in`) or
`propagator.trans_kernel_conj[bond_key]` (shift by `-b_bond·u`, used by `q_out`).
"""
function wlc_translate!(dest, src, kernel, buf_r, buf_c, P, iP, nd)
    n_orient = size(src, nd + 1)
    for i in 1:n_orient
        convolve!(selectdim(dest, nd + 1, i), selectdim(src, nd + 1, i),
                  selectdim(kernel, nd + 1, i), P, iP, buf_r, buf_c)
    end
    return dest
end

"""
    wlc_bend!(dest, src, eig_lm, sht, qlm_buf, nd)

Bending half-step: forward-transforms `src` (shape `(ngrid...,n_orient)`) to
spherical-harmonic coefficients `qlm_buf` via `sht_forward!`, multiplies by the
precomputed per-coefficient eigenvalue vector `eig_lm` (`expand_eig_to_lm`) — diagonal in
spherical-harmonic degree `l`, independent of order `m` and of `r` — then
inverse-transforms back to the orientation grid into `dest` via `sht_inverse!`.
"""
function wlc_bend!(dest, src, eig_lm, sht, qlm_buf, nd)
    sht_forward!(qlm_buf, src, sht)
    qlm_buf .*= reshape(eig_lm, ntuple(_ -> 1, nd)..., length(eig_lm))
    sht_inverse!(dest, qlm_buf, sht)
    return dest
end

"""
    _wlc_linear_sweep!(q_in_c, q_out_c, buf_r, buf_c, child_buf, qgrid_buf, qlm_buf, P, iP,
                        propagator, seg_spec, ef, nd)

Bottom-up (`q_in`)/top-down (`q_out`) discrete worm-like-chain sweep for one LINEAR chain
(node order == chain-position order). `q_in_c`/`q_out_c` have shape
`(ngrid..., n_orient, N_c)`.

`q_in[k](r,u)` is the statistical weight of the sub-chain `(1,...,k)` ending at node `k`
at position `r`, with bond `(k-1,k)`'s direction equal to `u`. Node 1 has no incoming
bond, so `q_in[1]` carries no orientation preference — an isotropic *probability
density* over `u`, `1/4π` (not a bare constant `1`):
```
q_in[1](r,u) = ef(α(1)) / 4π                                              (constant in u)
```
The `1/4π` is not optional: `compute_partition_functions`
(`src/models/SCFT/scft.jl`) computes `Q̃_c` from `q_in` *alone* (no compensating `q_out`
factor), so `q_in`'s own normalization directly sets `Q̃`'s absolute scale — without it,
a uniform (`Δw=0`) system would give `Q̃ = 4π` instead of the `Q̃ ≈ 1` every other
propagator convention (and the free-energy formulas built on it) assumes, since bending
and translation both exactly preserve a function that is constant over the *whole*
sphere (`κ̂_0 = 1`, `trans_kernel` at zero frequency `= 1`), so an unnormalized seed's
factor of `∫_{S²} du = 4π` would otherwise survive unchanged all the way to `q_in[N_c]`.
`q_out[N_c]`'s seed below is deliberately **not** given the same `1/4π` — `compute_densities!`
sums `∫du q_in(r,u,k)·q_out(r,u,k)` (a *product*), and `prefactor = n_molecules/(V_eff·Q̃)`
already divides out `q_in`'s `1/4π` via `Q̃`, so giving `q_out` its own `1/4π` too would
double-correct and make densities come out `4π`-fold too small; this asymmetry was
verified against a uniform-melt fixed point (`test/test_scft.jl`), which needs both
`Q̃ ≈ 1` *and* `ρ` conserved at bulk simultaneously to catch it.

For `k=2:N_c`, **bending then translation** — not the other order:
```
q_in[k](r,u) = ef(α(k)) · translate_{+b_bond·u}( bend(q_in[k-1](·,u)) )(r,u)
```
Bending combines over the *previous* bond direction `u'` with the kernel `K(u,u')` at
fixed `r`, producing an intermediate function still indexed by `u` (the *new*, k-1↔k
bond's direction); translation then shifts each `u`-slice of that intermediate by that
same `u`'s own displacement `b_bond·u`, because node `k`'s position is node `(k-1)`'s
position plus the `(k-1,k)` bond vector `b_bond·u`. (Swapping the order — translating
first by the *previous* node's orientation, then bending — would shift by the wrong
node's direction; see `WLCPropagator`'s docstring.)

Top-down mirrors this with the bond direction reversed: `q_out[k]` looks toward node
`k+1`, which sits at `r + b_bond·u` (not `r - b_bond·u`), so its translation uses the
*conjugate* kernel:
```
q_out[N_c](r,u) = ef(α(N_c))
q_out[k](r,u)   = ef(α(k)) · translate_{-b_bond·u}( bend(q_out[k+1](·,u)) )(r,u)     (k < N_c)
```
Each of `q_in[k]`/`q_out[k]` carries exactly one factor of `ef(α(k))` — the same
double-counting convention `_dgc_tree_sweep!` uses, corrected downstream via
`exp(Δw_α)` in `compute_densities!`.
"""
function _wlc_linear_sweep!(q_in_c, q_out_c, buf_r, buf_c, child_buf, qgrid_buf, qlm_buf, P, iP,
                             propagator, seg_spec, ef, nd)
    Nc = length(seg_spec)
    sht = propagator.sht

    selectdim(q_in_c, nd + 2, 1) .= ef(seg_spec[1]) ./ (4 * pi)
    for k in 2:Nc
        bond_key = minmax(seg_spec[k-1], seg_spec[k])
        prev = selectdim(q_in_c, nd + 2, k - 1)
        wlc_bend!(child_buf, prev, propagator.bend_eig[bond_key], sht, qlm_buf, nd)
        wlc_translate!(qgrid_buf, child_buf, propagator.trans_kernel[bond_key], buf_r, buf_c, P, iP, nd)
        selectdim(q_in_c, nd + 2, k) .= ef(seg_spec[k]) .* qgrid_buf
    end

    selectdim(q_out_c, nd + 2, Nc) .= ef(seg_spec[Nc])
    for k in (Nc-1):-1:1
        bond_key = minmax(seg_spec[k], seg_spec[k+1])
        nxt = selectdim(q_out_c, nd + 2, k + 1)
        wlc_bend!(child_buf, nxt, propagator.bend_eig[bond_key], sht, qlm_buf, nd)
        wlc_translate!(qgrid_buf, child_buf, propagator.trans_kernel_conj[bond_key], buf_r, buf_c, P, iP, nd)
        selectdim(q_out_c, nd + 2, k) .= ef(seg_spec[k]) .* qgrid_buf
    end
    return nothing
end

function propagate!(system, propagator::WLCPropagator, ρ, δfδρ_res, q_in, q_out, buf_r, buf_c,
                     child_buf, qgrid_buf, qlm_buf, P, iP)
    nd = dimension(system)
    species = system.species
    sequence = species.sequence
    nchains = length(sequence)

    ef(α) = exp.(.-selectdim(δfδρ_res, nd + 1, α))

    for c in 1:nchains
        seg_spec = sequence[c]
        _wlc_linear_sweep!(q_in[c], q_out[c], buf_r, buf_c, child_buf, qgrid_buf, qlm_buf, P, iP,
                            propagator, seg_spec, ef, nd)

        unique_species = unique(seg_spec)
        quad_weight = propagator.sht.quad_weight
        for α in unique_species
            seg_indices = findall(==(α), seg_spec)
            qq = selectdim(q_in[c], nd + 2, seg_indices[1]) .* selectdim(q_out[c], nd + 2, seg_indices[1])
            sum_qq = _orientation_marginalize(qq, quad_weight, nd + 1)
            for idx in seg_indices[2:end]
                qq = selectdim(q_in[c], nd + 2, idx) .* selectdim(q_out[c], nd + 2, idx)
                sum_qq = sum_qq .+ _orientation_marginalize(qq, quad_weight, nd + 1)
            end
            selectdim(δfδρ_res, nd + 1, α) .-= log.(sum_qq)
        end
    end
end

"""
    propagate!(system::SCFTSystem, ρ, w, cache_propagator; w_bulk, exp_field=nothing)

WLC analogue of `DiscreteGaussianChainPropagator`'s SCFT `propagate!`: runs
`_wlc_linear_sweep!` for every chain using shifted fields
`ef(α) = exp(w_bulk[α] - w_α)`.
"""
function propagate!(system::SCFTSystem, ρ, w, cache_propagator::NamedTuple;
                    w_bulk, exp_field=nothing)
    (; q_in, q_out, buf_r, buf_c, child_buf, qgrid_buf, qlm_buf, P, iP) = cache_propagator
    nd = dimension(system)
    propagator = system.propagator
    species = system.species
    sequence = species.sequence
    nchains = length(sequence)

    ef(α) = exp_field !== nothing ? exp_field[α] :
                exp.(w_bulk[α] .- selectdim(w, nd + 1, α))

    for c in 1:nchains
        seg_spec = sequence[c]
        _wlc_linear_sweep!(q_in[c], q_out[c], buf_r, buf_c, child_buf, qgrid_buf, qlm_buf, P, iP,
                            propagator, seg_spec, ef, nd)
    end
end
