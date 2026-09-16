include("eos.jl")

"""
    compute_fields!(system::SCFTSystem, ρ, w)

Compute the mean-field potential fields `w` from the density profiles `ρ`, for local Flory-Huggins interactions with Helfand compressibility:
```
 w_α(r) = Σ_β (χ_αβ / ρ₀) ρ_β(r) + (ζ / ρ₀)(ρ₊(r) / ρ₀ - 1)
```
where `ρ₊ = Σ_α ρ_α` is the total density, and `χ`/`ρ₀`/`ζ` come from `system.model` (an [`SCFTLatticeFluid`](@ref)).
"""
function compute_fields!(system::SCFTSystem, ρ, w; scratch=nothing)
    nd = dimension(system)
    nspecies = length(system.model.groups.flattenedgroups)
    FT = eltype(w)
    chi = FT.(system.model.params.chi.values)
    rho0 = FT(system.model.rho0)
    kappa = FT(system.model.kappa)

    # Use preallocated scratch if provided, otherwise allocate.
    # scratch must have the same shape/device as a single species slice of w.
    ρ_total = scratch !== nothing ? scratch : similar(selectdim(w, nd+1, 1))

    # Accumulate total density into scratch (in-place, no allocation)
    ρ_total .= zero(FT)
    for α in 1:nspecies
        ρ_total .+= selectdim(ρ, nd+1, α)
    end

    # Overwrite ρ_total in-place with the compressibility term to avoid a second allocation:
    # comp_term = (ζ / ρ₀)(ρ₊ / ρ₀ - 1)
    @. ρ_total = (kappa / rho0) * (ρ_total / rho0 - one(FT))

    # Field for each species
    for α in 1:nspecies
        w_α = selectdim(w, nd+1, α)
        w_α .= ρ_total
        for β in 1:nspecies
            if chi[α, β] != zero(FT)
                w_α .+= (chi[α, β] / rho0) .* selectdim(ρ, nd+1, β)
            end
        end
    end
end

"""
    compute_bulk_fields(model::EoSModel, bulk_densities::Vector{Float64})

Compute bulk (uniform) fields from bulk densities. Returns a vector of field values, one per species.
"""
function compute_bulk_fields(model::EoSModel, bulk_densities::AbstractVector{T}) where {T<:AbstractFloat}
    nspecies = length(bulk_densities)
    FT = eltype(bulk_densities)
    chi = FT.(model.params.chi.values)
    rho0 = FT(model.rho0)
    kappa = FT(model.kappa)

    ρ_total = sum(bulk_densities)
    comp_term = (kappa / rho0) * (ρ_total / rho0 - one(FT))

    w_bulk = zeros(FT, nspecies)
    for α in 1:nspecies
        w_bulk[α] = comp_term
        for β in 1:nspecies
            w_bulk[α] += (chi[α, β] / rho0) * bulk_densities[β]
        end
    end
    return w_bulk
end

"""
    compute_bulk_densities(system::SCFTSystem)

Return the bulk density of each species stored in `system` (`system.species.bulk_density`)
Returns a vector of length `nspecies` with element type `Float64` (see `SCFTSpecies`).
"""
compute_bulk_densities(system::SCFTSystem) = system.species.bulk_density

"""
    effective_volume(system::SCFTSystem, dz)

Compute the effective domain volume using the periodic trapezoidal rule:

```
V_eff = prod(dz) * prod(ngrid) = prod(L_i)
```
This is exact for any N and consistent with the default `:trapz` quadrature.
The previous Simpson-based fallback gave ~5% error for typical 3D grids because `structure_dz` returns `L/N` (periodic spacing) rather than `L/(N-1)` (non-periodic), causing the Simpson weights to underestimate the volume.
"""
function effective_volume(system::SCFTSystem, dz)
    ngrid = system.structure.ngrid
    return prod(dz) * prod(ngrid)
end

"""
    compute_partition_functions(system::SCFTSystem, w, w_bulk, q_in, dz)

Compute single-molecule partition functions from shifted propagators (chains and solvents, unified — a solvent is just an `N=1` molecule type flowing through the same propagator arrays):
```
Q̃_c = (1/V_eff) ∫ q̃_in[c](:, root_c) dr
```
where `q̃_in` is the bottom-up propagator computed with shifted fields `Δw = w - w_bulk`
(see `_dgc_tree_sweep!`, `src/propagator/discrete_gaussian_chain.jl`), and `root_c` is
chain `c`'s tree root (`chain_root`) — `q_in` at the root already folds in every branch
via the bottom-up recursion, so (unlike a linear chain's single well-defined "end") no
particular node needs picking beyond a valid root; by the sum-product invariant the
result doesn't depend on which node `compute_levels` happened to choose as root. For a
linear chain this reduces to evaluating the old forward propagator at its chain end.
For a uniform system at bulk densities, `Q̃ ≈ 1`.

Returns `Q::Vector{Float64}`, one entry per molecule type (`system.model.components`).
"""
function compute_partition_functions(system::SCFTSystem, w, w_bulk, q_in, dz;
                                     weights=nothing, V_eff=nothing, exp_field=nothing)
    nd = dimension(system)
    species = system.species
    nmol = length(species.sequence)
    FT = fptype(system.options)

    V_eff = V_eff !== nothing ? FT(V_eff) : FT(effective_volume(system, dz))

    Q = Vector{FT}(undef, nmol)
    for c in 1:nmol
        i_root = chain_root(species, c)
        if weights !== nothing
            # GPU-friendly: dot product with precomputed weight array — no host transfer
            Q[c] = sum(selectdim(q_in[c], nd+1, i_root) .* weights) / V_eff
        else
            # Periodic trapezoidal rule (matches `effective_volume`'s periodic convention,
            # since q_in lives on the periodic FFT grid — Simpson's `∫` assumes a
            # non-periodic domain with duplicated endpoints and is NOT consistent with
            # `dz = L/ngrid`/`V_eff = prod(dz)*prod(ngrid)` here).
            q_root = selectdim(q_in[c], nd+1, i_root)
            Q[c] = sum(q_root) * prod(dz) / V_eff
        end
    end

    return Q
end

"""
    compute_densities!(system::SCFTSystem, w, w_bulk, q_in, q_out, Q, ρ)

Compute density profiles from shifted propagators and partition functions, for every molecule type (chains and solvents, unified — a solvent is just an `N=1` molecule type, for which this formula reduces exactly to the old separate solvent formulas):
```
ρ_α(r) += prefactor * Σ_{k: α(k)=α} q̃_in(r,k) * q̃_out(r,k) * exp(Δw_α(r))
```
where `Δw = w - w_bulk`, and the exp(Δw) corrects for double-counting of the Boltzmann weight at node k (see `_dgc_tree_sweep!`'s docstring — `q_in`/`q_out` are node-indexed, not
position-indexed, so no `N+1-s` mirroring is needed here; a linear chain's `q_out` is
exactly the old `q_bwd` re-indexed from chain-position order into node order).
The shift factors cancel between numerator and denominator (Q̃), keeping values near O(1).
`prefactor = n_molecules/(V_eff*Q̃)` (canonical) or `bulk_density/(N*Q̃)` (grand canonical).
"""
function compute_densities!(system::SCFTSystem, w, w_bulk, q_in, q_out, Q, ρ;
                            V_eff=nothing, exp_field=nothing, inv_exp_field=nothing)
    nd = dimension(system)
    species = system.species
    nmol = length(species.sequence)
    dz = structure_dz(system.structure)
    FT = eltype(ρ)

    V_eff = V_eff !== nothing ? FT(V_eff) : FT(effective_volume(system, dz))

    # Zero out densities
    ρ .= zero(FT)

    for c in 1:nmol
        seg_spec = species.sequence[c]
        Nc = length(seg_spec)
        Qc = Q[c]

        # Prefactor depends on ensemble
        if species.ensemble[c] == :canonical
            prefactor = FT(species.n_molecules[c]) / (V_eff * Qc)
        else
            prefactor = FT(species.molecule_bulk_density[c]) / (FT(Nc) * Qc)
        end

        # For each node, add contribution to the appropriate species.
        # Double-count correction: exp(w_α - w_bulk_α) = 1/exp_field[α].
        # Use precomputed inv_exp_field if available to avoid recomputing per node.
        for k in 1:Nc
            α = seg_spec[k]
            inv_ef_α = inv_exp_field !== nothing ? inv_exp_field[α] :
                           exp.(selectdim(w, nd+1, α) .- w_bulk[α])
            selectdim(ρ, nd+1, α) .+= prefactor .* selectdim(q_in[c], nd+1, k) .*
                selectdim(q_out[c], nd+1, k) .* inv_ef_α
        end
    end
end

"""
    SCFTWLCSystem

Type alias for `SCFTSystem{<:EoSModel,<:DFTSpecies,<:DFTStructure,<:WLCPropagator,
<:DFTOptions,<:Any}` — an `SCFTSystem` using the discrete worm-like-chain propagator.
Used purely to give `compute_partition_functions`/`compute_densities!` a more specific
method that Julia's dispatch prefers automatically over the generic `system::SCFTSystem`
methods above, leaving those completely untouched for
`DiscreteGaussianChainPropagator`/`IdealPropagator` systems.

Written with the inline `<:X` syntax rather than an explicit `where {M,S,T,P<:...,O,EF}`
clause deliberately — the latter is NOT more specific than `system::SCFTSystem` in
Julia's dispatch (confirmed by direct testing) once the constrained parameter
(`WLCPropagator`) is itself a parametric type, since the two `where`-quantified
UnionAlls' relative specificity is compared structurally and the named-typevar form
doesn't register as strictly narrower; the `<:X` sugar desugars to fresh, unnamed type
variables that compare correctly instead.
"""
const SCFTWLCSystem = SCFTSystem{<:EoSModel,<:DFTSpecies,<:DFTStructure,<:WLCPropagator,<:DFTOptions,<:Any}

"""
    compute_partition_functions(system::SCFTWLCSystem, w, w_bulk, q_in, dz)

`WLCPropagator` analogue of `compute_partition_functions(system::SCFTSystem, ...)`.
`q_in[c]` carries an extra orientation axis (shape `(ngrid..., n_orient, N_c)`, see
`WLCPropagator`'s docstring), so `Q̃_c = (1/V_eff) ∫dr ∫du q̃_in(r,u,N_c)` needs an
orientation-quadrature-weighted sum (`_orientation_marginalize`,
`src/utils/spherical_harmonics.jl`) before the usual spatial integral — valid here
(unlike the density formula below) since only one `q` is being integrated, not a product
of two. `N_c` (not `chain_root`'s tree-root convention) is the right node to evaluate at:
v1 has no branching, so `q_in[c]`'s bottom-up sweep (`_wlc_linear_sweep!`) already ends at
the last chain position, not an internal tree root.
"""
function compute_partition_functions(system::SCFTWLCSystem, w, w_bulk, q_in, dz;
                                     weights=nothing, V_eff=nothing, exp_field=nothing)
    nd = dimension(system)
    species = system.species
    nmol = length(species.sequence)
    FT = fptype(system.options)
    quad_weight = system.propagator.sht.quad_weight

    V_eff = V_eff !== nothing ? FT(V_eff) : FT(effective_volume(system, dz))

    Q = Vector{FT}(undef, nmol)
    for c in 1:nmol
        Nc = length(species.sequence[c])
        q_end = selectdim(q_in[c], nd + 2, Nc)
        q_marg = _orientation_marginalize(q_end, quad_weight, nd + 1)
        if weights !== nothing
            Q[c] = sum(q_marg .* weights) / V_eff
        else
            Q[c] = sum(q_marg) * prod(dz) / V_eff
        end
    end

    return Q
end

"""
    compute_densities!(system::SCFTWLCSystem, w, w_bulk, q_in, q_out, Q, ρ)

`WLCPropagator` analogue of `compute_densities!(system::SCFTSystem, ...)`. The crucial
difference from the marginalize-then-use pattern above: `q_in(r,u,k)·q_out(r,u,k)` must
be multiplied elementwise *before* the orientation contraction — `∫du q_in·q_out ≠
(∫du q_in)(∫du q_out)` — so this multiplies the two orientation-resolved slices first,
contracts the orientation axis (`_orientation_marginalize`), and only then accumulates
the (now purely positional) result into `ρ`, mirroring the generic method's spatial-only
accumulation loop exactly.

`maier_saupe_field`, when not `nothing` (same `Vector`-indexed-by-species convention as
`propagate!`'s keyword of the same name, `src/propagator/wlc.jl`), makes the double-
counting correction factor orientation-*dependent*: `inv_ef_α(r,u) = exp(w_α(r) -
w_bulk[α] + w_MS,α(r,u))` — the reciprocal of `propagate!`'s
`ef(α)=exp(w_bulk[α]-w_α)·exp(-w_MS,α(r,u))`, so it must gain the *same* `w_MS,α(r,u)`
term (with a **positive** sign here, being the reciprocal) that `ef` picked up, not
just the position-only piece. This orientation-dependent factor must multiply `qq`
**before** the orientation contraction (`_orientation_marginalize`), not after —
unlike the position-only case above, `∫du (qq·inv_ef_α(u)) ≠ (∫du qq)·inv_ef_α` once
`inv_ef_α` depends on `u`. At `nu=0` (`maier_saupe_field=nothing`), this is exactly the
position-only branch below, unchanged.
"""
function compute_densities!(system::SCFTWLCSystem, w, w_bulk, q_in, q_out, Q, ρ;
                            V_eff=nothing, exp_field=nothing, inv_exp_field=nothing,
                            maier_saupe_field=nothing)
    nd = dimension(system)
    species = system.species
    nmol = length(species.sequence)
    dz = structure_dz(system.structure)
    FT = eltype(ρ)
    quad_weight = system.propagator.sht.quad_weight

    V_eff = V_eff !== nothing ? FT(V_eff) : FT(effective_volume(system, dz))

    ρ .= zero(FT)

    for c in 1:nmol
        seg_spec = species.sequence[c]
        Nc = length(seg_spec)
        Qc = Q[c]

        if species.ensemble[c] == :canonical
            prefactor = FT(species.n_molecules[c]) / (V_eff * Qc)
        else
            prefactor = FT(species.molecule_bulk_density[c]) / (FT(Nc) * Qc)
        end

        for k in 1:Nc
            α = seg_spec[k]
            qq = selectdim(q_in[c], nd + 2, k) .* selectdim(q_out[c], nd + 2, k)
            base_inv_ef_α = inv_exp_field !== nothing ? inv_exp_field[α] :
                                exp.(selectdim(w, nd + 1, α) .- w_bulk[α])
            if maier_saupe_field === nothing
                qq_marg = _orientation_marginalize(qq, quad_weight, nd + 1)
                selectdim(ρ, nd + 1, α) .+= prefactor .* qq_marg .* base_inv_ef_α
            else
                qq_weighted = qq .* base_inv_ef_α .* exp.(maier_saupe_field[α])
                qq_marg = _orientation_marginalize(qq_weighted, quad_weight, nd + 1)
                selectdim(ρ, nd + 1, α) .+= prefactor .* qq_marg
            end
        end
    end
end

"""
    _orientation_second_moments(system::SCFTWLCSystem, q_in, q_out, weights::AbstractMatrix)

Shared per-species, per-node marginalization loop underlying both
`orientation_order_parameter` (scalar `S(x)`) and `compute_orientation_tensor` (full
tensor `Q_ab(r)`). `weights` has shape `(n_orient, n_weight)` — one column per desired
`⟨w(u)⟩_α(r)` moment; all columns are evaluated against the *same* per-node product
`qq = q_in(r,u,k)·q_out(r,u,k)` in a single pass over chains/nodes, so multiple moments
(e.g. all 6 independent `u_a u_b` products) cost one pass, not one per moment.

Returns `(numer, denom, species_present)`: `numer` has shape `(ngrid..., nspecies,
n_weight)`, `Σ_{k:α(k)=α} ∫du weights[:,j](u) qq(r,u,k)`; `denom` has shape
`(ngrid..., nspecies)`, `Σ_{k:α(k)=α} ∫du qq(r,u,k)` (the un-normalized local density,
common to every moment); `species_present` is a length-`nspecies` `BitVector`, false for
any species that appears in no chain at all — callers should leave such a species'
output at its zero default rather than dividing through `denom`'s `eps` floor there
(which would silently produce a nonzero value from `0/eps`, not the "no such species"
zero the original scalar implementation returned).
"""
function _orientation_second_moments(system::SCFTWLCSystem, q_in, q_out, weights::AbstractMatrix)
    nd = dimension(system)
    species = system.species
    nspecies = length(system.model.groups.flattenedgroups)
    quad_weight = system.propagator.sht.quad_weight
    n_weight = size(weights, 2)
    FT = eltype(q_in[1])
    ngrid = system.structure.ngrid

    numer = zeros(FT, ngrid..., nspecies, n_weight)
    denom = zeros(FT, ngrid..., nspecies)
    species_present = falses(nspecies)
    qw_weights = quad_weight .* weights

    for α in 1:nspecies
        for c in eachindex(species.sequence)
            seg_spec = species.sequence[c]
            for k in findall(==(α), seg_spec)
                species_present[α] = true
                qq = selectdim(q_in[c], nd + 2, k) .* selectdim(q_out[c], nd + 2, k)
                selectdim(denom, nd + 1, α) .+= _orientation_marginalize(qq, quad_weight, nd + 1)
                for j in 1:n_weight
                    numer_αj = selectdim(selectdim(numer, nd + 2, j), nd + 1, α)
                    numer_αj .+= _orientation_marginalize(qq, view(qw_weights, :, j), nd + 1)
                end
            end
        end
    end
    return numer, denom, species_present
end

"""
    compute_orientation_tensor(system::SCFTWLCSystem, q_in, q_out)

Full traceless symmetric nematic order-parameter tensor `Q_ab^α(r) = (3⟨u_a u_b⟩_α(r) -
δ_ab)/2` for each WLC species `α`, generalizing `orientation_order_parameter`'s scalar
`S(x)` (a single diagonal component, for one distinguished `axis`) to the full tensor
needed by a Maier-Saupe mean field, which must couple to order along *any* direction.
Stores only the 6 independent components, ordered per `_Q_PAIRS`
(`src/utils/spherical_harmonics.jl`). Returns shape `(ngrid..., nspecies, 6)`.

A species present in no chain at all gets `Q_ab=0` everywhere (not the `-δ_ab/2` that a
naive `0/eps` ratio would give). A species that *is* present but has locally zero
density at some position gets `Q_ab=-δ_ab/2` there (the isotropic-orthogonal floor,
matching `orientation_order_parameter`'s analogous behavior) — these are different
cases: the former has no orientation distribution to speak of anywhere, the latter's
distribution is well-defined everywhere else and only vanishes locally.
"""
function compute_orientation_tensor(system::SCFTWLCSystem, q_in, q_out)
    nd = dimension(system)
    FT = eltype(q_in[1])
    u_nodes = system.propagator.sht.u_nodes
    weights = reduce(hcat, (u_nodes[a, :] .* u_nodes[b, :] for (a, b) in _Q_PAIRS))
    numer, denom, species_present = _orientation_second_moments(system, q_in, q_out, weights)
    denom_safe = max.(denom, eps(FT))

    Q = zeros(FT, size(numer)...)
    for (j, (a, b)) in enumerate(_Q_PAIRS)
        δab = a == b ? one(FT) : zero(FT)
        Q_j = selectdim(Q, nd + 2, j)
        for α in 1:length(species_present)
            species_present[α] || continue
            selectdim(Q_j, nd + 1, α) .= (3 .* selectdim(selectdim(numer, nd + 2, j), nd + 1, α) ./
                                           selectdim(denom_safe, nd + 1, α) .- δab) ./ 2
        end
    end
    return Q
end

"""
    compute_orientation_moments(system::SCFTWLCSystem, w, w_bulk, q_in, q_out, Q; V_eff=nothing, inv_exp_field=nothing, maier_saupe_field=nothing)

Density-normalized orientation second moment
```
R_ab^α(r) = prefactor_c · Σ_{k:α(k)=α} ∫dΩ u_a u_b q_in(r,u,k) q_out(r,u,k) · inv_ef_α(r,u)
```
for each WLC species `α` — the object the Maier-Saupe mean field
(`compute_maier_saupe_field`) and self-energy (`free_energy`'s `U_MS`) are built from.
`R` has *exactly* `ρ_α(r)`'s own normalization convention (`compute_densities!`'s
`prefactor = n_c/(V_eff·Qc)` and double-counting correction `inv_ef_α`) — it is **not**
the raw, un-normalized `q_in·q_out` product alone. This matters:  `q_in`/`q_out`'s
*overall* scale is gauge-dependent (only ratios like `ρ = prefactor·q_in·q_out` are
physically meaningful — this is exactly why `compute_densities!` divides by `Qc` via
`prefactor` in the first place), so omitting this normalization here would let `R`
drift to an arbitrary, ungrounded scale from iteration to iteration — this was
tried and empirically diverged (`R_tensor` growing unboundedly over hundreds of SCFT
iterations while `ρ`/`w` stayed bounded, since only `ρ` had the `1/Qc` grounding).
With this normalization, `Σ_a R_aa^α(r)` is exactly `ρ_α(r)` (since `Σ_a u_a^2 = 1`),
so `R` is bounded by `ρ_α(r)`'s own bound (`≤ρ0` in an incompressible melt) — matching
`ρ`'s own status closely enough that the standard SCFT double-counting identity
`wMSρ_sum = 2×U_MS` closes exactly (both `R` and `ρ` share the same
`prefactor/Qc`-normalized "density-like" status the argument in
`/Users/pierrewalker/.claude/plans/radiant-brewing-sprout.md` requires).

`maier_saupe_field`/`inv_exp_field`, when provided, must be the *same* values passed to
the `compute_densities!` call this shares `q_in`/`q_out`/`Q` with, for consistency.

Stores only the 6 independent components, ordered per `_Q_PAIRS`
(`src/utils/spherical_harmonics.jl`). Returns shape `(ngrid..., nspecies, 6)`.
"""
function compute_orientation_moments(system::SCFTWLCSystem, w, w_bulk, q_in, q_out, Q;
                                     V_eff=nothing, inv_exp_field=nothing, maier_saupe_field=nothing)
    nd = dimension(system)
    species = system.species
    nmol = length(species.sequence)
    dz = structure_dz(system.structure)
    FT = eltype(q_in[1])
    nspecies = length(system.model.groups.flattenedgroups)
    ngrid = system.structure.ngrid
    quad_weight = system.propagator.sht.quad_weight
    u_nodes = system.propagator.sht.u_nodes

    V_eff = V_eff !== nothing ? FT(V_eff) : FT(effective_volume(system, dz))
    weights = reduce(hcat, (u_nodes[a, :] .* u_nodes[b, :] for (a, b) in _Q_PAIRS))
    qw_weights = quad_weight .* weights

    R = zeros(FT, ngrid..., nspecies, 6)

    for c in 1:nmol
        seg_spec = species.sequence[c]
        Nc = length(seg_spec)
        Qc = Q[c]
        prefactor = species.ensemble[c] == :canonical ?
            FT(species.n_molecules[c]) / (V_eff * Qc) :
            FT(species.molecule_bulk_density[c]) / (FT(Nc) * Qc)

        for k in 1:Nc
            α = seg_spec[k]
            qq = selectdim(q_in[c], nd + 2, k) .* selectdim(q_out[c], nd + 2, k)
            base_inv_ef_α = inv_exp_field !== nothing ? inv_exp_field[α] :
                                exp.(selectdim(w, nd + 1, α) .- w_bulk[α])
            if maier_saupe_field === nothing
                for j in 1:6
                    Rj_α = selectdim(selectdim(R, nd + 2, j), nd + 1, α)
                    Rj_α .+= prefactor .* _orientation_marginalize(qq, view(qw_weights, :, j), nd + 1) .* base_inv_ef_α
                end
            else
                qq_weighted = qq .* base_inv_ef_α .* exp.(maier_saupe_field[α])
                for j in 1:6
                    Rj_α = selectdim(selectdim(R, nd + 2, j), nd + 1, α)
                    Rj_α .+= prefactor .* _orientation_marginalize(qq_weighted, view(qw_weights, :, j), nd + 1)
                end
            end
        end
    end
    return R
end

"""
    compute_maier_saupe_field(system::SCFTWLCSystem, R_tensor)

Orientation-dependent Maier-Saupe mean field
```
w_MS,α(r,u) = -(3/(2ρ0)) Σ_β ν_αβ Σ_ab R_ab^β(r) u_a u_b
```
evaluated at each of the propagator's `n_orient` orientation-grid nodes (`sht.u_nodes`),
for every species `α` (`ρ0 = system.model.rho0`, `ν = system.model.params.nu`).
`R_tensor` has shape `(ngrid...,nspecies,6)` (`compute_orientation_moments`,
`_Q_PAIRS`-ordered; the sum over the full 9 `(a,b)` pairs is recovered from the 6
stored components via `_Q_PAIR_WEIGHT`).

The prefactor `3/(2ρ0)` is fixed (not a free choice) by requiring this reduce exactly
to the classical bulk Maier-Saupe potential `w_MS = -ν·S·P_2(cosθ)` (validated
numerically — `ν*=5` isotropic-nematic linear instability — by the standalone solver
in the design plan) in the single-species, spatially-uniform, uniaxial limit.

Returns a `Vector` indexed by species (matching `exp_field`'s own indexing convention,
not a single array with a species dimension) — `result[α]` has shape
`(ngrid...,n_orient)`, ready to pass as `propagate!`'s `maier_saupe_field` keyword.
Species pairs with `ν_αβ=0` contribute nothing (skipped).
"""
function compute_maier_saupe_field(system::SCFTWLCSystem, R_tensor)
    nd = dimension(system)
    FT = eltype(R_tensor)
    nspecies = length(system.model.groups.flattenedgroups)
    nu = FT.(system.model.params.nu.values)
    rho0 = FT(system.model.rho0)
    u_nodes = system.propagator.sht.u_nodes
    n_orient = size(u_nodes, 2)
    ngrid = system.structure.ngrid
    prefactor = FT(3) / (FT(2) * rho0)

    result = Vector{Array{FT,length(ngrid) + 1}}(undef, nspecies)
    for α in 1:nspecies
        wα = zeros(FT, ngrid..., n_orient)
        for β in 1:nspecies
            nu[α, β] == zero(FT) && continue
            Rβ = selectdim(R_tensor, nd + 1, β)
            for (j, (a, b)) in enumerate(_Q_PAIRS)
                coeff = _Q_PAIR_WEIGHT[j] * prefactor * nu[α, β]
                Rj_β = selectdim(Rβ, nd + 1, j)
                uu = reshape(u_nodes[a, :] .* u_nodes[b, :], ntuple(_ -> 1, nd)..., n_orient)
                wα .-= coeff .* Rj_β .* uu
            end
        end
        result[α] = wα
    end
    return result
end

"""
    orientation_order_parameter(system::SCFTWLCSystem, q_in, q_out; axis::Int=1)

Local nematic order parameter `S_α(r) = (3⟨u_axis²⟩_α(r) - 1)/2` for each WLC species
`α`, where `⟨u_axis²⟩_α(r)` is the orientation-grid average of `u_axis²` weighted by the
local (un-normalized) orientational distribution `ψ_α(r,u) = Σ_{k:α(k)=α} q_in(r,u,k)
q_out(r,u,k)` — the same per-node product `compute_densities!` integrates over
orientation to build `ρ`, here kept resolved in `u` instead. `axis` selects which
Cartesian component of `u` (1=x, 2=y, 3=z) to measure alignment against — for a 1D/2D
structure, `axis` should be one of the *tracked* spatial dimensions
(`1:dimension(system)`) to ask "are bonds preferentially aligned along the
density-varying direction?", since `WLCPropagator` supports `dimension(structure) ∈
{1,2,3}` with orientation always tracked in full 3D (see `WLCPropagator`'s docstring).

`S=1`: bonds fully aligned along `axis`. `S=-1/2`: fully perpendicular (isotropic in the
plane normal to `axis`). `S=0`: isotropic in all directions — e.g. what every
`DiscreteGaussianChainPropagator` chain has, having no orientation to align in the first
place. The double-counted field factor `exp(Δw_α)` that `compute_densities!` divides out
when building `ρ` is a per-position scalar independent of `u`, so it cancels exactly in
this ratio and is not needed here.

Returns an `Array` of shape `(ngrid..., nspecies)`, matching `ρ`'s own shape convention.
Positions where a species' local density is exactly zero return `S=0` (rather than
`NaN`) since there is no orientation distribution to speak of there.
"""
function orientation_order_parameter(system::SCFTWLCSystem, q_in, q_out; axis::Int=1)
    nd = dimension(system)
    FT = eltype(q_in[1])
    u_axis2 = reshape(system.propagator.sht.u_nodes[axis, :] .^ 2, :, 1)
    numer, denom, species_present = _orientation_second_moments(system, q_in, q_out, u_axis2)
    numer_flat = dropdims(numer; dims=nd + 2)
    denom_safe = max.(denom, eps(FT))

    S = zeros(FT, size(numer_flat)...)
    for α in 1:length(species_present)
        species_present[α] || continue
        mean_u_axis2 = selectdim(numer_flat, nd + 1, α) ./ selectdim(denom_safe, nd + 1, α)
        selectdim(S, nd + 1, α) .= (3 .* mean_u_axis2 .- 1) ./ 2
    end
    return S
end

"""
    free_energy(system::SCFTSystem, ρ, w, Q; R_tensor=nothing)

Compute the SCFT free energy (mean-field Hamiltonian):
```
H = U_int + U_comp - Σ_K ∫w_K ρ_K dr - Σ_c n_c ln(Q̃_c) - Σ_c (bulk_density_c/N_c) V Q̃_c
```
summed over every molecule type (chains and solvents, unified — a solvent is just an `N=1` molecule type).
`Q` is the *shifted* partition function (Q̃) from propagators computed with `Δw = w - w_bulk`.
For grand-canonical molecule types, the fugacity/bulk correction exactly cancels the propagator's `exp(±w_bulk_sum)` shift factor so `Q̃` is used directly with no extra correction.
For canonical molecule types, the shift factor does *not* cancel inside `log`, so it must be subtracted explicitly.

`R_tensor` (WLC only, `nothing` for DGC/ideal systems — see `compute_orientation_moments`,
above), when provided, adds the Maier-Saupe orientational self-energy `U_MS` and its
double-counting correction `wMSρ_sum = 2×U_MS` (net contribution to `H`: `-U_MS`);
omitting it (`nu=0` or a non-WLC system) reproduces the exact pre-Maier-Saupe formula.
"""
function free_energy(system::SCFTSystem, ρ, w, Q;
                     V_eff=nothing, w_bulk=nothing, R_tensor=nothing)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    nspecies = length(system.model.groups.flattenedgroups)
    dz = structure_dz(system.structure)

    # Use V_eff and w_bulk from the iteration loop when provided, so the free energy
    # is computed consistently with the quadrature rule chosen for the iteration.
    # Fallback to recomputing from scratch (CPU Simpson) for standalone calls.
    if V_eff === nothing
        V_eff = effective_volume(system, dz)
    end
    if w_bulk === nothing
        w_bulk = compute_bulk_fields(system.model, compute_bulk_densities(system))
    end

    chi = system.model.params.chi.values
    rho0 = system.model.rho0
    kappa = system.model.kappa

    # Periodic composite-trapezoidal integral: sum(f)*prod(dz). Deliberately NOT the
    # general-purpose `∫` (Simpson's rule assuming non-periodic spacing dz=L/(N-1)) —
    # `structure_dz` returns the *periodic* spacing dz=L/N, so `∫` on a periodic grid
    # systematically undercounts by a factor (N-1)/N (the same discrepancy
    # `effective_volume`'s docstring already documents for volume/Q/density
    # normalization). Using the same periodic convention here keeps U_int/U_comp/wρ_sum
    # consistent with V_eff (and with each other) regardless of caller.
    per_∫(f) = sum(f) * prod(dz)

    # U_int = (1/ρ₀) ∫ Σ_{α<β} χ_αβ ρ_α ρ_β dr
    U_int_integrand = zeros(ngrid...)
    for α in 1:nspecies
        for β in (α+1):nspecies
            if chi[α, β] != 0.0
                U_int_integrand .+= chi[α, β] .* Array(selectdim(ρ, nd+1, α)) .* Array(selectdim(ρ, nd+1, β))
            end
        end
    end
    U_int = per_∫(U_int_integrand) / rho0

    # U_comp = (ζ / 2) ∫ (ρ₊/ρ₀ - 1)² dr
    # Consistent with w_comp = ζ/ρ₀ · (ρ₊/ρ₀ − 1) = δU_comp/δρ_α.
    ρ_total = zeros(ngrid...)
    for α in 1:nspecies
        ρ_total .+= Array(selectdim(ρ, nd+1, α))
    end
    U_comp_integrand = (kappa / 2.0) .* (ρ_total ./ rho0 .- 1.0) .^ 2
    U_comp = per_∫(U_comp_integrand)

    # -Σ_K ∫ w_K ρ_K dr
    wρ_sum = 0.0
    for α in 1:nspecies
        wρ_integrand = Array(selectdim(w, nd+1, α)) .* Array(selectdim(ρ, nd+1, α))
        wρ_sum += per_∫(wρ_integrand)
    end

    # Molecule-type contributions (chains and solvents, unified; a solvent is N_c=1).
    # Canonical: -n_c * ln(Q_c) where Q_c is the TRUE partition function.
    #   Q̃_c = Q_c * exp(Σ_s w_bulk[α(s)]), so ln(Q_c) = ln(Q̃_c) - Σ_s w_bulk[α(s)]
    #   -- the shift does NOT cancel inside log, so it must be subtracted explicitly.
    # Grand canonical: Ω = -kT V z Q_c, with z = bulk_density/(N_c * Q_c(bulk)) and
    #   Q_c(bulk) = exp(-Σ_s w_bulk[α(s)]) (uniform-field propagator). Substituting:
    #   Ω = -(bulk_density/N_c) V exp(Σ w_bulk) Q_c = -(bulk_density/N_c) V exp(Σ w_bulk) Q̃_c exp(-Σ w_bulk)
    #   the exp(±Σ w_bulk) factors cancel EXACTLY, leaving -(bulk_density/N_c) V Q̃_c with
    #   no correction at all -- unlike the canonical branch, this is *not* logarithmic.
    species = system.species
    molecule_sum = 0.0
    for c in eachindex(system.model.components)
        seg_spec = species.sequence[c]
        Nc = length(seg_spec)
        if species.ensemble[c] == :canonical
            w_bulk_sum = sum(w_bulk[seg_spec[s]] for s in 1:Nc)
            molecule_sum -= species.n_molecules[c] * (log(Q[c]) - w_bulk_sum)
        else
            molecule_sum -= (species.molecule_bulk_density[c] / Nc) * V_eff * Q[c]
        end
    end

    # Maier-Saupe orientational self-energy and its double-counting correction (WLC
    # only — R_tensor is `nothing` for DGC/ideal systems, which never build one).
    # U_MS = -(3/(4ρ0)) ΣΣ ν_αβ ∫ R^α:R^β dr; the correction wMSρ_sum = 2×U_MS exactly
    # (Euler's homogeneous-function theorem for a quadratic self-energy whose mean
    # field is its own exact functional derivative — see
    # /Users/pierrewalker/.claude/plans/radiant-brewing-sprout.md), so the net
    # contribution to H is U_MS - wMSρ_sum = -U_MS, mirroring U_int - wρ_sum = -U_int.
    U_MS = 0.0
    if R_tensor !== nothing
        nu = system.model.params.nu.values
        ms_prefactor = 3.0 / (4.0 * rho0)
        RR_integrand = zeros(ngrid...)
        for α in 1:nspecies
            Rα = selectdim(R_tensor, nd + 1, α)
            for β in 1:nspecies
                nu[α, β] == 0.0 && continue
                Rβ = selectdim(R_tensor, nd + 1, β)
                for j in eachindex(_Q_PAIRS)
                    RR_integrand .+= (_Q_PAIR_WEIGHT[j] * nu[α, β]) .*
                        Array(selectdim(Rα, nd + 1, j)) .* Array(selectdim(Rβ, nd + 1, j))
                end
            end
        end
        U_MS = -ms_prefactor * per_∫(RR_integrand)
    end
    wMSρ_sum = 2 * U_MS

    H = U_int + U_comp + U_MS - wρ_sum - wMSρ_sum + molecule_sum
    return H
end

free_energy(::SCFTSystem, ρ) = error(
    """free_energy(system::SCFTSystem, ρ) is not defined — SCFT free energy also depends on the field w and partition functions Q.
    Use free_energy(system, ρ, w, Q) instead."""
)
surface_tension(::SCFTSystem, ρ) = error(
    "surface_tension is not defined for SCFTSystem — SCFT systems are not vapor-liquid interface calculations against an EoSModel bulk phase."
)

export SCFTSystem
