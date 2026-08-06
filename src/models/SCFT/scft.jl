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
    free_energy(system::SCFTSystem, ρ, w, Q)

Compute the SCFT free energy (mean-field Hamiltonian):
```
H = U_int + U_comp - Σ_K ∫w_K ρ_K dr - Σ_c n_c ln(Q̃_c) - Σ_c (bulk_density_c/N_c) V Q̃_c
```
summed over every molecule type (chains and solvents, unified — a solvent is just an `N=1` molecule type).
`Q` is the *shifted* partition function (Q̃) from propagators computed with `Δw = w - w_bulk`.
For grand-canonical molecule types, the fugacity/bulk correction exactly cancels the propagator's `exp(±w_bulk_sum)` shift factor so `Q̃` is used directly with no extra correction.
For canonical molecule types, the shift factor does *not* cancel inside `log`, so it must be subtracted explicitly.
"""
function free_energy(system::SCFTSystem, ρ, w, Q;
                     V_eff=nothing, w_bulk=nothing)
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

    H = U_int + U_comp - wρ_sum + molecule_sum
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
