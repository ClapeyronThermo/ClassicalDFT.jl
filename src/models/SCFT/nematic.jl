#=
Bulk (spatially uniform) thermodynamics including nematic order.

For a uniform phase with molecule densities ρᵢ (per component, the same units as
`structure.ρbulk`) and uniaxial order Sα per species along a director n (taken as z), the
free energy density in k_BT per volume is

    f(ρ, S) = Σᵢ ρᵢ (ln ρᵢ − 1)                          ideal translational entropy
            + Σ_int bulk_f(int, ρ_seg, ρ₀)                isotropic interactions
            + (1/2ρ₀) Σ_αβ ν_αβ ρ_α ρ_β S_α S_β − Σᵢ ρᵢ ln qᵢ(η)      nematic part

with segment densities ρ_seg, η_α = (1/ρ₀) Σ_β ν_αβ ρ_β S_β the uniaxial Maier-Saupe field,
and qᵢ(η) molecule i's orientational partition function with segment weights
exp(η_α P₂(u·n)) and its own bending statistics, normalized so qᵢ(0) = 1. This is the
uniform limit of the SCFT free energy (`free_energy`), so a converged uniform nematic SCFT
solution has exactly this free energy density at the same `L_max`.

Stationarity in S gives S_α = (1/ρ_α) ∂/∂η_α Σᵢ ρᵢ ln qᵢ = ⟨P₂⟩ of α segments; S = 0
(isotropic) is always a solution. Chemical potentials and pressure follow from f at the
self-consistent S (envelope theorem).
=#

"""
    NematicBulk(system::SCFTSystem)
    NematicBulk(model::SCFTModel; mol_structure)

Precomputed orientation-space data for the bulk nematic free energy of `model`: each
molecule type's bond tree, the orientation grid (`L_max` of the model), bending kernels
and `P₂` on the grid. Molecule topology and sequence come from `system.species` or from
`mol_structure` (as for [`SCFTSystem`](@ref)).
"""
struct NematicBulk{M}
    model::M
    sht::SHTPlan{Float64}
    qw::Vector{Float64}
    P2::Vector{Float64}
    sequence::Vector{Vector{Int}}
    trees::Vector{OrientedTree}
    bend::Dict{Tuple{Int,Int},Vector{Float64}}
    gauss_bonds::Set{Tuple{Int,Int}}
    nu::Matrix{Float64}
    oriented::Vector{Bool}
end

function NematicBulk(model::SCFTModel, species::SCFTSpecies)
    sht = SHTPlan(model.L_max, Float64)
    qw = Array(sht.quad_weight)
    P2 = [(3 * sht.u_nodes[3, i]^2 - 1) / 2 for i in axes(sht.u_nodes, 2)]
    b, lp = segment_lengths(model), persistence_lengths(model)
    oriented = [is_oriented(c) for c in model.chains]
    trees = [_oriented_tree(species, c) for c in eachindex(species.sequence)]
    bend = Dict{Tuple{Int,Int},Vector{Float64}}()
    gauss = Set{Tuple{Int,Int}}()
    for (c, seq) in enumerate(species.sequence), k in eachindex(seq)
        p = trees[c].parent[k]
        p == 0 && continue
        key = minmax(seq[p], seq[k])
        if oriented[key[1]] && oriented[key[2]]
            haskey(bend, key) || (bend[key] = expand_eig_to_lm(bending_eigenvalues(_bond_kappa(b, lp, key...), model.L_max), sht))
        else
            push!(gauss, key)
        end
    end
    return NematicBulk(model, sht, qw, P2, species.sequence, trees, bend, gauss, Matrix{Float64}(maier_saupe_nu(model)), oriented)
end

NematicBulk(system::SCFTSystem) = NematicBulk(system.model, system.species)

function NematicBulk(model::SCFTModel; mol_structure)
    st = Uniform1DCart((0.0, 0.0), ones(length(model.components)), [0.0, 1.0], 2)
    species = get_species(expand_model(model, mol_structure), st)
    return NematicBulk(model, species)
end

"""
    orientational_logq(nb::NematicBulk, η) -> Vector

`ln qᵢ(η)` for every molecule type: the orientational partition function with segment
weights `exp(η_α P₂(u_z))`, relative to the isotropic state (`qᵢ(0) = 1`). Computed by the
same tree recursion as [`WLCPropagator`](@ref) without translation (a uniform phase):
rod bonds bend, branch points and Gaussian bonds average over orientation.
"""
function orientational_logq(nb::NematicBulk, η::AbstractVector)
    sht = nb.sht
    n = length(nb.P2)
    B = [exp.(η[α] .* nb.P2) for α in eachindex(η)]
    qlm = zeros(1, sht.nlm)
    tmp = zeros(1, n)
    out = Vector{Float64}(undef, length(nb.sequence))
    for (c, seq) in enumerate(nb.sequence)
        tree = nb.trees[c]
        N = length(seq)
        q = [zeros(1, n) for _ in 1:N]
        logscale = zeros(N)
        for k in Iterators.reverse(tree.order)
            q[k][1, :] .= B[seq[k]]
            free = length(tree.children[k]) >= 2
            for j in tree.children[k]
                key = minmax(seq[k], seq[j])
                if free || key in nb.gauss_bonds
                    q[k] .*= sum(nb.qw .* vec(q[j])) / (4π)
                else
                    wlc_bend!(tmp, q[j], nb.bend[key], sht, qlm, 1)
                    q[k] .*= tmp
                end
                logscale[k] += logscale[j]
            end
            m = maximum(q[k])
            q[k] ./= m
            logscale[k] += log(m)
        end
        root = tree.order[1]
        out[c] = log(sum(nb.qw .* vec(q[root])) / (4π)) + logscale[root]
    end
    return out
end

# Segment densities from molecule densities.
_segment_densities(nb::NematicBulk, ρ_mol) =
    [sum(ρ_mol[c] * count(==(α), seq) for (c, seq) in enumerate(nb.sequence); init = zero(eltype(ρ_mol)))
     for α in eachindex(nb.oriented)]

_ms_field(nb::NematicBulk, ρ_seg, S) = (nb.nu * (ρ_seg .* S)) ./ nb.model.rho0

"""
    nematic_free_energy(nb::NematicBulk, ρ_mol, S)

Nematic part of the bulk free energy density (k_BT per volume) at molecule densities
`ρ_mol` and per-species uniaxial order `S`:
`(1/2ρ₀) Σ ν_αβ ρ_α ρ_β S_α S_β − Σᵢ ρᵢ ln qᵢ(η)`. Zero at `S = 0`.
"""
function nematic_free_energy(nb::NematicBulk, ρ_mol, S)
    ρ = _segment_densities(nb, ρ_mol)
    η = _ms_field(nb, ρ, S)
    U = sum(nb.nu .* ((ρ .* S) * (ρ .* S)')) / (2 * nb.model.rho0)
    return U - sum(ρ_mol .* orientational_logq(nb, η))
end

# ⟨P₂⟩ of each species' segments in the field η, from ∂/∂η_α Σᵢ ρᵢ ln qᵢ (central
# differences; ln q is smooth in η).
function _segment_order(nb::NematicBulk, ρ_mol, ρ_seg, η)
    S = zeros(length(η))
    for α in eachindex(η)
        ρ_seg[α] > 0 && nb.oriented[α] || continue
        h = 1e-5 * max(1.0, abs(η[α]))
        ηp = copy(η); ηp[α] += h
        ηm = copy(η); ηm[α] -= h
        S[α] = sum(ρ_mol .* (orientational_logq(nb, ηp) .- orientational_logq(nb, ηm))) / (2h * ρ_seg[α])
    end
    return S
end

"""
    bulk_order_parameter(nb::NematicBulk, ρ_mol; S0=0.8, tol=1e-9, maxit=200)

Self-consistent uniaxial order `S` (one per species; zero for Gaussian species and for
species without Maier-Saupe coupling) at molecule densities `ρ_mol`, solving
`S = ⟨P₂⟩(η(S))` from `S0` (a number for every orientation-resolved species, or a vector):
fixed-point steps to approach the solution, then Newton steps with a finite-difference
Jacobian. `tol` should stay above the ~1e-10 accuracy of the finite-difference `⟨P₂⟩`.
Returns `(; S, Δf, converged)` with `Δf` the [`nematic_free_energy`](@ref).

Started from `S0 = 0` it returns the isotropic solution; a large `S0` finds the nematic
branch when one exists (otherwise it relaxes to `S = 0`).
"""
function bulk_order_parameter(nb::NematicBulk, ρ_mol; S0=0.8, tol=1e-9, maxit=200)
    ρ_seg = _segment_densities(nb, ρ_mol)
    S = S0 isa Real ? [nb.oriented[α] ? Float64(S0) : 0.0 for α in eachindex(nb.oriented)] : Float64.(collect(S0))
    active = [nb.oriented[α] && ρ_seg[α] > 0 && any(!iszero, view(nb.nu, α, :)) for α in eachindex(S)]
    S[.!active] .= 0
    F(S) = _segment_order(nb, ρ_mol, ρ_seg, _ms_field(nb, ρ_seg, S))
    idx = findall(active)
    converged = isempty(idx)
    for it in 1:maxit
        isempty(idx) && break
        g = F(S) .- S
        if maximum(abs, view(g, idx)) < tol
            converged = true
            break
        end
        if it <= 20 || maximum(abs, view(g, idx)) > 1e-3
            S .+= g                                        # fixed-point step
        else
            J = zeros(length(idx), length(idx))            # Newton on g(S) = F(S) − S
            for (c, α) in enumerate(idx)
                h = 1e-6
                Sp = copy(S); Sp[α] += h
                Sm = copy(S); Sm[α] -= h
                J[:, c] = ((F(Sp) .- Sp) .- (F(Sm) .- Sm))[idx] ./ (2h)
            end
            S[idx] .-= J \ g[idx]
        end
    end
    return (; S, Δf = nematic_free_energy(nb, ρ_mol, S), converged)
end

"""
    bulk_free_energy(nb::NematicBulk, ρ_mol, S)

Total bulk free energy density (k_BT per volume) of a uniform phase: ideal translational
entropy `Σᵢ ρᵢ(ln ρᵢ − 1)`, the isotropic interactions ([`bulk_free_energy_density`](@ref))
and the nematic part ([`nematic_free_energy`](@ref)).
"""
function bulk_free_energy(nb::NematicBulk, ρ_mol, S)
    ρ_seg = _segment_densities(nb, ρ_mol)
    ideal = sum(ρ_mol[i] > 0 ? ρ_mol[i] * (log(ρ_mol[i]) - 1) : 0.0 for i in eachindex(ρ_mol))
    return ideal + bulk_free_energy_density(nb.model, ρ_seg) + nematic_free_energy(nb, ρ_mol, S)
end

"""
    bulk_chemical_potential(nb::NematicBulk, ρ_mol, S)

Chemical potential of each molecule type (k_BT), `∂f/∂ρᵢ` at fixed `S`. Equal to the
total derivative when `S` is self-consistent (envelope theorem).
"""
function bulk_chemical_potential(nb::NematicBulk, ρ_mol, S)
    μ = zeros(length(ρ_mol))
    for i in eachindex(ρ_mol)
        h = 1e-6 * max(ρ_mol[i], 1e-8)
        ρp = Float64.(collect(ρ_mol)); ρp[i] += h
        ρm = Float64.(collect(ρ_mol)); ρm[i] -= h
        μ[i] = (bulk_free_energy(nb, ρp, S) - bulk_free_energy(nb, ρm, S)) / (2h)
    end
    return μ
end

"""
    bulk_pressure(nb::NematicBulk, ρ_mol, S)

Pressure (k_BT per volume), `Σᵢ ρᵢ μᵢ − f`.
"""
bulk_pressure(nb::NematicBulk, ρ_mol, S) =
    sum(ρ_mol .* bulk_chemical_potential(nb, ρ_mol, S)) - bulk_free_energy(nb, ρ_mol, S)

"""
    isotropic_nematic_coexistence(nb::NematicBulk, ρ_iso, ρ_nem; x_iso=nothing, S0=0.8,
                                  tol=1e-9, maxit=100)

Isotropic-nematic coexistence: equal chemical potentials of every molecule type and equal
pressure between an isotropic phase (`S = 0`) and a nematic phase (self-consistent `S`).
`ρ_iso`, `ρ_nem` are initial guesses for the molecule densities of the two phases.

For one component the unknowns are the two phase densities. For mixtures, the isotropic
phase composition is held fixed at `x_iso` (mole fractions; default: the composition of
`ρ_iso`) and the unknowns are its total density and the nematic densities — `n + 1`
equations for `n + 1` unknowns.

Solved by Newton's method on log-densities with a finite-difference Jacobian. Returns
`(; ρ_iso, ρ_nem, S_nem, μ, P, converged)`. A nematic phase that relaxes to `S = 0` means
there is no I–N transition near the guess.
"""
function isotropic_nematic_coexistence(nb::NematicBulk, ρ_iso, ρ_nem; x_iso=nothing, S0=0.8,
                                       tol=1e-9, maxit=100)
    n = length(ρ_iso)
    x = x_iso === nothing ? collect(ρ_iso) ./ sum(ρ_iso) : Float64.(collect(x_iso))
    S_warm = Ref{Any}(S0)
    function phases(y)
        ρI = exp(y[1]) .* x
        ρN = exp.(y[2:end])
        sol = bulk_order_parameter(nb, ρN; S0=S_warm[])
        return ρI, ρN, sol.S
    end
    function residual(y; update=false)
        ρI, ρN, SN = phases(y)
        update && (S_warm[] = SN)
        S_iso = zeros(length(SN))
        μI, μN = bulk_chemical_potential(nb, ρI, S_iso), bulk_chemical_potential(nb, ρN, SN)
        PI = sum(ρI .* μI) - bulk_free_energy(nb, ρI, S_iso)
        PN = sum(ρN .* μN) - bulk_free_energy(nb, ρN, SN)
        return vcat(μI .- μN, PI - PN)
    end
    y = vcat(log(sum(ρ_iso)), log.(collect(ρ_nem)))
    converged = false
    for _ in 1:maxit
        r = residual(y; update=true)
        if maximum(abs, r) < tol
            converged = true
            break
        end
        J = zeros(n + 1, n + 1)
        for k in 1:(n + 1)
            h = 1e-6
            yp = copy(y); yp[k] += h
            ym = copy(y); ym[k] -= h
            J[:, k] = (residual(yp) .- residual(ym)) ./ (2h)
        end
        δ = -(J \ r)
        step = 1.0
        while maximum(abs, residual(y .+ step .* δ)) > maximum(abs, r) && step > 1e-3
            step /= 2
        end
        y .+= step .* δ
    end
    ρI, ρN, SN = phases(y)
    μ = bulk_chemical_potential(nb, ρN, SN)
    return (; ρ_iso = ρI, ρ_nem = ρN, S_nem = SN, μ, P = bulk_pressure(nb, ρN, SN), converged)
end

export NematicBulk, orientational_logq, nematic_free_energy, bulk_order_parameter,
       bulk_free_energy, bulk_chemical_potential, bulk_pressure, isotropic_nematic_coexistence
