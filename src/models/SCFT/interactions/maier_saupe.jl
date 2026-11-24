"""
    MaierSaupe(nu)

Maier-Saupe orientational coupling between orientation-resolved species
([`WormLikeChain`](@ref), [`RigidRod`](@ref)). `nu` is a symmetric matrix in species order
(or a `Dict` of species-name pairs); unlike `chi`, a nonzero diagonal `nu[α,α]` is
meaningful (self-alignment). `nu = 5` is the isotropic-nematic linear instability of an
isolated, homogeneous, self-coupled species.

The orientational mean field is `u`-dependent, so it does not enter the scalar field `w`;
it is built from the orientation tensor state `R_tensor` that the solver carries
(see [`compute_maier_saupe_field`](@ref), [`compute_orientation_moments`](@ref)). Its
free-energy contribution is the self-energy minus its double-counting correction,
`U_MS − 2U_MS = −U_MS`, with `U_MS = −(3/(4ρ₀)) Σ_αβ ν_αβ ∫ R^α:R^β dr`.
"""
struct MaierSaupe{M} <: SCFTInteractionModel
    nu::M
end

bind_interaction(int::MaierSaupe, species::Vector{String}) =
    MaierSaupe(_species_matrix(int.nu, species, "MaierSaupe nu"))

function validate_interaction(int::MaierSaupe, model)
    for (α, c) in enumerate(model.chains)
        if !is_oriented(c) && any(!iszero, view(int.nu, α, :))
            throw(ArgumentError(
                "MaierSaupe couples species $(model.groups.flattenedgroups[α]), which uses a " *
                "$(chain_label(c)) representation with no bond orientation. Give it a WormLikeChain " *
                "or RigidRod representation, or set its row of nu to zero."))
        end
    end
    return nothing
end

has_maier_saupe(int::MaierSaupe) = any(!iszero, int.nu)

function field_energy(int::MaierSaupe, ρ, ρ_total, rho0, nd, per_∫, aux)
    R_tensor = get(aux, :R_tensor, nothing)
    R_tensor === nothing && return 0.0
    nu = int.nu
    n = size(nu, 1)
    integrand = zero(ρ_total)
    for α in 1:n
        Rα = selectdim(R_tensor, nd + 1, α)
        for β in 1:n
            iszero(nu[α, β]) && continue
            Rβ = selectdim(R_tensor, nd + 1, β)
            for j in eachindex(_Q_PAIRS)
                integrand .+= (_Q_PAIR_WEIGHT[j] * nu[α, β]) .*
                    Array(selectdim(Rα, nd + 1, j)) .* Array(selectdim(Rβ, nd + 1, j))
            end
        end
    end
    U_MS = -(3 / (4 * rho0)) * per_∫(integrand)
    return -U_MS   # U_MS − wMSρ_sum, with wMSρ_sum = 2 U_MS
end

export MaierSaupe
